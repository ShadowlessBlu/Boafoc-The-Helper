"""The voice loop: wake word -> listen -> transcribe -> Brain -> speak. Logs per-stage timings."""
from __future__ import annotations

import csv
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from .audio import Microphone, beep, make_vad, play_pcm, write_wav
from .fixups import fix_transcript, is_phantom

log = logging.getLogger(__name__)

_YES = re.compile(r"\b(yes|yeah|yep|sure|ok|okay|confirm|do it|go ahead)\b")
_NO = re.compile(r"\b(no|nope|don't|do not|cancel|stop)\b")


def parse_yes_no(text: str) -> bool:
    """Only a clear yes counts. Anything unclear or mixed is treated as no."""
    t = text.lower()
    return bool(_YES.search(t)) and not _NO.search(t)


class Speaker:
    """Synthesises sentence N+1 while sentence N is playing."""

    def __init__(self, tts, device: str):
        self.tts, self.device = tts, device

    def say(self, text: str, sentences: list[str]) -> float:
        """Speak; returns seconds until the first audio started."""
        return self.say_iter(iter(sentences))

    def say_iter(self, sentences, on_first=None) -> float:
        """Speak sentences as they arrive (e.g. from a reply the model is still writing): sentence
        N+1 is synthesised while N plays. Returns seconds until the first audio started."""
        t0 = time.perf_counter()
        q: queue.Queue = queue.Queue(maxsize=2)

        def producer():
            try:
                for s in sentences:
                    if s.strip():
                        q.put(self.tts.synthesize(s))
            except Exception:  # noqa: BLE001 - never leave the player waiting forever
                log.exception("speaking failed")
            finally:
                q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        first = None
        while (pcm := q.get()) is not None:
            if first is None:
                first = time.perf_counter() - t0
                if on_first is not None:
                    on_first()
            play_pcm(pcm, self.tts.sample_rate, self.device)
        return first or 0.0


class VoiceLoop:
    def __init__(self, brain, wake, stt, tts, cfg, lock: threading.Lock | None = None, status=None,
                 wake_control=None):
        from ..status import Status
        self.brain, self.wake, self.stt, self.cfg = brain, wake, stt, cfg
        self.wake_control = wake_control   # logs what each wake word did (wake_control.py)
        self.status = status or Status()  # what the face on the screen shows
        self.speaker = Speaker(tts, cfg.speaker_device)
        self.lock = lock or threading.Lock()
        self.vad = make_vad(cfg.vad_engine, cfg.vad_threshold, cfg.silero_threshold,
                            sample_rate=cfg.sample_rate, silence_ms=cfg.silence_ms,
                            max_s=cfg.max_utterance_s, start_timeout_s=cfg.listen_timeout_s)
        self.timing_path = Path(cfg.timing_log)
        # in RAM (/run/user/<uid>), never in the project folder: it's a recording of the user
        run_dir = Path(f"/run/user/{os.getuid()}") if hasattr(os, "getuid") else None
        self.keep_last = str(run_dir / "assistant_last_question.wav") if run_dir and run_dir.is_dir() else ""

    def _speak(self, text: str) -> float:
        from .engines import split_sentences
        self.status.set("speaking")
        try:
            return self.speaker.say(text, split_sentences(text))
        finally:
            self.status.set("idle")

    def _live(self, mic: Microphone):
        """Microphone audio, except while muted from the screen (POST /mute): then it is read and
        thrown away, so nothing can wake the assistant or be heard."""
        for c in mic.chunks():
            if self.status.get("muted"):
                continue
            yield c

    def _listen(self, mic: Microphone) -> tuple[str, float, float]:
        t0 = time.perf_counter()
        self.status.set("listening")
        pcm = self.vad.collect(self._live(mic))
        self.status.set("thinking")
        if pcm and self.keep_last:
            try:  # the last question's audio, overwritten each time, to compare speech-to-text engines
                write_wav(self.keep_last, pcm, self.cfg.sample_rate)
            except OSError:
                pass
        t1 = time.perf_counter()
        text = self.stt.transcribe(pcm) if pcm else ""
        return text, len(pcm) / 2 / self.cfg.sample_rate, time.perf_counter() - t1

    def _log(self, row: dict) -> None:
        self.timing_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.timing_path.exists()
        with self.timing_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def _wake_event(self, kind: str, word: str, score, heard: str = "") -> None:
        if self.wake_control is not None and word:
            try:
                self.wake_control.record(kind, word, float(score) if score is not None else None, heard)
            except Exception:  # noqa: BLE001 - the log must never stop the assistant answering
                log.exception("wake log failed")

    def run_once(self, mic: Microphone) -> None:
        self.status.set("idle", voice=True)
        score = self.wake.wait(self._live(mic))
        word = getattr(self.wake, "last_word", "")
        beep(self.cfg.sample_rate, device=self.cfg.speaker_device)
        text, audio_s, stt_s = self._listen(mic)
        fixed = fix_transcript(text)
        if fixed != text:
            log.info("heard %r, corrected to %r", text, fixed)
            text = fixed
        if text and is_phantom(text):
            log.info("heard %r, treated as nothing (Whisper writes this for noise)", text)
            self._wake_event("phantom", word, score)
            self.status.set("idle")
            return
        if not text:
            log.info("heard nothing")
            self._wake_event("no_speech", word, score)
            self.status.set("idle")
            return
        self._wake_event("replied", word, score, text)
        self.status.set("thinking", heard=text, said="", first_word_s=None)
        log.info("user: %s", text)

        def confirm(question: str) -> bool:
            self._speak(question.replace("_", " ") + " Say yes or no.")
            answer, _, _ = self._listen(mic)
            log.info("confirm answer: %r", answer)
            self.status.set("thinking")  # the brain carries on after the answer
            return parse_yes_no(answer)

        # Stream: the first sentence is spoken while the model is still writing the rest.
        t0 = time.perf_counter()
        first_ready: list[float] = []

        def timed(sentences):
            spoken = []
            for sentence in sentences:
                if not first_ready:
                    first_ready.append(time.perf_counter() - t0)
                spoken.append(sentence)
                self.status.set(said=" ".join(spoken))   # the screen shows the reply as it is spoken
                yield sentence

        with self.lock:
            first_audio_s = self.speaker.say_iter(timed(self.brain.respond(text, confirm=confirm)),
                                                  on_first=lambda: self.status.set("speaking"))
        reply = getattr(self.brain, "last_reply", "")
        log.info("bot: %s", reply)
        # The Bluetooth speaker is still playing for ~1.5 s after pw-cat returns; then drop everything
        # the mic recorded meanwhile, so the wake word never hears the assistant's own voice.
        time.sleep(1.6)
        dropped = mic.drain()
        log.debug("dropped %.1f s of audio recorded while speaking", dropped / 2 / self.cfg.sample_rate)
        self.status.set("idle", said=reply)
        brain_s = first_ready[0] if first_ready else first_audio_s  # until the first sentence was ready
        self.status.set(first_word_s=round(stt_s + first_audio_s, 1))
        self._log({"timestamp": datetime.now().isoformat(timespec="seconds"), "text": text,
                   "audio_s": round(audio_s, 2), "stt_s": round(stt_s, 2), "brain_s": round(brain_s, 2),
                   "tts_first_audio_s": round(max(0.0, first_audio_s - brain_s), 2),
                   "end_of_speech_to_first_audio_s": round(stt_s + first_audio_s, 2)})

    def run_forever(self) -> None:
        while True:
            try:
                with Microphone(self.cfg.mic_device, self.cfg.sample_rate) as mic:
                    while True:
                        self.run_once(mic)
            except KeyboardInterrupt:
                return
            except Exception:  # keep the assistant alive; systemd also restarts on crash
                self.status.set("idle", voice=False)
                log.exception("voice loop error, restarting microphone in 2 s")
                time.sleep(2)


def build_voice(cfg, wake_control=None):
    """Create the wake word, STT and TTS engines from VoiceConfig."""
    from .engines import OpenWakeWord, PiperTTS, PressEnter, VoskSTT, WhisperCppSTT
    if cfg.wake_engine == "personal":
        from .audio import SileroVAD
        from .personal_wake import PersonalWakeWord
        words = [w.strip().lower() for w in cfg.wake_word.split(",") if w.strip()]
        paths = [Path(cfg.personal_wake_dir) / f"{w}.npz" for w in words]
        missing = [str(p) for p in paths if not p.exists()]
        if not words or missing:
            raise FileNotFoundError(f"personal wake word not recorded yet: {missing or cfg.wake_word!r} "
                                    "(run python -m assistant.voice.enroll <word>)")
        wake = PersonalWakeWord(paths, vad=SileroVAD(0.5))
    elif cfg.wake_engine != "openwakeword":
        raise ValueError(f"unknown wake_engine {cfg.wake_engine!r}")
    else:
        vad = None
        if cfg.wake_word and cfg.vad_engine == "silero":
            try:
                from .audio import SileroVAD
                vad = SileroVAD(0.5)
            except (ImportError, OSError):
                vad = None
        wake = (OpenWakeWord(cfg.wake_word, cfg.wake_threshold, vad=vad, patience=cfg.wake_patience,
                             control=wake_control)
                if cfg.wake_word else PressEnter())
    if cfg.stt_engine == "whisper":
        stt = WhisperCppSTT(cfg.whisper_cmd, cfg.whisper_model, cfg.sample_rate, cfg.whisper_prompt,
                            cfg.whisper_fit_window)
    elif cfg.stt_engine == "vosk":
        stt = VoskSTT(cfg.vosk_model, cfg.sample_rate)
    else:
        raise ValueError(f"unknown stt_engine {cfg.stt_engine!r}")
    tts = PiperTTS(cfg.piper_cmd, cfg.piper_model)
    return wake, stt, tts
