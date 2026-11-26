"""Wrappers for the wake-word, speech-to-text and text-to-speech engines."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

from .audio import write_wav

log = logging.getLogger(__name__)


def resolve_cmd(cmd: list[str]) -> list[str]:
    """Find the program when it isn't on PATH: `.venv/bin/python -m assistant` doesn't put .venv/bin
    (where pip installs `piper`) on PATH, and services don't get ~/.local/bin (`whisper-cli`)."""
    if not cmd or os.sep in cmd[0] or shutil.which(cmd[0]):
        return cmd
    for d in (Path(sys.executable).parent, Path.home() / ".local" / "bin"):
        if (d / cmd[0]).exists():
            return [str(d / cmd[0])] + cmd[1:]
    return cmd


# ---------------- wake word ----------------
def wake_model_kwargs(params, name: str, models_dir: Path) -> dict:
    """Arguments for openwakeword.model.Model, for both API versions.

    0.5+ takes model names (`wakeword_models`). 0.4 (what pip installs on the Pi with Python 3.13,
    as newer versions need tflite-runtime) only takes ONNX file paths (`wakeword_model_paths`):
    "hey_jarvis" -> resources/models/hey_jarvis_v0.1.onnx.
    """
    if "wakeword_models" in params:
        return {"wakeword_models": [name], "inference_framework": "onnx"}
    if name.endswith(".onnx") or os.sep in name or "/" in name:
        return {"wakeword_model_paths": [name]}
    found = sorted(models_dir.glob(f"{name}_v*.onnx")) or sorted(models_dir.glob(f"{name}*.onnx"))
    if not found:
        raise FileNotFoundError(f"no openWakeWord model '{name}' in {models_dir}")
    return {"wakeword_model_paths": [str(found[-1])]}


class OpenWakeWord:
    """openWakeWord expects 16 kHz int16 frames of 1280 samples (80 ms).

    `model_name`: a built-in model ("hey_jarvis") or a trained .onnx file, or several separated by
    commas ("models/wake/uzi.onnx,models/wake/boafo.onnx"); `.last_word` says which one fired.
    With `vad` (Silero), a detection only counts if someone spoke within the last second, so
    fan noise can't wake the assistant (a trained model fired on room noise in testing).
    """
    FRAME_BYTES = 1280 * 2

    def __init__(self, model_name: str, threshold: float = 0.5, vad=None, patience: int = 1, control=None):
        import inspect

        import numpy as np
        import openwakeword
        from openwakeword.model import Model
        self.np = np
        params = inspect.signature(Model.__init__).parameters
        models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        kwargs: dict = {}
        self.thresholds: dict[str, float] = {}   # per word, from "file.onnx@0.8"
        for entry in (n.strip() for n in model_name.split(",") if n.strip()):
            name, _, thr = entry.partition("@")
            if thr:
                self.thresholds[Path(name).stem] = float(thr)
            for k, v in wake_model_kwargs(params, name, models_dir).items():
                kwargs[k] = kwargs.get(k, []) + v if isinstance(v, list) else v
        self.model = Model(**kwargs)
        self.threshold = threshold
        self.vad = vad
        self.last_word = ""
        # Frames (80 ms) in a row above the threshold before waking. Real wake words stay above it
        # for 6-12 frames; chance matches in conversation spike for one (training/tune_wake.py).
        self.patience = max(1, patience)
        # WakeControl (wake_control.py): words switched off and thresholds changed on the screen,
        # and the event log. Without it, every word is on with its threshold from config.toml.
        self.control = control

    def _on(self, key: str) -> bool:
        c = getattr(self, "control", None)
        if c is None:
            return True
        word = c.word_for(key)
        return word is None or c.enabled(word)

    def _record(self, kind: str, key: str, score: float) -> None:
        c = getattr(self, "control", None)
        if c is not None:
            c.record(kind, key, score)

    def _keep(self, frames: list[bytes]) -> None:
        """Save the audio behind this wake-up (RAM only, overwritten each time), to see later what
        triggered a false wake: /run/user/<uid>/assistant_last_wake.wav"""
        if not hasattr(os, "getuid"):
            return
        run_dir = Path(f"/run/user/{os.getuid()}")
        if run_dir.is_dir():
            try:
                write_wav(str(run_dir / "assistant_last_wake.wav"), b"".join(frames), 16000)
            except OSError:
                pass

    def threshold_for(self, word: str) -> float:
        """Model names are file stems ("boafo"; built-ins like "hey_jarvis_v0.1")."""
        c = getattr(self, "control", None)
        if c is not None and c.word_for(word) is not None:
            return c.threshold(c.word_for(word))
        for stem, thr in getattr(self, "thresholds", {}).items():
            if word == stem or word.startswith(stem):
                return thr
        return self.threshold

    NEAR_MISS = 0.15  # scores above this but below the threshold are logged, to tune the threshold

    def wait(self, chunks: Iterator[bytes]) -> float:
        """Block until a wake word is heard; returns its score (the word is in .last_word)."""
        buf = b""
        peak_word, peak, quiet_frames = "", 0.0, 0
        speech_until, ignored_at = 0.0, -9.0
        recent: list[bytes] = []       # the last ~3 s the detector heard, saved when it fires
        runs: dict[str, int] = {}
        if self.last_word:
            # After a wake-up the feature buffer still holds that wake word, which re-fired at ~1.00
            # within 0.4 s of listening again (2026-11-20/23). openWakeWord 0.4 has reset(), but it only
            # clears the score history, not the audio features, so this flush must always run:
            # 1.6 s of silence pushes the old word out (bench/wake_flush_test.py: 0.02 afterwards).
            silence = self.np.zeros(1280, dtype=self.np.int16)
            for _ in range(20):
                self.model.predict(silence)
        for c in chunks:
            if self.vad is not None and self.vad.is_speech(c):
                speech_until = time.monotonic() + 1.0
            buf += c
            while len(buf) >= self.FRAME_BYTES:
                frame, buf = buf[: self.FRAME_BYTES], buf[self.FRAME_BYTES:]
                recent = (recent + [frame])[-40:]
                scores = self.model.predict(self.np.frombuffer(frame, dtype=self.np.int16))
                scores = {w: s for w, s in scores.items() if self._on(w)}   # words switched off
                if not scores:
                    continue
                above = {w: s for w, s in scores.items() if s >= self.threshold_for(w)}
                runs = {w: runs.get(w, 0) + 1 for w in above}
                hits = {w: s for w, s in above.items() if runs[w] >= getattr(self, "patience", 1)}
                if hits:
                    word = max(hits, key=hits.get)
                    if self.vad is not None and time.monotonic() > speech_until:
                        if time.monotonic() - ignored_at > 1.0:  # noise, not speech: never a wake
                            log.info("wake word ignored, no speech heard: %s, score %.2f", word, hits[word])
                            self._record("ignored", word, hits[word])
                            ignored_at = time.monotonic()
                        continue
                    self.last_word = word
                    self._keep(recent)
                    log.info("wake word: %s, score %.2f (threshold %.2f)", word, hits[word],
                             self.threshold_for(word))
                    if hasattr(self.model, "reset"):  # not in openWakeWord 0.4
                        self.model.reset()
                    return hits[word]
                word = max(scores, key=scores.get)
                if scores[word] >= self.NEAR_MISS:
                    if scores[word] > peak:
                        peak_word, peak = word, scores[word]
                    quiet_frames = 0
                elif peak:
                    quiet_frames += 1
                    if quiet_frames >= 12:  # ~1 s after the attempt: log its peak once
                        log.info("wake word near miss: %s, score %.2f (threshold %.2f)", peak_word, peak,
                                 self.threshold_for(peak_word))
                        self._record("near_miss", peak_word, peak)
                        peak_word, peak, quiet_frames = "", 0.0, 0
        raise EOFError("microphone stream ended")


class PressEnter:
    """Fallback 'wake word' for testing without openWakeWord."""

    def wait(self, chunks) -> float:
        input("[press Enter and speak] ")
        return 1.0


# ---------------- speech to text ----------------
class VoskSTT:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Vosk model not found: {model_path} (run scripts/install_voice.sh)")
        self.model = Model(model_path)
        self.rate = sample_rate
        self.Recognizer = KaldiRecognizer

    def transcribe(self, pcm: bytes) -> str:
        rec = self.Recognizer(self.model, self.rate)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()


def audio_ctx_for(pcm_bytes: int, sample_rate: int = 16000, floor: int = 768) -> int:
    """whisper.cpp encoder window for this much 16-bit audio: 50 positions per second plus a margin
    (1500 = the full 30 s), at least `floor`. Encoding a 3 s question in the full window took 1.6 s
    on the Pi. Tested on 8 real clips (bench/whisper_ctx_check.py, 2026-11-20): sized to the clip,
    0.6-1.3 s but worse words and repetition loops; with a 768 floor, ~0.9 s faster on average but
    still 2 clearly worse transcripts and one 4 s outlier. So it is off by default."""
    seconds = pcm_bytes / 2 / sample_rate
    return min(1500, max(floor, math.ceil(seconds * 50) + 64))


class WhisperCppSTT:
    def __init__(self, cmd_template: str, model_path: str, sample_rate: int = 16000, prompt: str = "",
                 fit_window: bool = False):
        self.template, self.model, self.rate, self.prompt = cmd_template, model_path, sample_rate, prompt
        self.fit_window = fit_window

    def command(self, wav: str, pcm_bytes: int = 0) -> list[str]:
        cmd = resolve_cmd(shlex.split(self.template.format(model=self.model, wav=wav)))
        cmd += ["--prompt", self.prompt] if self.prompt else []
        if self.fit_window and pcm_bytes:
            cmd += ["-ac", str(audio_ctx_for(pcm_bytes, self.rate))]
        return cmd

    def transcribe(self, pcm: bytes) -> str:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            write_wav(wav, pcm, self.rate)
            out = subprocess.run(self.command(wav, len(pcm)), capture_output=True, text=True, timeout=120).stdout
        finally:
            os.unlink(wav)
        return clean_whisper_text(out)


def clean_whisper_text(out: str) -> str:
    text = " ".join(line.strip() for line in out.splitlines() if line.strip())
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)  # [BLANK_AUDIO], (music) ...
    return re.sub(r"\s+", " ", text).strip()


# ---------------- text to speech ----------------
def piper_pcm(voice, text: str) -> bytes:
    """Raw 16-bit mono PCM from a loaded PiperVoice, for both piper-tts APIs."""
    if hasattr(voice, "synthesize_stream_raw"):  # piper-tts 1.2
        return b"".join(voice.synthesize_stream_raw(text))
    return b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))  # 1.3+


class PiperTTS:
    """Text to speech with Piper.

    The voice is loaded once and kept in memory (the piper-tts Python package). Starting the
    `piper` program for every sentence reloaded the voice each time: ~1 s per sentence on the Pi 5
    (speaker test, 2026-11-19). The program is only used when the package can't be imported.
    """

    def __init__(self, cmd_template: str, model_path: str, keep_loaded: bool = True):
        self.cmd = resolve_cmd(shlex.split(cmd_template.format(model=model_path)))
        cfg = Path(model_path + ".json")
        self.sample_rate = 22050
        if cfg.exists():
            self.sample_rate = json.loads(cfg.read_text(encoding="utf-8"))["audio"]["sample_rate"]
        self.voice = None
        if keep_loaded:
            try:
                from piper import PiperVoice
                self.voice = PiperVoice.load(model_path)
            except ImportError:
                pass  # no package in this Python: use the program

    def synthesize(self, text: str) -> bytes:
        if self.voice is not None:
            return piper_pcm(self.voice, text)
        r = subprocess.run(self.cmd, input=text.encode(), capture_output=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace")[-300:])
        return r.stdout


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    """Split a reply so the first sentence can be spoken while the rest is synthesised."""
    parts = [p.strip() for p in _SENT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])
