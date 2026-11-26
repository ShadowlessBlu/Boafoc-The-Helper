"""Teach the assistant a personal wake word from your own voice (see personal_wake.py).

  python -m assistant.voice.enroll uzi aiya boafo [--count 5] [--config config.toml]

For each word, the speaker asks you to say it `count` times, then to say one normal sentence
(that is how other speech will sound, so the threshold can sit between the two). Saves
models/wake/<word>.npz: speech fingerprints only, no audio. Stop the assistant first
(`systemctl --user stop assistant`): it holds the microphone.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .audio import Microphone, SileroVAD, play_pcm
from .personal_wake import LENGTH, save_word, suggest_threshold

RATE = 16000
CHUNK = 960            # 30 ms, the size Silero scores
CONTEXT_S = 1.6        # audio fed to the feature model per example, ending just after the word
TAIL_S = 0.2           # kept after the last speech
PROMPT_LAG_S = 1.7     # the Bluetooth speaker is still saying the prompt for ~1.5 s after pw-cat returns


def record(device: str, seconds: float) -> bytes:
    with Microphone(device, RATE) as mic:
        out, need = b"", int(seconds * RATE) * 2
        for c in mic.chunks():
            out += c
            if len(out) >= need:
                return out[:need]
    return out


def speech_probs(vad: SileroVAD, pcm: bytes) -> np.ndarray:
    vad.reset()
    return np.array([vad.speech_probability(pcm[i:i + CHUNK]) for i in range(0, len(pcm) - CHUNK + 1, CHUNK)])


def example_clip(pcm: bytes, probs: np.ndarray) -> np.ndarray | None:
    """CONTEXT_S of audio ending TAIL_S after the last speech; padded at the front with this
    recording's own background (non-speech chunks), so the padding sounds like the room."""
    speech = np.flatnonzero(probs >= 0.5)
    if len(speech) == 0:
        return None
    x = np.frombuffer(pcm, dtype=np.int16)
    end = min(len(x), (speech[-1] + 1) * (CHUNK // 2) + int(TAIL_S * RATE))
    start = end - int(CONTEXT_S * RATE)
    clip = x[max(0, start):end]
    if start < 0:
        quiet = [x[i * (CHUNK // 2):(i + 1) * (CHUNK // 2)] for i in np.flatnonzero(probs < 0.2)]
        bg = np.concatenate(quiet) if quiet else np.zeros(CHUNK // 2, dtype=np.int16)
        pad = np.resize(bg, -start)
        clip = np.concatenate([pad, clip])
    return clip.astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="+")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--out", type=Path, default=Path("models/wake"))
    a = ap.parse_args()

    from openwakeword.utils import AudioFeatures

    from ..config import load_config
    from .engines import PiperTTS
    cfg = load_config(a.config).voice
    tts = PiperTTS(cfg.piper_cmd, cfg.piper_model)
    vad = SileroVAD(0.5)
    feats = AudioFeatures()

    def prompt(text: str) -> None:
        print(text, flush=True)
        play_pcm(tts.synthesize(text), tts.sample_rate, cfg.speaker_device)
        time.sleep(2.5)  # the Bluetooth speaker plays ~1.5 s after pw-cat returns

    def ask_and_record(text: str, seconds: float) -> bytes:
        """Record from the moment the prompt is sent: people answer as soon as they hear it,
        which on Bluetooth is ~1.5 s later. The part where the speaker is still talking is dropped."""
        print(text, flush=True)
        play_pcm(tts.synthesize(text), tts.sample_rate, cfg.speaker_device)
        pcm = record(cfg.mic_device, PROMPT_LAG_S + seconds)
        return pcm[int(PROMPT_LAG_S * RATE) * 2:]

    for word in a.words:
        prompt(f"Let's record the wake word {word}. Say it once each time I ask.")
        templates = []
        tries = 0
        while len(templates) < a.count and tries < a.count + 3:
            tries += 1
            pcm = ask_and_record(f"Say {word}." if templates else f"Say {word}, now.", 3.0)
            clip = example_clip(pcm, speech_probs(vad, pcm))
            if clip is None:
                print("  (heard no speech, asking again)")
                continue
            f = feats.embed_clips(clip[None, :], batch_size=1)[0]
            templates.append(f[-LENGTH:])
            print(f"  example {len(templates)}: ok")
        if len(templates) < 2:
            prompt(f"Sorry, I couldn't hear {word} clearly. Skipping it.")
            continue
        pcm = ask_and_record("Thank you. Now say a normal sentence, for example: what's the weather like today.", 5.0)
        other = feats.embed_clips(np.frombuffer(pcm, dtype=np.int16)[None, :], batch_size=1)[0]
        threshold, stats = suggest_threshold(templates, other)
        path = a.out / f"{word.lower()}.npz"
        save_word(path, templates, threshold, stats)
        print(f"{word}: {len(templates)} examples, threshold {threshold:.2f} "
              f"(examples match each other {stats['match_median']:.2f}, worst {stats['match_min']:.2f}; "
              f"other speech up to {stats['other_speech_max']:.2f}) -> {path}")
        if stats["other_speech_max"] >= stats["match_median"]:
            print(f"  warning: other speech matched {word} as well as your examples did; "
                  "it may wake by mistake. Try a longer word, or record again closer to the mic.")
    prompt("All done.")


if __name__ == "__main__":
    main()
