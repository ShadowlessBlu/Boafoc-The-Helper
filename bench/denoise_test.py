"""Does removing the fan's steady noise help speech-to-text? (experiment, numpy only)

Spectral gating: learn the noise spectrum from the parts Silero says are not speech, then
turn down every frequency bin that isn't clearly above that noise. Writes the cleaned
audio next to the input and runs Whisper (with the assistant's hint) on both.

  .venv/bin/python bench/denoise_test.py [file.wav] [--strength 1.5]
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.voice.audio import SileroVAD, write_wav  # noqa: E402
from assistant.voice.engines import clean_whisper_text, resolve_cmd  # noqa: E402

N, HOP = 512, 128


def stft(x: np.ndarray) -> np.ndarray:
    w = np.hanning(N)
    frames = [x[i:i + N] * w for i in range(0, len(x) - N, HOP)]
    return np.fft.rfft(np.array(frames), axis=1)


def istft(S: np.ndarray, length: int) -> np.ndarray:
    w = np.hanning(N)
    out, norm = np.zeros(length), np.zeros(length)
    for k, frame in enumerate(np.fft.irfft(S, n=N, axis=1)):
        i = k * HOP
        out[i:i + N] += frame * w
        norm[i:i + N] += w ** 2
    return out / np.maximum(norm, 1e-8)


def denoise(x: np.ndarray, noise_mask: np.ndarray, strength: float) -> np.ndarray:
    """x: float samples; noise_mask: per STFT frame, True where there is no speech."""
    S = stft(x)
    mag = np.abs(S)
    noise = mag[noise_mask].mean(axis=0) if noise_mask.any() else np.percentile(mag, 20, axis=0)
    gain = np.clip((mag - strength * noise) / np.maximum(mag, 1e-8), 0.1, 1.0)  # floor: keep 10%
    return istft(S * gain, len(x))


def whisper(cfg, wav: str) -> tuple[str, float]:
    cmd = resolve_cmd(shlex.split(cfg.whisper_cmd.format(model=cfg.whisper_model, wav=wav)))
    if cfg.whisper_prompt:
        cmd += ["--prompt", cfg.whisper_prompt]
    t = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    return clean_whisper_text(out), time.perf_counter() - t


def main() -> None:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", default=f"/run/user/{uid}/assistant_last_question.wav")
    ap.add_argument("--strength", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    a = ap.parse_args()
    cfg = load_config(ROOT / "config.toml").voice
    with wave.open(a.wav) as w:
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)

    vad = SileroVAD(0.5)
    probs = np.array([vad.speech_probability(raw[i:i + 960]) for i in range(0, len(raw) - 959, 960)])
    # STFT frame k covers samples k*HOP .. k*HOP+N; map it to the 30 ms VAD chunk at its centre
    centres = (np.arange(0, len(x) - N, HOP) + N // 2) // 480
    mask = probs[np.minimum(centres, len(probs) - 1)] < 0.2
    print(f"{a.wav}: {len(x) / 16000:.1f} s, {mask.mean():.0%} of frames used as the noise sample")

    text, dt = whisper(cfg, a.wav)
    print(f"original         {dt:4.1f} s  {text!r}")
    for s in a.strength:
        t = time.perf_counter()
        y = denoise(x, mask, s)
        dn = time.perf_counter() - t
        out = str(Path(a.wav).with_name(Path(a.wav).stem + f"_clean{s}.wav"))
        write_wav(out, np.clip(y, -32768, 32767).astype(np.int16).tobytes(), 16000)
        text, dt = whisper(cfg, out)
        print(f"cleaned x{s:<4}    {dt:4.1f} s  {text!r}   (filter {dn * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
