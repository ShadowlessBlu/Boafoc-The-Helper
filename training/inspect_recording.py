"""What's in a training recording? Length, how much is speech (Silero), loudness and speech over
time, and a few short Whisper transcripts, to check it before it's used for training.

  .venv/bin/python training/inspect_recording.py ~/wake-train/real/talk1.wav [--samples 30 120 200]
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.voice.audio import SileroVAD, write_wav  # noqa: E402
from assistant.voice.engines import clean_whisper_text, resolve_cmd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", type=Path)
    ap.add_argument("--samples", type=float, nargs="*", default=[])
    a = ap.parse_args()
    with wave.open(str(a.wav)) as w:
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    vad = SileroVAD(0.5)
    p = np.array([vad.speech_probability(raw[i:i + 960]) for i in range(0, len(raw) - 959, 960)])
    print(f"{a.wav.name}: {len(x) / 16000:.0f} s, speech in {np.mean(p >= 0.5):.0%} (mean prob {p.mean():.2f}), "
          f"RMS {np.sqrt(np.mean(x ** 2)):.0f}")
    step = 10 * 16000
    print("per 10 s, loudness:", " ".join(f"{np.sqrt(np.mean(x[i:i + step] ** 2)):.0f}" for i in range(0, len(x), step)))
    k = 333  # ~10 s of 30 ms chunks
    print("per 10 s, speech:  ", " ".join(f"{p[i:i + k].mean():.2f}" for i in range(0, len(p), k)))
    cfg = load_config(ROOT / "config.toml").voice
    for t in a.samples:
        seg = raw[int(t * 16000) * 2:int((t + 8) * 16000) * 2]
        if not seg:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        write_wav(path, seg, 16000)
        cmd = resolve_cmd(shlex.split(cfg.whisper_cmd.format(model=cfg.whisper_model, wav=path)))
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        Path(path).unlink()
        print(f"at {t:.0f} s: {clean_whisper_text(out)!r}")


if __name__ == "__main__":
    main()
