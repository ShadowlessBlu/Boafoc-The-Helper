"""Cut a recording of one word said many times into one clip per utterance (Silero finds them).

  .venv/bin/python training/cut_examples.py ~/wake-train/real/word_uzi.wav ~/wake-train/real/uzi

Each burst of speech (gaps under 0.3 s joined) between 0.15 and 1.6 s long is saved as
NN.wav (just the word, plus 50 ms either side). Longer bursts are probably not the word alone
and are skipped (listed). Prints a summary and, with --check, what Whisper hears in a few.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.voice.audio import SileroVAD, write_wav  # noqa: E402
from assistant.voice.engines import clean_whisper_text, resolve_cmd  # noqa: E402

CHUNK = 480  # samples (30 ms)


def bursts(p: np.ndarray, gap: int = 10, threshold: float = 0.5) -> list[tuple[int, int]]:
    """(start, end) chunk indices of speech, joining gaps shorter than `gap` chunks."""
    out: list[list[int]] = []
    for i in np.flatnonzero(p >= threshold):
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(s, e + 1) for s, e in out]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", type=int, default=5, help="transcribe this many clips with Whisper")
    a = ap.parse_args()
    with wave.open(str(a.wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    vad = SileroVAD(0.5)
    raw = x.tobytes()
    p = np.array([vad.speech_probability(raw[i:i + CHUNK * 2]) for i in range(0, len(raw) - CHUNK * 2 + 1, CHUNK * 2)])
    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("*.wav"):
        old.unlink()
    kept, skipped = [], []
    pad = 800  # 50 ms
    for s, e in bursts(p):
        dur = (e - s) * CHUNK / 16000
        if not 0.15 <= dur <= 1.6:
            skipped.append(round(dur, 2))
            continue
        clip = x[max(0, s * CHUNK - pad):min(len(x), e * CHUNK + pad)]
        path = a.out / f"{len(kept):02d}.wav"
        write_wav(str(path), clip.tobytes(), 16000)
        kept.append((path, dur))
    print(f"{a.wav.name}: {len(kept)} examples kept "
          f"(length {min((d for _, d in kept), default=0):.2f}-{max((d for _, d in kept), default=0):.2f} s), "
          f"{len(skipped)} skipped {skipped[:8]}")
    cfg = load_config(ROOT / "config.toml").voice
    for path, _ in kept[:: max(1, len(kept) // max(1, a.check))][: a.check]:
        cmd = resolve_cmd(shlex.split(cfg.whisper_cmd.format(model=cfg.whisper_model, wav=str(path))))
        print(f"  {path.name}: {clean_whisper_text(subprocess.run(cmd, capture_output=True, text=True).stdout)!r}")


if __name__ == "__main__":
    main()
