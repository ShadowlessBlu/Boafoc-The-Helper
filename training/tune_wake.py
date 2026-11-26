"""Pick a threshold and "patience" for each wake word from real recordings.

Streams held-out audio through the real openWakeWord detector, frame by frame (80 ms) like the
assistant, and counts:
  hits         of the user's 5 held-out examples (not used in training), each placed in room noise
  false wakes  in the held-out last minute of real talk
for thresholds x patience (frames in a row above the threshold before it wakes).

  python training/tune_wake.py --work C:/Users/DELL/wake-train
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train_wake_word as t  # noqa: E402

SR = 16000


def scores(model_path: Path, audio: np.ndarray) -> np.ndarray:
    from openwakeword.model import Model
    m = Model(wakeword_model_paths=[str(model_path)])
    return np.array([max(m.predict(audio[i:i + 1280]).values()) for i in range(0, len(audio) - 1280, 1280)])


def wakes(sc: np.ndarray, thr: float, patience: int, refractory: int = 25) -> list[int]:
    """Frames where the assistant would wake: `patience` frames in a row >= thr; then 2 s pause."""
    out, run, last = [], 0, -10**9
    for i, s in enumerate(sc):
        run = run + 1 if s >= thr else 0
        if run >= patience and i - last > refractory:
            out.append(i)
            last, run = i, 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path.home() / "wake-train")
    ap.add_argument("--models", type=Path, default=ROOT.parent / "models" / "wake")
    a = ap.parse_args()
    real = a.work / "real2"
    with wave.open(str(a.work / "room_noise.wav")) as w:
        noise = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    with wave.open(str(real / "talk.wav")) as w:
        talk = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    talk_test = talk[len(talk) - 60 * SR:]
    rng = np.random.default_rng(1)
    for word in ("uzi", "aiya", "boafo"):
        held = t.real_word_clips(real, word)[-5:]
        # each example in 3 s of room noise at the level it was recorded (no rescaling)
        parts = []
        for c in held:
            bg = noise[rng.integers(0, len(noise) - 3 * SR):][:3 * SR].copy()
            bg[SR:SR + len(c)] += c
            parts.append(bg)
        pos_audio = np.clip(np.concatenate(parts), -32768, 32767).astype(np.int16)
        model = a.models / f"{word}.onnx"
        sp, sn = scores(model, pos_audio), scores(model, talk_test)
        seg = 3 * SR // 1280
        print(f"\n{word}: hits of 5 held-out examples / false wakes in 1 min of real talk")
        print("  thr   " + "  ".join(f"patience {p}" for p in (1, 2, 3, 4)))
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            row = []
            for p in (1, 2, 3, 4):
                hit = len({f // seg for f in wakes(sp, thr, p)})
                row.append(f"{hit}/5 , {len(wakes(sn, thr, p)):>2}  ")
            print(f"  {thr:<5} " + "  ".join(row))
        print(f"  frames >= 0.5 per example: {[int((sp[i * seg:(i + 1) * seg] >= 0.5).sum()) for i in range(5)]}")


if __name__ == "__main__":
    main()
