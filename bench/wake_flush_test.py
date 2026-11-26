"""Does feeding digital silence (zeros) to the wake-word models make them fire? (self-wake hunt)

Feeds the live models: 1) real room noise, 2) 20 frames of zeros (what the post-wake flush did),
3) room noise again, printing the highest score per phase and word.
  .venv/bin/python bench/wake_flush_test.py
"""
from __future__ import annotations

import inspect
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.voice.engines import wake_model_kwargs  # noqa: E402


def main() -> None:
    import openwakeword
    from openwakeword.model import Model
    cfg = load_config(ROOT / "config.toml").voice
    params = inspect.signature(Model.__init__).parameters
    models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    kwargs: dict = {}
    for entry in cfg.wake_word.split(","):
        name = entry.strip().partition("@")[0]
        for k, v in wake_model_kwargs(params, name, models_dir).items():
            kwargs[k] = kwargs.get(k, []) + v if isinstance(v, list) else v
    m = Model(**kwargs)
    with wave.open(str(Path.home() / "wake-train" / "room_noise.wav")) as w:
        noise = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    def run(label: str, frames: list[np.ndarray]) -> None:
        peak: dict[str, float] = {}
        runs: dict[str, int] = {}
        longest: dict[str, int] = {}
        for f in frames:
            for word, s in m.predict(f).items():
                peak[word] = max(peak.get(word, 0.0), s)
                runs[word] = runs.get(word, 0) + 1 if s >= 0.5 else 0
                longest[word] = max(longest.get(word, 0), runs[word])
        print(f"{label:28s} " + "  ".join(f"{w} max {peak[w]:.2f} (run {longest[w]})" for w in sorted(peak)))

    noise_frames = [noise[i:i + 1280] for i in range(0, 1280 * 40, 1280)]
    run("room noise (3.2 s)", noise_frames)
    run("zeros (1.6 s, the flush)", [np.zeros(1280, dtype=np.int16)] * 20)
    run("room noise again (3.2 s)", [noise[i:i + 1280] for i in range(1280 * 40, 1280 * 80, 1280)])
    run("quiet noise x0.1 (1.6 s)", [(noise[i:i + 1280] * 0.1).astype(np.int16) for i in range(0, 1280 * 20, 1280)])

    # The real case: the user's own "Uzi" (from the guided recording), then a flush, then room noise.
    sys.path.insert(0, str(ROOT / "training"))
    from train_wake_word import real_word_clips
    uzi = real_word_clips(Path.home() / "wake-train" / "real2", "uzi")[0].astype(np.int16)
    for flush in (20, 30, 40):
        m.reset() if hasattr(m, "reset") else None
        run("room noise (prime)", noise_frames)
        clip = np.concatenate([uzi, np.zeros(1280 * 2 - len(uzi) % 1280 + 1280, dtype=np.int16)])
        run("user's 'Uzi'", [clip[i:i + 1280] for i in range(0, len(clip) - 1279, 1280)])
        run(f"zeros x{flush} ({flush * 0.08:.1f} s)", [np.zeros(1280, dtype=np.int16)] * flush)
        run("room noise after the flush", [noise[i:i + 1280] for i in range(1280 * 80, 1280 * 110, 1280)])
        print()


if __name__ == "__main__":
    main()
