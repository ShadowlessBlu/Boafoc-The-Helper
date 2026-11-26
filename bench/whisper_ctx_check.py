"""Does a shorter Whisper window (--audio-ctx sized to the clip) change what it hears?

whisper.cpp always encodes a 30 s window (audio_ctx 1500, 50 per second). For a 3 s question most
of that is padding. This transcribes slices of a real recording both ways and compares.

  .venv/bin/python bench/whisper_ctx_check.py ~/wake-train/real2/talk.wav
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.voice.audio import write_wav  # noqa: E402
from assistant.voice.engines import audio_ctx_for, clean_whisper_text, resolve_cmd  # noqa: E402


def main() -> None:
    c = load_config(ROOT / "config.toml").voice
    with wave.open(sys.argv[1]) as w:
        raw = w.readframes(w.getnframes())

    def run(path: str, ac: int) -> tuple[float, str]:
        cmd = resolve_cmd(shlex.split(c.whisper_cmd.format(model=c.whisper_model, wav=path)))
        cmd += ["--prompt", c.whisper_prompt] + (["-ac", str(ac)] if ac else [])
        t = time.perf_counter()
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        return time.perf_counter() - t, clean_whisper_text(out)

    same = n = 0
    tf = td = 0.0
    for start, dur in ((10, 3), (40, 4), (70, 5), (100, 6), (130, 8), (165, 3), (190, 5), (215, 10)):
        seg = raw[start * 32000:(start + dur) * 32000]
        write_wav("/tmp/seg.wav", seg, 16000)
        ac = audio_ctx_for(len(seg))
        t1, full = run("/tmp/seg.wav", 0)
        t2, short = run("/tmp/seg.wav", ac)
        n += 1
        ok = full.lower().strip(". ") == short.lower().strip(". ")
        same += ok
        tf, td = tf + t1, td + t2
        print(f"{dur:2d} s  full {t1:4.2f} s | ctx {ac:4d} {t2:4.2f} s | {'same' if ok else 'DIFFERENT'}")
        if not ok:
            print(f"      full : {full[:100]!r}")
            print(f"      short: {short[:100]!r}")
    print(f"identical {same}/{n}; total time full {tf:.1f} s vs short {td:.1f} s")


if __name__ == "__main__":
    main()
