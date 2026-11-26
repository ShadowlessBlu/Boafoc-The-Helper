"""Compare two Whisper hints on slices of a real recording (does a new hint hurt ordinary speech?).

  .venv/bin/python bench/hint_check.py ~/wake-train/real2/talk.wav "old hint" "new hint"
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.voice.audio import write_wav  # noqa: E402
from assistant.voice.engines import clean_whisper_text, resolve_cmd  # noqa: E402


def main() -> None:
    wav, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
    c = load_config(ROOT / "config.toml").voice
    with wave.open(wav) as w:
        raw = w.readframes(w.getnframes())

    def run(path: str, hint: str) -> str:
        cmd = resolve_cmd(shlex.split(c.whisper_cmd.format(model=c.whisper_model, wav=path))) + ["--prompt", hint]
        return clean_whisper_text(subprocess.run(cmd, capture_output=True, text=True).stdout)

    same = 0
    slices = ((10, 3), (40, 4), (70, 5), (100, 6), (130, 8), (165, 3), (190, 5), (215, 10))
    for start, dur in slices:
        write_wav("/tmp/hint_seg.wav", raw[start * 32000:(start + dur) * 32000], 16000)
        a, b = run("/tmp/hint_seg.wav", old), run("/tmp/hint_seg.wav", new)
        same += a.lower().strip(". ") == b.lower().strip(". ")
        print(f"{start:3d}s  old: {a[:90]!r}\n      new: {b[:90]!r}")
    print(f"identical {same}/{len(slices)}")


if __name__ == "__main__":
    main()
