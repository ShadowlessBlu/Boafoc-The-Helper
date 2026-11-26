"""Compare speech-to-text engines on one recording (default: the assistant's last question).

  .venv/bin/python bench/stt_compare.py [file.wav] [--hint "text"]

Runs Vosk, Whisper tiny.en and base.en (each plain, and with a context hint if given), prints
what each heard and how long it took. The assistant keeps the last question's audio in RAM at
/run/user/<uid>/assistant_last_question.wav (overwritten each time).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.voice.engines import VoskSTT, clean_whisper_text, resolve_cmd  # noqa: E402


def whisper(cmd_template: str, model: str, wav: str, hint: str = "") -> tuple[str, float]:
    cmd = resolve_cmd(shlex.split(cmd_template.format(model=model, wav=wav)))
    if hint:
        cmd += ["--prompt", hint]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode:
        return f"(failed: {r.stderr.strip()[-150:]})", time.perf_counter() - t
    return clean_whisper_text(r.stdout), time.perf_counter() - t


def main() -> None:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", default=f"/run/user/{uid}/assistant_last_question.wav")
    ap.add_argument("--hint", default="", help="context for Whisper (its --prompt option)")
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    a = ap.parse_args()
    c = load_config(a.config).voice
    with wave.open(a.wav) as w:
        pcm = w.readframes(w.getnframes())
    print(f"{a.wav}: {len(pcm) / 2 / 16000:.1f} s")
    t = time.perf_counter()
    text = VoskSTT(c.vosk_model, 16000).transcribe(pcm)
    print(f"vosk small          {time.perf_counter() - t:5.2f} s  {text!r}")
    models = Path(c.whisper_model).parent
    for name in ("tiny.en", "base.en"):
        for hint in ([""] + ([a.hint] if a.hint else [])):
            text, dt = whisper(c.whisper_cmd, str(models / f"ggml-{name}.bin"), a.wav, hint)
            print(f"whisper {name:8s} {'hint ' if hint else 'plain'} {dt:5.2f} s  {text!r}")


if __name__ == "__main__":
    main()
