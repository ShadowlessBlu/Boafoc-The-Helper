"""Where does a voice reply's time go? Measures, on the Pi, for a few typed questions:

  LLM: Ollama's own timings for the assistant's real prompt: load, prompt reading (prefill:
       tokens and seconds, and how many were reused from the cache), writing (tokens, seconds)
  Whisper: whisper-cli on the last question's audio (or a Piper clip): total, with the model's
       own load/encode/decode split, and with shorter --audio-ctx windows

  .venv/bin/python bench/latency_breakdown.py [--wav file.wav]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.__main__ import build  # noqa: E402
from assistant.brain import REPLY_SCHEMA  # noqa: E402
from assistant.config import load_config  # noqa: E402
from assistant.voice.engines import PiperTTS, clean_whisper_text, resolve_cmd  # noqa: E402

QUESTIONS = ["What is the capital of Ghana?", "Give me one short tip for staying focused.",
             "Tell me something interesting about octopuses."]


def ollama(cfg, messages) -> dict:
    body = {"model": cfg.llm.model, "messages": messages, "stream": False, "format": REPLY_SCHEMA,
            "options": {"num_ctx": cfg.llm.num_ctx, "temperature": cfg.llm.temperature,
                        "num_predict": cfg.llm.num_predict},
            "keep_alive": cfg.llm.keep_alive, "think": cfg.llm.think}
    req = urllib.request.Request(cfg.llm.host + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    d["wall_s"] = time.perf_counter() - t
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", default=f"/run/user/{os.getuid()}/assistant_last_question.wav")
    a = ap.parse_args()
    cfg = load_config(ROOT / "config.toml")
    tmp = tempfile.TemporaryDirectory()
    cfg.memory.db_path = str(Path(tmp.name) / "lat.db")
    brain, _ = build(cfg)

    print("== LLM (qwen3, the assistant's real prompt)")
    for q in QUESTIONS:
        msgs = brain._messages(q)
        d = ollama(cfg, msgs)
        ns = 1e9
        print(f"{q[:40]:42s} wall {d['wall_s']:4.1f} s | load {d.get('load_duration', 0) / ns:4.2f} | "
              f"prompt {d.get('prompt_eval_count', 0):3d} tok {d.get('prompt_eval_duration', 0) / ns:4.2f} s | "
              f"reply {d.get('eval_count', 0):3d} tok {d.get('eval_duration', 0) / ns:4.2f} s "
              f"({d.get('eval_count', 0) / max(d.get('eval_duration', 1) / ns, 1e-9):.1f} tok/s)")

    print("== Whisper tiny.en")
    wav = a.wav
    if not Path(wav).exists():
        tts = PiperTTS(cfg.voice.piper_cmd, cfg.voice.piper_model)
        wav = str(Path(tmp.name) / "q.wav")
        import numpy as np
        from scipy.signal import resample_poly
        from assistant.voice.audio import write_wav
        x = np.frombuffer(tts.synthesize("What is the capital of Ghana?"), dtype=np.int16).astype(float)
        write_wav(wav, resample_poly(x, 160, 441).astype(np.int16).tobytes(), 16000)
    for ctx in (0, 768, 512, 384, 256):
        cmd = resolve_cmd(shlex.split(cfg.voice.whisper_cmd.format(model=cfg.voice.whisper_model, wav=wav)))
        cmd = [c for c in cmd if c != "-np"] + ["--prompt", cfg.voice.whisper_prompt]
        if ctx:
            cmd += ["-ac", str(ctx)]
        t = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t
        times = dict(re.findall(r"(load|encode|decode|batchd|total) time =\s+([\d.]+) ms", r.stderr))
        print(f"audio_ctx {ctx or 'full':>4}: wall {wall:4.2f} s | load {float(times.get('load', 0)) / 1000:4.2f} "
              f"encode {float(times.get('encode', 0)) / 1000:4.2f} decode "
              f"{(float(times.get('decode', 0)) + float(times.get('batchd', 0))) / 1000:4.2f} | "
              f"{clean_whisper_text(r.stdout)!r}")


if __name__ == "__main__":
    main()
