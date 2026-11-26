"""Benchmark models through the Ollama API (EXP-001 Phase A).

For each model x prompt x run it records: time to first token, prefill and decode
tokens/s (from Ollama's own timers), load time, model memory reported by Ollama,
and max temperature / min clock / memory drop / throttling from the Pi.

Usage:
  python bench/bench_ollama.py --models qwen3:0.6b llama3.2:1b --runs 3 --num-ctx 2048
Results are appended to experiments/results/bench_ollama.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sysmon import SysMon, read_temp_c  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ROOT / "prompts" / "standard_v1.txt"
OUT = ROOT / "experiments" / "results" / "bench_ollama.csv"

FIELDS = ["timestamp", "host", "model", "prompt_id", "run", "num_ctx", "num_thread",
          "load_s", "ttft_ms", "prompt_tokens", "prefill_tps", "gen_tokens", "decode_tps", "total_s",
          "ollama_size_mb", "max_temp_c", "min_arm_mhz", "min_mem_avail_mb", "mem_drop_mb", "throttled"]


def load_prompts(path: Path) -> list[tuple[str, str]]:
    prompts, pid, buf = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            if pid:
                prompts.append((pid, "\n".join(buf).strip()))
            pid, buf = line[4:].strip(), []
        elif pid:
            buf.append(line)
    if pid:
        prompts.append((pid, "\n".join(buf).strip()))
    return prompts


def post_stream(host: str, path: str, body: dict):
    req = urllib.request.Request(host + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.strip():
                yield json.loads(line)


def get_json(host: str, path: str) -> dict:
    with urllib.request.urlopen(host + path, timeout=10) as r:
        return json.loads(r.read())


def ollama_model_size_mb(host: str, model: str) -> float | str:
    try:
        for m in get_json(host, "/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return round(m.get("size", 0) / 1024 ** 2)
    except OSError:
        pass
    return ""


def run_once(host, model, prompt, num_ctx, num_thread, max_tokens) -> dict:
    options = {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": 0, "seed": 42}
    if num_thread:
        options["num_thread"] = num_thread
    body = {"model": model, "prompt": prompt, "stream": True, "options": options, "keep_alive": "10m"}
    t0 = time.perf_counter()
    ttft = None
    final = {}
    for chunk in post_stream(host, "/api/generate", body):
        # Thinking models (Qwen3) stream "thinking" text before any "response": the first token of
        # either kind is when the model started producing output.
        if ttft is None and (chunk.get("response") or chunk.get("thinking")):
            ttft = (time.perf_counter() - t0) * 1000
        if chunk.get("done"):
            final = chunk
    ns = 1e9
    pe_n, pe_d = final.get("prompt_eval_count", 0), final.get("prompt_eval_duration", 0)
    ev_n, ev_d = final.get("eval_count", 0), final.get("eval_duration", 0)
    return {
        "load_s": round(final.get("load_duration", 0) / ns, 2),
        "ttft_ms": round(ttft) if ttft else "",
        "prompt_tokens": pe_n,
        "prefill_tps": round(pe_n / (pe_d / ns), 2) if pe_d else "",
        "gen_tokens": ev_n,
        "decode_tps": round(ev_n / (ev_d / ns), 2) if ev_d else "",
        "total_s": round(final.get("total_duration", 0) / ns, 2),
    }


def wait_cool(limit_c: float, timeout_s: int = 300) -> None:
    t = read_temp_c()
    if t is None or t <= limit_c:
        return
    print(f"  cooling down: {t:.1f} °C > {limit_c} °C", flush=True)
    end = time.time() + timeout_s
    while time.time() < end:
        time.sleep(5)
        t = read_temp_c()
        if t is not None and t <= limit_c:
            return


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434"))
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--only", nargs="*", help="prompt ids to run (default all)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--num-ctx", type=int, default=2048)
    p.add_argument("--num-thread", type=int, default=0, help="0 = Ollama default")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--cool-to", type=float, default=55.0, help="wait until CPU is below this before each run")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)

    prompts = [x for x in load_prompts(a.prompts) if not a.only or x[0] in a.only]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    with a.out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for model in a.models:
            print(f"\n== {model}: warm-up (load) ...", flush=True)
            run_once(a.host, model, "Hello", a.num_ctx, a.num_thread, 8)
            size = ollama_model_size_mb(a.host, model)
            decode = []
            for pid, text in prompts:
                for r in range(1, a.runs + 1):
                    wait_cool(a.cool_to)
                    with SysMon() as mon:
                        res = run_once(a.host, model, text, a.num_ctx, a.num_thread, a.max_tokens)
                    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                           "host": platform.node(), "model": model, "prompt_id": pid, "run": r,
                           "num_ctx": a.num_ctx, "num_thread": a.num_thread or "default",
                           "ollama_size_mb": size, **res, **mon.summary()}
                    w.writerow(row)
                    f.flush()
                    if res["decode_tps"] != "":
                        decode.append(res["decode_tps"])
                    print(f"  {pid} run {r}: TTFT {res['ttft_ms']} ms, prefill {res['prefill_tps']} t/s, "
                          f"decode {res['decode_tps']} t/s, max {row['max_temp_c']} °C", flush=True)
            if decode:
                sd = statistics.stdev(decode) if len(decode) > 1 else 0
                print(f"  -> {model}: decode {statistics.mean(decode):.2f} ± {sd:.2f} t/s, "
                      f"Ollama size {size} MB")
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()
