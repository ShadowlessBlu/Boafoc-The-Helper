"""Compare Hugging Face models (base vs pruned/distilled) on the PC: perplexity, speed, sample answers.

Speed here is PyTorch on the PC CPU, NOT the Pi. Use it only for relative comparisons
(e.g. "pruned is 15% faster than base"); measure real speed on the Pi with llama.cpp.

Usage:
  python research/eval_hf.py --models Qwen/Qwen3-0.6B research/models/qwen3-0.6b-pruned4 --threads 4
Appends to experiments/results/eval_hf.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "research" / "eval_text_v1.txt"
OUT = ROOT / "experiments" / "results" / "eval_hf.csv"
PROMPTS = [
    "What is the capital of Ghana? Answer in one sentence.",
    "What is 17 times 23?",
    "Explain in one sentence what a relay does.",
]


def perplexity(model, tok, text: str, window: int = 512) -> tuple[float, int]:
    import torch
    ids = tok(text, return_tensors="pt").input_ids[0]
    nll, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(ids) - 1, window):
            chunk = ids[start:start + window + 1].unsqueeze(0)
            if chunk.shape[1] < 2:
                break
            out = model(chunk, labels=chunk)
            n = chunk.shape[1] - 1
            nll += out.loss.item() * n
            count += n
    return math.exp(nll / count), count


def generate(model, tok, prompt: str, max_new: int = 48) -> tuple[str, float, int]:
    import torch
    msgs = [{"role": "user", "content": prompt}]
    kwargs = {"enable_thinking": False} if "qwen3" in (model.config._name_or_path or "").lower() or \
        getattr(model.config, "model_type", "") == "qwen3" else {}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
    ids = tok(text, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    new = out[0][ids.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), dt, len(new)


def main(argv=None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    text = TEXT.read_text(encoding="utf-8")

    rows = []
    for name in a.models:
        print(f"\n== {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        params = sum(x.numel() for x in model.parameters())
        ppl, ntok = perplexity(model, tok, text)
        print(f"  layers {model.config.num_hidden_layers}, params {params/1e6:.0f}M, perplexity {ppl:.2f} ({ntok} tokens)")
        gen_t, gen_n, samples = 0.0, 0, []
        generate(model, tok, "Hi", 4)  # warm-up
        for pr in PROMPTS:
            ans, dt, n = generate(model, tok, pr)
            gen_t += dt
            gen_n += n
            samples.append(ans)
            print(f"  Q: {pr}\n  A: {ans[:160]!r}")
        tps = gen_n / gen_t if gen_t else 0
        print(f"  generation: {tps:.1f} tokens/s on {a.threads} CPU threads (PC)")
        rows.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
                     "model": name, "layers": model.config.num_hidden_layers, "params_m": round(params / 1e6),
                     "perplexity": round(ppl, 3), "eval_tokens": ntok, "threads": a.threads,
                     "pc_cpu_tokens_per_s": round(tps, 2), "samples": json.dumps(samples, ensure_ascii=False)})
        del model

    if rows and not a.no_save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        new = not OUT.exists()
        with OUT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nSaved to {OUT}")
    print("\n| Model | Layers | Params (M) | Perplexity | PC CPU tok/s |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['layers']} | {r['params_m']} | {r['perplexity']} | {r['pc_cpu_tokens_per_s']} |")


if __name__ == "__main__":
    main()
