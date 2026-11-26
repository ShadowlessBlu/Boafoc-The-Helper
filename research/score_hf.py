"""Score Hugging Face models on the assistant test set, without Ollama (PC-side twin of experiments/score.py).

Uses the same system prompt, test set and checks as the Pi scorer, so results are comparable
in kind (generation settings differ slightly: greedy decoding, no JSON grammar constraint).

Usage:
  python research/score_hf.py --models Qwen/Qwen3-0.6B research/models/qwen3-0.6b-pruned4mid-healed/merged
Appends to experiments/results/score_hf_v1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from assistant.brain import build_system_prompt, parse_reply  # noqa: E402
from assistant.memory import Memory  # noqa: E402
from assistant.tools import Devices, build_tools, describe_for_prompt  # noqa: E402
from score import check  # noqa: E402


def main(argv=None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--set", type=Path, default=ROOT / "experiments" / "test_set_v1.json")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-new", type=int, default=80)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)

    data = json.loads(a.set.read_text(encoding="utf-8"))
    tools_text = describe_for_prompt(build_tools(Memory(), Devices()))
    out = ROOT / "experiments" / "results" / f"score_hf_{data['version']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = []

    for name in a.models:
        print(f"\n== {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        qwen3 = getattr(model.config, "model_type", "") == "qwen3"
        score, cats, rows = 0, {}, []
        for item in data["items"]:
            msgs = [{"role": "system", "content": build_system_prompt(tools_text, item.get("facts"))},
                    {"role": "user", "content": item["q"]}]
            kw = {"enable_thinking": False} if qwen3 else {}
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            ids = tok(text, return_tensors="pt")
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            secs = time.perf_counter() - t0
            raw = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            say, proposal = parse_reply(raw)
            ok, why = check(item["check"], say, proposal)
            score += ok
            c = cats.setdefault(item["category"], [0, 0])
            c[0] += ok
            c[1] += 1
            print(f"  {'PASS' if ok else 'FAIL'} {item['id']:4} {secs:5.1f}s {why}", flush=True)
            rows.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "model": name,
                         "id": item["id"], "category": item["category"], "pass": int(ok), "seconds": round(secs, 1),
                         "why": why, "raw": raw[:300]})
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        line = f"{name}: {score}/{len(data['items'])}  " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in cats.items())
        print(line, flush=True)
        summary.append(line)
        del model

    print("\n" + "\n".join(summary))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
