"""Score a model on the assistant test set (EXP-001 Q3 / Q5 / Q7 quality measure).

Each question is sent with the real assistant system prompt and JSON reply format,
so the score reflects how the model behaves inside the assistant. A tool question only
passes if the Governor would also let the call run: choosing the right tool with an argument
the Governor refuses still fails the user (found 2026-11-19: get_time {"timezone": "local"}).

Usage:
  python experiments/score.py --model qwen3:0.6b [--set experiments/test_set_v1.json] [--show]
Appends one row per question to experiments/results/score_<set version>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from assistant.brain import REPLY_SCHEMA, build_system_prompt, parse_reply  # noqa: E402
from assistant.governor import Governor, Verdict  # noqa: E402
from assistant.llm import LLMError, OllamaClient, ReplyTooLong  # noqa: E402
from assistant.memory import Memory  # noqa: E402
from assistant.tools import Devices, build_tools, describe_for_prompt  # noqa: E402

sys.path.insert(0, str(ROOT / "bench"))
from bench_ollama import wait_cool  # noqa: E402

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def check(item_check: dict, say: str, proposal: dict | None) -> tuple[bool, str]:
    kind = item_check["type"]
    low = say.lower()
    if kind == "contains_any":
        hay = low.replace(" ", "") if item_check.get("strip_spaces") else low
        ok = any(v.lower() in hay for v in item_check["values"])
        return ok, "" if ok else f"none of {item_check['values']} in reply"
    if kind == "regex":
        ok = re.search(item_check["pattern"], say, re.I) is not None
        return ok, "" if ok else "regex did not match"
    if kind == "number":
        nums = [float(n.replace(",", "")) for n in _NUM.findall(say)]
        ok = any(abs(n - item_check["value"]) < 1e-6 for n in nums)
        return ok, "" if ok else f"{item_check['value']} not in {nums[:5]}"
    if kind == "no_tool":
        return proposal is None, "" if proposal is None else f"proposed {proposal.get('name')}"
    if kind == "tool":
        if proposal is None:
            return False, "no tool proposed"
        if proposal.get("name") != item_check["name"]:
            return False, f"wrong tool {proposal.get('name')}"
        args = proposal.get("args") or {}
        for k, sub in item_check.get("args", {}).items():
            if sub.lower() not in str(args.get(k, "")).lower():
                return False, f"arg {k}={args.get(k)!r}"
        verdict, reason = governor_verdict(proposal)
        if verdict is Verdict.DENY:
            return False, f"governor denied: {reason}"
        return True, ""
    raise ValueError(f"unknown check type {kind}")


def governor_verdict(proposal: dict | None) -> tuple[Verdict | None, str]:
    """What the real Governor decides about this call (a fresh one each time, so rate limits
    from earlier questions don't count). CONFIRM passes: the user would be asked."""
    if proposal is None:
        return None, ""
    d = Governor(build_tools(Memory(), Devices())).check(proposal)
    return d.verdict, d.reason


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--set", type=Path, default=ROOT / "experiments" / "test_set_v1.json")
    p.add_argument("--num-ctx", type=int, default=2048)
    p.add_argument("--show", action="store_true", help="print every reply")
    p.add_argument("--cool-to", type=float, default=0,
                   help="before each question, wait (max 2 min) until the CPU is below this many degC; 0 = off")
    p.add_argument("--timeout", type=float, default=90,
                   help="seconds before a reply counts as hung (slowest real answer on the Pi: 21 s; runaway replies never end)")
    p.add_argument("--num-predict", type=int, default=256,
                   help="reply token cap, as in the assistant (0 = none, as in runs before 2026-11-19)")
    p.add_argument("--think", choices=["default", "on", "off"], default="default",
                   help="Qwen3-style thinking (default = the model's own default)")
    a = p.parse_args(argv)

    data = json.loads(a.set.read_text(encoding="utf-8"))
    tools_text = describe_for_prompt(build_tools(Memory(), Devices()))
    llm = OllamaClient(a.host, a.model, num_ctx=a.num_ctx, temperature=0,
                       timeout=a.timeout, think={"default": None, "on": True, "off": False}[a.think],
                       num_predict=a.num_predict)

    out = ROOT / "experiments" / "results" / f"score_{data['version']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    score, by_cat, rows = 0, {}, []
    for item in data["items"]:
        if a.cool_to:  # long scoring runs reached 85.6 degC and throttled, which inflates seconds/answer
            wait_cool(a.cool_to, timeout_s=120)
        messages = [{"role": "system", "content": build_system_prompt(tools_text, item.get("facts"))},
                    {"role": "user", "content": item["q"]}]
        t0 = time.perf_counter()
        try:
            raw = llm.chat(messages, fmt=REPLY_SCHEMA)
        except ReplyTooLong as e:
            raw, err = None, f"cut off: {e}"
        except LLMError as e:  # one hung question (qwen3:0.6b ran on for 300 s) must not lose the whole run
            raw = None
            err = str(e)
        secs = time.perf_counter() - t0
        if raw is None:
            say, proposal, ok, why = "", None, False, f"error: {err}"
        else:
            say, proposal = parse_reply(raw)
            ok, why = check(item["check"], say, proposal)
        score += ok
        c = by_cat.setdefault(item["category"], [0, 0])
        c[0] += ok
        c[1] += 1
        mark = "PASS" if ok else "FAIL"
        print(f"{mark} {item['id']:4} {secs:5.1f}s  {why}")
        if a.show or not ok:
            print(f"      say={say[:120]!r} tool={proposal}")
        rows.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                     "model": a.model + ("" if a.think == "default" else f" (think {a.think})"),
                     "set": data["version"], "id": item["id"], "category": item["category"],
                     "pass": int(ok), "seconds": round(secs, 2), "why": why,
                     "say": say[:300], "tool": json.dumps(proposal) if proposal else "",
                     "governor": (governor_verdict(proposal)[0].value if proposal else ""),
                     "tokens": llm.last_tokens if raw is not None else ""})

    if not new:  # older files have fewer columns: start a new one rather than misalign rows
        with out.open(encoding="utf-8") as f:
            if f.readline().strip().split(",") != list(rows[0]):
                out, new = out.with_name(out.stem + "_r2.csv"), not out.with_name(out.stem + "_r2.csv").exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\n{a.model}: {score}/{len(data['items'])}   " +
          "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_cat.items()))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
