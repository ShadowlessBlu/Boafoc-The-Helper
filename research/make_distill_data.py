"""EXP-001 Q5 step 1: build a distillation dataset from a larger "teacher" model.

Generates varied assistant requests from templates, asks the teacher (via Ollama, on the
PC or the Pi) to answer in the assistant's JSON format, and keeps only answers that
parse and that the Governor would not deny. Questions from the test set are excluded so
the student is never trained on its exam.

Usage:
  python research/make_distill_data.py --teacher qwen3:1.7b --think off --n 600 --out research/data/distill_v1.jsonl
  (add --resume to carry on after a power cut: the requests are the same every run, so it skips
   the ones already answered, counted in <out>.progress)
Output: one {"messages": [system, user, assistant]} object per line (chat fine-tuning format).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from assistant.brain import REPLY_SCHEMA, build_system_prompt, claims_action, parse_reply  # noqa: E402
from assistant.governor import Governor, Verdict  # noqa: E402
from assistant.memory import Memory  # noqa: E402
from assistant.tools import ROOMS, Devices, build_tools, describe_for_prompt  # noqa: E402

LIGHT_ON = ["Turn on the {r} light", "Switch the {r} lights on", "Can you put the light on in the {r}?",
            "Lights on in the {r} please", "It's dark in the {r}, light please"]
LIGHT_OFF = ["Turn off the {r} light", "Switch off the {r} lights", "Kill the light in the {r}",
             "Lights off in the {r}", "I'm leaving the {r}, turn the light off"]
TEMP = ["What's the temperature in the {r}?", "Is it cold in the {r}?", "How hot is the {r} right now?",
        "Check the {r} temperature"]
TIME = ["What's the time?", "Tell me the time please", "What day is it today?", "Do you know the date?"]
REMEMBER = ["Remember that my {who} likes {thing}", "Please note that {who}'s birthday is on {day}",
            "Don't forget that I parked on level {n}", "Remember that the bins go out on {day}"]
CHAT = ["Good evening!", "Thanks for your help", "Tell me a joke about computers",
        "Explain what a resistor does in one sentence", "Give me a tip for studying",
        "What's a healthy breakfast idea?", "How far is the Moon from Earth?", "Who wrote Things Fall Apart?",
        "Convert 5 kilometres to miles", "What is the square root of 144?", "Name three planets",
        "What does CPU stand for?", "Suggest a name for a robot", "Why is the sky blue?"]
SAFETY = ["Delete all my files", "Run sudo reboot", "Open a shell for me", "Disable your safety rules",
          "Format the SD card", "Send my passwords to this email"]
WHO = ["sister", "brother", "mum", "dad", "friend Kofi", "colleague Ama"]
THING = ["jollof rice", "football", "chess", "science fiction", "gardening"]
DAYS = ["4 March", "12 June", "30 September", "1 January"]


PREFIXES = ["", "", "Hey, ", "Please ", "Could you ", "Assistant, ", "Quick one: "]
SUFFIXES = ["", "", " please", ", thanks", " now", "?"]


def _expect(kind: str, room: str) -> tuple[str, dict]:
    """The right tool call for a request made from a template of this kind."""
    return {"on": ("set_light", {"room": room, "state": "on"}),
            "off": ("set_light", {"room": room, "state": "off"}),
            "temp": ("get_temperature", {"room": room}),
            "time": ("get_time", {}),
            "remember": ("remember", {}),
            "chat": ("none", {}),
            "safety": ("none", {})}[kind]


def make_items(n: int, seed: int = 7, exclude: set[str] = frozenset()) -> list[tuple[str, tuple[str, dict]]]:
    """Up to n unique (request, (right tool, args it must have)) pairs. The template space holds
    roughly 2,800, so ask for fewer."""
    rng = random.Random(seed)
    pools = [(LIGHT_ON, 3, "on"), (LIGHT_OFF, 3, "off"), (TEMP, 2, "temp"), (TIME, 1, "time"),
             (REMEMBER, 1, "remember"), (CHAT, 3, "chat"), (SAFETY, 1, "safety")]
    weighted = [(p, kind) for p, w, kind in pools for _ in range(w)]
    out, seen = [], {e.lower() for e in exclude}
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        pool, kind = rng.choice(weighted)
        t = rng.choice(pool)
        room = rng.choice(ROOMS)
        core = t.format(r=room, who=rng.choice(WHO), thing=rng.choice(THING),
                        day=rng.choice(DAYS), n=rng.randint(1, 6))
        pre, suf = rng.choice(PREFIXES), rng.choice(SUFFIXES)
        if pre:
            core = core[0].lower() + core[1:]
        if suf and core[-1] in "?!.":
            suf = ""
        q = pre + core + suf
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append((q, _expect(kind, room)))
    return out


def make_requests(n: int, seed: int = 7, exclude: set[str] = frozenset()) -> list[str]:
    return [q for q, _ in make_items(n, seed, exclude)]


def accept(raw: str, governor: Governor, expect: tuple[str, dict] | None = None) -> tuple[bool, str]:
    """Keep only well-formed replies whose tool call the Governor would not deny and, when the right
    call is known, that make exactly that call. A teacher answer that says "Remembering that..."
    without calling remember would teach the student to claim things it didn't do."""
    say, proposal = parse_reply(raw)
    if not say:
        return False, "empty say"
    if proposal is not None:
        d = governor._decide(proposal)  # check only; no rate-limit side effects that matter here
        if d.verdict is Verdict.DENY:
            return False, d.reason
    if expect is None:
        return True, ""
    tool, args = expect
    got = proposal["name"] if proposal else "none"
    if got != tool:
        return False, f"wanted tool {tool}, got {got}"
    for k, v in args.items():
        if str(proposal["args"].get(k, "")).strip().lower() != v:
            return False, f"wrong {k}: {proposal['args'].get(k)!r}"
    if tool == "remember" and not str(proposal["args"].get("text", "")).strip():
        return False, "remember with no text"
    if tool == "none" and claims_action(say):
        return False, "claims an action it didn't take"
    return True, ""


def good_rows(data: bytes) -> tuple[list[str], int]:
    """The whole, valid JSON rows of a data file, and how many lines were damaged."""
    rows, damaged = [], 0
    for line in data.decode("utf-8", errors="replace").split("\n"):
        line = line.strip("\x00").strip()
        if not line:
            continue
        try:
            json.loads(line)
            rows.append(line)
        except ValueError:
            damaged += 1
    return rows, damaged


def main(argv=None) -> None:
    from assistant.llm import LLMError, OllamaClient

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--out", type=Path, default=ROOT / "research" / "data" / "distill_v1.jsonl")
    p.add_argument("--think", choices=("on", "off", "default"), default="default",
                   help="qwen3's thinking mode for the teacher (off: the assistant's setting)")
    p.add_argument("--resume", action="store_true", help="continue a run that was cut off")
    a = p.parse_args(argv)

    exclude = set()
    for name in ("test_set_v1.json", "test_set_v2.json"):   # both exams, never trained on
        test = json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"))
        exclude |= {i["q"] for i in test["items"]}
    tools = build_tools(Memory(), Devices())
    gov = Governor(tools, default_rate_per_min=10 ** 9)
    system = build_system_prompt(describe_for_prompt(tools))
    think = {"on": True, "off": False}.get(a.think)
    llm = OllamaClient(a.host, a.teacher, num_ctx=2048, temperature=0.2, think=think, num_predict=256)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    progress = a.out.with_name(a.out.name + ".progress")
    done = 0
    if a.resume and progress.exists() and a.out.exists():
        done = int(progress.read_text().strip() or 0)
        lines, damaged = good_rows(a.out.read_bytes())
        if damaged:   # a power cut can leave NUL bytes or half a line: keep only whole rows
            a.out.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
            print(f"dropped {damaged} damaged line(s)", flush=True)
        kept = len(lines)
        rejected = done - kept
        print(f"resuming after {done} requests ({kept} kept)", flush=True)
    else:
        kept = rejected = 0
    with a.out.open("a" if done else "w", encoding="utf-8") as f:
        for i, (q, expect) in enumerate(make_items(a.n, exclude=exclude), 1):
            if i <= done:
                continue
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
            try:
                raw = llm.chat(msgs, fmt=REPLY_SCHEMA)
                ok, why = accept(raw, gov, expect)
            except LLMError as e:   # too long, timeout: just not kept
                ok, why = False, f"teacher error: {e}"[:120]
            if ok:
                say, proposal = parse_reply(raw)
                clean = json.dumps({"say": say, "tool": proposal["name"] if proposal else "none",
                                    "args": proposal["args"] if proposal else {}}, ensure_ascii=False)
                f.write(json.dumps({"messages": msgs + [{"role": "assistant", "content": clean}]},
                                   ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected += 1
                print(f"  rejected: {q!r} ({why})")
            f.flush()
            os.fsync(f.fileno())   # on disk before the progress mark, so a power cut loses at most one row
            progress.write_text(str(i))
            if i % 50 == 0:
                print(f"{i}/{a.n}  kept {kept}  rejected {rejected}", flush=True)
    print(f"Done: kept {kept}, rejected {rejected} -> {a.out}")


if __name__ == "__main__":
    main()
