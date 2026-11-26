"""How many tokens does the assistant's prompt cost? (Prefill time = tokens / prefill speed.)

Counts the system prompt (with and without saved notes) and a typical conversation turn with the
Qwen3 tokenizer, then estimates the prefill time on the Pi from EXP-001's measured prefill speeds.
Needs `transformers` and a local Qwen3 tokenizer (any folder in research/models/ has one).

Usage: python bench/prompt_tokens.py [--tokenizer research/models/qwen3-0.6b-pruned2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from assistant.brain import build_system_prompt  # noqa: E402
from assistant.memory import Memory  # noqa: E402
from assistant.tools import Devices, build_tools, describe_for_prompt  # noqa: E402

# Uncached prefill speeds measured on the Pi (Ollama, 2026-11-18, long_context prompt, run 1)
PREFILL_TPS = {"qwen3:0.6b": 167.8, "qwen3:1.7b": 67.9, "llama3.2:1b Q4_K_M": 108.9, "llama3.2:3b": 36.4}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", default=str(ROOT / "research" / "models" / "qwen3-0.6b-pruned2"))
    a = p.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    tools = build_tools(Memory(), Devices())
    names = frozenset(tools)
    notes = ["The user's sister's birthday is 4 March.", "The user's dog is called Bingo.",
             "The user parked on level 3 of the mall car park."]
    system = build_system_prompt(describe_for_prompt(tools), None, names)
    system_notes = build_system_prompt(describe_for_prompt(tools), notes, names)
    history = [{"role": "user", "content": "Turn on the kitchen light."},
               {"role": "assistant", "content": "Turning on the kitchen light."}] * 3  # history = 6 messages
    question = {"role": "user", "content": "What's the temperature in the bedroom?"}

    def chat_tokens(msgs):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rows = [
        ("System prompt only", chat_tokens([{"role": "system", "content": system}])),
        ("+ 3 saved notes", chat_tokens([{"role": "system", "content": system_notes}])),
        ("+ notes + 6 history messages + question",
         chat_tokens([{"role": "system", "content": system_notes}, *history, question])),
    ]
    print("| Prompt | Tokens | " + " | ".join(f"{m} (s)" for m in PREFILL_TPS) + " |")
    print("|---|---|" + "---|" * len(PREFILL_TPS))
    for label, n in rows:
        print(f"| {label} | {n} | " + " | ".join(f"{n / s:.1f}" for s in PREFILL_TPS.values()) + " |")
    print(f"\nSystem prompt text: {len(system)} characters")


if __name__ == "__main__":
    main()
