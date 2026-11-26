"""How often each wake word fired, and how many of those wake-ups led to a real request.

Reads logs/assistant.log. A wake-up counts as "used" when Whisper then heard words (a "user:" line
with letters in it, and not just a filler like "Thank you." or ">>"); otherwise it was most likely
a false wake (nobody was talking to the assistant, or it heard itself). Without knowing what was
really said this can't separate a false wake from a real one where the request was missed, so treat
"not used" as an upper bound on false wakes.

  .venv/bin/python bench/wake_report.py [logs/assistant.log] [--since 2026-11-21]
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

WAKE = re.compile(r"^(\S+ \S+) INFO assistant\.voice\.engines: wake word: (\w+)[^,]*, score ([\d.]+)")
IGNORED = re.compile(r"^(\S+ \S+) INFO assistant\.voice\.engines: wake word ignored, no speech heard: (\w+)")
USER = re.compile(r"^(\S+ \S+) INFO assistant\.voice\.loop: user: (.*)$")
READY = re.compile(r"voice loop ready")
FILLER = {"", "you", "thank you", "thanks", "bye", "okay", "ok", "yeah", "hmm", "so"}


def is_request(text: str) -> bool:
    t = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    return bool(t) and t not in FILLER and len(t) > 3


def report(lines, since: str = "") -> dict:
    """Per word: {"wakes", "used", "unused", "ignored", "scores_unused"}."""
    stats: dict[str, dict] = defaultdict(lambda: {"wakes": 0, "used": 0, "unused": 0, "ignored": 0,
                                                   "scores_unused": []})
    pending: tuple[str, float] | None = None     # the last wake, waiting for what was heard

    def close(used: bool):
        nonlocal pending
        if pending:
            word, score = pending
            stats[word]["used" if used else "unused"] += 1
            if not used:
                stats[word]["scores_unused"].append(score)
        pending = None

    for line in lines:
        if since and line[:10] < since:
            continue
        if m := WAKE.match(line):
            close(False)
            word = m.group(2)
            stats[word]["wakes"] += 1
            pending = (word, float(m.group(3)))
        elif m := USER.match(line):
            close(is_request(m.group(2)))
        elif m := IGNORED.match(line):
            stats[m.group(2)]["ignored"] += 1
        elif READY.search(line):
            close(False)
    close(False)
    return dict(stats)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else ""
    if since in args:
        args.remove(since)
    path = Path(args[0] if args else Path(__file__).resolve().parents[1] / "logs" / "assistant.log")
    text = path.read_bytes().decode("utf-8", errors="replace")
    stats = report(text.splitlines(), since)
    print(f"{'word':8} {'wakes':>6} {'request':>8} {'no req.':>8} {'blocked':>8}  scores of wakes with no request")
    for word, s in sorted(stats.items()):
        scores = sorted(s["scores_unused"])
        spread = f"{scores[0]:.2f}-{scores[-1]:.2f}, median {scores[len(scores) // 2]:.2f}" if scores else "-"
        print(f"{word:8} {s['wakes']:6} {s['used']:8} {s['unused']:8} {s['ignored']:8}  {spread}")
    print("blocked = fired without speech (Silero), never woke the assistant")


if __name__ == "__main__":
    main()
