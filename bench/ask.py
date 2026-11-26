"""Ask the running assistant something (typed, through its API on the Pi) and show the reply and
the lights before/after. Uses the API token from config.toml without printing it.

  .venv/bin/python bench/ask.py "Turn on the office light."
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402

API = "http://127.0.0.1:8765"


def lights() -> dict:
    with urllib.request.urlopen(API + "/state", timeout=5) as r:
        return json.loads(r.read())["home"]["lights"]


def ask(text: str, token: str) -> dict:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    req = urllib.request.Request(API + "/ask", data=json.dumps({"text": text}).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main() -> None:
    text = " ".join(sys.argv[1:]) or "Turn on the office light."
    token = load_config(ROOT / "config.toml").api.token
    print("lights before:", lights())
    r = ask(text, token)
    print(f"you: {text}\nassistant: {r['reply']}  ({r['seconds']} s)")
    time.sleep(1.5)
    print("lights after: ", lights())


if __name__ == "__main__":
    main()
