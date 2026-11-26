"""Watch MQTT topics with timestamps while sending light commands (assistant login from config.toml).

  .venv/bin/python bench/mqtt_watch.py office off on off [--gap 4]
Prints every message on home/<room>/# and home/<room>-node/# for the whole run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("room")
    ap.add_argument("states", nargs="+", choices=["on", "off"])
    ap.add_argument("--gap", type=float, default=4.0)
    a = ap.parse_args()
    m = load_config(ROOT / "config.toml").mqtt
    auth = ["-h", m.host, "-p", str(m.port), "-u", m.username, "-P", m.password]
    room = a.room.replace(" ", "_")
    t0 = time.perf_counter()
    sub = subprocess.Popen(["mosquitto_sub", *auth, "-v", "-t", f"home/{room}/#", "-t", f"home/{room}-node/#"],
                           stdout=subprocess.PIPE, text=True)

    def reader():
        for line in sub.stdout:
            print(f"{time.perf_counter() - t0:6.2f}s  recv  {line.strip()}", flush=True)
    threading.Thread(target=reader, daemon=True).start()
    time.sleep(1.0)
    for st in a.states:
        subprocess.run(["mosquitto_pub", *auth, "-q", "1", "-t", f"home/{room}/light/set", "-m", st], check=True)
        print(f"{time.perf_counter() - t0:6.2f}s  SENT  {st}", flush=True)
        time.sleep(a.gap)
    sub.terminate()


if __name__ == "__main__":
    main()
