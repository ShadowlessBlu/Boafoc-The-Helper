"""Send light commands straight over MQTT (with the assistant's login from config.toml, not printed)
and show what the node reports back.

  .venv/bin/python bench/mqtt_light.py office on off on [--gap 4]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
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
    ap.add_argument("--node", default="")
    a = ap.parse_args()
    m = load_config(ROOT / "config.toml").mqtt
    auth = ["-h", m.host, "-p", str(m.port), "-u", m.username, "-P", m.password]
    room = a.room.replace(" ", "_")
    node = a.node or f"{room}-node"
    for st in a.states:
        subprocess.run(["mosquitto_pub", *auth, "-t", f"home/{room}/light/set", "-m", st], check=True)
        print(f"{time.strftime('%H:%M:%S')} sent {st}", flush=True)
        time.sleep(a.gap)
    r = subprocess.run(["mosquitto_sub", *auth, "-t", f"home/{room}/light/state", "-t", f"home/{node}/status",
                        "-v", "-W", "2"], capture_output=True, text=True)
    print("node reports:", r.stdout.strip().replace("\n", " | ") or "(nothing)")


if __name__ == "__main__":
    main()
