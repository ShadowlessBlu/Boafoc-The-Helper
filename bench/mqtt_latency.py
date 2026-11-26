"""MQTT round-trip latency to an ESP32 node (MQTT note experiment; needs paho-mqtt>=2).

The firmware echoes anything on <base>/<node>/ping to <base>/<node>/pong.
Usage: python bench/mqtt_latency.py --node livingroom-node --count 100 --qos 0 1
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "mqtt_latency.csv"


def main(argv=None) -> None:
    import paho.mqtt.client as mqtt

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--user", default="assistant")
    p.add_argument("--node", required=True)
    p.add_argument("--base", default="home")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--qos", type=int, nargs="+", default=[0, 1])
    a = p.parse_args(argv)

    got: dict[str, float] = {}
    ev = threading.Event()
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="latency-bench")
    c.username_pw_set(a.user, os.environ.get("ASSISTANT_MQTT_PASSWORD") or None)
    c.on_message = lambda cl, ud, m: (got.__setitem__(m.payload.decode(), time.perf_counter()), ev.set())
    c.connect(a.host, a.port)
    c.subscribe(f"{a.base}/{a.node}/pong", qos=1)
    c.loop_start()
    time.sleep(0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "node", "qos", "count", "lost", "mean_ms", "median_ms", "p95_ms", "max_ms"])
        for qos in a.qos:
            rtts, lost = [], 0
            for i in range(a.count):
                tag = f"{qos}-{i}-{time.time_ns()}"
                ev.clear()
                t0 = time.perf_counter()
                c.publish(f"{a.base}/{a.node}/ping", tag, qos=qos)
                if ev.wait(2.0) and tag in got:
                    rtts.append((got.pop(tag) - t0) * 1000)
                else:
                    lost += 1
                time.sleep(0.05)
            if rtts:
                rtts.sort()
                row = [datetime.now().isoformat(timespec="seconds"), a.node, qos, a.count, lost,
                       round(statistics.mean(rtts), 1), round(statistics.median(rtts), 1),
                       round(rtts[int(0.95 * (len(rtts) - 1))], 1), round(rtts[-1], 1)]
            else:
                row = [datetime.now().isoformat(timespec="seconds"), a.node, qos, a.count, lost, "", "", "", ""]
            w.writerow(row)
            print(f"QoS {qos}: lost {lost}/{a.count}, mean {row[5]} ms, p95 {row[7]} ms")
    c.loop_stop()
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
