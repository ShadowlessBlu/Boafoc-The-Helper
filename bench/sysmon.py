"""Background sampler for Pi temperature, ARM clock, throttling and free memory.

Works on any Linux; on non-Pi machines the vcgencmd fields stay empty.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time


def _vcgencmd(*args: str) -> str:
    if not shutil.which("vcgencmd"):
        return ""
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def read_temp_c() -> float | None:
    m = re.search(r"([\d.]+)", _vcgencmd("measure_temp"))
    if m:
        return float(m.group(1))
    try:  # generic Linux fallback
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except OSError:
        return None


def read_arm_mhz() -> float | None:
    m = re.search(r"=(\d+)", _vcgencmd("measure_clock", "arm"))
    return int(m.group(1)) / 1e6 if m else None


def read_throttled() -> str:
    out = _vcgencmd("get_throttled")
    return out.split("=")[-1] if out else ""


def read_mem_available_mb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


class SysMon:
    """with SysMon() as m: ...  then m.summary() gives peaks over the block."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append({
                "t": time.time(),
                "temp_c": read_temp_c(),
                "arm_mhz": read_arm_mhz(),
                "mem_avail_mb": read_mem_available_mb(),
            })
            self._stop.wait(self.interval)

    def __enter__(self) -> "SysMon":
        self.start_mem = read_mem_available_mb()
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()
        self.throttled = read_throttled()

    def summary(self) -> dict:
        def vals(k):
            return [s[k] for s in self.samples if s[k] is not None]
        temps, clocks, mems = vals("temp_c"), vals("arm_mhz"), vals("mem_avail_mb")
        return {
            "max_temp_c": max(temps) if temps else "",
            "min_arm_mhz": min(clocks) if clocks else "",
            "min_mem_avail_mb": round(min(mems)) if mems else "",
            "mem_drop_mb": round(self.start_mem - min(mems)) if mems and self.start_mem else "",
            "throttled": getattr(self, "throttled", ""),
        }
