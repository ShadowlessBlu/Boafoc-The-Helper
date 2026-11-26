"""What the assistant is doing right now, for the face on the 7-inch screen (GET /state).

The voice loop and the API set the mode (idle, listening, thinking, speaking) and the last
exchange; the page polls /state a few times a second. Pi readings (temperature, memory,
throttling) come from /sys and /proc and are cached, so polling costs almost nothing.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

MODES = ("idle", "listening", "thinking", "speaking")


def read_temp_c(path: str = "/sys/class/thermal/thermal_zone0/temp") -> float | None:
    try:
        return round(int(Path(path).read_text().strip()) / 1000, 1)
    except (OSError, ValueError):
        return None


def read_memory_gb(path: str = "/proc/meminfo") -> tuple[float, float] | None:
    """(used, total) in GB, where used = total - available, like `free`."""
    try:
        info = {}
        for line in Path(path).read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0])  # kB
        total, avail = info["MemTotal"], info["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        return None
    return round((total - avail) / 1024 ** 2, 1), round(total / 1024 ** 2, 1)


def read_input_volts() -> float | None:
    """The Pi 5's 5 V input, from the power chip (vcgencmd pmic_read_adc EXT5V_V)."""
    try:
        out = subprocess.run(["vcgencmd", "pmic_read_adc", "EXT5V_V"], capture_output=True, text=True,
                             timeout=2).stdout
        return round(float(out.strip().split("=")[1].rstrip("V")), 2)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def read_fan_rpm() -> int | None:
    for p in Path("/sys/devices/platform/cooling_fan/hwmon").glob("*/fan1_input"):
        try:
            return int(p.read_text().strip())
        except (OSError, ValueError):
            pass
    return None


def read_uptime_s() -> int | None:
    try:
        return int(float(Path("/proc/uptime").read_text().split()[0]))
    except (OSError, ValueError, IndexError):
        return None


def read_load() -> float | None:
    try:
        import os
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


def read_throttled() -> bool | None:
    """True if the Pi is throttling or capped right now (bits 0-3 of get_throttled)."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return bool(int(out.strip().split("=")[1], 16) & 0xF)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class Status:
    def __init__(self, model: str = "", home: Callable[[], dict] | None = None,
                 clock: Callable[[], float] = time.time, stats_every_s: float = 2.0):
        self.model = model
        self.home = home or (lambda: {})
        self.clock = clock
        self.stats_every_s = stats_every_s
        self._lock = threading.Lock()
        self._state = {"mode": "idle", "since": clock(), "heard": "", "said": "", "first_word_s": None,
                       "voice": False, "muted": False, "wake_words": []}
        self._stats: dict = {}
        self._stats_at = -1e9

    def set(self, mode: str | None = None, **fields) -> None:
        if mode is not None and mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        with self._lock:
            if mode is not None and mode != self._state["mode"]:
                self._state["mode"], self._state["since"] = mode, self.clock()
            self._state.update(fields)

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def pi_stats(self) -> dict:
        now = self.clock()
        if now - self._stats_at >= self.stats_every_s:
            mem = read_memory_gb()
            self._stats = {"temp_c": read_temp_c(), "mem_used_gb": mem[0] if mem else None,
                           "mem_total_gb": mem[1] if mem else None, "throttled": read_throttled(),
                           "volts": read_input_volts(), "fan_rpm": read_fan_rpm(), "uptime_s": read_uptime_s(),
                           "load": read_load()}
            self._stats_at = now
        return self._stats

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._state)
        try:
            home = self.home()
        except Exception:  # noqa: BLE001 - the screen must keep working
            home = {}
        return {**s, "model": self.model, "pi": self.pi_stats(), "home": home, "time": self.clock()}
