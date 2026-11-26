"""Tool registry. Device tools are stubs that keep state in memory until MQTT is wired in."""
from __future__ import annotations

from datetime import datetime
import threading
from typing import Callable

from .governor import Param, Risk, ToolSpec
from .memory import Memory

ROOMS = ("living room", "bedroom", "kitchen", "office")
TIME_FMT = "%A %d %B %Y, %H:%M"


# Spoken answers for simple read-only tools, so the result doesn't need a second LLM call to
# become a sentence (that call made tool answers 9.5-13.6 s on the Pi instead of 4-7 s).
def say_time(result: str) -> str:
    t = datetime.strptime(result, TIME_FMT)
    hour = t.hour % 12 or 12
    return f"It's {hour}:{t:%M} {'AM' if t.hour < 12 else 'PM'} on {t:%A}, {t.day} {t:%B}."


def say_temperature(result: str) -> str:
    room, sep, value = result.partition(": ")
    if sep:
        return f"It's {value.replace(' °C', ' degrees')} in the {room}."
    return result[:1].upper() + result[1:] + "."


class DeviceNoAnswer(RuntimeError):
    """The device didn't confirm the command (off, out of Wi-Fi range, or not connected to MQTT)."""


class Devices:
    """The device layer. Without MQTT it is a stand-in that keeps state in memory; with the MQTT
    bridge, commands are published and a light only counts as switched once its device reports the
    new state (2026-11-23: the assistant said "Turning on the office light" while the ESP32 had lost
    its connection, and recorded "on" although nothing happened)."""

    def __init__(self, publish: Callable[[str, str], None] | None = None, confirm_s: float = 0.0):
        self.lights = {r: "off" for r in ROOMS}
        self.temps: dict[str, float] = {}
        self.publish = publish or (lambda topic, payload: None)
        self.confirm_s = confirm_s           # > 0: wait this long for the device's state report
        self._reports: dict[str, int] = {}   # state reports received per room
        self._cv = threading.Condition()

    def report_light(self, room: str, state: str) -> None:
        """A device reported its light state (MQTT <base>/<room>/light/state)."""
        with self._cv:
            self.lights[room] = state
            self._reports[room] = self._reports.get(room, 0) + 1
            self._cv.notify_all()

    def connected(self) -> list[str]:
        """Rooms whose device has reported at least once since the assistant started."""
        with self._cv:
            return sorted(self._reports)

    def set_light(self, room: str, state: str) -> str:
        with self._cv:
            before = self._reports.get(room, 0)
        self.publish(f"home/{room.replace(' ', '_')}/light/set", state)  # raises if MQTT is down
        if self.confirm_s <= 0:                # no real devices: the stand-in just records it
            self.lights[room] = state
            return f"{room} light is now {state}"
        with self._cv:
            answered = self._cv.wait_for(lambda: self._reports.get(room, 0) > before, timeout=self.confirm_s)
            actual = self.lights.get(room)
        if not answered and before == 0:        # nothing has ever reported for this room
            raise DeviceNoAnswer(f"there's no device for the {room} light yet")
        if not answered:
            raise DeviceNoAnswer(f"the {room} light didn't answer. Is its device switched on?")
        if actual != state:
            raise DeviceNoAnswer(f"the {room} light reports {actual}, not {state}")
        return f"{room} light is now {state}"

    def get_temperature(self, room: str) -> str:
        t = self.temps.get(room)
        return f"{room}: {t:.1f} °C" if t is not None else f"no temperature sensor in the {room} yet"

    def unlock_door(self, door: str) -> str:
        self.publish(f"home/{door}_door/lock/set", "unlock")
        return f"{door} door unlocked"


def build_tools(memory: Memory, devices: Devices, min_score: float = 0.45) -> dict[str, ToolSpec]:
    def forget_question(args: dict) -> str | None:
        hit = memory.best_match(str(args.get("text", "")), min_score)
        return f"Should I forget this note: {hit[1]}?" if hit else None

    tools = [
        ToolSpec("get_time", "Get the current date and time.", Risk.READ,
                 func=lambda: datetime.now().strftime(TIME_FMT), speak=say_time),
        ToolSpec("get_temperature", "Read the temperature in a room.", Risk.READ,
                 {"room": Param(enum=ROOMS)}, func=devices.get_temperature, speak=say_temperature),
        ToolSpec("set_light", "Turn a room's light on or off.", Risk.LOW,
                 {"room": Param(enum=ROOMS), "state": Param(enum=("on", "off"))},
                 rate_per_min=20, func=devices.set_light),
        ToolSpec("remember", "Save a fact about the user for later.", Risk.LOW,
                 {"text": Param(max_len=300)}, rate_per_min=10,
                 func=memory.remember_new),
        # Sensitive: a misheard sentence could delete the wrong note, so it names the note and waits
        # for a spoken yes (memory rules, 2026-11-25).
        ToolSpec("forget", "Delete a note the user saved earlier; text describes which one.", Risk.SENSITIVE,
                 {"text": Param(max_len=300)}, rate_per_min=5,
                 func=lambda text: memory.forget_matching(text, min_score), confirm_text=forget_question,
                 speak=lambda r: ("I don't have a note like that." if r.startswith("no saved note")
                                  else "Done, I've forgotten that note.")),
        ToolSpec("unlock_door", "Unlock a door.", Risk.SENSITIVE,
                 {"door": Param(enum=("front", "back"))}, rate_per_min=3, func=devices.unlock_door),
        # Never offered to the LLM; listed so the Governor denies it explicitly if proposed.
        ToolSpec("run_shell", "Run a shell command.", Risk.FORBIDDEN, {"command": Param()}),
    ]
    return {t.name: t for t in tools}


def describe_for_prompt(tools: dict[str, ToolSpec]) -> str:
    lines = []
    for t in tools.values():
        if t.risk is Risk.FORBIDDEN:
            continue
        if t.params:
            args = ", ".join(
                f'"{n}": ' + ("|".join(f'"{e}"' for e in p.enum) if p.enum else p.type)
                for n, p in t.params.items())
        else:
            args = ""
        lines.append(f"- {t.name}({{{args}}}): {t.description}")
    return "\n".join(lines)
