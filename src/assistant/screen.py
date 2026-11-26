"""The assistant's face and dashboard on the Pi's own 7-inch touch screen (800x480), drawn with pygame.

Same design as the web page at /face (which other devices use), but a small Python window
instead of a browser. It reads the assistant's /state, so it also runs while the assistant is
down (grey "offline" eyes) and keeps going when the assistant restarts.

Touch (main dashboard):
  face            -> full-screen face; there, only a triple-tap goes back
  clock           -> clock page
  Settings > Exit -> stop the whole assistant (no listening) and close this screen; tap twice to
                     confirm. The desktop icon starts everything again; boot is unchanged.
  Last reply card -> history (Clear history: tap twice; saved notes and the audit log stay)
  Home card -> lights and temperatures      Pi card -> Pi status
  Home page: tap a light's row to switch it (it only changes once the device confirms)
  Mic button      -> mute / unmute the microphone      Settings button -> settings
  Settings > Wake words -> each word on/off and its threshold (at once, kept after a restart), with
                     what it did today or in the last 7 days; Recent wakes -> mark each as real or false
Every other page has a Back button and returns to the dashboard after a minute without touches.
The screen stays on whatever page you chose while the assistant listens, thinks and speaks: the
dashboard and the full face show it live, other pages get an activity bar at the bottom with what
was heard and the reply as it is spoken.

  python -m assistant.screen                 full screen (on the Pi)
  python -m assistant.screen --window        800x480 window (testing on the PC; Esc closes it)
  python -m assistant.screen --demo          cycle the four states, no assistant needed
  python -m assistant.screen --shot out.png [--page main|face|clock|history|home|pi|settings|wake|wakes]
                                             draw one frame with sample data to a PNG and exit

Needs pygame-ce (`pip install pygame-ce`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

W, H = 800, 480
BG, CARD, LINE = (11, 10, 16), (21, 20, 27), (36, 34, 44)
TEXT, DIM, ACCENT, WARN, OFF = (233, 230, 242), (143, 138, 163), (185, 166, 255), (250, 199, 117), (74, 70, 88)
LABEL = {"idle": "Waiting for the wake word", "listening": "Listening…", "thinking": "Thinking…",
         "speaking": "Speaking", "offline": "Assistant offline", "muted": "Microphone off"}
MODES = ("idle", "listening", "thinking", "speaking")
PAGES = ("main", "face", "clock", "history", "home", "pi", "settings", "wake", "wakes")
FONTS = "notosans,dejavusans,freesans,segoeui,arial"
TAP_WINDOW_S = 0.6      # taps closer than this count together (triple-tap)
AUTO_BACK_S = 60.0      # pages other than the face return to the dashboard after this long untouched

SAMPLE = {"mode": "idle", "voice": True, "llm_ok": True, "mqtt": "disabled", "heard": "What time is it?",
          "said": "It's 4:39 PM on Thursday, 19 November.", "first_word_s": 4.7, "muted": False,
          "model": "qwen3:1.7b",
          "wake_words": ["models/wake/uzi.onnx@0.5", "models/wake/aiya.onnx@0.7", "models/wake/boafo.onnx@0.8"],
          "pi": {"temp_c": 51.0, "mem_used_gb": 2.6, "mem_total_gb": 7.9, "throttled": False, "volts": 5.07,
                 "fan_rpm": 3100, "uptime_s": 7380, "load": 0.42},
          "home": {"lights": {"living room": "off", "bedroom": "off", "kitchen": "on", "office": "off"},
                   "temps": {}, "devices": ["office"]}}
SAMPLE_HISTORY = [{"role": "user", "content": "What time is it?"},
                  {"role": "assistant", "content": "It's 4:39 PM on Thursday, 19 November."},
                  {"role": "user", "content": "What is the capital of Ghana?"},
                  {"role": "assistant", "content": "The capital of Ghana is Accra."}]


def _sample_stats(woke, no_speech, replied, near, real=0, false=0):
    return {"woke": woke, "no_speech": no_speech, "phantom": woke - no_speech - replied, "replied": replied,
            "near_miss": near, "ignored": 0, "real": real, "false": false}


SAMPLE_WAKE = {"days": 1, "step": 0.05, "lowest": 0.3, "highest": 0.95, "words": [
    {"word": "uzi", "enabled": True, "threshold": 0.5, "default": 0.5, "stats": _sample_stats(25, 17, 8, 822, 2, 3)},
    {"word": "aiya", "enabled": True, "threshold": 0.75, "default": 0.7, "stats": _sample_stats(22, 6, 16, 271)},
    {"word": "boafo", "enabled": False, "threshold": 0.8, "default": 0.8,
     "stats": _sample_stats(28, 12, 15, 1442, 0, 6)}],
    "recent": [{"id": "3", "time": 1764011471.0, "word": "boafo", "score": 1.0, "heard": "Everyone out to me.",
                "label": "false"},
               {"id": "2", "time": 1764011395.0, "word": "uzi", "score": 0.77, "heard": "Turn on the office light.",
                "label": "real"},
               {"id": "1", "time": 1764011300.0, "word": "aiya", "score": 0.97, "heard": "Good morning, everyone.",
                "label": ""}]}


# ---------------- pure parts (tested without pygame) ----------------
def face_shapes(mode: str, t: float) -> list[tuple]:
    """The visor face in a 400x240 box, as drawing commands. Same geometry as the web page.

    ("poly", points) filled eye with round corners; ("bar", x, y, w, h); ("dot", x, y, r, alpha);
    ("ring", x, y, r, alpha).
    """
    h = 58 if mode == "listening" else 20 if mode == "thinking" else 40
    if mode in ("idle", "offline") and (t % 4.2) < 0.14:
        h = 4  # blink
    if mode == "offline":
        h = 10
    if mode == "muted":
        h = 4  # eyes closed
    slant = 0 if mode == "listening" else 12
    dx = math.sin(t * 3) * 14 if mode == "thinking" else 0
    out: list[tuple] = []
    for x0, d in ((135, 1), (265, -1)):
        x, y, w = x0 + dx, 105, 44
        xo, xi = x - d * w, x + d * w
        out.append(("poly", [(xo, y - h / 2), (xi, y - h / 2 + slant), (xi, y + h / 2), (xo, y + h / 2)]))
    if mode == "speaking":
        for i in range(11):
            b = 6 + abs(math.sin(t * 9 + i * 1.3) * math.sin(t * 4 + i)) * 34
            out.append(("bar", 150 + i * 9.5, 190 - b / 2, 5, b))
    elif mode == "thinking":
        for i in range(3):
            out.append(("dot", 180 + i * 20, 190, 6, 0.25 + 0.75 * max(0.0, math.sin(t * 5 - i * 0.9))))
    elif mode == "listening":
        r = 8 + (t * 30 % 30)
        out += [("dot", 200, 190, 6, 1.0), ("ring", 200, 190, r, 1 - (r - 8) / 30)]
    return out


def clock_text(now: datetime, seconds: bool = False) -> str:
    sec = f":{now:%S}" if seconds else ""
    return f"{now.hour % 12 or 12}:{now:%M}{sec} {'AM' if now.hour < 12 else 'PM'}"


def date_text(now: datetime) -> str:
    return f"{now:%A} {now.day} {now:%B}"


def wake_word_list(entries: list[str]) -> list[str]:
    """'models/wake/uzi.onnx@0.5' -> 'Uzi (0.5)'; built-ins like 'hey_jarvis' -> 'Hey jarvis'."""
    out = []
    for e in entries:
        name, _, thr = e.partition("@")
        base = name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ")
        out.append(base[:1].upper() + base[1:] + (f" ({thr})" if thr else ""))
    return out


def duration_text(seconds: int | None) -> str:
    if seconds is None:
        return "–"
    d, rest = divmod(int(seconds), 86400)
    h, rest = divmod(rest, 3600)
    m = rest // 60
    return (f"{d} d " if d else "") + f"{h} h {m} min"


def describe(st: dict | None) -> dict:
    """Everything the dashboard writes, from a /state reply (None = assistant unreachable)."""
    if st is None:
        return {"mode": "offline", "label": LABEL["offline"], "temp": "–", "temp_warn": False, "mem": "",
                "thr": "", "thr_warn": False, "said": "Nothing yet", "said_dim": True, "heard": "",
                "lights": "", "temps": "", "muted": False,
                "dots": {"Model": False, "Voice": False, "Devices": False}}
    pi = st.get("pi") or {}
    mode = st.get("mode", "idle")
    muted = bool(st.get("muted"))
    if muted and mode == "idle":
        mode = "muted"
    label = ("Ready · voice off" if mode == "idle" and not st.get("voice") else LABEL.get(mode, mode))
    temp = pi.get("temp_c")
    lights = [k for k, v in ((st.get("home") or {}).get("lights") or {}).items() if v == "on"]
    temps = ((st.get("home") or {}).get("temps") or {}).items()
    heard = st.get("heard") or ""
    if heard and st.get("first_word_s"):
        heard += f" · {st['first_word_s']} s"
    return {
        "mode": mode, "label": label, "muted": muted,
        "temp": f"{temp:.0f} °C" if temp is not None else "–", "temp_warn": temp is not None and temp >= 80,
        "mem": (f"{pi['mem_used_gb']:.1f} / {pi['mem_total_gb']:.0f} GB memory"
                if pi.get("mem_used_gb") is not None else ""),
        "thr": "" if pi.get("throttled") is None else "Throttling" if pi["throttled"] else "No throttling",
        "thr_warn": bool(pi.get("throttled")),
        "said": st.get("said") or "Nothing yet", "said_dim": not st.get("said"),
        "heard": f"You: {heard}" if heard else "",
        "lights": f"Lights on: {', '.join(lights)}" if lights else "All lights off",
        "temps": " · ".join(f"{k} {v:.1f} °C" for k, v in temps) or "No temperature sensors yet",
        "dots": {"Model": bool(st.get("llm_ok")), "Voice": bool(st.get("voice")),
                 "Devices": st.get("mqtt") == "connected"},
    }


def activity_lines(d: dict) -> tuple[str, str]:
    """What the activity bar / face caption shows while the assistant is busy: (state, text).
    Your words appear once speech-to-text is done; the reply as each sentence starts playing."""
    heard = d["heard"].split(" · ")[0] if d["heard"] else ""
    if d["mode"] == "listening":
        return "Listening…", ""
    if d["mode"] == "thinking":
        return "Thinking…", heard
    if d["mode"] == "speaking":
        return "Speaking", "" if d["said_dim"] else d["said"]
    return d["label"], ""


def page_to_show(page: str, mode: str) -> str:
    """The page never changes by itself when the assistant is busy (the user's choice, 2026-11-20)."""
    return page


def card_rects() -> dict[str, tuple[int, int, int, int]]:
    """The three dashboard cards (x, y, w, h): Pi, Last reply, Home."""
    unit = (W - 48 - 28) / 3.5
    xs, ws = (24, 24 + unit + 14, 24 + unit * 2.5 + 28), (unit, unit * 1.5, unit)
    return {name: (int(x), 256, int(w), 200) for name, x, w in zip(("pi", "reply", "home"), xs, ws)}


MAIN_REGIONS = {"face": (482, 16, 300, 180), "clock": (20, 8, 450, 140),
                "mute": (300, 182, 84, 34), "settings": (392, 182, 84, 34), **card_rects()}
BACK = (0, 0, 150, 64)
SETTINGS_REGIONS = {"mic": (560, 104, 200, 52), "dim": (560, 176, 200, 52), "wake": (560, 248, 200, 52),
                    "exit": (560, 404, 200, 52)}
EXIT_CONFIRM_S = 3.0    # Exit needs a second tap within this time
HOME_ROW_Y, HOME_ROW_STEP, HOME_ROWS = 116, 58, 5
HOME_REGIONS = {f"light:{i}": (40, HOME_ROW_Y + i * HOME_ROW_STEP, 720, 48) for i in range(HOME_ROWS)}
# Wake words page: one row per word (on/off, threshold - and +, its numbers), the period, Recent wakes
WAKE_ROW_Y, WAKE_ROW_STEP, WAKE_ROWS = 92, 78, 4
WAKE_REGIONS = {"period": (630, 12, 150, 40), "recent": (40, 416, 260, 48)}
for _i in range(WAKE_ROWS):
    _y = WAKE_ROW_Y + _i * WAKE_ROW_STEP
    WAKE_REGIONS.update({f"wtoggle:{_i}": (150, _y + 14, 84, 40), f"wdown:{_i}": (252, _y + 14, 48, 40),
                         f"wup:{_i}": (380, _y + 14, 48, 40)})
# Recent wakes page: newest first, six per screen, Real / False on each
WAKES_ROW_Y, WAKES_ROW_STEP, WAKES_ROWS = 72, 56, 6
WAKES_REGIONS = {"older": (40, 420, 130, 44), "newer": (184, 420, 130, 44)}
for _i in range(WAKES_ROWS):
    _y = WAKES_ROW_Y + _i * WAKES_ROW_STEP
    WAKES_REGIONS.update({f"real:{_i}": (600, _y + 6, 76, 36), f"false:{_i}": (688, _y + 6, 76, 36)})
HISTORY_REGIONS = {"clear": (600, 12, 180, 40)}   # Clear history: two taps, like Exit
BACK_TO = {"wake": "settings", "wakes": "wake"}   # sub-pages go back one level; the others to the dashboard


def region_at(page: str, pos: tuple[int, int]) -> str | None:
    x, y = pos
    extra = {"settings": SETTINGS_REGIONS, "home": HOME_REGIONS, "wake": WAKE_REGIONS,
             "wakes": WAKES_REGIONS, "history": HISTORY_REGIONS}.get(page, {})
    regions = MAIN_REGIONS if page == "main" else {"back": BACK, **extra}
    if page == "face":
        return "face"
    for name, (rx, ry, rw, rh) in regions.items():
        if rx <= x < rx + rw and ry <= y < ry + rh:
            return name
    return None


class App:
    """Which page is showing, and what touches do. Pure logic, so it can be tested without a screen.
    tap() and tick() return actions: ("page", name), ("exit",), ("toggle_mute",), ("dim", on),
    ("light", row) for the Home page's n-th light; on the wake pages ("wake_toggle", row),
    ("wake_step", row, -1|+1), ("wake_period", days), ("wake_label", index in the recent list, label)."""

    def __init__(self):
        self.page = "main"
        self.dim = False
        self.wake_days = 1        # Wake words page: today (1) or the last 7 days
        self.wakes_offset = 0     # Recent wakes page: how many newer wakes are scrolled past
        self.last_touch = 0.0
        self.taps: list[float] = []          # recent taps counting towards a triple-tap
        self.exit_armed: float | None = None   # first tap on Exit: waiting for the confirming tap

    def tap(self, pos: tuple[int, int], now: float) -> list[tuple]:
        self.last_touch = now
        region = region_at(self.page, pos)
        if self.page == "face":  # nothing on the full face reacts, except a triple-tap
            if self._count(now) >= 3:
                return self._go("main")
            return []
        if self.page == "main":
            self.taps = []
            if region == "mute":
                return [("toggle_mute",)]
            target = {"face": "face", "clock": "clock", "pi": "pi", "reply": "history", "home": "home",
                      "settings": "settings"}.get(region or "")
            return self._go(target) if target else []
        if region in ("exit", "clear"):   # both need a second tap within EXIT_CONFIRM_S
            if self.exit_armed is not None and now - self.exit_armed <= EXIT_CONFIRM_S:
                self.exit_armed = None
                return [("exit",) if region == "exit" else ("clear_history",)]
            self.exit_armed = now
            return []
        self.exit_armed = None
        if region == "back":
            return self._go(BACK_TO.get(self.page, "main"))
        if self.page == "settings" and region == "wake":
            return self._go("wake")
        if self.page == "wake":
            return self._wake_tap(region or "")
        if self.page == "wakes":
            return self._wakes_tap(region or "")
        if region == "mic":
            return [("toggle_mute",)]
        if region == "dim":
            self.dim = not self.dim
            return [("dim", self.dim)]
        if region and region.startswith("light:"):
            return [("light", int(region[6:]))]
        return []

    def _wake_tap(self, region: str) -> list[tuple]:
        if region == "period":
            self.wake_days = 1 if self.wake_days == 7 else 7
            return [("wake_period", self.wake_days)]
        if region == "recent":
            self.wakes_offset = 0
            return self._go("wakes")
        name, _, row = region.partition(":")
        if name == "wtoggle":
            return [("wake_toggle", int(row))]
        if name in ("wdown", "wup"):
            return [("wake_step", int(row), -1 if name == "wdown" else 1)]
        return []

    def _wakes_tap(self, region: str) -> list[tuple]:
        if region == "older":
            self.wakes_offset += WAKES_ROWS
            return []
        if region == "newer":
            self.wakes_offset = max(0, self.wakes_offset - WAKES_ROWS)
            return []
        name, _, row = region.partition(":")
        if name in ("real", "false"):
            return [("wake_label", self.wakes_offset + int(row), name)]
        return []

    def tick(self, now: float) -> list[tuple]:
        if self.exit_armed is not None and now - self.exit_armed > EXIT_CONFIRM_S:
            self.exit_armed = None
        if self.page not in ("main", "face") and now - self.last_touch > AUTO_BACK_S:
            return self._go("main")
        return []

    def _count(self, now: float) -> int:
        self.taps = [t for t in self.taps if now - t < TAP_WINDOW_S] + [now]
        return len(self.taps)

    def _go(self, page: str) -> list[tuple]:
        self.page, self.taps, self.exit_armed = page, [], None
        return [("page", page)]


class Poller:
    """Fetches /state in the background; .state is None while the assistant can't be reached.
    Also fetches /history on request and sends mute on/off."""

    def __init__(self, url: str, token: str = "", start: bool = True):
        self.base, self.token = url.rstrip("/"), token
        self.url = self.base + "/state"
        self.state: dict | None = None
        self.history: list[dict] = []
        self.fails = 0
        self.pending: dict[str, str] = {}             # room -> state asked for, until the reply
        self.light_msg: tuple[str, float] | None = None  # last light result and when
        self.wake: dict | None = None                    # GET /wake, while a wake page is open
        self.wake_msg: tuple[str, float] | None = None   # why a wake change was refused, and when
        self.history_msg: tuple[str, float] | None = None  # result of Clear history, and when
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def _request(self, path: str, body: dict | None = None):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read())

    def _run(self) -> None:
        while True:
            try:
                self.state, self.fails = self._request("/state"), 0
            except (OSError, ValueError):
                self.fails += 1
                if self.fails >= 3:
                    self.state = None
            busy = self.state is not None and self.state.get("mode") != "idle"
            time.sleep(0.25 if busy else 1.0)

    def fetch_history(self) -> None:
        def run():
            try:
                self.history = self._request("/history").get("messages", [])
            except (OSError, ValueError):
                pass
        threading.Thread(target=run, daemon=True).start()

    def set_light(self, room: str, state: str) -> None:
        """Tap on the Home page. While it waits for the device, the row shows "…"; the result
        (or why it failed) is shown under the list."""
        if room in self.pending:
            return
        self.pending[room] = state

        def run():
            try:
                body = json.dumps({"room": room, "state": state}).encode()
                headers = {"Content-Type": "application/json"}
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"
                req = urllib.request.Request(self.base + "/light", data=body, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        msg = json.loads(r.read())["message"]
                except urllib.error.HTTPError as e:
                    msg = json.loads(e.read() or b"{}").get("message") or f"Couldn't switch it (error {e.code})."
            except (OSError, ValueError, KeyError):
                msg = "Couldn't reach the assistant."
            self.light_msg = (msg, time.monotonic())
            self.pending.pop(room, None)
        threading.Thread(target=run, daemon=True).start()

    def fetch_wake(self, days: int) -> None:
        """GET /wake into .wake ({"off": True} when the assistant has no wake words)."""
        def run():
            try:
                self.wake = self._request(f"/wake?days={days}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    self.wake = {"off": True}
            except (OSError, ValueError):
                pass
        threading.Thread(target=run, daemon=True).start()

    def change_wake(self, path: str, body: dict) -> None:
        """POST /wake or /wake/label; the reply is the new state, or why it was refused (shown on the page)."""
        def run():
            try:
                self.wake = self._request(path, body)
            except urllib.error.HTTPError as e:
                try:
                    msg = json.loads(e.read() or b"{}").get("error") or f"Refused (error {e.code})."
                except ValueError:
                    msg = f"Refused (error {e.code})."
                self.wake_msg = (msg, time.monotonic())
            except (OSError, ValueError):
                self.wake_msg = ("Couldn't reach the assistant.", time.monotonic())
        threading.Thread(target=run, daemon=True).start()

    def clear_history(self) -> None:
        def run():
            try:
                n = self._request("/history/clear", {}).get("deleted", 0)
                self.history = []
                self.history_msg = (f"History cleared ({n} messages).", time.monotonic())
            except (OSError, ValueError):
                self.history_msg = ("Couldn't clear the history.", time.monotonic())
        threading.Thread(target=run, daemon=True).start()

    def set_mute(self, muted: bool) -> None:
        def run():
            try:
                self._request("/mute", {"muted": muted})
                if self.state is not None:
                    self.state["muted"] = muted
            except (OSError, ValueError):
                pass
        threading.Thread(target=run, daemon=True).start()


# ---------------- drawing ----------------
class Painter:
    SS = 2  # the face is drawn at 2x and smoothed down, for clean edges

    def __init__(self, pg):
        self.pg = pg
        self._fonts: dict = {}
        self._text: dict = {}
        self._faces: dict = {}

    def font(self, size: int):
        if size not in self._fonts:
            self._fonts[size] = self.pg.font.SysFont(FONTS, size)
        return self._fonts[size]

    def text(self, s: str, size: int, color) -> "object":
        key = (s, size, color)
        if key not in self._text:
            if len(self._text) > 400:
                self._text.clear()
            self._text[key] = self.font(size).render(s, True, color)
        return self._text[key]

    def blit_center(self, screen, s: str, size: int, color, cx: int, cy: int) -> None:
        img = self.text(s, size, color)
        screen.blit(img, img.get_rect(center=(cx, cy)))

    def fit(self, s: str, size: int, width: int) -> str:
        f = self.font(size)
        if f.size(s)[0] <= width:
            return s
        while s and f.size(s + "…")[0] > width:
            s = s[:-1]
        return s.rstrip() + "…"

    def wrap(self, s: str, size: int, width: int, lines: int) -> list[str]:
        f, out, cur = self.font(size), [], ""
        for word in s.split():
            test = f"{cur} {word}".strip()
            if f.size(test)[0] <= width:
                cur = test
            else:
                out.append(cur)
                cur = word
        out.append(cur)
        if len(out) > lines:
            out = out[:lines]
            out[-1] = self.fit(out[-1] + " …", size, width) if not out[-1].endswith("…") else out[-1]
        return out

    def draw_face(self, screen, mode: str, t: float, rect=(W - 18 - 300, 16, 300, 180)) -> None:
        pg = self.pg
        x0, y0, w, h = rect
        k = (w / 400) * self.SS
        c = OFF if mode in ("offline", "muted") else ACCENT
        if (w, h) not in self._faces:
            self._faces[(w, h)] = pg.Surface((w * self.SS, h * self.SS), pg.SRCALPHA)
        f = self._faces[(w, h)]
        f.fill((0, 0, 0, 0))
        for shape in face_shapes(mode, t):
            if shape[0] == "poly":
                pts = [(x * k, y * k) for x, y in shape[1]]
                pg.draw.polygon(f, c, pts)
                pg.draw.lines(f, c, True, pts, int(10 * k))
                for p in pts:  # round corners, like stroke-linejoin: round
                    pg.draw.circle(f, c, p, 5 * k)
            elif shape[0] == "bar":
                _, x, y, bw, bh = shape
                pg.draw.rect(f, c, (x * k, y * k, bw * k, bh * k), border_radius=int(2.5 * k))
            elif shape[0] == "dot":
                _, x, y, r, a = shape
                pg.draw.circle(f, (*c, int(255 * a)), (x * k, y * k), r * k)
            elif shape[0] == "ring":
                _, x, y, r, a = shape
                pg.draw.circle(f, (*c, int(255 * max(0.0, a))), (x * k, y * k), r * k, max(1, int(2 * k)))
        screen.blit(pg.transform.smoothscale(f, (w, h)), (x0, y0))

    def card(self, screen, rect, title: str) -> None:
        x, y, w, h = rect
        self.pg.draw.rect(screen, CARD, rect, border_radius=14)
        self.pg.draw.rect(screen, LINE, rect, 1, border_radius=14)
        screen.blit(self.text(title, 14, DIM), (x + 18, y + 16))

    def pill(self, screen, rect, label: str, on: bool) -> None:
        self.pg.draw.rect(screen, CARD, rect, border_radius=rect[3] // 2)
        self.pg.draw.rect(screen, ACCENT if on else LINE, rect, 1, border_radius=rect[3] // 2)
        self.blit_center(screen, label, 15, ACCENT if on else DIM, rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)

    # --- pages
    def draw(self, screen, d: dict, now: datetime, t: float) -> None:
        """The main dashboard."""
        screen.fill(BG)
        screen.blit(self.text(clock_text(now), 80, TEXT), (28, 18))
        screen.blit(self.text(date_text(now), 20, DIM), (32, 116))
        screen.blit(self.text(d["label"], 19, ACCENT), (32, 154))
        x = 32
        for name, ok in d["dots"].items():
            self.pg.draw.circle(screen, ACCENT if ok else OFF, (x + 4, 201), 4.5)
            screen.blit(self.text(name, 14, DIM), (x + 15, 192))
            x += 15 + self.font(14).size(name)[0] + 18
        self.pill(screen, MAIN_REGIONS["mute"], "Mic off" if d.get("muted") else "Mic on", not d.get("muted"))
        self.pill(screen, MAIN_REGIONS["settings"], "Settings", False)
        self.draw_face(screen, d["mode"], t)

        cards = card_rects()
        (x1, y1, w1, _), (x2, _, w2, _), (x3, _, w3, _) = cards["pi"], cards["reply"], cards["home"]
        self.card(screen, cards["pi"], "Pi")
        screen.blit(self.text(d["temp"], 30, WARN if d["temp_warn"] else TEXT), (x1 + 18, y1 + 44))
        screen.blit(self.text(d["mem"], 16, DIM), (x1 + 18, y1 + 86))
        screen.blit(self.text(d["thr"], 16, WARN if d["thr_warn"] else DIM), (x1 + 18, y1 + 112))

        self.card(screen, cards["reply"], "Last reply")
        y = y1 + 44
        for line in self.wrap(d["said"], 18, w2 - 36, 4):
            screen.blit(self.text(line, 18, DIM if d["said_dim"] else TEXT), (x2 + 18, y))
            y += 24
        if d["heard"]:
            screen.blit(self.text(self.fit(d["heard"], 14, w2 - 36), 14, DIM), (x2 + 18, y + 8))

        self.card(screen, cards["home"], "Home")
        screen.blit(self.text(self.fit(d["lights"], 16, w3 - 36), 16, TEXT), (x3 + 18, y1 + 44))
        y = y1 + 70
        for line in self.wrap(d["temps"], 16, w3 - 36, 3):
            screen.blit(self.text(line, 16, DIM), (x3 + 18, y))
            y += 22

    def header(self, screen, title: str) -> None:
        screen.fill(BG)
        self.pg.draw.rect(screen, CARD, (16, 12, 110, 40), border_radius=20)
        self.blit_center(screen, "‹ Back", 18, TEXT, 71, 32)
        self.blit_center(screen, title, 26, TEXT, W // 2, 32)

    def draw_full_face(self, screen, d: dict, t: float) -> None:
        screen.fill(BG)
        busy = d["mode"] in ("listening", "thinking", "speaking")
        if not busy:
            self.draw_face(screen, d["mode"], t, rect=(100, 40, 600, 360))
            self.blit_center(screen, d["label"], 20, DIM, W // 2, 440)
            return
        self.draw_face(screen, d["mode"], t, rect=(150, 10, 500, 300))   # room for the words below
        state, text = activity_lines(d)
        self.blit_center(screen, state, 20, ACCENT, W // 2, 330)
        y = 362
        for line in self.wrap(text, 22, W - 80, 4) if text else []:
            self.blit_center(screen, line, 22, TEXT if d["mode"] == "speaking" else DIM, W // 2, y)
            y += 30

    def draw_activity_bar(self, screen, d: dict, t: float) -> None:
        """Bottom bar on the other pages while the assistant is busy."""
        pg = self.pg
        bar = pg.Surface((W, 92), pg.SRCALPHA)
        bar.fill((21, 20, 27, 245))
        screen.blit(bar, (0, H - 92))
        pg.draw.line(screen, ACCENT, (0, H - 92), (W, H - 92), 2)
        self.draw_face(screen, d["mode"], t, rect=(12, H - 86, 120, 72))
        state, text = activity_lines(d)
        screen.blit(self.text(state, 18, ACCENT), (148, H - 82))
        y = H - 56
        for line in self.wrap(text, 17, W - 170, 2) if text else []:
            screen.blit(self.text(line, 17, TEXT if d["mode"] == "speaking" else DIM), (148, y))
            y += 22

    def draw_clock(self, screen, now: datetime) -> None:
        self.header(screen, "Clock")
        self.blit_center(screen, clock_text(now, seconds=True), 120, TEXT, W // 2, 200)
        self.blit_center(screen, date_text(now), 30, DIM, W // 2, 300)
        self.blit_center(screen, "Timers and reminders will appear here.", 18, DIM, W // 2, 400)

    def draw_history(self, screen, messages: list[dict], armed: bool = False, note: str = "") -> None:
        self.header(screen, "History")
        cl = HISTORY_REGIONS["clear"]
        self.pg.draw.rect(screen, (60, 30, 36) if armed else CARD, cl, border_radius=cl[3] // 2)
        self.pg.draw.rect(screen, WARN, cl, 1, border_radius=cl[3] // 2)
        self.blit_center(screen, "Tap again to clear" if armed else "Clear history", 15, WARN,
                         cl[0] + cl[2] // 2, cl[1] + cl[3] // 2)
        if note or armed:
            msg = note or "Deletes every question and reply. Saved notes stay."
            self.blit_center(screen, msg, 15, WARN if armed else TEXT, W // 2, 66)
        if not messages:
            self.blit_center(screen, "No conversation yet", 22, DIM, W // 2, 240)
            self.blit_center(screen, "Questions and replies are kept 30 days.", 16, DIM, W // 2, 276)
            return
        lines: list[tuple[str, tuple]] = []
        for m in messages:
            who, color = ("You", DIM) if m.get("role") == "user" else ("Assistant", TEXT)
            for i, line in enumerate(self.wrap(f"{who}: {m.get('content', '')}", 18, W - 64, 3)):
                lines.append((line, color))
            lines.append(("", DIM))
        room = (H - 80) // 25
        y = 76
        for line, color in lines[-room:]:  # newest at the bottom
            if line:
                screen.blit(self.text(line, 18, color), (32, y))
            y += 25

    def draw_home(self, screen, st: dict | None, pending: dict | None = None, note: str = "") -> None:
        self.header(screen, "Home")
        home = (st or {}).get("home") or {}
        temps = home.get("temps") or {}
        devices = set(home.get("devices") or [])
        pending = pending or {}
        screen.blit(self.text("Lights", 18, DIM), (40, 80))
        rows = list((home.get("lights") or {}).items())[:HOME_ROWS]
        for i, (room, state) in enumerate(rows):
            y = HOME_ROW_Y + i * HOME_ROW_STEP
            on = state == "on"
            self.pg.draw.rect(screen, CARD, (40, y, 720, 48), border_radius=12)
            screen.blit(self.text(room.capitalize(), 20, TEXT), (60, y + 12))
            if room not in devices:
                screen.blit(self.text("no device yet", 15, DIM), (250, y + 15))
            t = temps.get(room)
            if t is not None:
                screen.blit(self.text(f"{t:.1f} °C", 18, DIM), (430, y + 13))
            label = "…" if room in pending else ("On" if on else "Off")
            self.pill(screen, (640, y + 8, 100, 32), label, on)
        self.blit_center(screen, self.fit(note or "Tap a light to switch it.", 16, W - 80), 16,
                         TEXT if note else DIM, W // 2, HOME_ROW_Y + len(rows) * HOME_ROW_STEP + 12)

    def draw_pi(self, screen, st: dict | None, d: dict) -> None:
        self.header(screen, "Pi status")
        st = st or {}
        pi = st.get("pi") or {}
        rows = [
            ("Temperature", d["temp"], d["temp_warn"]),
            ("CPU load (1 min)", f"{pi['load']:.2f}" if pi.get("load") is not None else "–", False),
            ("Memory", d["mem"].replace(" memory", "") or "–", False),
            ("Throttling", d["thr"] or "–", d["thr_warn"]),
            ("Input voltage", f"{pi['volts']:.2f} V" if pi.get("volts") else "–",
             bool(pi.get("volts")) and pi["volts"] < 4.8),
            ("Fan", f"{pi['fan_rpm']} rpm" if pi.get("fan_rpm") is not None else "–", False),
            ("Uptime", duration_text(pi.get("uptime_s")), False),
            ("Model", f"{st.get('model', '–')}  ({'running' if st.get('llm_ok') else 'not answering'})",
             not st.get("llm_ok")),
            ("Voice", ("on, mic muted" if st.get("muted") else "on") if st.get("voice") else "off", False),
            ("Wake words", ", ".join(wake_word_list(st.get("wake_words") or [])) or "–", False),
        ]
        y = 76
        for label, value, warn in rows:
            screen.blit(self.text(label, 18, DIM), (48, y))
            screen.blit(self.text(self.fit(value, 18, 470), 18, WARN if warn else TEXT), (290, y))
            y += 38

    def draw_settings(self, screen, d: dict, st: dict | None, dim: bool, exit_armed: bool = False) -> None:
        self.header(screen, "Settings")
        st = st or {}
        screen.blit(self.text("Microphone", 22, TEXT), (48, 116))
        screen.blit(self.text("Off: nothing is heard, the eyes close", 15, DIM), (48, 146))
        self.pill(screen, SETTINGS_REGIONS["mic"], "Off" if d.get("muted") else "On", not d.get("muted"))
        screen.blit(self.text("Dim screen", 22, TEXT), (48, 188))
        self.pill(screen, SETTINGS_REGIONS["dim"], "On" if dim else "Off", dim)
        y = 262
        screen.blit(self.text("Wake words", 22, TEXT), (48, 256))
        screen.blit(self.text(self.fit(", ".join(wake_word_list(st.get("wake_words") or [])) or "–", 15, 490),
                              15, DIM), (48, 286))
        self.pill(screen, SETTINGS_REGIONS["wake"], "Open ›", False)
        ex = SETTINGS_REGIONS["exit"]
        self.pg.draw.rect(screen, (60, 30, 36) if exit_armed else CARD, ex, border_radius=ex[3] // 2)
        self.pg.draw.rect(screen, WARN, ex, 1, border_radius=ex[3] // 2)
        self.blit_center(screen, "Tap again to exit" if exit_armed else "Exit", 17, WARN,
                         ex[0] + ex[2] // 2, ex[1] + ex[3] // 2)
        screen.blit(self.text("Exit the assistant", 22, TEXT), (48, 410))
        screen.blit(self.text("Stops listening and closes this screen. The desktop icon starts it again.", 14, DIM),
                    (48, 440))
        hints = ["Triple-tap the full face to come back to the dashboard.",
                 "Pages return to the dashboard after a minute without touches."]
        y += 80
        for h in hints:
            screen.blit(self.text(h, 15, DIM), (48, y))
            y += 24

    def button(self, screen, rect, label: str, on: bool = False, size: int = 17) -> None:
        """A rounded button; `on` draws it highlighted (a chosen mark, a word that is on)."""
        x, y, w, h = rect
        self.pg.draw.rect(screen, (44, 38, 70) if on else CARD, rect, border_radius=10)
        self.pg.draw.rect(screen, ACCENT if on else LINE, rect, 1, border_radius=10)
        self.blit_center(screen, label, size, ACCENT if on else TEXT, x + w // 2, y + h // 2)

    def draw_wake(self, screen, data: dict | None, note: str = "", days: int = 1) -> None:
        self.header(screen, "Wake words")
        x, y, w, h = WAKE_REGIONS["period"]
        self.button(screen, (x, y, w, h), "Today" if days == 1 else "Last 7 days", True, 16)
        if not data or data.get("off") or "words" not in data:
            msg = "The assistant has no wake words (voice is off)." if data and data.get("off") else "Loading…"
            self.blit_center(screen, msg, 20, DIM, W // 2, 240)
            return
        screen.blit(self.text("Threshold: higher = fewer wakes", 13, DIM), (252, 70))
        cols = (("woke", "woke"), ("no_speech", "no speech"), ("replied", "replied"), ("near_miss", "near miss"))
        for i, wd in enumerate(data["words"][:WAKE_ROWS]):
            y = WAKE_ROW_Y + i * WAKE_ROW_STEP
            on = wd["enabled"]
            self.pg.draw.rect(screen, CARD, (24, y + 4, 752, 70), border_radius=12)
            screen.blit(self.text(wd["word"].capitalize(), 21, TEXT if on else DIM), (40, y + 12))
            screen.blit(self.text(false_text(wd["stats"]), 13, DIM), (40, y + 44))
            self.pill(screen, WAKE_REGIONS[f"wtoggle:{i}"], "On" if on else "Off", on)
            self.button(screen, WAKE_REGIONS[f"wdown:{i}"], "−", size=22)
            self.button(screen, WAKE_REGIONS[f"wup:{i}"], "+", size=22)
            self.blit_center(screen, f"{wd['threshold']:.2f}", 20, TEXT if on else DIM, 340, y + 30)
            if abs(wd["threshold"] - wd.get("default", wd["threshold"])) > 1e-9:
                self.blit_center(screen, f"was {wd['default']:.2f}", 12, DIM, 340, y + 54)
            for k, (key, label) in enumerate(cols):
                cx = 486 + k * 76
                self.blit_center(screen, str(wd["stats"].get(key, 0)), 20, TEXT if on else DIM, cx, y + 26)
                self.blit_center(screen, label, 12, DIM, cx, y + 52)
        self.button(screen, WAKE_REGIONS["recent"], "Recent wakes ›")
        hint = note or "Changes take effect at once and are kept after a restart."
        screen.blit(self.text(self.fit(hint, 15, 450), 15, WARN if note else DIM), (320, 432))

    def draw_wakes(self, screen, data: dict | None, offset: int, now: datetime) -> None:
        self.header(screen, "Recent wakes")
        recent = (data or {}).get("recent") or []
        if not recent:
            self.blit_center(screen, "No questions after a wake word since the assistant started.", 18, DIM,
                             W // 2, 220)
            self.blit_center(screen, "Each one appears here, to mark it real or false.", 16, DIM, W // 2, 252)
            return
        offset = clamp_offset(offset, len(recent))
        for i, r in enumerate(recent[offset:offset + WAKES_ROWS]):
            y = WAKES_ROW_Y + i * WAKES_ROW_STEP
            self.pg.draw.rect(screen, CARD, (24, y, 752, 48), border_radius=12)
            screen.blit(self.text(wake_time_text(r["time"], now), 15, DIM), (40, y + 6))
            screen.blit(self.text(r["word"].capitalize(), 15, ACCENT), (40, y + 26))
            screen.blit(self.text(self.fit(f"“{r.get('heard', '')}”", 18, 450), 18, TEXT), (136, y + 13))
            self.button(screen, WAKES_REGIONS[f"real:{i}"], "Real", r.get("label") == "real", 16)
            self.button(screen, WAKES_REGIONS[f"false:{i}"], "False", r.get("label") == "false", 16)
        if offset + WAKES_ROWS < len(recent):
            self.button(screen, WAKES_REGIONS["older"], "Older ›", size=16)
        if offset > 0:
            self.button(screen, WAKES_REGIONS["newer"], "‹ Newer", size=16)
        screen.blit(self.text("Real: you were talking to it. Tap again to clear.", 14, DIM), (336, 434))


def home_note(light_msg: tuple[str, float] | None, now: float, keep_s: float = 6.0) -> str:
    """The last light result, shown under the list for a few seconds."""
    return light_msg[0] if light_msg and now - light_msg[1] < keep_s else ""


def light_to_switch(st: dict | None, row: int) -> tuple[str, str] | None:
    """(room, new state) for a tap on the Home page's row, or None if there's no such row."""
    rooms = list((((st or {}).get("home") or {}).get("lights") or {}).items())[:HOME_ROWS]
    if not 0 <= row < len(rooms):
        return None
    room, state = rooms[row]
    return room, ("off" if state == "on" else "on")


def wake_request(data: dict | None, act: tuple, days: int = 1) -> tuple[str, dict] | None:
    """(path, body) to POST for a wake page action, from the last GET /wake; None if nothing to do."""
    if not data or "words" not in data:
        return None
    words, recent = data["words"], data.get("recent") or []
    if act[0] in ("wake_toggle", "wake_step"):
        if not 0 <= act[1] < len(words):
            return None
        w = words[act[1]]
        if act[0] == "wake_toggle":
            return "/wake", {"word": w["word"], "enabled": not w["enabled"], "days": days}
        thr = round(w["threshold"] + act[2] * data.get("step", 0.05), 2)
        thr = min(data.get("highest", 0.95), max(data.get("lowest", 0.3), thr))
        return None if thr == w["threshold"] else ("/wake", {"word": w["word"], "threshold": thr, "days": days})
    if act[0] == "wake_label" and 0 <= act[1] < len(recent):
        r = recent[act[1]]
        return "/wake/label", {"id": r["id"], "label": "" if r.get("label") == act[2] else act[2], "days": days}
    return None


def clamp_offset(offset: int, n: int) -> int:
    """Recent wakes: never scroll past the last screenful."""
    return max(0, min(offset, (max(n, 1) - 1) // WAKES_ROWS * WAKES_ROWS))


def false_text(stats: dict) -> str:
    marked = stats.get("real", 0) + stats.get("false", 0)
    return f"{stats.get('false', 0)}/{marked} false" if marked else "not marked"


def wake_time_text(ts: float, now: datetime) -> str:
    t = datetime.fromtimestamp(ts)
    return f"{t:%H:%M}" if t.date() == now.date() else f"{t:%a %H:%M}"


def render(painter, screen, app: App, st: dict | None, history: list[dict], now: datetime, t: float,
           home_extra: tuple = (), wake_extra: tuple = (), history_note: str = "") -> None:
    d = describe(st)
    busy = d["mode"] in ("listening", "thinking", "speaking")
    page = page_to_show(app.page, d["mode"])
    if page == "main":
        painter.draw(screen, d, now, t)
    elif page == "face":
        painter.draw_full_face(screen, d, t)
    elif page == "clock":
        painter.draw_clock(screen, now)
    elif page == "history":
        painter.draw_history(screen, history, app.exit_armed is not None, history_note)
    elif page == "home":
        painter.draw_home(screen, st, *home_extra)
    elif page == "pi":
        painter.draw_pi(screen, st, d)
    elif page == "settings":
        painter.draw_settings(screen, d, st, app.dim, app.exit_armed is not None)
    elif page == "wake":
        painter.draw_wake(screen, *(wake_extra or (None, "")), days=app.wake_days)
    elif page == "wakes":
        painter.draw_wakes(screen, (wake_extra or (None,))[0], app.wakes_offset, now)
    if busy and page not in ("main", "face"):
        painter.draw_activity_bar(screen, d, t)
    if app.dim:
        shade = painter.pg.Surface((W, H), painter.pg.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        screen.blit(shade, (0, 0))


def stop_everything(url: str) -> None:
    """Settings > Exit: stop the assistant service (no listening, no API) and let the language model
    leave memory now instead of after its keep-alive. The desktop icon starts it all again."""
    import subprocess
    subprocess.run(["systemctl", "--user", "stop", "assistant.service"], check=False, timeout=30)
    try:
        from .config import load_config
        llm = load_config("config.toml").llm
        req = urllib.request.Request(llm.host + "/api/generate",
                                     data=json.dumps({"model": llm.model, "keep_alive": 0}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).close()
    except Exception:  # noqa: BLE001 - best effort; Ollama unloads it later anyway
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="the assistant's API")
    ap.add_argument("--token", default=os.environ.get("ASSISTANT_VIEW_TOKEN", ""),
                    help="view token, only needed for an assistant on another machine")
    ap.add_argument("--window", action="store_true", help="800x480 window instead of full screen")
    ap.add_argument("--demo", action="store_true", help="cycle the four states with sample data")
    ap.add_argument("--shot", metavar="PNG", help="draw one frame with sample data, save it and exit")
    ap.add_argument("--mode", choices=MODES + ("offline", "muted"), default="speaking", help="state for --shot")
    ap.add_argument("--page", choices=PAGES, default="main", help="page for --shot")
    a = ap.parse_args()

    if a.shot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame as pg
    pg.init()
    if a.shot:
        screen = pg.Surface((W, H))
    else:
        flags = 0 if a.window else pg.FULLSCREEN | pg.SCALED
        screen = pg.display.set_mode((W, H), flags)
        pg.display.set_caption("Assistant")
        pg.mouse.set_visible(False)
    painter = Painter(pg)
    app = App()

    if a.shot:
        if a.mode == "offline":
            st = None
        elif a.mode == "muted":
            st = {**SAMPLE, "mode": "idle", "muted": True}
        else:
            st = {**SAMPLE, "mode": a.mode}
        app.page = a.page
        render(painter, screen, app, st, SAMPLE_HISTORY, datetime.now(), 1.0, wake_extra=(SAMPLE_WAKE, ""))
        pg.image.save(screen, a.shot)
        print(f"saved {a.shot}")
        return

    poller = None if a.demo else Poller(a.url, a.token)
    debug = os.environ.get("ASSISTANT_SCREEN_DEBUG") == "1"
    clock, t0, last_history, last_wake = pg.time.Clock(), time.monotonic(), -1e9, -1e9
    while True:
        now = time.monotonic()
        actions: list[tuple] = []
        for e in pg.event.get():
            if debug and e.type not in (pg.ACTIVEEVENT, pg.VIDEOEXPOSE):
                print(f"event {pg.event.event_name(e.type)} {e.dict}", flush=True)
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)):
                pg.quit()
                return
            if e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                acts = app.tap(e.pos, now)
                print(f"tap {e.pos} on {app.page}: {acts}", flush=True)  # in the screen's log
                actions += acts
            elif e.type == pg.FINGERDOWN:  # raw touch; SDL also sends it as a mouse click (handled above)
                print(f"finger {e.x:.2f},{e.y:.2f}", flush=True)
        actions += app.tick(now)
        st = {**SAMPLE, "mode": MODES[int((now - t0) // 3) % 4]} if a.demo else poller.state
        for act in actions:
            if act[0] == "exit":
                pg.quit()
                if not a.demo:
                    stop_everything(a.url)
                return
            if act[0] == "toggle_mute" and poller is not None and st is not None:
                poller.set_mute(not st.get("muted"))
            if act == ("page", "history"):
                last_history = -1e9
            if act[0] == "light" and poller is not None:
                target = light_to_switch(st, act[1])
                if target:
                    poller.set_light(*target)
            if act[0] in ("page", "wake_period") and act[-1] in ("wake", "wakes", 1, 7):
                last_wake = -1e9                       # fetch at once on opening / switching period
            if act[0] == "clear_history" and poller is not None:
                poller.clear_history()
            if act[0] in ("wake_toggle", "wake_step", "wake_label") and poller is not None:
                req = wake_request(poller.wake, act, app.wake_days)
                if req:
                    poller.change_wake(*req)
        history = SAMPLE_HISTORY if a.demo else (poller.history if poller else [])
        if app.page == "history" and poller is not None and now - last_history > 5:
            poller.fetch_history()
            last_history = now
        if app.page in ("wake", "wakes") and poller is not None and now - last_wake > 3:
            poller.fetch_wake(app.wake_days)
            last_wake = now
        wake_data = SAMPLE_WAKE if a.demo else (poller.wake if poller else None)
        if wake_data and app.page == "wakes":
            app.wakes_offset = clamp_offset(app.wakes_offset, len(wake_data.get("recent") or []))
        extra = (poller.pending, home_note(poller.light_msg, now)) if poller else ()
        wake_extra = (wake_data, home_note(poller.wake_msg, now) if poller else "")
        render(painter, screen, app, st, history, datetime.now(), now - t0, extra, wake_extra,
               home_note(poller.history_msg, now) if poller else "")
        pg.display.flip()
        d_mode = describe(st)["mode"]
        clock.tick(10 if d_mode in ("idle", "offline", "muted") and app.page != "clock" else 24)


if __name__ == "__main__":
    main()
