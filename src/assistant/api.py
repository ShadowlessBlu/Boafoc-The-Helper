"""Small JSON HTTP API (standard library only).

GET  /health  -> status of the LLM server, database, MQTT and uptime (no auth)
GET  /face    -> the assistant's face and dashboard for the 7-inch screen (static page)
GET  /history -> recent questions and answers (same access rule as /state)
POST /history/clear -> delete every question and reply (the screen's Clear history; same access as /mute).
                 Saved facts and the audit log stay. {"deleted": n}
POST /mute    -> {"muted": true|false}: microphone off/on (from the Pi itself, or with the full token)
POST /light   -> {"room": "office", "state": "on"|"off"}: the Home page's light buttons (same access as
                 /mute). Goes through the Governor (room check, rate limit, audit) like a spoken
                 command, and only reports success once the device confirms:
                 {"ok": true|false, "room", "state", "message"}
GET  /state   -> what the assistant is doing, last exchange, Pi readings (polled by /face).
                 Only from the Pi itself, or with the token or the read-only view token
                 (other devices open /face#view=<view token>): it contains what was said.
GET  /wake?days=1|7 -> the wake words (on/off, threshold), what each did over today or the last 7 days,
                 and the recent wakes with what was heard (same access as /state)
POST /wake    -> {"word": "boafo", "enabled": false} and/or {"threshold": 0.85}: takes effect at once
                 and is kept (same access as /mute). Replies with the new GET /wake, or 400 + "error"
POST /wake/label -> {"id": "...", "label": "real"|"false"|""}: mark a recent wake (same access as /mute)
POST /ask     -> {"text": "..."} -> {"reply": "...", "seconds": 1.2}
                 needs "Authorization: Bearer <token>" when a token is configured.

Sensitive actions are never confirmed over the API: they are declined, and the
user has to confirm them by voice on the device.
"""
from __future__ import annotations

import hmac
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .governor import Verdict
from .status import Status

MAX_BODY = 8 * 1024
FACE_PAGE = Path(__file__).parent / "face" / "index.html"
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


class AssistantAPI:
    def __init__(self, brain, token: str = "", llm_host: str = "",
                 mqtt_status: Callable[[], str] | None = None, lock: threading.Lock | None = None,
                 status: Status | None = None, view_token: str = "", wake_control=None):
        self.brain = brain
        self.wake = wake_control   # voice/wake_control.py; None when voice is off
        self.token = token
        self.view_token = view_token
        self.llm_host = llm_host.rstrip("/")
        self.mqtt_status = mqtt_status or (lambda: "disabled")
        self.lock = lock or threading.Lock()   # shared with the voice loop: one request at a time
        self.started = time.time()
        self.status = status or Status()
        self._llm_check = (0.0, None)  # (when, ok) - /state is polled often, the LLM check is cached

    def llm_ok(self) -> bool | None:
        when, ok = self._llm_check
        if self.llm_host and time.time() - when > 10:
            try:
                with urllib.request.urlopen(self.llm_host + "/api/tags", timeout=2) as r:
                    ok = r.status == 200
            except OSError:
                ok = False
            self._llm_check = (time.time(), ok)
        return ok

    def history(self, n: int = 12) -> dict:
        with self.lock:
            return {"messages": self.brain.memory.recent_messages(max(1, min(n, 50)))}

    def state(self) -> dict:
        return {**self.status.snapshot(), "llm_ok": self.llm_ok(), "mqtt": self.mqtt_status()}

    def health(self) -> dict:
        checks = {}
        if self.llm_host:
            try:
                with urllib.request.urlopen(self.llm_host + "/api/tags", timeout=3) as r:
                    checks["llm"] = "ok" if r.status == 200 else f"http {r.status}"
            except OSError as e:
                checks["llm"] = f"down: {e.__class__.__name__}"
        try:
            with self.lock:
                self.brain.memory.db.execute("SELECT 1").fetchone()
            checks["database"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["database"] = f"error: {e}"
        checks["mqtt"] = self.mqtt_status()
        ok = all(v in ("ok", "disabled", "connected") for v in checks.values())
        return {"status": "ok" if ok else "degraded", "checks": checks,
                "uptime_s": round(time.time() - self.started)}

    def authorized(self, header: str | None) -> bool:
        if not self.token:
            return True
        return self._bearer_is(header, self.token)

    @staticmethod
    def _bearer_is(header: str | None, secret: str) -> bool:
        if not secret or not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:].encode(), secret.encode())

    def may_view(self, client_ip: str, header: str | None) -> bool:
        """/state: the Pi itself, or either token. Without a token the API only listens on the Pi."""
        return (client_ip in LOOPBACK or self._bearer_is(header, self.token)
                or self._bearer_is(header, self.view_token))

    def ask(self, text: str) -> dict:
        t0 = time.perf_counter()
        with self.lock:
            self.status.set("thinking", heard=text, said="", first_word_s=None)
            try:
                reply = self.brain.handle(text, confirm=lambda q: False)
            finally:
                self.status.set("idle")
        seconds = round(time.perf_counter() - t0, 2)
        self.status.set(said=reply)
        return {"reply": reply, "seconds": seconds}

    def light(self, room: str, state: str) -> tuple[int, dict]:
        """Switch a light from a screen tap. Doesn't take the voice lock: a tap shouldn't wait
        until a spoken reply has finished. The audit is written in the background, with the lock."""
        brain = self.brain
        d = brain.governor.check({"name": "set_light", "args": {"room": room, "state": state}})
        if d.verdict is not Verdict.ALLOW:
            self._audit(d, "not run (screen tap)")
            return (429 if "rate limit" in d.reason else 400), {"ok": False, "room": room, "state": state,
                                                                  "message": d.reason}
        try:
            result, ok = str(brain.governor.execute(d)), True
        except Exception as e:  # noqa: BLE001 - device didn't answer, MQTT down...
            result, ok = str(e), False
        self._audit(d, ("" if ok else "error: ") + result + " (screen tap)")
        msg = f"The {room} light is {state}." if ok else f"That didn't work: {result}"
        return 200, {"ok": ok, "room": room, "state": state, "message": msg}

    def wake_change(self, data: dict) -> tuple[int, dict]:
        """POST /wake and /wake/label. Bad input or a refused change (the last word off) -> 400."""
        if self.wake is None:
            return 404, {"error": "No wake words: voice is off."}
        try:
            if "id" in data:
                self.wake.label(str(data["id"]), str(data.get("label") or ""))
            else:
                enabled = data.get("enabled")
                threshold = data.get("threshold")
                self.wake.update(str(data["word"]), None if enabled is None else bool(enabled),
                                 None if threshold is None else float(threshold))
                self.status.set(wake_words=self.wake.entries())
        except (KeyError, TypeError, ValueError) as e:
            return 400, {"error": str(e) if isinstance(e, ValueError) else "expected JSON {\"word\": ...}"}
        return 200, self.wake.snapshot(7 if data.get("days") == 7 else 1)

    def _audit(self, d, outcome: str) -> None:
        def run():
            with self.lock:
                self.brain.memory.log_audit(d.tool, d.args, d.verdict.value, d.reason, outcome)
        threading.Thread(target=run, daemon=True).start()

    def make_server(self, host: str, port: int) -> ThreadingHTTPServer:
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, obj: dict | bytes, ctype: str = "application/json") -> None:
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    h = api.health()
                    self._send(200 if h["status"] == "ok" else 503, h)
                elif path == "/face":
                    self._send(200, FACE_PAGE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/history":
                    if not api.may_view(self.client_address[0], self.headers.get("Authorization")):
                        return self._send(403, {"error": "only from the Pi itself, or with a token"})
                    self._send(200, api.history())
                elif path == "/state":
                    if not api.may_view(self.client_address[0], self.headers.get("Authorization")):
                        return self._send(403, {"error": "only from the Pi itself, or with a token"})
                    self._send(200, api.state())
                elif path == "/wake":
                    if not api.may_view(self.client_address[0], self.headers.get("Authorization")):
                        return self._send(403, {"error": "only from the Pi itself, or with a token"})
                    if api.wake is None:
                        return self._send(404, {"error": "No wake words: voice is off."})
                    query = self.path.partition("?")[2]
                    days = 7 if "days=7" in query.split("&") else 1
                    self._send(200, api.wake.snapshot(days))
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                if self.path == "/mute":
                    return self._mute()
                if self.path == "/light":
                    return self._light()
                if self.path in ("/wake", "/wake/label"):
                    return self._wake()
                if self.path == "/history/clear":
                    return self._clear_history()
                if self.path != "/ask":
                    return self._send(404, {"error": "not found"})
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                # Read the body before any early reply: answering 401 and closing with unread data
                # makes some clients see "connection reset" instead of the 401 (seen 1 in 40 on Windows).
                raw = self.rfile.read(length) if 0 < length <= MAX_BODY else b""
                if not api.authorized(self.headers.get("Authorization")):
                    return self._send(401, {"error": "unauthorized"})
                if length <= 0 or length > MAX_BODY:
                    return self._send(413 if length > MAX_BODY else 400, {"error": "bad body size"})
                try:
                    data = json.loads(raw)
                    text = str(data["text"]).strip()
                except (ValueError, KeyError, TypeError):
                    return self._send(400, {"error": "expected JSON {\"text\": ...}"})
                if not text:
                    return self._send(400, {"error": "empty text"})
                try:
                    self._send(200, api.ask(text))
                except Exception as e:  # noqa: BLE001
                    self._send(502, {"error": str(e)})

            def _mute(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                raw = self.rfile.read(length) if 0 < length <= MAX_BODY else b""
                local = self.client_address[0] in LOOPBACK
                if not (local or (api.token and api.authorized(self.headers.get("Authorization")))):
                    return self._send(403, {"error": "only from the Pi itself, or with the token"})
                try:
                    muted = bool(json.loads(raw)["muted"])
                except (ValueError, KeyError, TypeError):
                    return self._send(400, {"error": "expected JSON {\"muted\": true|false}"})
                api.status.set(muted=muted)
                self._send(200, {"muted": muted})

            def _light(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                raw = self.rfile.read(length) if 0 < length <= MAX_BODY else b""
                local = self.client_address[0] in LOOPBACK
                if not (local or (api.token and api.authorized(self.headers.get("Authorization")))):
                    return self._send(403, {"error": "only from the Pi itself, or with the token"})
                try:
                    data = json.loads(raw)
                    room, state = str(data["room"]), str(data["state"])
                except (ValueError, KeyError, TypeError):
                    return self._send(400, {"error": "expected JSON {\"room\": ..., \"state\": \"on\"|\"off\"}"})
                code, body = api.light(room, state)
                self._send(code, body)

            def _clear_history(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                if 0 < length <= MAX_BODY:
                    self.rfile.read(length)
                local = self.client_address[0] in LOOPBACK
                if not (local or (api.token and api.authorized(self.headers.get("Authorization")))):
                    return self._send(403, {"error": "only from the Pi itself, or with the token"})
                with api.lock:
                    n = api.brain.memory.clear_history()
                api.status.set(heard="", said="")
                self._send(200, {"deleted": n})

            def _wake(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                raw = self.rfile.read(length) if 0 < length <= MAX_BODY else b""
                local = self.client_address[0] in LOOPBACK
                if not (local or (api.token and api.authorized(self.headers.get("Authorization")))):
                    return self._send(403, {"error": "only from the Pi itself, or with the token"})
                try:
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ValueError
                except ValueError:
                    return self._send(400, {"error": "expected a JSON object"})
                if self.path == "/wake/label" and "id" not in data:
                    return self._send(400, {"error": "expected JSON {\"id\": ..., \"label\": ...}"})
                if self.path == "/wake":
                    data.pop("id", None)
                code, body = api.wake_change(data)
                self._send(code, body)

            def log_message(self, fmt, *args):  # keep the console quiet
                pass

        return ThreadingHTTPServer((host, port), Handler)
