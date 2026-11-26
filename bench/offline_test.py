"""Offline test: does the whole assistant work with the internet unplugged?

Start it on the Pi, then unplug the internet at the router (the cable into the router from the wall, or
switch the router's internet off; keep the home Wi-Fi on so the lights can still talk to the Pi):
    bash bench/offline_test.sh            (from the PC: ssh pi5 "bash ~/assistant/bench/offline_test.sh")
It says out loud when it's waiting for the internet to go, runs every check once it's gone, says the
result, then waits (up to 30 min) for the internet to come back and writes the report:
    experiments/results/offline_test_<date-time>.md     progress: logs/offline_test.log
    --now      run the checks at once, internet or not (a dry run; the report says it wasn't offline)
    --quiet    don't speak or switch the office light

What it checks (with a separate in-memory database, so your history and notes are not touched):
  readiness   bench/offline_check.sh (config hosts local, model files on disk, no outside connections)
  service     the running assistant's /health and /state answer (model server, database, MQTT)
  model       a plain question ("What is 12 times 7?" -> 84)
  time        "What time is it?" through the get_time tool, and the clock it reads
  memory      a note saved with embeddings, found again by a question, then "forget" with a yes
  speech      Piper speaks a sentence on the speaker, and Whisper turns it back into the same words
  wake word   the wake word models load
  lights      the office light switched and confirmed by its ESP32 over MQTT (then put back)
  telemetry   ONNX Runtime's Microsoft telemetry is off (it was on by default until 2026-11-25)
  screen      the touchscreen app is running
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
API = "http://127.0.0.1:8765"
PROBES = (("1.1.1.1", 443), ("8.8.8.8", 53), ("9.9.9.9", 443))


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def internet_up() -> bool:
    for host, port in PROBES:
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except OSError:
            pass
    return False


def wait_for(want_up: bool, max_s: float, settle: int = 3) -> bool:
    """Wait until the internet has been up (or down) for `settle` probes in a row."""
    end, streak = time.monotonic() + max_s, 0
    while time.monotonic() < end:
        streak = streak + 1 if internet_up() == want_up else 0
        if streak >= settle:
            return True
        time.sleep(5)
    return False


def api(path: str, body: dict | None = None, timeout: float = 15):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Checks:
    def __init__(self, cfg, quiet: bool):
        self.cfg, self.quiet = cfg, quiet
        self.results: list[tuple[str, bool | None, str, float]] = []   # name, ok (None = skipped), detail, s
        self._tts = None

    def run(self, name: str, fn) -> None:
        t0 = time.monotonic()
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - a failing check is a result, not a crash
            ok, detail = False, f"{type(e).__name__}: {e}"
        secs = time.monotonic() - t0
        self.results.append((name, ok, detail, secs))
        log(f"{'PASS' if ok else 'SKIP' if ok is None else 'FAIL'}  {name}: {detail} ({secs:.1f} s)")

    # --- helpers
    def tts(self):
        if self._tts is None:
            from assistant.voice.engines import PiperTTS
            self._tts = PiperTTS(self.cfg.voice.piper_cmd, self.cfg.voice.piper_model)
        return self._tts

    def say(self, text: str) -> None:
        if self.quiet:
            return
        try:
            from assistant.voice.audio import play_pcm
            tts = self.tts()
            play_pcm(tts.synthesize(text), tts.sample_rate, self.cfg.voice.speaker_device)
        except Exception as e:  # noqa: BLE001
            log(f"(couldn't speak: {e})")

    def brain(self):
        from assistant.__main__ import build
        cfg = copy.deepcopy(self.cfg)
        cfg.memory.db_path = ":memory:"          # never touches the real history or notes
        return build(cfg)[0]

    # --- checks
    def readiness(self):
        out = subprocess.run(["bash", str(ROOT / "bench" / "offline_check.sh")], capture_output=True, text=True,
                             timeout=120, cwd=ROOT).stdout
        self.readiness_text = out
        missing = [ln for ln in out.splitlines() if ln.startswith("MISSING")]
        outside = "none (only local" not in out
        return (not missing and not outside,
                ("missing: " + "; ".join(missing) if missing else "all model files on disk")
                + ("; connections to outside hosts found" if outside else "; no outside connections"))

    def service(self):
        try:
            h = api("/health")
        except urllib.error.HTTPError as e:   # 503 = degraded, still has the body
            h = json.loads(e.read())
        st = api("/state")
        checks = h.get("checks", {})
        ok = checks.get("llm") == "ok" and checks.get("database") == "ok"
        return ok, (f"health {h.get('status')} ({', '.join(f'{k} {v}' for k, v in checks.items())}); "
                    f"model {st.get('model')}, voice {'on' if st.get('voice') else 'off'}")

    def model(self):
        reply = self.brain().handle("What is 12 times 7?")
        return "84" in reply, f"reply: {reply[:120]!r}"

    def time_tool(self):
        reply = self.brain().handle("What time is it?")
        now = datetime.now()
        want = {f"{(now.hour % 12) or 12}:{now:%M}"}
        prev = datetime.fromtimestamp(time.time() - 60)
        want.add(f"{(prev.hour % 12) or 12}:{prev:%M}")
        synced = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], capture_output=True,
                                text=True).stdout.strip()
        return any(w in reply for w in want), (f"reply: {reply!r}; clock synced: {synced or '?'} (offline it keeps "
                                               "running, but a power cut would leave it wrong: no RTC battery)")

    def memory(self):
        b = self.brain()
        b.memory.remember("The spare key is in the blue box in the hallway.")
        if not b.memory.list_memories()[0][1]:
            return False, "note not saved"
        emb = b.memory.db.execute("SELECT embedding IS NOT NULL FROM memories").fetchone()[0]
        reply = b.handle("Where is the spare key?")
        found = "blue box" in reply.lower()
        asked: list[str] = []
        b.handle("Forget the note about the spare key.", confirm=lambda q: asked.append(q) or True)
        gone = not b.memory.list_memories()
        return found and gone and bool(asked), (f"embeddings {'yes' if emb else 'NO (keyword search)'}; "
                                                f"recall: {reply[:80]!r}; forget asked {len(asked)}x, note "
                                                f"{'deleted' if gone else 'still there'}")

    def speech(self):
        from assistant.voice.engines import WhisperCppSTT
        import numpy as np
        sentence = "What time is it in the kitchen?"
        tts = self.tts()
        pcm = tts.synthesize(sentence)
        if not self.quiet:
            from assistant.voice.audio import play_pcm
            play_pcm(pcm, tts.sample_rate, self.cfg.voice.speaker_device)
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        n = int(len(x) * 16000 / tts.sample_rate)
        pcm16 = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16).tobytes()
        v = self.cfg.voice
        heard = WhisperCppSTT(v.whisper_cmd, v.whisper_model, 16000, v.whisper_prompt).transcribe(pcm16)
        words = set(re.findall(r"[a-z]+", heard.lower()))
        ok = {"time", "kitchen"} <= words
        return ok, f"spoke {len(pcm) / 2 / tts.sample_rate:.1f} s of audio; Whisper heard {heard!r}"

    def wake(self):
        v = self.cfg.voice
        from assistant.voice.engines import OpenWakeWord
        w = OpenWakeWord(v.wake_word, v.wake_threshold, patience=v.wake_patience)
        names = sorted(getattr(w.model, "models", {}) or {})
        return bool(names), f"loaded: {', '.join(names) or 'none'}"

    def lights(self):
        st = api("/state")
        home = st.get("home") or {}
        if "office" not in (home.get("devices") or []):
            return None, "skipped: the office ESP32 hasn't reported since the assistant started (switched off?)"
        if self.quiet:
            return None, "skipped (--quiet)"
        before = (home.get("lights") or {}).get("office", "off")
        other = "off" if before == "on" else "on"
        t0 = time.monotonic()
        r1 = api("/light", {"room": "office", "state": other})
        secs = time.monotonic() - t0
        time.sleep(1)
        r2 = api("/light", {"room": "office", "state": before})
        ok = bool(r1.get("ok") and r2.get("ok"))
        return ok, f"office {other}: {r1.get('message')} ({secs:.1f} s); back {before}: {r2.get('message')}"

    def telemetry(self):
        """ONNX Runtime's Microsoft telemetry (found 2026-11-25) must stay off: the switch is set when
        the assistant package loads, and its event store must not have been written since the service started."""
        import os
        import assistant  # noqa: F401 - sets ORT_DISABLE_TELEMETRY
        flag = os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
        store = Path.home() / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime"
        pid = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", "assistant.service"],
                             capture_output=True, text=True).stdout.strip()
        started = Path(f"/proc/{pid}").stat().st_mtime if pid and pid != "0" else time.time()
        # -shm changes whenever anything merely opens the database (reading it counts), so it's ignored
        written = ([p.name for p in store.glob("*") if p.stat().st_mtime > started and not p.name.endswith("-shm")]
                   if store.exists() else [])
        detail = (f"switch {'set' if flag else 'NOT set'}; " + (f"written since start: {written}" if written else
                  "event store not written since the service started" if store.exists() else "no event store"))
        return flag and not written, detail

    def screen(self):
        out = subprocess.run(["pgrep", "-f", r"assistant\.screen"], capture_output=True, text=True).stdout.split()
        return bool(out), "touchscreen app running" if out else "touchscreen app NOT running"


def write_report(path: Path, c: Checks, offline: bool, started: datetime, offline_at, back_at) -> None:
    passed = sum(1 for r in c.results if r[1])
    failed = sum(1 for r in c.results if r[1] is False)
    lines = [f"# Offline test {started:%Y-%m-%d %H:%M}", "",
             f"- Internet: **{'off during the checks' if offline else 'ON during the checks (dry run, not a real offline test)'}**"
             + (f" (gone at {offline_at:%H:%M:%S})" if offline_at else ""),
             f"- Back online: {back_at:%H:%M:%S}" if back_at else "- Back online: not seen (or not waited for)",
             f"- Model: {c.cfg.llm.model}", f"- Result: **{passed} passed, {failed} failed, "
             f"{len(c.results) - passed - failed} skipped**", "",
             "| Check | Result | Detail | Time |", "|---|---|---|---|"]
    for name, ok, detail, secs in c.results:
        res = "pass" if ok else "skipped" if ok is None else "**FAIL**"
        lines.append(f"| {name} | {res} | {detail.replace('|', '/')} | {secs:.1f} s |")
    if getattr(c, "readiness_text", ""):
        lines += ["", "## Readiness check output", "```", c.readiness_text.strip(), "```"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--now", action="store_true", help="don't wait for the internet to go (dry run)")
    ap.add_argument("--quiet", action="store_true", help="no speaking, no light switching")
    ap.add_argument("--wait-min", type=float, default=20, help="how long to wait for the internet to go")
    a = ap.parse_args()

    import os
    os.chdir(ROOT)
    from assistant.config import load_config
    cfg = load_config("config.toml")
    c = Checks(cfg, a.quiet)
    started = datetime.now()
    log(f"offline test started (model {cfg.llm.model})")
    offline_at = back_at = None
    if a.now:
        offline = not internet_up()
    else:
        c.say("Offline test. Unplug the internet now. I'll start when it's gone.")
        log("waiting for the internet to go (unplug it at the router)")
        offline = wait_for(False, a.wait_min * 60)
        if not offline:
            log("the internet never went away: running the checks anyway (not a real offline test)")
            c.say("The internet is still on. I'll run the checks anyway.")
        else:
            offline_at = datetime.now()
            log("internet is gone: running the checks")
            c.say("The internet is gone. Starting the checks.")
    for name, fn in (("readiness", c.readiness), ("service", c.service), ("model", c.model),
                     ("time", c.time_tool), ("memory", c.memory), ("speech", c.speech), ("wake word", c.wake),
                     ("lights", c.lights), ("telemetry", c.telemetry), ("screen", c.screen)):
        c.run(name, fn)
    still_offline = not internet_up()
    offline = offline and still_offline
    passed = sum(1 for r in c.results if r[1])
    failed = [r[0] for r in c.results if r[1] is False]
    summary = f"{passed} of {len(c.results)} checks passed" + (f". Failed: {', '.join(failed)}" if failed else "")
    log(summary)
    report = ROOT / "experiments" / "results" / f"offline_test_{started:%Y%m%d-%H%M}.md"
    write_report(report, c, offline, started, offline_at, None)
    if not a.now:
        c.say(f"Offline test finished. {summary}. You can plug the internet back in.")
        log("waiting up to 30 min for the internet to come back")
        if wait_for(True, 30 * 60, settle=2):
            back_at = datetime.now()
            log("internet is back")
        write_report(report, c, offline, started, offline_at, back_at)
    log(f"report: {report}")


if __name__ == "__main__":
    main()
