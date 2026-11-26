import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import _path  # noqa: F401
from assistant.api import AssistantAPI
from assistant.status import Status
from assistant.voice.fixups import is_phantom
from assistant.voice.wake_control import WakeControl, parse_entries

ENTRIES = "models/wake/uzi.onnx@0.5,models/wake/aiya.onnx@0.7,models/wake/boafo.onnx@0.8"


class Clock:
    def __init__(self, when: str = "2026-11-24 18:00:00"):
        self.t = datetime.fromisoformat(when).timestamp()

    def __call__(self):
        self.t += 0.01   # every event gets its own id
        return self.t


class WakeControlTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.clock = Clock()

    def tearDown(self):
        self.dir.cleanup()

    def make(self, entries=ENTRIES):
        return WakeControl(entries, 0.5, self.root / "data" / "wake_settings.json", self.root / "logs" / "wake",
                           clock=self.clock)

    def test_entries_from_config(self):
        self.assertEqual(parse_entries(ENTRIES, 0.5), {"uzi": 0.5, "aiya": 0.7, "boafo": 0.8})
        self.assertEqual(parse_entries("hey_jarvis", 0.4), {"hey_jarvis": 0.4})
        c = self.make("hey_jarvis")
        self.assertEqual(c.word_for("hey_jarvis_v0.1"), "hey_jarvis")
        self.assertIsNone(c.word_for("alexa"))

    def test_switch_off_and_tune_are_kept(self):
        c = self.make()
        c.update("boafo", enabled=False)
        c.update("aiya", threshold=0.7501)
        self.assertFalse(c.enabled("boafo"))
        self.assertEqual(c.entries(), ["uzi@0.5", "aiya@0.75", "boafo@off"])
        again = self.make()                      # after a restart
        self.assertFalse(again.enabled("boafo"))
        self.assertEqual(again.threshold("aiya"), 0.75)

    def test_thresholds_stay_in_range(self):
        c = self.make()
        c.update("uzi", threshold=0.1)
        self.assertEqual(c.threshold("uzi"), 0.3)
        c.update("uzi", threshold=2)
        self.assertEqual(c.threshold("uzi"), 0.95)

    def test_last_word_stays_on(self):
        c = self.make()
        c.update("uzi", enabled=False)
        c.update("aiya", enabled=False)
        with self.assertRaises(ValueError):
            c.update("boafo", enabled=False)
        self.assertTrue(c.enabled("boafo"))
        with self.assertRaises(ValueError):
            c.update("alexa", enabled=False)

    def test_bad_settings_file_never_leaves_it_deaf(self):
        p = self.root / "data" / "wake_settings.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"words": {w: {"enabled": False} for w in ("uzi", "aiya", "boafo")}}))
        self.assertTrue(all(self.make().enabled(w) for w in ("uzi", "aiya", "boafo")))
        p.write_text("not json")
        self.assertEqual(self.make().threshold("boafo"), 0.8)

    def test_counts_and_marks(self):
        c = self.make()
        a = c.record("replied", "uzi", 0.9, "Turn on the office light.")
        b = c.record("replied", "boafo", 1.0, "Everyone out to me.")
        c.record("no_speech", "boafo", 0.95)
        c.record("phantom", "boafo", 0.85)
        c.record("near_miss", "boafo", 0.4)
        c.record("ignored", "uzi", 0.8)
        self.assertIsNone(c.record("near_miss", "alexa", 0.4))
        c.label(a, "real")
        c.label(b, "real")
        c.label(b, "false")                       # the latest mark counts
        s = c.stats(1)
        self.assertEqual(s["boafo"], {"replied": 1, "no_speech": 1, "phantom": 1, "ignored": 0, "near_miss": 1,
                                      "woke": 3, "real": 0, "false": 1})
        self.assertEqual((s["uzi"]["woke"], s["uzi"]["ignored"], s["uzi"]["real"]), (1, 1, 1))
        self.assertEqual([r["heard"] for r in c.recent()], ["Everyone out to me.", "Turn on the office light."])
        self.assertEqual(c.recent()[0]["label"], "false")
        with self.assertRaises(ValueError):
            c.label("nope", "real")
        with self.assertRaises(ValueError):
            c.label(a, "maybe")
        text = (self.root / "logs" / "wake" / "2026-11-24.csv").read_text()
        self.assertNotIn("office", text)          # what was said never goes to the file

    def test_today_and_last_week(self):
        c = self.make()
        c.record("replied", "aiya", 0.9)
        self.clock.t += 2 * 86400
        c.record("no_speech", "aiya", 0.9)
        self.assertEqual(c.stats(1)["aiya"]["woke"], 1)
        self.assertEqual(c.stats(7)["aiya"]["woke"], 2)
        self.clock.t += 61 * 86400
        self.make()                                # old days are deleted at start
        self.assertEqual(sorted(p.name for p in (self.root / "logs" / "wake").iterdir()), [])

    def test_snapshot(self):
        c = self.make()
        c.update("boafo", enabled=False)
        snap = c.snapshot(7)
        self.assertEqual(snap["days"], 7)
        self.assertEqual([(w["word"], w["enabled"]) for w in snap["words"]],
                         [("uzi", True), ("aiya", True), ("boafo", False)])


class PhantomTests(unittest.TestCase):
    def test_noise_words(self):
        for t in ("Thank you.", ">> >>", "thank you very much!", "...", "Bye.", "  "):
            self.assertTrue(is_phantom(t), t)
        for t in ("Thank you, turn off the light.", "What time is it?", "Turn on the office light."):
            self.assertFalse(is_phantom(t), t)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
class EngineTests(unittest.TestCase):
    def test_switched_off_word_never_wakes_and_events_are_logged(self):
        import numpy as np
        from assistant.voice.engines import OpenWakeWord

        class Model:
            def __init__(self):
                self.k = 0

            def predict(self, frame):
                self.k += 1
                return {"boafo": 0.99, "uzi": 0.9 if self.k > 30 else 0.3 if self.k < 5 else 0.0}

        with tempfile.TemporaryDirectory() as d:
            c = WakeControl(ENTRIES, 0.5, Path(d) / "s.json", Path(d) / "wake", clock=Clock())
            c.update("boafo", enabled=False)
            w = object.__new__(OpenWakeWord)
            w.np, w.model, w.threshold, w.vad, w.last_word, w.thresholds, w.patience = np, Model(), 0.5, None, "", {}, 1
            w.control = c
            self.assertEqual(w.wait(iter([bytes(2560)] * 80)), 0.9)   # one 80 ms frame per chunk
            self.assertEqual(w.last_word, "uzi")       # boafo scored 0.99 all along, but it's off
            self.assertEqual(c.stats(1)["uzi"]["near_miss"], 1)
            c.update("uzi", threshold=0.95)
            self.assertEqual(w.threshold_for("uzi"), 0.95)


class VoiceLoopTests(unittest.TestCase):
    def run_once(self, heard):
        from assistant.voice import loop as loop_mod

        class Cfg:
            speaker_device, vad_threshold, sample_rate, silence_ms, max_utterance_s = "x", 500, 16000, 800, 10
            vad_engine, silero_threshold, listen_timeout_s = "energy", 0.5, 7.0
            timing_log = "unused.csv"

        class Wake:
            last_word = "boafo"

            def wait(self, chunks):
                return 0.97

        events = []

        class Control:
            def record(self, kind, word, score=None, heard=""):
                events.append((kind, word, score, heard))

        lp = loop_mod.VoiceLoop(None, Wake(), None, type("T", (), {"sample_rate": 22050})(), Cfg(),
                                status=Status(), wake_control=Control())
        lp._listen = lambda mic: (heard, 1.0, 0.5)
        beep = loop_mod.beep
        loop_mod.beep = lambda *a, **k: None
        try:
            lp.run_once(None)
        finally:
            loop_mod.beep = beep
        return events

    def test_phantom_and_silence_get_no_reply(self):
        self.assertEqual(self.run_once("Thank you."), [("phantom", "boafo", 0.97, "")])
        self.assertEqual(self.run_once(""), [("no_speech", "boafo", 0.97, "")])


class WakeApiTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.control = WakeControl(ENTRIES, 0.5, Path(self.dir.name) / "s.json", Path(self.dir.name) / "wake")
        self.status = Status()
        self.api = AssistantAPI(None, token="secret", status=self.status, wake_control=self.control)
        self.server = self.api.make_server("127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.dir.cleanup()

    def call(self, path, body=None):
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_get_change_and_label(self):
        code, snap = self.call("/wake?days=7")
        self.assertEqual((code, snap["days"], len(snap["words"])), (200, 7, 3))
        code, snap = self.call("/wake", {"word": "boafo", "enabled": False})
        self.assertEqual(code, 200)
        self.assertFalse(snap["words"][2]["enabled"])
        self.assertIn("boafo@off", self.status.get("wake_words"))
        code, snap = self.call("/wake", {"word": "uzi", "threshold": 0.55})
        self.assertEqual(snap["words"][0]["threshold"], 0.55)
        eid = self.control.record("replied", "uzi", 0.9, "What time is it?")
        code, snap = self.call("/wake/label", {"id": eid, "label": "real"})
        self.assertEqual((code, snap["recent"][0]["label"]), (200, "real"))

    def test_refusals(self):
        self.call("/wake", {"word": "uzi", "enabled": False})
        self.call("/wake", {"word": "aiya", "enabled": False})
        code, body = self.call("/wake", {"word": "boafo", "enabled": False})
        self.assertEqual(code, 400)
        self.assertIn("at least one", body["error"])
        self.assertEqual(self.call("/wake", {"enabled": False})[0], 400)
        self.assertEqual(self.call("/wake/label", {"label": "real"})[0], 400)
        self.assertEqual(self.call("/wake", ["x"])[0], 400)

    def test_no_wake_words(self):
        self.api.wake = None
        self.assertEqual(self.call("/wake")[0], 404)
        self.assertEqual(self.call("/wake", {"word": "uzi", "enabled": False})[0], 404)
