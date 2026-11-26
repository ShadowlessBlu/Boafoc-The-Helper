import array
import importlib.util
import json
import math
import os
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import _path  # noqa: F401
from assistant.api import AssistantAPI
from assistant.brain import Brain
from assistant.config import load_config
from assistant.governor import Governor
from assistant.memory import Memory
from assistant.mqtt_bridge import handle_message
from assistant.tools import Devices, build_tools
from assistant.voice.audio import EnergyVAD, SileroVAD, make_vad, play_command, rms
from assistant.voice.calibrate import set_threshold, suggest
from assistant.voice.engines import (PiperTTS, WhisperCppSTT, audio_ctx_for, clean_whisper_text, piper_pcm, resolve_cmd, split_sentences,
                                     wake_model_kwargs)
from assistant.status import Status, read_memory_gb
from assistant.voice.loop import VoiceLoop, parse_yes_no
from test_brain import ScriptedLLM


def tone(ms, amp, rate=16000):
    n = rate * ms // 1000
    return array.array("h", (int(amp * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n))).tobytes()


class ConfigTests(unittest.TestCase):
    def test_defaults_and_env(self):
        os.environ["ASSISTANT_API_TOKEN"] = "abc"
        try:
            cfg = load_config("does-not-exist.toml")
        finally:
            del os.environ["ASSISTANT_API_TOKEN"]
        self.assertEqual(cfg.api.token, "abc")
        self.assertEqual(cfg.voice.sample_rate, 16000)

    def test_example_config_loads(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = load_config(os.path.join(root, "config.example.toml"))
        self.assertFalse(cfg.mqtt.enabled)

    def test_unknown_key_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write("[llm]\nmodle = 'x'\n")
        try:
            with self.assertRaises(ValueError):
                load_config(f.name)
        finally:
            os.unlink(f.name)


class ApiTests(unittest.TestCase):
    def setUp(self):
        mem = Memory()
        self.llm = ScriptedLLM(*[{"say": "Hi there.", "tool": "none", "args": {}}] * 3)
        brain = Brain(self.llm, Governor(build_tools(mem, Devices())), mem)
        self.api = AssistantAPI(brain, token="secret")
        self.server = self.api.make_server("127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, body, token="secret"):
        req = urllib.request.Request(self.url + "/ask", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {token}"} if token else {})})
        return urllib.request.urlopen(req, timeout=5)

    def test_health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
            h = json.loads(r.read())
        self.assertEqual(h["checks"]["database"], "ok")
        self.assertEqual(h["status"], "ok")

    def test_ask(self):
        with self.post({"text": "hello"}) as r:
            self.assertEqual(json.loads(r.read())["reply"], "Hi there.")

    def test_auth_required(self):
        for tok in (None, "wrong"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.post({"text": "hello"}, token=tok)
            self.assertEqual(cm.exception.code, 401)
            cm.exception.close()

    def test_bad_body(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({"nope": 1})
        self.assertEqual(cm.exception.code, 400)
        cm.exception.close()

    def test_sensitive_never_confirmed_over_api(self):
        self.llm.replies = [{"say": "Unlocking", "tool": "unlock_door", "args": {"door": "front"}}]
        with self.post({"text": "unlock the door"}) as r:
            self.assertEqual(json.loads(r.read())["reply"], "Okay, I won't do that.")

    def get_state(self, token=None):
        req = urllib.request.Request(self.url + "/state",
                                     headers={"Authorization": f"Bearer {token}"} if token else {})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def test_face_page_served(self):
        with urllib.request.urlopen(self.url + "/face?demo", timeout=5) as r:
            self.assertIn("text/html", r.headers["Content-Type"])
            page = r.read().decode()
        self.assertIn('fetch("/state"', page)
        self.assertNotIn("http", page.split("<script>")[1].split("fetch")[0])  # no outside files

    def test_state_shows_last_exchange(self):
        st = self.get_state()
        self.assertEqual((st["mode"], st["said"]), ("idle", ""))
        self.post({"text": "hello"}).close()
        st = self.get_state()
        self.assertEqual((st["mode"], st["heard"], st["said"]), ("idle", "hello", "Hi there."))
        self.assertIn("pi", st)

    def test_state_only_local_or_with_token(self):
        import assistant.api as api_mod
        saved, api_mod.LOOPBACK = api_mod.LOOPBACK, ()   # pretend the request comes from elsewhere
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.get_state()
            self.assertEqual(cm.exception.code, 403)
            cm.exception.close()
            self.assertEqual(self.get_state(token="secret")["mode"], "idle")
            self.api.view_token = "look-only"
            self.assertEqual(self.get_state(token="look-only")["mode"], "idle")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.get_state(token="wrong")
            cm.exception.close()
        finally:
            api_mod.LOOPBACK = saved

    def test_mute_and_history(self):
        req = urllib.request.Request(self.url + "/mute", data=json.dumps({"muted": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(json.loads(r.read()), {"muted": True})
        self.assertTrue(self.get_state()["muted"])
        self.post({"text": "hello"}).close()
        with urllib.request.urlopen(self.url + "/history", timeout=5) as r:
            msgs = json.loads(r.read())["messages"]
        self.assertEqual([m["content"] for m in msgs][-2:], ["hello", "Hi there."])

    def test_mute_not_from_other_devices_without_token(self):
        import assistant.api as api_mod
        saved, api_mod.LOOPBACK = api_mod.LOOPBACK, ()
        try:
            req = urllib.request.Request(self.url + "/mute", data=b'{"muted": true}',
                                         headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 403)
            cm.exception.close()
        finally:
            api_mod.LOOPBACK = saved

    def light(self, body, token=None):
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
        req = urllib.request.Request(self.url + "/light", data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_light_switch_from_the_screen(self):
        code, body = self.light({"room": "office", "state": "on"})
        self.assertEqual((code, body["ok"]), (200, True))
        self.assertEqual(self.get_state()["home"], {})   # this test's Status has no home callback
        self.assertEqual(self.api.brain.governor.tools["set_light"].func.__self__.lights["office"], "on")

    def test_light_bad_room_is_refused_by_the_governor(self):
        code, body = self.light({"room": "garage", "state": "on"})
        self.assertEqual((code, body["ok"]), (400, False))
        self.assertIn("must be one of", body["message"])
        code, _ = self.light({"room": "office"})
        self.assertEqual(code, 400)

    def test_light_device_that_does_not_answer(self):
        devices = self.api.brain.governor.tools["set_light"].func.__self__
        devices.confirm_s = 0.05
        code, body = self.light({"room": "office", "state": "on"})
        self.assertEqual((code, body["ok"]), (200, False))
        self.assertIn("no device for the office light yet", body["message"])
        self.assertEqual(devices.lights["office"], "off")

    def test_light_not_from_other_devices_without_token(self):
        import assistant.api as api_mod
        saved, api_mod.LOOPBACK = api_mod.LOOPBACK, ()
        try:
            self.assertEqual(self.light({"room": "office", "state": "on"})[0], 403)
            self.assertEqual(self.light({"room": "office", "state": "on"}, token="secret")[0], 200)
        finally:
            api_mod.LOOPBACK = saved

    def test_view_token_cannot_ask(self):
        self.api.view_token = "look-only"
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({"text": "unlock the door"}, token="look-only")
        self.assertEqual(cm.exception.code, 401)
        cm.exception.close()

    def test_view_link(self):
        from assistant.__main__ import view_link
        from assistant.config import Config
        cfg = Config()
        self.assertIn("only listens on the Pi", view_link(cfg))
        cfg.api.host, cfg.api.view_token = "100.78.110.4", "abc"
        self.assertEqual(view_link(cfg), "http://100.78.110.4:8765/face#view=abc")


class StatusTests(unittest.TestCase):
    def test_modes_and_since(self):
        t = [100.0]
        st = Status("qwen3:1.7b", home=lambda: {"lights": {"kitchen": "on"}}, clock=lambda: t[0])
        st.set("listening")
        t[0] = 103.0
        st.set("listening", heard="hi")  # same mode: 'since' stays
        snap = st.snapshot()
        self.assertEqual((snap["mode"], snap["since"], snap["heard"]), ("listening", 100.0, "hi"))
        self.assertEqual(snap["home"]["lights"]["kitchen"], "on")
        with self.assertRaises(ValueError):
            st.set("dancing")

    def test_broken_home_does_not_break_the_screen(self):
        st = Status(home=lambda: 1 / 0)
        self.assertEqual(st.snapshot()["home"], {})

    def test_read_memory(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "meminfo"
            p.write_text("MemTotal:        8255488 kB\nMemFree:  100 kB\nMemAvailable:    5761024 kB\n")
            self.assertEqual(read_memory_gb(str(p)), (2.4, 7.9))
            self.assertIsNone(read_memory_gb(str(p) + "x"))

    def test_microphone_chunks_are_whole_even_from_short_reads(self):
        from assistant.voice.audio import Microphone
        mic = Microphone(chunk_ms=30)                      # 960 bytes per chunk
        pieces = [b"a" * 500, b"b" * 700, b"c" * 760, b""]

        class Out:
            def read(self, n):
                return pieces.pop(0)[:n] if pieces else b""

        mic.proc = type("P", (), {"stdout": Out()})()
        self.assertEqual([len(c) for c in mic.chunks()], [960])   # 500+460, then the rest is short of a chunk

    @unittest.skipIf(sys.platform == "win32", "select() on pipes needs Linux")
    def test_drain_drops_waiting_audio(self):
        from assistant.voice.audio import Microphone
        r, w = os.pipe()
        os.write(w, b"x" * 5000)
        mic = Microphone()
        mic.proc = type("P", (), {"stdout": os.fdopen(r, "rb", buffering=0)})()
        self.assertEqual(mic.drain(), 5000)
        self.assertEqual(mic.drain(), 0)
        os.close(w)

    def test_muted_audio_never_reaches_the_wake_word(self):
        class Cfg:
            speaker_device, vad_threshold, sample_rate, silence_ms, max_utterance_s = "x", 500, 16000, 800, 10
            vad_engine, silero_threshold, listen_timeout_s = "energy", 0.5, 7.0
            timing_log = "unused.csv"

        st = Status()
        loop = VoiceLoop(None, None, None, type("T", (), {"sample_rate": 22050})(), Cfg(), status=st)

        class Mic:
            def chunks(self):
                for i in range(6):
                    if i == 3:
                        st.set(muted=False)
                    yield bytes([i]) * 4

        st.set(muted=True)
        self.assertEqual([c[0] for c in loop._live(Mic())], [3, 4, 5])

    def test_voice_loop_reports_speaking_then_idle(self):
        seen = []

        class Rec(Status):
            def set(self, mode=None, **kw):
                if mode:
                    seen.append(mode)
                super().set(mode, **kw)

        class Cfg:
            speaker_device, vad_threshold, sample_rate, silence_ms, max_utterance_s = "x", 500, 16000, 800, 10
            vad_engine, silero_threshold, listen_timeout_s = "energy", 0.5, 7.0
            timing_log = "unused.csv"

        loop = VoiceLoop(None, None, None, type("T", (), {"sample_rate": 22050})(), Cfg(), status=Rec())
        loop.speaker.say = lambda text, sentences: 0.3
        self.assertEqual(loop._speak("Hello."), 0.3)
        self.assertEqual(seen, ["speaking", "idle"])


class MqttTests(unittest.TestCase):
    def setUp(self):
        self.dev, self.online = Devices(), {}

    def h(self, topic, payload):
        return handle_message(self.dev, self.online, "home", topic, payload.encode())

    def test_temperature(self):
        self.h("home/bedroom/temperature", "21.5")
        self.assertEqual(self.dev.temps["bedroom"], 21.5)
        self.assertIn("21.5", self.dev.get_temperature("bedroom"))

    def test_bad_values_ignored(self):
        self.assertIsNone(self.h("home/bedroom/temperature", "hot"))
        self.assertIsNone(self.h("home/bedroom/temperature", "500"))
        self.assertIsNone(self.h("home/garage/temperature", "20"))
        self.assertIsNone(self.h("other/bedroom/temperature", "20"))
        self.assertIsNone(self.h("home/kitchen/light/state", "maybe"))
        self.assertEqual(self.dev.temps, {})

    def test_light_state_and_status(self):
        self.h("home/living_room/light/state", "on")
        self.assertEqual(self.dev.lights["living room"], "on")
        self.h("home/livingroom-node/status", "offline")
        self.assertEqual(self.online["livingroom-node"], "offline")

    def test_light_counts_only_when_the_device_confirms(self):
        dev = Devices(confirm_s=0.3)
        dev.publish = lambda t, p: handle_message(dev, {}, "home", "home/office/light/state", p.encode())
        self.assertEqual(dev.set_light("office", "on"), "office light is now on")
        self.assertEqual(dev.lights["office"], "on")

    def test_no_answer_from_the_device_is_an_error(self):
        from assistant.tools import DeviceNoAnswer
        dev = Devices(confirm_s=0.2)
        dev.publish = lambda t, p: None          # sent, but the ESP32 is offline
        with self.assertRaises(DeviceNoAnswer):
            dev.set_light("office", "on")
        self.assertEqual(dev.lights["office"], "off")

    def test_assistant_says_it_failed_when_the_light_does_not_answer(self):
        mem = Memory()
        dev = Devices(confirm_s=0.2)
        llm = ScriptedLLM({"say": "Turning on the office light.", "tool": "set_light",
                           "args": {"room": "office", "state": "on"}})
        reply = Brain(llm, Governor(build_tools(mem, dev)), mem).handle("office light on")
        self.assertIn("no device for the office light", reply)   # nothing ever reported for it

    def test_offline_publish_does_not_change_state(self):
        def offline(t, p):
            raise ConnectionError("MQTT broker not connected")
        self.dev.publish = offline
        with self.assertRaises(ConnectionError):
            self.dev.set_light("kitchen", "on")
        self.assertEqual(self.dev.lights["kitchen"], "off")


class MqttPublishTests(unittest.TestCase):
    def bridge(self):
        from assistant.mqtt_bridge import MqttBridge
        b = MqttBridge.__new__(MqttBridge)   # no paho or broker needed
        b.connected, b.CONNECT_WAIT_S, b.sent = threading.Event(), 0.5, []

        class Client:
            def publish(self, topic, payload, qos):
                b.sent.append((topic, payload))
                return type("Info", (), {"wait_for_publish": lambda self, timeout: None})()
        b.client = Client()
        return b

    def test_waits_briefly_for_the_broker(self):
        b = self.bridge()
        threading.Timer(0.1, b.connected.set).start()   # connects just after the tap
        b.publish("home/office/light/set", "on")
        self.assertEqual(b.sent, [("home/office/light/set", "on")])

    def test_plain_message_when_not_connected(self):
        b = self.bridge()
        b.CONNECT_WAIT_S = 0.05
        dev = Devices(b.publish, confirm_s=1.0)
        with self.assertRaises(ConnectionError) as cm:
            dev.set_light("office", "on")
        self.assertEqual(str(cm.exception),
                         "I'm not connected to the home devices right now. Try again in a few seconds.")
        self.assertEqual(dev.lights["office"], "off")


class VoiceTests(unittest.TestCase):
    def test_play_command(self):
        """ALSA devices use aplay; "pipewire" uses pw-cat (reaches Bluetooth speakers), optionally a named node."""
        self.assertEqual(play_command(22050, "default")[:3], ["aplay", "-q", "-D"])
        pw = play_command(22050, "pipewire")
        self.assertEqual(pw[0], "pw-cat")
        self.assertIn("22050", pw)
        self.assertEqual(pw[-1], "-")
        self.assertNotIn("--target", pw)
        named = play_command(16000, "pipewire:bluez_output.2B_BA_EB_6A_E4_B3.1")
        self.assertEqual(named[named.index("--target") + 1], "bluez_output.2B_BA_EB_6A_E4_B3.1")

    def test_rms(self):
        self.assertEqual(rms(b"\x00\x00" * 100), 0)
        self.assertAlmostEqual(rms(tone(100, 1000)), 1000 / math.sqrt(2), delta=20)

    def test_vad_collects_utterance(self):
        vad = EnergyVAD(threshold=300, silence_ms=300, pre_ms=60)
        chunks = [tone(30, 10)] * 10 + [tone(30, 3000)] * 20 + [tone(30, 10)] * 30
        pcm = vad.collect(iter(chunks))
        n_chunks = len(pcm) // vad.chunk_bytes
        self.assertEqual(n_chunks, 2 + 20 + 10)  # pre-roll + speech + silence tail

    def test_vad_timeout_and_max(self):
        self.assertEqual(EnergyVAD(300, start_timeout_s=0.3).collect(iter([tone(30, 10)] * 50)), b"")
        vad = EnergyVAD(300, max_s=0.3, pre_ms=0)
        self.assertEqual(len(vad.collect(iter([tone(30, 3000)] * 100))) // vad.chunk_bytes, 10)

    def test_yes_no(self):
        for t in ["yes", "Yeah go ahead", "ok do it"]:
            self.assertTrue(parse_yes_no(t), t)
        for t in ["no", "", "what?", "yes no", "don't do it", "cancel"]:
            self.assertFalse(parse_yes_no(t), t)

    def test_split_sentences(self):
        self.assertEqual(split_sentences("Light on. It is 21 degrees! Anything else?"),
                         ["Light on.", "It is 21 degrees!", "Anything else?"])
        self.assertEqual(split_sentences("Pi 5 runs at 2.4 GHz."), ["Pi 5 runs at 2.4 GHz."])
        self.assertEqual(split_sentences("  "), [])

    def test_whisper_cleanup(self):
        self.assertEqual(clean_whisper_text(" [BLANK_AUDIO]\n Turn on the light. \n"), "Turn on the light.")


class CalibrateTest(unittest.TestCase):
    def test_suggest_geometric_midpoint(self):
        stats, value = suggest([100.0] * 20, [900.0] * 20)
        self.assertEqual(value, 300)
        self.assertAlmostEqual(stats["snr_db"], 20 * math.log10(9), places=3)

    def test_speech_with_pauses_uses_the_loud_frames(self):
        # counting: half pauses at room level, half words; the median would sit at room level
        speech = [250.0] * 50 + [1000.0] * 50
        self.assertEqual(suggest([250.0] * 20, speech)[1], 500)

    def test_no_suggestion_when_speech_is_not_clearly_louder(self):
        self.assertIsNone(suggest([300.0] * 10, [200.0] * 10)[1])
        self.assertIsNone(suggest([300.0] * 10, [400.0] * 10)[1])

    def test_set_threshold_keeps_the_rest_of_the_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "config.toml"
            p.write_text("[voice]\nvad_threshold = 500.0         # tune it\nsilence_ms = 800\n", encoding="utf-8")
            set_threshold(p, 742)
            self.assertEqual(p.read_text(encoding="utf-8"),
                             "[voice]\nvad_threshold = 742.0         # tune it\nsilence_ms = 800\n")
            self.assertEqual(load_config(p).voice.vad_threshold, 742.0)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_silero_vad_uses_the_model_not_loudness(self):
        class Fake:  # says "speech" for chunks whose first sample is 1, whatever their loudness
            resets = 0

            def reset_states(self):
                Fake.resets += 1

            def predict(self, x, frame_size=480):
                return 0.9 if x[0] == 1 else 0.1

        speech = array.array("h", [1] + [40] * 479).tobytes()        # quiet, but speech
        fan = array.array("h", [0] + [3000] * 479).tobytes()         # loud, but noise
        vad = SileroVAD(0.5, model=Fake(), silence_ms=90, pre_ms=0)
        out = vad.collect(iter([fan] * 5 + [speech] * 4 + [fan] * 3 + [speech] * 50))
        self.assertEqual(out, speech * 4 + fan * 3)   # loud fan noise ends it, quiet speech doesn't start late
        self.assertEqual(Fake.resets, 1)
        self.assertEqual(SileroVAD(0.5, model=Fake(), start_timeout_s=0.3).collect(iter([fan] * 50)), b"")

    def test_make_vad(self):
        self.assertIs(type(make_vad("energy", 450, 0.5)), EnergyVAD)
        with self.assertRaises(ValueError):
            make_vad("magic", 450, 0.5)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_wake_word_needs_speech(self):
        import numpy as np
        from assistant.voice.engines import OpenWakeWord

        class Model:
            def predict(self, frame):
                return {"uzi": 0.9, "boafo": 0.2}

        class Vad:
            def __init__(self, speech):
                self.speech = speech

            def is_speech(self, chunk):
                return self.speech

        def make(speech):
            w = object.__new__(OpenWakeWord)
            w.np, w.model, w.threshold, w.vad, w.last_word = np, Model(), 0.5, Vad(speech), ""
            w.thresholds = {}
            return w

        with self.assertRaises(EOFError):          # fan noise scoring 0.9: never wakes
            make(False).wait(iter([b"\0" * 960] * 40))
        w = make(True)
        self.assertEqual(w.wait(iter([b"\0" * 960] * 40)), 0.9)
        self.assertEqual(w.last_word, "uzi")
        w = make(True)
        w.thresholds = {"uzi": 0.95}                # stricter for uzi: 0.9 isn't enough now
        with self.assertRaises(EOFError):
            w.wait(iter([b"\0" * 960] * 40))
        self.assertEqual(w.threshold_for("hey_jarvis_v0.1"), 0.5)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_flush_runs_even_if_the_model_has_reset(self):
        import numpy as np
        from assistant.voice.engines import OpenWakeWord

        class Model:  # like openWakeWord 0.4: reset() exists but only clears scores
            def __init__(self):
                self.silent = 0

            def reset(self):
                pass

            def predict(self, frame):
                if not frame.any():
                    self.silent += 1
                return {"uzi": 0.0}

        w = object.__new__(OpenWakeWord)
        w.np, w.model, w.threshold, w.vad, w.thresholds, w.patience = np, Model(), 0.5, None, {}, 1
        w.last_word = "uzi"                       # a wake-up just happened
        with self.assertRaises(EOFError):
            w.wait(iter([]))
        self.assertEqual(w.model.silent, 20)      # the old word was pushed out

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_wake_patience_ignores_single_frame_spikes(self):
        import numpy as np
        from assistant.voice.engines import OpenWakeWord

        class Model:  # one high frame, then low, then four high in a row
            def __init__(self):
                self.k = 0

            def predict(self, frame):
                self.k += 1
                return {"uzi": 0.9 if self.k in (3, 7, 8, 9, 10) else 0.1}

        w = object.__new__(OpenWakeWord)
        w.np, w.model, w.threshold, w.vad, w.last_word, w.thresholds, w.patience = np, Model(), 0.5, None, "", {}, 3
        w.wait(iter([bytes(960)] * 60))
        self.assertEqual(w.model.k, 9)   # woke on the third high frame in a row, not on the spike

    def test_whisper_prompt(self):
        stt = WhisperCppSTT("whisper-cli -m {model} -f {wav} -nt", "m.bin", 16000, "A question.")
        self.assertEqual(stt.command("a.wav")[-2:], ["--prompt", "A question."])
        self.assertNotIn("--prompt", WhisperCppSTT("whisper-cli -m {model} -f {wav}", "m.bin").command("a.wav"))
        self.assertNotIn("-ac", stt.command("a.wav", 3 * 32000))                  # off by default
        fit = WhisperCppSTT("whisper-cli -m {model} -f {wav}", "m.bin", fit_window=True)
        self.assertEqual(fit.command("a.wav", 3 * 32000)[-2:], ["-ac", "768"])     # 15 s minimum
        self.assertEqual(audio_ctx_for(3 * 32000, floor=0), 214)                    # 3 s -> 150 + 64
        self.assertEqual(audio_ctx_for(60 * 32000), 1500)                           # never above 30 s

    def test_wake_model_both_apis(self):
        new = {"self": 0, "wakeword_models": 0, "inference_framework": 0}
        self.assertEqual(wake_model_kwargs(new, "hey_jarvis", pathlib.Path(".")),
                         {"wakeword_models": ["hey_jarvis"], "inference_framework": "onnx"})
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "hey_jarvis_v0.1.onnx").write_bytes(b"")
            old = {"self": 0, "wakeword_model_paths": 0}
            self.assertEqual(wake_model_kwargs(old, "hey_jarvis", pathlib.Path(d)),
                             {"wakeword_model_paths": [str(pathlib.Path(d) / "hey_jarvis_v0.1.onnx")]})
            with self.assertRaises(FileNotFoundError):
                wake_model_kwargs(old, "hey_nobody", pathlib.Path(d))

    def test_piper_pcm_both_apis(self):
        class Old:  # piper-tts 1.2
            def synthesize_stream_raw(self, text):
                yield b"ab"
                yield b"cd"

        class Chunk:
            def __init__(self, b):
                self.audio_int16_bytes = b

        class New:  # piper-tts 1.3+
            def synthesize(self, text):
                return iter([Chunk(b"ab"), Chunk(b"cd")])

        self.assertEqual(piper_pcm(Old(), "hi"), b"abcd")
        self.assertEqual(piper_pcm(New(), "hi"), b"abcd")

    def test_piper_uses_loaded_voice(self):
        tts = PiperTTS("no-such-piper --model {model}", "missing.onnx", keep_loaded=False)
        self.assertIsNone(tts.voice)
        tts.voice = type("V", (), {"synthesize_stream_raw": lambda self, t: iter([t.encode()])})()
        self.assertEqual(tts.synthesize("hi"), b"hi")  # no program started

    def test_resolve_cmd(self):
        self.assertEqual(resolve_cmd(["/opt/x/piper", "-m", "a"]), ["/opt/x/piper", "-m", "a"])
        self.assertEqual(resolve_cmd(["no-such-program-xyz", "-v"]), ["no-such-program-xyz", "-v"])
        py = pathlib.Path(sys.executable).name  # the interpreter is always in its own folder
        self.assertTrue(resolve_cmd([py, "-V"])[0].endswith(py))


if __name__ == "__main__":
    unittest.main()
