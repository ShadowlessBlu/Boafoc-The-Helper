"""Memory rules (2026-11-25): history kept 30 days, audit 90, facts until forgotten by voice (with a yes),
Clear history on the screen, deleted rows overwritten in the file."""
import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import _path  # noqa: F401
from assistant.api import AssistantAPI
from assistant.config import load_config
from assistant.memory import Memory
from assistant.screen import HISTORY_REGIONS, App
from test_brain import make


class RetentionTests(unittest.TestCase):
    def test_prune_by_age(self):
        mem = Memory()
        now = time.time()
        for age_days, table in ((40, "messages"), (10, "messages"), (100, "audit"), (50, "audit")):
            mem.db.execute(f"INSERT INTO {table} (ts, " + ("role, content" if table == "messages" else "tool")
                           + ") VALUES (?" + (",'user','hi'" if table == "messages" else ",'x'") + ")",
                           (now - age_days * 86400,))
        mem.remember("My sister's birthday is 3 May.")
        self.assertEqual(mem.prune(30, 90, now), (1, 1))
        self.assertEqual(mem.prune(30, 90, now), (0, 0))
        self.assertEqual(mem.prune(0, 0, now + 1e9), (0, 0))          # 0 = keep forever
        self.assertEqual(len(mem.list_memories()), 1)                 # facts are never pruned

    def test_defaults(self):
        cfg = load_config("does-not-exist.toml")
        self.assertEqual((cfg.memory.history_days, cfg.memory.audit_days), (30, 90))

    def test_clear_history_keeps_notes_and_audit(self):
        mem = Memory()
        mem.add_message("user", "hi")
        mem.add_message("assistant", "Hello.")
        mem.remember("I like tea.")
        mem.log_audit("get_time", {}, "allow", "read action")
        self.assertEqual(mem.clear_history(), 2)
        self.assertEqual(mem.recent_messages(), [])
        self.assertEqual(len(mem.list_memories()), 1)
        self.assertEqual(len(mem.audit_log()), 1)

    def test_deleted_text_is_gone_from_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "a.db"
            mem = Memory(db)
            mem.remember("secret-pin-4821 is my code")
            mem.forget(1)
            mem.add_message("user", "my-diary-entry-xyz")
            mem.clear_history()
            mem.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            mem.db.close()
            raw = db.read_bytes() + (Path(str(db) + "-wal").read_bytes() if Path(str(db) + "-wal").exists() else b"")
            self.assertNotIn(b"secret-pin-4821", raw)
            self.assertNotIn(b"my-diary-entry-xyz", raw)


class NoDuplicateNotesTests(unittest.TestCase):
    def test_recall_question_does_not_save_a_copy(self):
        brain, _, mem, _ = make({"say": "The spare key is in the blue box.", "tool": "remember",
                                 "args": {"text": "The spare key is in the blue box!"}})
        mem.remember("The spare key is in the blue box.")
        self.assertEqual(brain.handle("Where is the spare key?"), "The spare key is in the blue box.")
        self.assertEqual(len(mem.list_memories()), 1)
        self.assertEqual(mem.remember_new("My bike is red."), "saved as memory #2")
        self.assertEqual(mem.remember_new("my bike is RED"), "already saved as memory #2")

    def test_near_identical_meaning_by_embedding(self):
        vecs = {"The spare key is in the blue box.": [1.0, 0.0], "Spare key: blue box.": [0.99, 0.05],
                "My bike is red.": [0.0, 1.0]}
        mem = Memory(embedder=lambda texts: [vecs[t] for t in texts])
        mem.remember("The spare key is in the blue box.")
        self.assertEqual(mem.duplicate_of("Spare key: blue box."), 1)
        self.assertIsNone(mem.duplicate_of("My bike is red."))


class VoiceForgetTests(unittest.TestCase):
    def call(self, answer):
        return {"say": "I'll forget that.", "tool": "forget", "args": {"text": "my sister's birthday"}}

    def test_names_the_note_and_waits_for_yes(self):
        brain, _, mem, _ = make(self.call(None), self.call(None))
        mem.remember("My sister's birthday is 3 May.")
        mem.remember("The wifi router is in the office.")
        asked = []
        reply = brain.handle("forget my sister's birthday", confirm=lambda q: asked.append(q) or False)
        self.assertEqual(asked, ["Should I forget this note: My sister's birthday is 3 May.?"])
        self.assertEqual(reply, "Okay, I won't do that.")
        self.assertEqual(len(mem.list_memories()), 2)
        reply = brain.handle("forget my sister's birthday", confirm=lambda q: True)
        self.assertEqual(reply, "Done, I've forgotten that note.")
        self.assertEqual([t for _, t in mem.list_memories()], ["The wifi router is in the office."])
        self.assertEqual(mem.audit_log(1)[0][1], "forget")

    def test_nothing_to_forget(self):
        brain, _, mem, _ = make(self.call(None))
        asked = []
        reply = brain.handle("forget my sister's birthday", confirm=lambda q: asked.append(q) or True)
        self.assertEqual((asked, reply), ([], "I don't have a note like that."))

    def test_forget_sentences_skip_the_model(self):
        from assistant.brain import quick_forget
        self.assertEqual(quick_forget("Forget what I told you about my birthday."), "about my birthday")
        self.assertEqual(quick_forget("Please delete the note about the wifi password"), "about the wifi password")
        self.assertEqual(quick_forget("Can you forget that my sister lives in Kumasi?"), "my sister lives in kumasi")
        self.assertIsNone(quick_forget("Don't forget to turn off the light."))
        self.assertIsNone(quick_forget("Remember my birthday is 3 May."))
        self.assertIsNone(quick_forget("Delete everything."))              # not about a note: the model decides
        brain, llm, mem, _ = make()                         # no scripted reply: the model must not be asked
        mem.remember("My birthday is 3 May.")
        asked = []
        reply = brain.handle("Forget what I told you about my birthday.", confirm=lambda q: asked.append(q) or True)
        self.assertEqual(asked, ["Should I forget this note: My birthday is 3 May.?"])
        self.assertEqual((reply, mem.list_memories(), llm.calls), ("Done, I've forgotten that note.", [], []))

    def test_never_over_the_api(self):
        brain, _, mem, _ = make(self.call(None))
        mem.remember("My sister's birthday is 3 May.")
        self.assertEqual(brain.handle("forget my sister's birthday"), "Okay, I won't do that.")
        self.assertEqual(len(mem.list_memories()), 1)


class NoTelemetryTests(unittest.TestCase):
    def test_importing_the_assistant_turns_onnxruntime_telemetry_off(self):
        import os
        import assistant  # noqa: F401
        self.assertEqual(os.environ.get("ORT_DISABLE_TELEMETRY"), "1")


class ClearHistoryApiAndScreenTests(unittest.TestCase):
    def test_api(self):
        brain, _, mem, _ = make()
        mem.add_message("user", "hi")
        api = AssistantAPI(brain, token="secret")
        server = api.make_server("127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/history/clear"
            req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                self.assertEqual(json.loads(r.read()), {"deleted": 1})
        finally:
            server.shutdown()
            server.server_close()

    def test_screen_needs_two_taps(self):
        app = App()
        app.page = "history"
        x, y, w, h = HISTORY_REGIONS["clear"]
        mid = (x + w // 2, y + h // 2)
        self.assertEqual(app.tap(mid, 1.0), [])
        self.assertEqual(app.tap(mid, 2.0), [("clear_history",)])
        self.assertEqual(app.tap(mid, 10.0), [])
        self.assertEqual(app.tap(mid, 20.0), [])                     # too slow: armed again, not cleared
