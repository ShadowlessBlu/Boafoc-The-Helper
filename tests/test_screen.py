import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime

import _path  # noqa: F401
from assistant.screen import (MAIN_REGIONS, SAMPLE, App, activity_lines, card_rects, clock_text, date_text, describe,
                              duration_text, face_shapes, page_to_show, region_at, wake_word_list)

HAS_PYGAME = importlib.util.find_spec("pygame") is not None


class FaceShapesTest(unittest.TestCase):
    def kinds(self, mode, t=1.0):
        return [s[0] for s in face_shapes(mode, t)]

    def test_each_state_has_its_own_mouth(self):
        self.assertEqual(self.kinds("idle"), ["poly", "poly"])
        self.assertEqual(self.kinds("speaking").count("bar"), 11)
        self.assertEqual(self.kinds("thinking").count("dot"), 3)
        self.assertIn("ring", self.kinds("listening"))

    def eye_height(self, mode, t=1.0):
        pts = face_shapes(mode, t)[0][1]
        return pts[3][1] - pts[0][1]

    def test_eyes_open_wide_when_listening_and_narrow_when_thinking(self):
        self.assertGreater(self.eye_height("listening"), self.eye_height("idle"))
        self.assertLess(self.eye_height("thinking"), self.eye_height("idle"))

    def test_blink(self):
        self.assertEqual(self.eye_height("idle", t=4.2 * 3 + 0.05), 4)


class DescribeTest(unittest.TestCase):
    def test_sample(self):
        d = describe(SAMPLE)
        self.assertEqual((d["label"], d["temp"], d["lights"]), ("Waiting for the wake word", "51 °C", "Lights on: kitchen"))
        self.assertEqual(d["heard"], "You: What time is it? · 4.7 s")
        self.assertTrue(d["dots"]["Model"])

    def test_offline_and_voice_off(self):
        self.assertEqual(describe(None)["mode"], "offline")
        self.assertEqual(describe({"mode": "idle", "voice": False})["label"], "Ready · voice off")

    def test_hot_pi_is_flagged(self):
        d = describe({"mode": "idle", "pi": {"temp_c": 82.0, "throttled": True}})
        self.assertTrue(d["temp_warn"] and d["thr_warn"])
        self.assertEqual(d["thr"], "Throttling")

    def test_clock(self):
        self.assertEqual(clock_text(datetime(2026, 11, 19, 0, 5)), "12:05 AM")
        self.assertEqual(clock_text(datetime(2026, 11, 19, 17, 11)), "5:11 PM")
        self.assertEqual(date_text(datetime(2026, 11, 19)), "Thursday 19 November")


def centre(name):
    x, y, w, h = MAIN_REGIONS[name]
    return (x + w // 2, y + h // 2)


class TouchTest(unittest.TestCase):
    def test_cards_and_buttons_open_their_pages(self):
        for region, page in (("face", "face"), ("pi", "pi"), ("reply", "history"), ("home", "home"),
                             ("settings", "settings")):
            app = App()
            self.assertEqual(app.tap(centre(region), 10.0), [("page", page)], region)
            self.assertEqual(app.page, page)

    def test_clock_opens_at_once_and_never_exits(self):
        app = App()
        self.assertEqual(app.tap(centre("clock"), 10.0), [("page", "clock")])
        app = App()
        for k in range(5):                                            # many quick taps: never an exit
            self.assertNotIn(("exit",), app.tap(centre("clock"), 10.0 + k * 0.1))

    def test_exit_button_needs_a_second_tap(self):
        app = App()
        app.tap(centre("settings"), 1.0)
        exit_pos = (660, 430)
        self.assertEqual(app.tap(exit_pos, 2.0), [])                   # armed: "Tap again to exit"
        self.assertEqual(app.tap(exit_pos, 2.5), [("exit",)])
        app.tap(exit_pos, 10.0)
        self.assertEqual(app.tap(exit_pos, 14.0), [])                  # too late: armed again instead
        app.tick(20.0)
        self.assertIsNone(app.exit_armed)

    def test_full_face_only_answers_a_triple_tap(self):
        app = App()
        app.tap(centre("face"), 1.0)
        self.assertEqual(app.page, "face")
        self.assertEqual(app.tap((60, 30), 5.0), [])                   # where Back would be: nothing
        self.assertEqual(app.tap((400, 400), 9.0), [])
        self.assertEqual(app.tap((400, 400), 9.2), [])
        self.assertEqual(app.tap((10, 470), 9.4), [("page", "main")])
        self.assertEqual(app.tick(500.0), [])                          # the face never times out

    def test_back_and_auto_return(self):
        app = App()
        app.tap(centre("pi"), 1.0)
        self.assertEqual(app.tap((60, 30), 2.0), [("page", "main")])
        app.tap(centre("home"), 3.0)
        self.assertEqual(app.tick(50.0), [])
        self.assertEqual(app.tick(64.0), [("page", "main")])

    def test_mute_and_dim(self):
        app = App()
        self.assertEqual(app.tap(centre("mute"), 1.0), [("toggle_mute",)])
        app.tap(centre("settings"), 2.0)
        self.assertEqual(app.tap((650, 130), 3.0), [("toggle_mute",)])
        self.assertEqual(app.tap((650, 200), 4.0), [("dim", True)])
        self.assertTrue(app.dim)

    def test_home_rows_switch_lights(self):
        from assistant.screen import HOME_REGIONS, home_note, light_to_switch
        app = App()
        app.tap(centre("home"), 1.0)
        x, y, w, h = HOME_REGIONS["light:3"]
        self.assertEqual(app.tap((x + w // 2, y + h // 2), 2.0), [("light", 3)])
        st = {"home": {"lights": {"living room": "off", "bedroom": "off", "kitchen": "on", "office": "off"}}}
        self.assertEqual(light_to_switch(st, 3), ("office", "on"))
        self.assertEqual(light_to_switch(st, 2), ("kitchen", "off"))
        self.assertIsNone(light_to_switch(st, 4))
        self.assertIsNone(light_to_switch(None, 0))
        self.assertEqual(app.tap((60, 30), 3.0), [("page", "main")])   # Back still works
        self.assertEqual(home_note(("The office light is on.", 10.0), 12.0), "The office light is on.")
        self.assertEqual(home_note(("The office light is on.", 10.0), 20.0), "")

    def test_regions_do_not_overlap_on_the_dashboard(self):
        rects = list(MAIN_REGIONS.values())
        for i, (ax, ay, aw, ah) in enumerate(rects):
            for bx, by, bw, bh in rects[i + 1:]:
                self.assertFalse(ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah)
        self.assertEqual(set(card_rects()), {"pi", "reply", "home"})

    def test_busy_assistant_never_changes_the_page(self):
        for page in ("main", "face", "pi", "history", "home", "clock", "settings"):
            for mode in ("listening", "thinking", "speaking", "idle"):
                self.assertEqual(page_to_show(page, mode), page)

    def test_activity_lines(self):
        base = {**SAMPLE, "heard": "What time is it?", "said": "It's 4:39 PM."}
        self.assertEqual(activity_lines(describe({**base, "mode": "listening"})), ("Listening…", ""))
        self.assertEqual(activity_lines(describe({**base, "mode": "thinking"})), ("Thinking…", "You: What time is it?"))
        self.assertEqual(activity_lines(describe({**base, "mode": "speaking"})), ("Speaking", "It's 4:39 PM."))

    def test_wake_pages(self):
        from assistant.screen import (SAMPLE_WAKE, SETTINGS_REGIONS, WAKE_REGIONS, WAKES_REGIONS, clamp_offset,
                                      false_text, wake_request)

        def mid(r):
            return r[0] + r[2] // 2, r[1] + r[3] // 2
        app = App()
        app.tap(centre("settings"), 1.0)
        self.assertEqual(app.tap(mid(SETTINGS_REGIONS["wake"]), 2.0), [("page", "wake")])
        self.assertEqual(app.tap(mid(WAKE_REGIONS["wtoggle:2"]), 3.0), [("wake_toggle", 2)])
        self.assertEqual(app.tap(mid(WAKE_REGIONS["wdown:1"]), 4.0), [("wake_step", 1, -1)])
        self.assertEqual(app.tap(mid(WAKE_REGIONS["wup:0"]), 5.0), [("wake_step", 0, 1)])
        self.assertEqual(app.tap(mid(WAKE_REGIONS["period"]), 6.0), [("wake_period", 7)])
        self.assertEqual(app.tap(mid(WAKE_REGIONS["recent"]), 7.0), [("page", "wakes")])
        self.assertEqual(app.tap(mid(WAKES_REGIONS["false:1"]), 8.0), [("wake_label", 1, "false")])
        app.tap(mid(WAKES_REGIONS["older"]), 9.0)
        self.assertEqual(app.tap(mid(WAKES_REGIONS["real:0"]), 10.0), [("wake_label", 6, "real")])
        self.assertEqual(app.tap((60, 30), 11.0), [("page", "wake")])      # Back goes up one level
        self.assertEqual(app.tap((60, 30), 12.0), [("page", "settings")])

        self.assertEqual(wake_request(SAMPLE_WAKE, ("wake_toggle", 2), 7),
                         ("/wake", {"word": "boafo", "enabled": True, "days": 7}))
        self.assertEqual(wake_request(SAMPLE_WAKE, ("wake_step", 1, 1)),
                         ("/wake", {"word": "aiya", "threshold": 0.8, "days": 1}))
        low = {**SAMPLE_WAKE, "words": [{**SAMPLE_WAKE["words"][0], "threshold": 0.3}]}
        self.assertIsNone(wake_request(low, ("wake_step", 0, -1)))       # already at the lowest
        self.assertEqual(wake_request(SAMPLE_WAKE, ("wake_label", 0, "false")),    # tap again: cleared
                         ("/wake/label", {"id": "3", "label": "", "days": 1}))
        self.assertEqual(wake_request(SAMPLE_WAKE, ("wake_label", 2, "real"))[1]["label"], "real")
        self.assertIsNone(wake_request(SAMPLE_WAKE, ("wake_label", 9, "real")))
        self.assertIsNone(wake_request(None, ("wake_toggle", 0)))
        self.assertEqual((clamp_offset(12, 7), clamp_offset(6, 6), clamp_offset(6, 13)), (6, 0, 6))
        self.assertEqual(false_text(SAMPLE_WAKE["words"][0]["stats"]), "3/5 false")

    def test_wake_page_regions_do_not_overlap(self):
        from assistant.screen import WAKE_REGIONS, WAKES_REGIONS
        for regions in (WAKE_REGIONS, WAKES_REGIONS):
            rects = list(regions.values()) + [(0, 0, 150, 64)]
            for i, (ax, ay, aw, ah) in enumerate(rects):
                for bx, by, bw, bh in rects[i + 1:]:
                    self.assertFalse(ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah)
                self.assertLessEqual(ax + aw, 800)
                self.assertLessEqual(ay + ah, 480)

    def test_texts(self):
        self.assertEqual(wake_word_list(["models/wake/boafo.onnx@0.8", "hey_jarvis"]), ["Boafo (0.8)", "Hey jarvis"])
        self.assertEqual(duration_text(90061), "1 d 1 h 1 min")
        d = describe({**SAMPLE, "mode": "idle", "muted": True})
        self.assertEqual((d["mode"], d["label"]), ("muted", "Microphone off"))
        self.assertEqual(region_at("history", (60, 30)), "back")


@unittest.skipUnless(HAS_PYGAME, "pygame-ce not installed")
class ScreenshotTest(unittest.TestCase):
    def test_draws_every_state_without_a_display(self):
        src = pathlib.Path(__file__).resolve().parents[1] / "src"
        with tempfile.TemporaryDirectory() as d:
            for mode in ("idle", "listening", "thinking", "speaking", "offline", "muted"):
                png = pathlib.Path(d) / f"{mode}.png"
                subprocess.run([sys.executable, "-m", "assistant.screen", "--shot", str(png), "--mode", mode],
                               check=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(src)})
                self.assertGreater(png.stat().st_size, 3000)
            for page in ("face", "clock", "history", "home", "pi", "settings", "wake", "wakes"):
                for mode in ("idle", "thinking"):          # busy: activity bar / caption, same page
                    png = pathlib.Path(d) / f"page_{page}_{mode}.png"
                    subprocess.run([sys.executable, "-m", "assistant.screen", "--shot", str(png), "--page", page,
                                    "--mode", mode], check=True, capture_output=True,
                                   env={**os.environ, "PYTHONPATH": str(src)})
                self.assertGreater(png.stat().st_size, 3000)


if __name__ == "__main__":
    unittest.main()
