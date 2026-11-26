import sys
import unittest
from pathlib import Path

import _path  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from wake_report import is_request, report  # noqa: E402

LOG = """2026-11-23 13:14:54,603 INFO assistant.voice.engines: wake word: uzi, score 0.92 (threshold 0.50)
2026-11-23 13:15:04,145 INFO assistant.voice.loop: user: Turn on the office light.
2026-11-23 13:15:20,548 INFO assistant.voice.engines: wake word: boafo, score 1.00 (threshold 0.80)
2026-11-23 13:15:27,110 INFO assistant.voice.loop: user: >>
2026-11-23 13:15:50,520 INFO assistant.voice.engines: wake word ignored, no speech heard: boafo, score 0.64
2026-11-23 13:16:20,548 INFO assistant.voice.engines: wake word: boafo, score 0.93 (threshold 0.80)
2026-11-23 13:17:00,858 INFO assistant: voice loop ready (wake word: x)
2026-11-22 09:00:00,000 INFO assistant.voice.engines: wake word: aiya, score 0.80 (threshold 0.70)"""


class WakeReportTest(unittest.TestCase):
    def test_counts_requests_and_unused_wakes(self):
        r = report(LOG.splitlines())
        self.assertEqual((r["uzi"]["wakes"], r["uzi"]["used"]), (1, 1))
        self.assertEqual((r["boafo"]["wakes"], r["boafo"]["unused"], r["boafo"]["ignored"]), (2, 2, 1))
        self.assertEqual(r["boafo"]["scores_unused"], [1.0, 0.93])   # a restart closes the open wake

    def test_since_and_fillers(self):
        self.assertNotIn("aiya", report(LOG.splitlines(), since="2026-11-23"))
        self.assertFalse(is_request("Thank you."))
        self.assertFalse(is_request(">>"))
        self.assertTrue(is_request("What time is it?"))
