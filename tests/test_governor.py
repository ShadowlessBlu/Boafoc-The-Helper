import unittest

import _path  # noqa: F401
from assistant.governor import Governor, Verdict
from assistant.memory import Memory
from assistant.tools import Devices, build_tools


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class GovernorTests(unittest.TestCase):
    def setUp(self):
        self.devices = Devices()
        self.log = []
        self.clock = FakeClock()
        self.gov = Governor(build_tools(Memory(), self.devices), audit=self.log.append, clock=self.clock)

    def test_read_tool_allowed(self):
        self.assertIs(self.gov.check({"name": "get_time", "args": {}}).verdict, Verdict.ALLOW)

    def test_low_tool_allowed_and_runs(self):
        d = self.gov.check({"name": "set_light", "args": {"room": "kitchen", "state": "on"}})
        self.assertIs(d.verdict, Verdict.ALLOW)
        self.gov.execute(d)
        self.assertEqual(self.devices.lights["kitchen"], "on")

    def test_sensitive_needs_confirmation(self):
        d = self.gov.check({"name": "unlock_door", "args": {"door": "front"}})
        self.assertIs(d.verdict, Verdict.CONFIRM)
        with self.assertRaises(PermissionError):
            self.gov.execute(d)
        self.assertIn("unlocked", self.gov.execute(d, confirmed=True))

    def test_forbidden_denied(self):
        d = self.gov.check({"name": "run_shell", "args": {"command": "rm -rf /"}})
        self.assertIs(d.verdict, Verdict.DENY)
        with self.assertRaises(PermissionError):
            self.gov.execute(d, confirmed=True)

    def test_unknown_tool_denied(self):
        self.assertIs(self.gov.check({"name": "format_disk", "args": {}}).verdict, Verdict.DENY)

    def test_malformed_proposals_denied(self):
        for bad in [None, "set_light", [], {"args": {}}, {"name": "set_light", "args": "kitchen"}]:
            self.assertIs(self.gov.check(bad).verdict, Verdict.DENY, bad)

    def test_argument_validation(self):
        cases = [
            {"room": "kitchen"},                                   # missing
            {"room": "garage", "state": "on"},                     # not in enum
            {"room": "kitchen", "state": True},                    # wrong type
            {"room": "kitchen", "state": "on", "brightness": 5},   # extra
        ]
        for args in cases:
            d = self.gov.check({"name": "set_light", "args": args})
            self.assertIs(d.verdict, Verdict.DENY, args)

    def test_read_tool_ignores_invented_arguments(self):
        # qwen3:1.7b on the Pi: "What time is it?" -> get_time {"timezone": "local"}
        d = self.gov.check({"name": "get_time", "args": {"timezone": "local"}})
        self.assertIs(d.verdict, Verdict.ALLOW)
        self.assertEqual(d.args, {})
        self.assertIn("timezone", d.reason)
        self.assertTrue(self.gov.execute(d))  # runs with the cleaned arguments

    def test_extra_arguments_still_denied_with_side_effects(self):
        d = self.gov.check({"name": "unlock_door", "args": {"door": "front", "force": True}})
        self.assertIs(d.verdict, Verdict.DENY)  # sensitive
        d = self.gov.check({"name": "set_light", "args": {"room": "kitchen", "state": "on", "x": 1}})
        self.assertIs(d.verdict, Verdict.DENY)  # low risk, but changes something

    def test_long_string_denied(self):
        d = self.gov.check({"name": "remember", "args": {"text": "x" * 1000}})
        self.assertIs(d.verdict, Verdict.DENY)

    def test_rate_limit_and_window(self):
        p = {"name": "unlock_door", "args": {"door": "front"}}  # 3 per minute
        verdicts = [self.gov.check(p).verdict for _ in range(4)]
        self.assertEqual(verdicts[:3], [Verdict.CONFIRM] * 3)
        self.assertIs(verdicts[3], Verdict.DENY)
        self.clock.t += 61
        self.assertIs(self.gov.check(p).verdict, Verdict.CONFIRM)

    def test_every_decision_audited(self):
        self.gov.check({"name": "get_time", "args": {}})
        self.gov.check({"name": "nope"})
        self.assertEqual(len(self.log), 2)


if __name__ == "__main__":
    unittest.main()
