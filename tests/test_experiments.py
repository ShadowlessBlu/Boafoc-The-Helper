import json
import unittest

import _path  # noqa: F401
import kv_calc
from bench_ollama import DEFAULT_PROMPTS, load_prompts
from score import ROOT, check


class KVCalcTests(unittest.TestCase):
    def test_kv_formula(self):
        a = kv_calc.PRESETS["llama3.2-1b"]
        # 2 * 16 layers * 8 kv heads * 64 dim * 2 bytes = 32 KiB per token
        self.assertEqual(kv_calc.kv_cache_bytes(a, 1, "f16"), 32 * 1024)
        self.assertEqual(kv_calc.kv_cache_bytes(a, 2048, "f16"), 64 * 1024 ** 2)

    def test_quant_halves(self):
        p = 1e9
        self.assertAlmostEqual(kv_calc.weights_bytes(p, "f16") / kv_calc.weights_bytes(p, "q8_0"), 16 / 8.5)

    def test_cli_runs(self):
        kv_calc.main(["--model", "qwen3-0.6b", "--ctx", "1024"])


class PromptFileTests(unittest.TestCase):
    def test_standard_prompts(self):
        ids = [pid for pid, text in load_prompts(DEFAULT_PROMPTS) if text]
        self.assertEqual(ids, ["short", "explain", "assistant", "long_context"])


class ScoreCheckTests(unittest.TestCase):
    def test_test_set_valid(self):
        data = json.loads((ROOT / "experiments" / "test_set_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["items"]), 20)
        self.assertEqual(len({i["id"] for i in data["items"]}), 20)

    def test_test_set_v2_extends_v1(self):
        v1 = json.loads((ROOT / "experiments" / "test_set_v1.json").read_text(encoding="utf-8"))
        v2 = json.loads((ROOT / "experiments" / "test_set_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(len(v2["items"]), 60)
        self.assertEqual(len({i["id"] for i in v2["items"]}), 60)
        self.assertEqual(v2["items"][:20], v1["items"])  # v1 questions unchanged, so scores compare
        for item in v2["items"]:  # every check type is one the scorer knows
            check(item["check"], "", None)

    def test_number(self):
        self.assertTrue(check({"type": "number", "value": 391}, "17 x 23 = 391.", None)[0])
        self.assertFalse(check({"type": "number", "value": 391}, "It is 390.", None)[0])

    def test_larger_regex(self):
        data = json.loads((ROOT / "experiments" / "test_set_v1.json").read_text(encoding="utf-8"))
        c = next(i["check"] for i in data["items"] if i["id"] == "r04")
        self.assertTrue(check(c, "9.9 is larger than 9.11.", None)[0])
        self.assertTrue(check(c, "The larger number is 9.9.", None)[0])
        self.assertTrue(check(c, "9.9", None)[0])
        self.assertFalse(check(c, "9.11 is larger than 9.9.", None)[0])

    def test_three_words(self):
        c = {"type": "regex", "pattern": "^\\W*\\w+\\W+\\w+\\W+\\w+\\W*$"}
        self.assertTrue(check(c, "Vast, blue, restless.", None)[0])
        self.assertFalse(check(c, "The sea is vast.", None)[0])

    def test_backwards_with_spaces(self):
        c = {"type": "contains_any", "values": ["yrrebpsar"], "strip_spaces": True}
        self.assertTrue(check(c, "Y R R E B P S A R", None)[0])

    def test_tool_check(self):
        c = {"type": "tool", "name": "set_light", "args": {"room": "living", "state": "on"}}
        self.assertTrue(check(c, "", {"name": "set_light", "args": {"room": "living room", "state": "on"}})[0])
        self.assertFalse(check(c, "", {"name": "set_light", "args": {"room": "kitchen", "state": "on"}})[0])
        self.assertFalse(check(c, "", None)[0])
        self.assertTrue(check({"type": "no_tool"}, "hi", None)[0])

    def test_tool_check_needs_the_governor_to_agree(self):
        c = {"type": "tool", "name": "set_light", "args": {"room": "kitchen"}}
        ok, why = check(c, "", {"name": "set_light", "args": {"room": "kitchen", "state": "on", "brightness": 5}})
        self.assertFalse(ok)
        self.assertIn("governor denied", why)
        ok, _ = check({"type": "tool", "name": "get_time"}, "", {"name": "get_time", "args": {"timezone": "local"}})
        self.assertTrue(ok)  # read-only tools drop invented arguments (2026-11-19)
        self.assertTrue(check({"type": "tool", "name": "unlock_door"}, "",
                              {"name": "unlock_door", "args": {"door": "front"}})[0])  # confirm = the user is asked


if __name__ == "__main__":
    unittest.main()
