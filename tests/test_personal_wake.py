import importlib.util
import pathlib
import tempfile
import unittest

import _path  # noqa: F401

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class PersonalWakeTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from assistant.voice import personal_wake as pw
        self.np, self.pw = np, pw
        self.rng = np.random.default_rng(0)
        self.word = self.rng.normal(size=(pw.LENGTH, 96))          # "your word"

    def noisy(self, base, amount):
        return base + self.rng.normal(scale=amount, size=base.shape)

    def test_similarity_finds_the_word_a_little_before_now(self):
        pw, np = self.pw, self.np
        t = pw.normalise(self.word)
        live = pw.normalise(np.vstack([self.rng.normal(size=(2, 96)), self.word, self.rng.normal(size=(2, 96))]))
        self.assertGreater(pw.similarity(live, t), 0.99)       # found with a shift of 2
        self.assertLess(pw.similarity(live, t, max_shift=0), 0.5)

    def test_threshold_sits_between_the_word_and_other_speech(self):
        pw = self.pw
        examples = [self.noisy(self.word, 0.3) for _ in range(5)]
        other = self.rng.normal(size=(30, 96))
        thr, stats = pw.suggest_threshold(examples, other)
        self.assertLess(stats["other_speech_max"], thr)
        self.assertLess(thr, stats["match_median"])

    def test_save_and_load(self):
        pw = self.pw
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "boafo.npz"
            pw.save_word(p, [self.word, self.word * 2], 0.71, {"match_median": 0.9})
            templates, thr = pw.load_word(p)
        self.assertEqual(len(templates), 2)
        self.assertAlmostEqual(thr, 0.71, places=5)
        self.assertAlmostEqual(float(self.np.linalg.norm(templates[0][0])), 1.0, places=5)

    def make(self, script, speech=True):
        """A wake word fed by a fake feature model: frame k of `script` is what it 'hears'."""
        pw, np = self.pw, self.np

        class Features:
            def __init__(self):
                self.seen = []

            def __call__(self, frame):
                self.seen.append(script[min(len(self.seen), len(script) - 1)])

            def get_features(self, n):
                return np.array(self.seen[-n:])[None]

        class Vad:
            def is_speech(self, chunk):
                return speech

        with tempfile.TemporaryDirectory() as d:
            for name, base in (("uzi", self.word), ("aiya", self.rng.normal(size=self.word.shape))):
                pw.save_word(pathlib.Path(d) / f"{name}.npz", [base], 0.8, {})
            return pw.PersonalWakeWord([pathlib.Path(d) / "uzi.npz", pathlib.Path(d) / "aiya.npz"],
                                       vad=Vad(), features=Features())

    def chunks(self, n):
        return iter([b"\0" * 960] * n)  # 30 ms chunks; contents unused by the fakes

    def test_fires_on_the_right_word_after_warmup(self):
        np = self.np
        noise = list(self.rng.normal(size=(40, 96)))
        w = self.make(noise + list(self.word) + noise)
        self.assertGreater(w.wait(self.chunks(400)), 0.8)
        self.assertEqual(w.last_word, "uzi")

    def test_only_while_someone_speaks(self):
        w = self.make([row for row in self.word] * 50, speech=False)
        with self.assertRaises(EOFError):
            w.wait(self.chunks(200))


if __name__ == "__main__":
    unittest.main()
