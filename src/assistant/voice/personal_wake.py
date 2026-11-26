"""Personal wake words: any word, in any language, learned from a few of your own recordings.

openWakeWord's ready-made models only cover a few English phrases ("hey jarvis", "alexa"...), and
training a new one needs thousands of synthetic clips from English voices, which mispronounce
Twi words like "Boafo" or "Aiya". Instead, this compares live audio with your own examples:

- Enrolment (`python -m assistant.voice.enroll boafo`): you say the word ~5 times. Each recording
  is turned into a sequence of speech "fingerprints" by openWakeWord's feature model (one 96-number
  vector per 80 ms) and saved to models/wake/<word>.npz. Only these numbers are kept, never audio.
- Listening: every 80 ms the latest fingerprints are compared with each saved example (cosine
  similarity, allowing a small time shift). It fires when the best match passes the threshold
  set at enrolment and Silero VAD says someone is actually speaking, so fan noise can't trigger it.

Several words can listen at once (wake_word = "uzi,aiya,boafo"); the log says which one fired and
the near misses of each, to compare them.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

FRAME = 1280            # samples per feature step (80 ms at 16 kHz)
MAX_SHIFT = 3           # live window may lag the example by up to 3 steps (240 ms)
LENGTH = 5              # frames per example: ~1.1 s of sound ending just after the word
WARMUP_S = 1.0          # ignore the first second after (re)starting: the feature buffer is filling


def normalise(feats: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is a cosine similarity."""
    return feats / np.maximum(np.linalg.norm(feats, axis=-1, keepdims=True), 1e-8)


def similarity(live: np.ndarray, template: np.ndarray, max_shift: int = MAX_SHIFT) -> float:
    """Best mean cosine similarity between the template and the newest frames of `live`,
    trying shifts 0..max_shift (the word may have ended a little before now). Rows normalised."""
    n = len(template)
    best = -1.0
    for s in range(max_shift + 1):
        end = len(live) - s
        if end - n < 0:
            break
        best = max(best, float(np.mean(np.sum(live[end - n:end] * template, axis=1))))
    return best


def save_word(path: Path, templates: list[np.ndarray], threshold: float, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"t{i}": normalise(t).astype(np.float32) for i, t in enumerate(templates)}
    np.savez(path, threshold=np.float32(threshold), **arrays,
             **{f"stat_{k}": np.float32(v) for k, v in stats.items()})


def load_word(path: Path) -> tuple[list[np.ndarray], float]:
    with np.load(path) as d:
        templates = [d[k] for k in sorted((k for k in d.files if k.startswith("t") and k[1:].isdigit()),
                                          key=lambda k: int(k[1:]))]
        return templates, float(d["threshold"])


def windows(feats: np.ndarray, n: int) -> list[np.ndarray]:
    """Every run of n consecutive frames (a sliding window over a longer recording)."""
    return [feats[i:i + n] for i in range(0, len(feats) - n + 1)]


def suggest_threshold(templates: list[np.ndarray], other: np.ndarray | None) -> tuple[float, dict]:
    """Halfway between how well the examples match each other (each against the rest) and how
    well other speech matches them (`other`: features of a normal sentence, any length).
    Silero lets only speech through, so other speech, not fan noise, is the real competitor."""
    norm = [normalise(t) for t in templates]
    pos = [max(similarity(o, t, 0) for j, o in enumerate(norm) if j != i) for i, t in enumerate(norm)]
    neg = 0.0
    if other is not None and len(other) >= len(norm[0]):
        neg = max(similarity(w, t, 0) for w in windows(normalise(other), len(norm[0])) for t in norm)
    p = float(np.median(pos))
    threshold = (p + neg) / 2 if neg < p else p - 0.02
    return round(threshold, 3), {"match_median": p, "match_min": float(min(pos)), "other_speech_max": neg}


class PersonalWakeWord:
    def __init__(self, paths: list[Path], vad=None, features=None, threshold_override: float = 0.0):
        if features is None:
            from openwakeword.utils import AudioFeatures
            features = AudioFeatures()
        self.features = features
        self.vad = vad
        self.words: dict[str, tuple[list[np.ndarray], float]] = {}
        for p in paths:
            templates, thr = load_word(p)
            self.words[p.stem] = (templates, threshold_override or thr)
        self.last_word = ""

    def scores(self) -> dict[str, float]:
        live = normalise(self.features.get_features(LENGTH + MAX_SHIFT)[0])
        return {w: max(similarity(live, t) for t in ts) for w, (ts, _) in self.words.items()}

    def wait(self, chunks: Iterator[bytes]) -> float:
        """Block until one of the words is heard; returns its score (the word is in .last_word)."""
        buf = b""
        steps, warmup = 0, int(WARMUP_S * 16000 / FRAME)
        speech_until = 0.0
        peaks: dict[str, float] = {}
        quiet_steps = 0
        for c in chunks:
            if self.vad is not None and self.vad.is_speech(c):
                speech_until = time.monotonic() + 0.6   # the word just ended, or is going on
            buf += c
            while len(buf) >= FRAME * 2:
                frame, buf = buf[: FRAME * 2], buf[FRAME * 2:]
                self.features(np.frombuffer(frame, dtype=np.int16))
                steps += 1
                if steps <= warmup:
                    continue
                if self.vad is not None and time.monotonic() > speech_until:
                    if peaks:
                        quiet_steps += 1
                        if quiet_steps >= 6:   # ~0.5 s after an attempt: log its best scores once
                            log.info("wake word near miss: %s", ", ".join(
                                f"{w} {s:.2f}/{self.words[w][1]:.2f}" for w, s in sorted(peaks.items())))
                            peaks, quiet_steps = {}, 0
                    continue
                quiet_steps = 0
                for w, s in self.scores().items():
                    if s >= self.words[w][1]:
                        log.info("wake word: %s (score %.2f, threshold %.2f)", w, s, self.words[w][1])
                        self.last_word = w
                        return s
                    if s >= self.words[w][1] - 0.15:
                        peaks[w] = max(peaks.get(w, 0.0), s)
        raise EOFError("microphone stream ended")
