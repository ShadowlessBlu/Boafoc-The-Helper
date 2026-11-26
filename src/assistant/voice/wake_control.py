"""Wake words you can switch on and off and tune from the screen, and what each one did.

Settings: the words and thresholds come from [voice] wake_word in config.toml; changes made on the
screen (POST /wake) are kept in data/wake_settings.json and win over config.toml. They take effect
at once, without a restart. At least one word stays on: Mic off is for not listening at all.

Events: one CSV per day in logs/wake/ (kept 60 days), one row per event, never what was said:
  replied    the word woke the assistant, it heard a question and answered
  no_speech  the word woke it, then nobody spoke
  phantom    the word woke it and Whisper made words up from noise ("Thank you.", ">>"): no reply
  ignored    the word scored above its threshold with no speech around it (fan, music): no wake
  near_miss  the word came close (above 0.15) without reaching its threshold
  label      you marked a "replied" wake as real (you were talking to it) or false
The words heard after recent wakes are kept in memory only (the Recent wakes page), for marking.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

STEP, LOWEST, HIGHEST = 0.05, 0.30, 0.95
WAKES = ("replied", "no_speech", "phantom")
KINDS = WAKES + ("ignored", "near_miss", "label")
LABELS = ("real", "false", "")
FIELDS = ["time", "id", "kind", "word", "score", "threshold", "label"]


def parse_entries(wake_word: str, default_threshold: float) -> dict[str, float]:
    """"models/wake/uzi.onnx@0.5,hey_jarvis" -> {"uzi": 0.5, "hey_jarvis": default}"""
    out: dict[str, float] = {}
    for entry in (e.strip() for e in wake_word.split(",") if e.strip()):
        name, _, thr = entry.partition("@")
        stem = name.replace("\\", "/").rsplit("/", 1)[-1]
        stem = stem[:-5] if stem.endswith(".onnx") else stem
        out[stem] = float(thr) if thr else float(default_threshold)
    return out


def _empty() -> dict:
    return {k: 0 for k in WAKES + ("ignored", "near_miss", "woke", "real", "false")}


class WakeControl:
    def __init__(self, wake_word: str, default_threshold: float = 0.5,
                 settings_path: str | Path = "data/wake_settings.json", log_dir: str | Path = "logs/wake",
                 clock: Callable[[], float] = time.time, keep_days: int = 60):
        self.defaults = parse_entries(wake_word, default_threshold)
        self.settings_path, self.log_dir = Path(settings_path), Path(log_dir)
        self.clock, self.keep_days = clock, keep_days
        self._lock = threading.Lock()
        self._enabled = {w: True for w in self.defaults}
        self._threshold = dict(self.defaults)
        self._recent: deque[dict] = deque(maxlen=60)
        self._files: dict[Path, tuple[float, list[dict]]] = {}   # parsed day files, by mtime
        self._load()
        self._prune()

    # ---------------- settings ----------------
    def _load(self) -> None:
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8")).get("words", {})
        except (OSError, ValueError, AttributeError):
            return
        for w, s in saved.items():
            if w not in self.defaults or not isinstance(s, dict):
                continue   # a word no longer in config.toml
            if "threshold" in s:
                try:
                    self._threshold[w] = self._clamp(float(s["threshold"]))
                except (TypeError, ValueError):
                    pass
            if "enabled" in s:
                self._enabled[w] = bool(s["enabled"])
        if not any(self._enabled.values()):   # never leave the assistant deaf by a bad file
            self._enabled = {w: True for w in self.defaults}

    def _save(self) -> None:
        data = {"words": {w: {"enabled": self._enabled[w], "threshold": self._threshold[w]}
                          for w in self.defaults}}
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.settings_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.settings_path)

    @staticmethod
    def _clamp(thr: float) -> float:
        return round(min(HIGHEST, max(LOWEST, thr)), 2)

    def words(self) -> list[str]:
        return list(self.defaults)

    def word_for(self, key: str) -> str | None:
        """A model's score key ("uzi", "hey_jarvis_v0.1") -> its word, or None."""
        if key in self.defaults:
            return key
        return next((w for w in self.defaults if key.startswith(w)), None)

    def enabled(self, word: str) -> bool:
        with self._lock:
            return self._enabled.get(word, True)

    def threshold(self, word: str) -> float:
        with self._lock:
            return self._threshold.get(word, 0.5)

    def update(self, word: str, enabled: bool | None = None, threshold: float | None = None) -> None:
        """Raises ValueError (with a message for the screen) for an unknown word or the last word off."""
        if word not in self.defaults:
            raise ValueError(f"No wake word called {word!r}.")
        with self._lock:
            if enabled is False and self._enabled[word] and sum(self._enabled.values()) == 1:
                raise ValueError("Keep at least one wake word on. Use Mic off to stop listening.")
            changes = []
            if enabled is not None and bool(enabled) != self._enabled[word]:
                self._enabled[word] = bool(enabled)
                changes.append("on" if enabled else "off")
            if threshold is not None:
                thr = self._clamp(float(threshold))
                if thr != self._threshold[word]:
                    self._threshold[word] = thr
                    changes.append(f"threshold {thr:.2f}")
            if changes:
                self._save()
        if changes:
            log.info("wake word %s: %s (from the screen)", word, ", ".join(changes))

    def entries(self) -> list[str]:
        """For the Pi and Settings pages: ["uzi@0.5", "boafo@off"]."""
        with self._lock:
            return [f"{w}@{self._threshold[w]:g}" if self._enabled[w] else f"{w}@off" for w in self.defaults]

    # ---------------- events ----------------
    def _day_file(self, day: date) -> Path:
        return self.log_dir / f"{day.isoformat()}.csv"

    def _write(self, row: dict) -> None:
        now = datetime.fromtimestamp(self.clock())
        path = self._day_file(now.date())
        row = {"time": now.isoformat(timespec="seconds"), **row}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                os.fsync(f.fileno())   # the Pi loses power without warning
        except OSError as e:
            log.warning("couldn't write the wake log: %s", e)

    def record(self, kind: str, key: str, score: float | None = None, heard: str = "") -> str | None:
        """Log an event for the model key that caused it; returns its id (None for an unknown word)."""
        if kind not in KINDS or kind == "label":
            raise ValueError(f"unknown wake event {kind!r}")
        word = self.word_for(key)
        if word is None:
            return None
        eid = f"{self.clock() * 1000:.0f}"
        thr = self.threshold(word)
        self._write({"id": eid, "kind": kind, "word": word,
                     "score": "" if score is None else f"{score:.2f}", "threshold": f"{thr:.2f}"})
        if kind == "replied":
            with self._lock:
                self._recent.appendleft({"id": eid, "time": self.clock(), "word": word,
                                         "score": None if score is None else round(score, 2),
                                         "heard": heard, "label": ""})
        return eid

    def label(self, eid: str, label: str) -> None:
        """Mark a replied wake as real or false ("" clears the mark)."""
        if label not in LABELS:
            raise ValueError("Label must be real, false or empty.")
        with self._lock:
            rec = next((r for r in self._recent if r["id"] == eid), None)
            if rec is None:
                raise ValueError("That wake is no longer in the list.")
            rec["label"] = label
            word = rec["word"]
        self._write({"id": eid, "kind": "label", "word": word, "label": label})

    def recent(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._recent]

    # ---------------- statistics ----------------
    def _rows(self, path: Path) -> list[dict]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        cached = self._files.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                rows = [r for r in csv.DictReader(f) if r.get("kind") in KINDS]
        except OSError:
            rows = []
        self._files[path] = (mtime, rows)
        return rows

    def stats(self, days: int = 1) -> dict[str, dict]:
        """Counts per word over today and the days before it (days=7: the last week)."""
        today = datetime.fromtimestamp(self.clock()).date()
        out = {w: _empty() for w in self.defaults}
        wakes: dict[str, str] = {}      # id -> word, for replied wakes in the window
        labels: dict[str, str] = {}     # id -> latest label
        for i in range(max(1, days) - 1, -1, -1):
            for r in self._rows(self._day_file(today - timedelta(days=i))):
                w, kind = r.get("word"), r.get("kind")
                if w not in out:
                    continue
                if kind == "label":
                    labels[r.get("id", "")] = r.get("label", "")
                    continue
                out[w][kind] += 1
                if kind in WAKES:
                    out[w]["woke"] += 1
                if kind == "replied":
                    wakes[r.get("id", "")] = w
        for eid, lab in labels.items():
            if eid in wakes and lab in ("real", "false"):
                out[wakes[eid]][lab] += 1
        return out

    def snapshot(self, days: int = 1) -> dict:
        """Everything the Wake words pages show (GET /wake)."""
        stats = self.stats(days)
        with self._lock:
            words = [{"word": w, "enabled": self._enabled[w], "threshold": self._threshold[w],
                      "default": self.defaults[w], "stats": stats[w]} for w in self.defaults]
        return {"days": max(1, days), "words": words, "recent": self.recent(),
                "step": STEP, "lowest": LOWEST, "highest": HIGHEST}

    def _prune(self) -> None:
        cutoff = datetime.fromtimestamp(self.clock()).date() - timedelta(days=self.keep_days)
        for p in self.log_dir.glob("20*.csv"):
            try:
                if date.fromisoformat(p.stem) < cutoff:
                    p.unlink()
            except (ValueError, OSError):
                pass
