"""SQLite store for conversation history, long-term memories and the audit log.

Search uses embeddings (cosine similarity) when an embedder is given, otherwise
simple keyword overlap so it still works without an embedding model.
"""
from __future__ import annotations

import json
import math
from array import array
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable

Embedder = Callable[[list[str]], list[list[float]]]

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, text TEXT NOT NULL, embedding TEXT);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, tool TEXT, args TEXT, verdict TEXT, reason TEXT, outcome TEXT);
"""

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "my", "me", "i", "you", "your", "of", "to", "and", "what",
         "on", "in", "at", "for", "it", "do", "does", "that", "this", "was", "be", "with"}


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def encode_vec(v) -> bytes:
    """Store an embedding as a float32 blob: 3 KB for 768 dims, against ~16 KB as JSON text."""
    return array("f", v).tobytes()


def decode_vec(stored, np=None):
    """Read a stored embedding: float32 blob, or JSON text (databases written before 2026-11-18).
    Returns a numpy float32 array if `np` (the numpy module) is given, else a list."""
    if isinstance(stored, (bytes, memoryview)):
        if np is not None:
            return np.frombuffer(stored, dtype=np.float32)
        a = array("f")
        a.frombytes(bytes(stored))
        return a.tolist()
    v = json.loads(stored)
    return np.asarray(v, dtype=np.float32) if np is not None else v


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Memory:
    def __init__(self, db_path: str | Path = ":memory:", embedder: Embedder | None = None):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # The API and voice loop share one connection; callers serialise access with a lock.
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        # Deleted rows are overwritten with zeros instead of lingering in free pages, so "forgotten"
        # really means gone from the file (memory rules, 2026-11-25).
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript(SCHEMA)
        self.embedder = embedder
        # Parsed embeddings, so each search doesn't re-read JSON for every memory
        # (the memory_search benchmark showed that dominated search time).
        self._vec_cache: dict[int, list[float]] = {}
        # numpy matrix of all vectors and their norms, rebuilt only when the set of ids changes.
        # On the Pi, rebuilding it from lists cost ~2 s at 5,000 memories; the search itself ~1.5 ms.
        self._mat_ids: tuple[int, ...] = ()
        self._mat = None
        self._mat_norms = None

    # --- conversation ---
    def add_message(self, role: str, content: str) -> None:
        self.db.execute("INSERT INTO messages (ts, role, content) VALUES (?,?,?)", (time.time(), role, content))
        self.db.commit()

    def recent_messages(self, n: int = 8) -> list[dict]:
        rows = self.db.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def clear_history(self) -> int:
        """Delete every question and reply (the screen's Clear history). Saved facts and the audit log stay."""
        cur = self.db.execute("DELETE FROM messages")
        self.db.commit()
        return cur.rowcount

    # --- retention (memory rules, 2026-11-25: history 30 days, audit 90 days, facts until forgotten) ---
    def prune(self, history_days: float, audit_days: float, now: float | None = None) -> tuple[int, int]:
        """Delete messages and audit rows older than these many days (0 = keep). Returns (messages, audit)."""
        now = time.time() if now is None else now
        counts = []
        for table, days in (("messages", history_days), ("audit", audit_days)):
            if days and days > 0:
                counts.append(self.db.execute(f"DELETE FROM {table} WHERE ts < ?", (now - days * 86400,)).rowcount)
            else:
                counts.append(0)
        self.db.commit()
        return counts[0], counts[1]

    # --- long-term memory ---
    def remember(self, text: str) -> int:
        emb = encode_vec(self.embedder([text])[0]) if self.embedder else None
        cur = self.db.execute("INSERT INTO memories (ts, text, embedding) VALUES (?,?,?)", (time.time(), text, emb))
        self.db.commit()
        return cur.lastrowid

    def duplicate_of(self, text: str, same: float = 0.95) -> int | None:
        """The id of a note that already says this (same words, or embeddings at least `same` alike)."""
        norm = " ".join(_WORD.findall(text.lower()))
        for mid, t in self.list_memories():
            if " ".join(_WORD.findall(t.lower())) == norm:
                return mid
        if self.embedder:
            hits = self.search(text, 1, same)
            if hits and hits[0][0] >= same:
                return hits[0][1]
        return None

    def remember_new(self, text: str) -> str:
        """The `remember` tool. Asked "Where is the spare key?", the distilled student answered and
        also called remember with the note it had just been shown (offline test, 2026-11-25), so
        every recall question saved a copy. A note that is already there is not saved again."""
        dup = self.duplicate_of(text)
        if dup is not None:
            return f"already saved as memory #{dup}"
        return f"saved as memory #{self.remember(text)}"

    def forget(self, memory_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        self._vec_cache.pop(memory_id, None)
        return cur.rowcount > 0

    def _vectors(self, ids, np=None) -> list:
        """Decoded vectors for these ids; only ids not yet cached are read from SQLite.
        With `np` they are numpy arrays (the choice is fixed per process: numpy is there or not)."""
        missing = [i for i in ids if i not in self._vec_cache]
        for k in range(0, len(missing), 500):  # stay under SQLite's variable limit
            chunk = missing[k:k + 500]
            q = f"SELECT id, embedding FROM memories WHERE id IN ({','.join('?' * len(chunk))})"
            for mid, emb in self.db.execute(q, chunk):
                self._vec_cache[mid] = decode_vec(emb, np)
        return [self._vec_cache[i] for i in ids]

    def best_match(self, text: str, min_score: float = 0.45) -> tuple[int, str] | None:
        """The saved fact that best matches a description ("my sister's birthday"), or None."""
        hits = self.search(text, 1, min_score)
        return (hits[0][1], hits[0][2]) if hits else None

    def forget_matching(self, text: str, min_score: float = 0.45) -> str:
        """The voice "forget" tool, after the user said yes to the fact it named."""
        hit = self.best_match(text, min_score)
        if hit is None:
            return "no saved note matches that"
        self.forget(hit[0])
        return f"forgot note #{hit[0]}: {hit[1]}"

    def list_memories(self) -> list[tuple[int, str]]:
        return self.db.execute("SELECT id, text FROM memories ORDER BY id").fetchall()

    def search(self, query: str, top_k: int = 3, min_score: float = 0.45) -> list[tuple[float, int, str]]:
        # Don't read the embeddings here: at 5,000 memories that is ~80 MB of JSON per search.
        rows = self.db.execute("SELECT id, text, embedding IS NOT NULL FROM memories ORDER BY id").fetchall()
        if not rows:
            return []
        scored = []
        if self.embedder and all(r[2] for r in rows):
            q = self.embedder([query])[0]
            ids = tuple(r[0] for r in rows)
            try:
                import numpy as np
                if self._mat is None or ids != self._mat_ids:
                    self._mat = np.vstack(self._vectors(ids, np))
                    self._mat_norms = np.linalg.norm(self._mat, axis=1)
                    self._mat_ids = ids
                qa = np.asarray(q, dtype=np.float32)
                sims = (self._mat @ qa) / (self._mat_norms * np.linalg.norm(qa) + 1e-12)
                k = min(top_k, len(rows))
                best = np.argpartition(-sims, k - 1)[:k]
                scored = [(float(sims[i]), rows[i][0], rows[i][1]) for i in best]
            except ImportError:
                vecs = self._vectors(ids)
                scored = [(cosine(q, v), mid, text) for v, (mid, text, _) in zip(vecs, rows)]
        else:
            qw = _words(query)
            for mid, text, _ in rows:
                tw = _words(text)
                score = len(qw & tw) / len(qw | tw) if qw and tw else 0.0
                scored.append((score, mid, text))
            min_score = min(min_score, 0.1)  # keyword overlap scores run lower
        scored.sort(reverse=True)
        return [s for s in scored[:top_k] if s[0] >= min_score]

    # --- audit ---
    def log_audit(self, tool: str, args: dict, verdict: str, reason: str, outcome: str = "") -> None:
        self.db.execute("INSERT INTO audit (ts, tool, args, verdict, reason, outcome) VALUES (?,?,?,?,?,?)",
                        (time.time(), tool, json.dumps(args), verdict, reason, outcome))
        self.db.commit()

    def audit_log(self, n: int = 20) -> list[tuple]:
        return self.db.execute("SELECT ts, tool, args, verdict, reason, outcome FROM audit "
                               "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
