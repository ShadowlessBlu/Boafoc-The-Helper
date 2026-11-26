"""Databases / Memory System experiment: how does memory search time grow with stored memories?

Stores N random unit vectors (embedding size like nomic-embed-text = 768) in the real
SQLite memory table and times Memory.search() (pure Python cosine). If numpy is
installed, it also times a vectorised numpy search over the same data for comparison.
No LLM needed; the embedder is a fixed random generator.

Usage: python bench/memory_search.py --sizes 100 1000 5000 --dim 768 --queries 5
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from assistant.memory import Memory, decode_vec, encode_vec  # noqa: E402

# v1 timed the first search in the mean; v2 stored JSON text; v3 stores float32 blobs (--json for the old way)
OUT = ROOT / "experiments" / "results" / "memory_search_v3.csv"


def rand_vec(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def run(size: int, dim: int, queries: int, seed: int = 0, as_json: bool = False) -> dict:
    rng = random.Random(seed)
    mem = Memory()
    enc = json.dumps if as_json else encode_vec
    rows = [(time.time(), f"memory {i}", enc(rand_vec(rng, dim))) for i in range(size)]
    mem.db.executemany("INSERT INTO memories (ts, text, embedding) VALUES (?,?,?)", rows)
    mem.db.commit()
    qvecs = [rand_vec(rng, dim) for _ in range(queries)]
    it = iter(qvecs)
    mem.embedder = lambda texts: [next(it)]

    # memory.search() as the assistant uses it. The first call parses every stored vector
    # (and builds the numpy matrix if numpy is installed); later calls reuse that cache.
    t0 = time.perf_counter()
    mem.search("q", top_k=3, min_score=-1)
    first_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for _ in range(queries - 1):
        mem.search("q", top_k=3, min_score=-1)
    py_ms = (time.perf_counter() - t0) / max(queries - 1, 1) * 1000
    try:
        import numpy  # noqa: F401
        path = "numpy"
    except ImportError:
        path = "python"

    np_ms = ""
    try:
        import numpy as np
        t0 = time.perf_counter()
        mat = np.vstack([decode_vec(e, np) for (e,) in mem.db.execute("SELECT embedding FROM memories")])
        load_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        for q in qvecs:
            scores = mat @ np.asarray(q, dtype=np.float32)
            np.argpartition(-scores, 3)[:3]
        np_ms = round((time.perf_counter() - t0) / queries * 1000, 2)
        np_load = round(load_ms)
    except ImportError:
        np_load = ""
    db_mb = sum(len(r[2]) for r in rows) / 1024 ** 2
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
            "machine": platform.machine(), "memories": size, "dim": dim, "search_path": path,
            "storage": "json" if as_json else "float32",
            "first_search_ms": round(first_ms, 1), "search_ms": round(py_ms, 2),
            "raw_numpy_ms": np_ms, "numpy_load_ms": np_load,
            "stored_mb": round(db_mb, 1)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--json", action="store_true", help="store vectors as JSON text (the pre-v3 format)")
    a = p.parse_args(argv)
    results = []
    print(f"Storage: {'JSON text' if a.json else 'float32 blobs'}")
    print("| Memories | Path | First search (ms) | Later searches (ms) | Raw numpy matmul (ms) | Stored embeddings (MB) |")
    print("|---|---|---|---|---|---|")
    for n in a.sizes:
        r = run(n, a.dim, a.queries, as_json=a.json)
        results.append(r)
        print(f"| {n} | {r['search_path']} | {r['first_search_ms']} | {r['search_ms']} | {r['raw_numpy_ms'] or '-'} "
              f"| {r['stored_mb']} |",
              flush=True)
    if not a.no_save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        new = not OUT.exists()
        with OUT.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            if new:
                w.writeheader()
            w.writerows(results)
        print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
