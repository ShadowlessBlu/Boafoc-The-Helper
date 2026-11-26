"""Turn the EXP-001 result CSVs into Markdown tables for the vault.

Reads experiments/results/{bench_ollama,score_v1,redteam_v1,memory_search}.csv (whichever exist)
and prints Markdown. Medians are used so one slow run doesn't skew a row.

Usage: python bench/summarize.py [--host NAME] > summary.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "experiments" / "results"


def rows(name: str) -> list[dict]:
    p = R / name
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def med(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=1) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def latest_per_model(rs: list[dict]) -> dict[str, list[dict]]:
    """Keep only each model's most recent full run (rows are appended per run)."""
    by = defaultdict(list)
    for r in rs:
        by[r["model"]].append(r)
    out = {}
    for m, lst in by.items():
        n = len({r["id"] for r in lst})
        out[m] = lst[-n:]
    return out


def speed_table(host: str | None) -> tuple[str, dict]:
    rs = [r for r in rows("bench_ollama.csv") if not host or r["host"] == host]
    if not rs:
        return "", {}
    by = defaultdict(list)
    for r in rs:
        by[(r["model"], r["prompt_id"])].append(r)
    models = list(dict.fromkeys(r["model"] for r in rs))
    prompts = list(dict.fromkeys(r["prompt_id"] for r in rs))
    lines = ["| Model | Ollama RAM (MB) | " + " | ".join(f"{p} decode t/s" for p in prompts)
             + " | Prefill t/s (long_context, run 1) | First token, est. (s) | Max °C | Throttled |",
             "|---|---|" + "---|" * len(prompts) + "---|---|---|---|"]
    stats = {}
    for m in models:
        cells, decs = [], []
        for p in prompts:
            d = med(num(r["decode_tps"]) for r in by[(m, p)])
            decs.append(d)
            cells.append(fmt(d))
        allr = [r for (mm, _), lst in by.items() if mm == m for r in lst]
        ram = med(num(r["ollama_size_mb"]) for r in allr)
        lc = by.get((m, "long_context")) or by[(m, prompts[-1])]
        # Runs 2+ reuse Ollama's prompt cache, so their prefill t/s is not real: use run 1 only.
        lc = [r for r in lc if r["run"] == "1"] or lc
        pre = med(num(r["prefill_tps"]) for r in lc)
        ptok = med(num(r["prompt_tokens"]) for r in lc)
        ttft = ptok / pre if pre and ptok else None
        tmax = max((num(r["max_temp_c"]) or 0) for r in allr)
        thr = sorted({r["throttled"] for r in allr if r["throttled"] not in ("", "0x0")}) or ["no"]
        stats[m] = {"decode": med(decs), "ram_mb": ram}
        lines.append(f"| {m} | {fmt(ram, 0)} | " + " | ".join(cells)
                     + f" | {fmt(pre)} | {fmt(ttft, 2)} | {fmt(tmax)} | {', '.join(thr)} |")
    return "\n".join(lines), stats


def score_table(name: str = "score_v1.csv") -> tuple[str, dict]:
    per = latest_per_model(rows(name))
    if not per:
        return "", {}
    cats = list(dict.fromkeys(r["category"] for lst in per.values() for r in lst))
    n_q = max(len(lst) for lst in per.values())
    lines = [f"| Model | Score /{n_q} | " + " | ".join(cats) + " | Median s/answer |",
             "|---|---|" + "---|" * len(cats) + "---|"]
    scores = {}
    for m, lst in per.items():
        s = sum(int(r["pass"]) for r in lst)
        scores[m] = (s, med(num(r["seconds"]) for r in lst), len(lst))
        cc = []
        for c in cats:
            cl = [r for r in lst if r["category"] == c]
            cc.append(f"{sum(int(r['pass']) for r in cl)}/{len(cl)}")
        sec = med(num(r["seconds"]) for r in lst)
        lines.append(f"| {m} | **{s}** | " + " | ".join(cc) + f" | {fmt(sec)} |")
    return "\n".join(lines), scores


def redteam_table() -> str:
    per = latest_per_model(rows("redteam_v1.csv"))
    if not per:
        return ""
    lines = ["| Model | Model resisted | Governor caught | Breach |", "|---|---|---|---|"]
    for m, lst in per.items():
        c = defaultdict(int)
        for r in lst:
            c[r["outcome"]] += 1
        lines.append(f"| {m} | {c['model resisted']} | {c['governor caught']} | **{c['BREACH']}** |")
    return "\n".join(lines)


def memory_table(host: str | None) -> str:
    rs = [r for r in rows("memory_search_v3.csv") or rows("memory_search_v2.csv") if not host or r["host"] == host]
    if rs:
        lines = ["| Memories | Path | First search (ms) | Later searches (ms) | Raw numpy (ms) |",
                 "|---|---|---|---|---|"]
        for r in rs[-3:]:
            lines.append(f"| {r['memories']} | {r['search_path']} | {r['first_search_ms']} | {r['search_ms']} "
                         f"| {r['raw_numpy_ms'] or '–'} |")
        return "\n".join(lines)
    rs = [r for r in rows("memory_search.csv") if not host or r["host"] == host]
    if not rs:
        return ""
    lines = ["| Memories | Python search (ms) | numpy search (ms) | numpy load (ms) |", "|---|---|---|---|"]
    for r in rs[-3:]:
        lines.append(f"| {r['memories']} | {r['python_search_ms']} | {r['numpy_search_ms'] or '–'} "
                     f"| {r.get('numpy_load_ms') or '–'} |")
    return "\n".join(lines)


def tradeoff_table(speed: dict, scores: dict) -> str:
    """Score rows may be "model (think off)": they share the base model's speed and RAM."""
    ms = [m for m in scores if m.split(" (")[0] in speed]
    if not ms:
        return ""
    lines = ["| Model | Score | Median s/answer | Decode t/s | RAM (GB) | (score × t/s) ÷ RAM | Meets ≥5 t/s |",
             "|---|---|---|---|---|---|---|"]
    ranked = []
    for m in ms:
        sp = speed[m.split(" (")[0]]
        d, ram = sp["decode"], (sp["ram_mb"] or 0) / 1024
        sc, secs, n_q = scores[m]
        val = (sc / n_q * d / ram) if d and ram else None
        ranked.append((val or 0, m, d, ram, sc, secs))
    for val, m, d, ram, sc, secs in sorted(ranked, reverse=True):
        lines.append(f"| {m} | {sc}/{n_q} | {fmt(secs)} | {fmt(d)} | {fmt(ram, 2)} | {fmt(val, 2)} | "
                     f"{'yes' if d and d >= 5 else 'no'} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", help="only rows from this host (bench + memory CSVs)")
    a = p.parse_args(argv)
    speed_md, speed = speed_table(a.host)
    score_md, scores = score_table()
    score2_md, scores2 = score_table("score_v2.csv")
    for title, md in (("Speed and RAM (medians of 3 runs)", speed_md), ("Quality (test_set_v1)", score_md),
                      ("Quality (test_set_v2, 60 questions)", score2_md),
                      ("Governor red-team", redteam_table()), ("Memory search", memory_table(a.host)),
                      ("Q7 trade-off (v1 scores)", tradeoff_table(speed, scores)),
                      ("Q7 trade-off (v2 scores)", tradeoff_table(speed, scores2))):
        if md:
            print(f"### {title}\n{md}\n")


if __name__ == "__main__":
    main()
