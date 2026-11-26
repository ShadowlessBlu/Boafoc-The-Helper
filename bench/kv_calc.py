"""Predict LLM memory use on the Pi before measuring it (EXP-001 Q1, Q2, Q6).

weights  = params x bits_per_weight / 8
KV cache = 2 (K and V) x layers x kv_heads x head_dim x n_ctx x bytes_per_value

Usage:
  python bench/kv_calc.py --model all
  python bench/kv_calc.py --model llama3.2-3b --ctx 2048 8192 32768 --kv q8_0
  python bench/kv_calc.py --config path/to/config.json --params 1.2e9
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

# Architecture values from each model's Hugging Face config.json.
# Double-check against the real config.json before quoting them in results.
@dataclass(frozen=True)
class Arch:
    name: str
    params: float      # total parameters
    layers: int
    kv_heads: int
    head_dim: int
    note: str = ""

PRESETS: dict[str, Arch] = {
    "qwen3-0.6b":  Arch("Qwen3-0.6B",   0.6e9, 28, 8, 128),
    "gemma3-1b":   Arch("Gemma-3-1B",   1.0e9, 26, 1, 256,
                        "most layers use a 512-token sliding window, so real KV is smaller than predicted"),
    "llama3.2-1b": Arch("Llama-3.2-1B", 1.24e9, 16, 8, 64),
    "qwen3-1.7b":  Arch("Qwen3-1.7B",   1.7e9, 28, 8, 128),
    "llama3.2-3b": Arch("Llama-3.2-3B", 3.21e9, 28, 8, 128),
}

# Approximate average bits per weight for llama.cpp quant types.
# K-quants mix precisions, so these are averages (to check against real GGUF file sizes).
BITS_PER_WEIGHT = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0,
    "q8_0": 8.5, "q6_k": 6.56, "q5_k_m": 5.69, "q5_0": 5.5,
    "q4_k_m": 4.89, "q4_k_s": 4.58, "q4_0": 4.5, "q3_k_m": 3.91, "q3_k_s": 3.5, "q2_k": 2.96,
}

KV_BYTES = {"f32": 4.0, "f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}


def weights_bytes(params: float, quant: str) -> float:
    return params * BITS_PER_WEIGHT[quant.lower()] / 8


def kv_cache_bytes(arch: Arch, n_ctx: int, kv_type: str = "f16") -> float:
    return 2 * arch.layers * arch.kv_heads * arch.head_dim * n_ctx * KV_BYTES[kv_type.lower()]


def arch_from_config(path: str, params: float) -> Arch:
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    c = c.get("text_config", c)
    heads = c["num_attention_heads"]
    head_dim = c.get("head_dim") or c["hidden_size"] // heads
    return Arch(c.get("_name_or_path", path), params, c["num_hidden_layers"],
                c.get("num_key_value_heads", heads), head_dim)


GB = 1024 ** 3
MB = 1024 ** 2


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", help=f"one of {', '.join(PRESETS)} or 'all'")
    p.add_argument("--config", help="HF config.json instead of a preset (needs --params)")
    p.add_argument("--params", type=float, help="parameter count, e.g. 1.2e9")
    p.add_argument("--quant", nargs="+", default=["f16", "q8_0", "q4_k_m", "q3_k_s"])
    p.add_argument("--ctx", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    p.add_argument("--kv", default="f16", choices=list(KV_BYTES))
    p.add_argument("--overhead-mb", type=float, default=300,
                   help="runtime overhead guess (compute buffers etc.) to add to totals")
    a = p.parse_args(argv)

    if a.config:
        if not a.params:
            p.error("--config needs --params")
        archs = [arch_from_config(a.config, a.params)]
    elif a.model == "all":
        archs = list(PRESETS.values())
    else:
        archs = [PRESETS[a.model]]

    for arch in archs:
        print(f"\n### {arch.name}  ({arch.params/1e9:.2f}B params, {arch.layers} layers, "
              f"{arch.kv_heads} KV heads x {arch.head_dim})")
        if arch.note:
            print(f"Note: {arch.note}")
        kv_per_tok = kv_cache_bytes(arch, 1, a.kv)
        print(f"KV cache per token ({a.kv}): {kv_per_tok/1024:.1f} KiB")
        header = "| Quant | Weights (GB) | " + " | ".join(f"Total @ {c} ctx (GB)" for c in a.ctx) + " |"
        print(header)
        print("|" + "---|" * (2 + len(a.ctx)))
        for q in a.quant:
            w = weights_bytes(arch.params, q)
            totals = [(w + kv_cache_bytes(arch, c, a.kv) + a.overhead_mb * MB) / GB for c in a.ctx]
            print(f"| {q} | {w/GB:.2f} | " + " | ".join(f"{t:.2f}" for t in totals) + " |")
        print("| KV only | - | " + " | ".join(f"{kv_cache_bytes(arch, c, a.kv)/MB:.0f} MB" for c in a.ctx) + " |")


if __name__ == "__main__":
    main()
