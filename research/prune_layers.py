"""EXP-001 Q4: structured pruning by removing whole transformer layers.

Method (Gromov et al. 2024, "The Unreasonable Ineffectiveness of the Deeper Layers"):
for each block of n consecutive layers, measure the angular distance between the hidden
state entering the block and the one leaving it, averaged over calibration text. The block
that changes the representation least is removed.

Runs on a PC/Colab with PyTorch (CPU works for <=1B models, just slowly):
  pip install torch transformers accelerate
  python research/prune_layers.py --model Qwen/Qwen3-0.6B --drop-n 4 --out research/models/qwen3-0.6b-pruned4
Then convert for the Pi:
  python ~/src/llama.cpp/convert_hf_to_gguf.py research/models/qwen3-0.6b-pruned4 --outtype f16
  llama-quantize <f16.gguf> <q4_k_m.gguf> Q4_K_M
Pruned models usually need "healing" with a short LoRA run (research/train_lora.py).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CALIBRATION = [
    "Turn on the living room light and tell me the temperature in the bedroom.",
    "The Raspberry Pi 5 has a quad-core Arm Cortex-A76 processor and up to 8 GB of memory.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Remember that my sister's birthday is in March, and remind me a week before.",
    "Accra is the capital of Ghana, located on the Gulf of Guinea.",
    "To compute the area of a circle, multiply pi by the square of the radius.",
    "Yesterday I walked to the market, bought tomatoes and onions, and cooked stew.",
    "MQTT is a lightweight publish and subscribe protocol used by many IoT devices.",
]


def angular_distance(a, b):
    import torch
    cos = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).clamp(-1, 1)
    return (torch.arccos(cos) / math.pi).mean().item()


def block_scores(model, tok, n: int, texts: list[str]) -> list[float]:
    import torch
    layers = model.config.num_hidden_layers
    sums = [0.0] * (layers - n + 1)
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").to(model.device)
            hs = model(**ids, output_hidden_states=True).hidden_states  # layers+1 tensors
            for start in range(layers - n + 1):
                # last-token representation, as in the paper
                sums[start] += angular_distance(hs[start][:, -1], hs[start + n][:, -1])
    return [s / len(texts) for s in sums]


def drop_layers(model, drop: list[int]):
    import torch
    keep = [i for i in range(model.config.num_hidden_layers) if i not in set(drop)]
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    for new_idx, layer in enumerate(model.model.layers):  # keep KV-cache indexing consistent
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
    model.config.num_hidden_layers = len(keep)
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = [model.config.layer_types[i] for i in keep]
    return model


def main(argv=None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face id or local path")
    p.add_argument("--drop-n", type=int, help="remove the most redundant block of n layers")
    p.add_argument("--drop", type=int, nargs="*", help="explicit layer indices to remove instead")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if not a.drop and not a.drop_n:
        p.error("give --drop-n or --drop")

    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if dev == "cuda" else torch.float32)
    model.to(dev).eval()
    L = model.config.num_hidden_layers
    params_before = sum(x.numel() for x in model.parameters())

    report = {"model": a.model, "layers_before": L, "params_before": params_before}
    if a.drop:
        drop = sorted(a.drop)
    else:
        scores = block_scores(model, tok, a.drop_n, CALIBRATION)
        best = min(range(len(scores)), key=scores.__getitem__)
        drop = list(range(best, best + a.drop_n))
        report["block_scores"] = [round(s, 4) for s in scores]
        print("angular distance per block start:", report["block_scores"])
    print(f"removing layers {drop} of {L}")
    drop_layers(model, drop)
    report.update(dropped=drop, layers_after=model.config.num_hidden_layers,
                  params_after=sum(x.numel() for x in model.parameters()))

    a.out.mkdir(parents=True, exist_ok=True)
    model.to(torch.float16).save_pretrained(a.out)
    tok.save_pretrained(a.out)
    (a.out / "prune_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "block_scores"}, indent=2))


if __name__ == "__main__":
    main()
