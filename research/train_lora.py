"""EXP-001 Q5 (distillation) and Q4 healing: LoRA fine-tune a small model on chat JSONL.

Runs on a GPU, or (slowly) on a CPU: Q4 healing ran on this PC's CPU in ~12 min for 108 examples.
Use --answer-only for distillation: the loss is then only on the teacher's reply, not on the system
prompt and question the student is given anyway.
  pip install torch transformers peft datasets accelerate
  python research/train_lora.py --model Qwen/Qwen3-0.6B --data research/data/distill_v1.jsonl \
      --out research/models/qwen3-0.6b-distill --rank 8 --epochs 2
The merged model in <out>/merged converts to GGUF like any HF model
(see research/README.md), then runs on the Pi with Ollama or llama.cpp.
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path


def main(argv=None) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation: effective batch = batch x accum (batch 8 ran out of memory on a 16 GB PC: the 152k-word output layer alone needed 1.35 GB)")
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--resume", action="store_true",
                   help="continue from the newest checkpoint in <out>/checkpoints (saved every epoch), if any")
    p.add_argument("--answer-only", action="store_true",
                   help="loss only on the last (assistant) message, with thinking off as on the Pi")
    a = p.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16
                                                 if torch.cuda.is_available() else torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(a.data))["train"].train_test_split(test_size=a.val_frac, seed=0)

    def encode(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, truncation=True, max_length=a.max_len)

    def template(messages, **kw):
        try:   # Qwen3: thinking off, as the assistant runs it (an empty think block, then the reply)
            return tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, **kw)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, **kw)

    def encode_answer_only(ex):
        prompt = template(ex["messages"][:-1], add_generation_prompt=True)
        full = template(ex["messages"])
        if not full.startswith(prompt):
            raise ValueError("chat template: the prompt is not a prefix of the full example")
        n = len(tok(prompt)["input_ids"])
        enc = tok(full, truncation=True, max_length=a.max_len)
        enc["labels"] = [-100] * min(n, len(enc["input_ids"])) + enc["input_ids"][n:]
        return enc

    ds = ds.map(encode_answer_only if a.answer_only else encode, remove_columns=["messages"])
    collator = (DataCollatorForSeq2Seq(tok, label_pad_token_id=-100) if a.answer_only
                else DataCollatorForLanguageModeling(tok, mlm=False))
    import inspect
    wanted = dict(
        output_dir=str(a.out / "checkpoints"), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=10, report_to=[],
        bf16=torch.cuda.is_available(), lr_scheduler_type="cosine", warmup_ratio=0.05)
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    args = TrainingArguments(**{k: v for k, v in wanted.items() if k in accepted})
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                      data_collator=collator)
    if sys.platform == "win32":   # the 2026-11-23 run died when the PC idle-slept; ask Windows to stay
        import ctypes             # awake while this process runs (ends with it; a closed lid still sleeps)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
    ckpts = sorted((a.out / "checkpoints").glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[1]))
    if a.resume and ckpts:
        print(f"resuming from {ckpts[-1]}", flush=True)
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if a.resume and ckpts else None)
    metrics = trainer.evaluate()

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out / "merged")
    tok.save_pretrained(a.out / "merged")
    (a.out / "train_report.json").write_text(json.dumps({**vars(a), "data": str(a.data), "out": str(a.out),
                                                          "eval": metrics}, indent=2, default=str))
    print("eval:", metrics)


if __name__ == "__main__":
    main()
