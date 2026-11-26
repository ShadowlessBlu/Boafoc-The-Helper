# EXP-001 Phase C: pruning and distillation

Training runs on the PC (its CPU is enough for a LoRA on a 0.6B model: about 1.2 h of steps for 561 examples × 2 epochs); the Pi makes the distillation data and runs and scores the results.

## Q5: distillation (teacher → student)
Done on 2026-11-23/24: the student scored 48/60 against 46 for the base through the same pipeline, and 16/16 on tool calls against 13. Full results in the vault's Experiment Log (EXP-001, Q5). How it was run:

1. **Data, on the Pi** (survives power cuts: resumes from a `@reboot` entry, fsyncs each row):
   ```bash
   bash research/run_distill_data.sh            # qwen3:1.7b, thinking off, 600 requests
   bash research/run_distill_data.sh --status
   ```
   An answer is kept only if it makes the right tool call for its template and passes the Governor; both test sets are excluded.
2. **Everything else, from the PC** (Git Bash, project folder; each step is skipped once done):
   ```bash
   bash research/finish_distill.sh
   ```
   It runs these steps in order:
   - **fetch** the data;
   - **train** a LoRA with the loss on the teacher's reply only (`--answer-only --batch 2 --accum 4 --resume`; batch 8 ran out of memory, and it resumes from the epoch checkpoint);
   - **convert** the student and the base model, saved the same way, to f16 GGUF, with llama.cpp's converter in `research/tools/llama_convert`;
   - **copy** them to the Pi over the home network when it can, size-checked;
   - **create** the Q4_K_M Ollama models `q06-pipe-base` and `q06-distill`;
   - **score** qwen3:0.6b, the control and the student on test set v2, in the background.
3. **Results:**
   ```bash
   ssh pi5 'cd ~/assistant && bash research/q5_pi.sh result'
   ```

## Q4: layer pruning
```bash
python research/prune_layers.py --model Qwen/Qwen3-0.6B --drop-n 4 --out research/models/qwen3-0.6b-pruned4
```
- `prune_report.json` lists the angular distance for each block, which layers were removed, and the parameters before and after.
- Optionally heal the pruned model with `train_lora.py` on the distillation data.
- Convert and quantize as above, then on the Pi run `bench/bench_llamacpp.sh` (speed) and `experiments/score.py` (quality).
- Hypothesis: speed rises roughly in proportion to the layers removed, and quality falls. Healing recovers part of that loss.

## Fair comparison
Use the same quant (Q4_K_M), context, threads, prompts and cooling for base, pruned and distilled models. Record everything in the Experiment Log.
