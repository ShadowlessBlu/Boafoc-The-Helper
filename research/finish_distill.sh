#!/bin/bash
# EXP-001 Q5 after the Pi's data run (research/run_distill_data.sh) has finished. Run on the PC, in
# Git Bash, from the project folder:   bash research/finish_distill.sh
# Steps (each is skipped once done, so it can be re-run after a failure):
#   fetch    copy research/data/distill_v1.jsonl from the Pi
#   train    LoRA on Qwen3-0.6B, answer-only loss (PC CPU, ~1-3 h for 561 rows x 2 epochs; resumes
#            from the last epoch checkpoint; keep the PC from sleeping, sleep killed the 2026-11-23 run)
#   convert  merged student and the base model (control) to f16 GGUF, with llama.cpp's converter
#            copied from the Pi (research/tools/llama_convert; shim/ stands in for sentencepiece)
#   copy     both GGUFs to the Pi's ~/models (size-checked)
#   create   on the Pi: Q4_K_M + Ollama models q06-pipe-base and q06-distill (research/q5_pi.sh)
#   score    start the scoring on the Pi (background, resumes after a power cut)
# Pi steps wait for the Pi while it reboots. Results: ssh pi5 'cd ~/assistant && bash research/q5_pi.sh result'
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-research/Scripts/python
OUT=research/models/qwen3-0.6b-distill
G=research/models/gguf
DONE=research/models/.q5_done
mkdir -p "$DONE" "$G" research/data
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ | head -1)

pi() {  # run on the Pi; wait up to ~10 min for it if it is rebooting after a power cut
  local i
  for i in $(seq 1 30); do ssh -n -o ConnectTimeout=10 pi5 true 2>/dev/null && break; sleep 20; done
  ssh pi5 "$@"
}
fetch() {
  pi "grep -aq '=== .* finished' ~/assistant/research/data/distill_v1.log" \
    || { echo "the Pi's data run hasn't finished yet"; return 1; }
  scp -q pi5:assistant/research/data/distill_v1.jsonl research/data/ && wc -l research/data/distill_v1.jsonl
}
train() {
  HF_HUB_OFFLINE=1 "$PY" research/train_lora.py --model Qwen/Qwen3-0.6B \
    --data research/data/distill_v1.jsonl --out "$OUT" --answer-only --batch 2 --accum 4 --resume
}
convert_one() {  # <hf folder> <name>; succeeds only if the converter says so
  local log="$G/convert_$2.log"
  (cd research/tools/llama_convert && PYTHONPATH="gguf-py;.;shim" HF_HUB_OFFLINE=1 "../../../$PY" \
     convert_hf_to_gguf.py "$1" --outtype f16 --outfile "../../models/gguf/qwen3-0.6b-$2-f16.gguf") > "$log" 2>&1
  grep -E "successfully|Error" "$log" | tail -2
  grep -q "successfully exported" "$log"
}
convert() {
  # The control goes through the same save as the student: the original files carry a separate
  # lm_head (311 tensors, output.weight Q6_K after quantizing), while save_pretrained of the tied
  # model drops it (310 tensors, token_embd then Q6_K). Without this the two differ in more than training.
  local base=research/models/qwen3-0.6b-base-hf
  [ -f "$base/config.json" ] || HF_HUB_OFFLINE=1 "$PY" -c "
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch; m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32); m.save_pretrained(sys.argv[1])
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B').save_pretrained(sys.argv[1])" "$base" || return 1
  [ -f "$G/qwen3-0.6b-base-f16.gguf" ] || convert_one "$(cd "$base" && pwd)" base || return 1
  [ -f "$G/qwen3-0.6b-distill-f16.gguf" ] || convert_one "$(cd "$OUT/merged" && pwd)" distill
}
copy() {
  # capped at ~5 MB/s: a full-speed 1.5 GB copy on top of other load preceded a power cut (2026-11-23).
  # Copied to .part and renamed, and checked by size; only a missing or wrong-size file is copied again.
  local f size there i ip host=pi5 opts=()
  pi "mkdir -p ~/models" || return 1
  # Over Tailscale the copy went through a relay at ~0.28 MB/s; on the same network it's direct.
  # HostKeyAlias checks the LAN address against the Pi's known Tailscale host key.
  for ip in $(pi "hostname -I"); do
    case $ip in 192.168.*|10.*|172.1[6-9].*|172.2?.*|172.3[01].*)
      if ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o HostKeyAlias=100.78.110.4 "shadowlessblu@$ip" true 2>/dev/null; then
        host="shadowlessblu@$ip"; opts=(-o HostKeyAlias=100.78.110.4); break
      fi;;
    esac
  done
  echo "copying via $host"
  for f in qwen3-0.6b-base-f16.gguf qwen3-0.6b-distill-f16.gguf; do
    size=$(stat -c %s "$G/$f")
    for i in 1 2 3; do
      there=$(pi "stat -c %s models/$f 2>/dev/null")
      [ "$there" = "$size" ] && break
      scp -q -l 40000 "${opts[@]}" "$G/$f" "$host:models/$f.part" && pi "mv models/$f.part models/$f"
    done
    there=$(pi "stat -c %s models/$f 2>/dev/null")
    [ "$there" = "$size" ] || { echo "$f: $there bytes on the Pi, $size here"; return 1; }
    echo "$f ok ($size bytes)"
  done
}
create() {
  tar cf - research/q5_pi.sh | pi "cd ~/assistant && tar xf - && bash research/q5_pi.sh create"
}
score() {
  pi "cd ~/assistant && bash research/q5_pi.sh score"
}

for s in fetch train convert copy create score; do
  if [ -f "$DONE/$s" ]; then echo "skip $s (done)"; continue; fi
  echo "=== $(date +%T) $s"
  if $s; then touch "$DONE/$s"; else echo "!!! $s failed"; exit 1; fi
done
echo "=== $(date +%T) scoring started on the Pi: ssh pi5 'cd ~/assistant && bash research/q5_pi.sh result'"
