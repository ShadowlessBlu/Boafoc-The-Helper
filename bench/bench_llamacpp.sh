#!/usr/bin/env bash
# EXP-001 with llama.cpp directly: speed per quant (Q2) and peak RAM vs context (Q1, Q6).
# Usage: bash bench/bench_llamacpp.sh <folder with .gguf files> [ctx list]
#   e.g. bash bench/bench_llamacpp.sh ~/models "512 2048 4096 8192"
# Env:   THREADS=4  KV=f16|q8_0  REPS=3
# Notes:
#  - llama-bench sizes the context to the prompt + generation lengths, so "-p CTX" fills
#    the KV cache to about CTX tokens. Peak RSS includes the memory-mapped weights it touched.
#  - Quantized KV (KV=q8_0) needs flash attention (-fa 1).
set -euo pipefail

DIR=${1:?folder with .gguf files}
CTXS=${2:-"512 2048 4096 8192"}
THREADS=${THREADS:-4}
KV=${KV:-f16}
REPS=${REPS:-3}
cd "$(dirname "$0")/.."
OUTDIR=experiments/results
mkdir -p "$OUTDIR"
SPEED=$OUTDIR/llamacpp_speed.csv
MEM=$OUTDIR/llamacpp_mem.csv

command -v llama-bench >/dev/null || { echo "llama-bench not on PATH (run scripts/install_llm.sh)"; exit 1; }
[ -f "$MEM" ] || echo "timestamp,model,file_mb,n_ctx,kv_type,threads,peak_rss_mb,max_temp_c,throttled" > "$MEM"

FA=""
[ "$KV" != "f16" ] && FA="-fa 1"

temp() { vcgencmd measure_temp 2>/dev/null | grep -o '[0-9.]*' || echo ""; }
cool() { while t=$(temp); [ -n "$t" ] && awk "BEGIN{exit !($t > 55)}"; do echo "  cooling ($t C)"; sleep 5; done; }

for f in "$DIR"/*.gguf; do
  name=$(basename "$f" .gguf)
  size_mb=$(( $(stat -c %s "$f") / 1048576 ))
  echo "== $name ($size_mb MB)"

  # Speed: standard pp512 / tg128
  cool
  llama-bench -m "$f" -t "$THREADS" -p 512 -n 128 -r "$REPS" -ctk "$KV" -ctv "$KV" $FA -o csv \
    | { if [ -s "$SPEED" ]; then tail -n +2; else cat; fi; } >> "$SPEED"

  # Memory vs context
  for c in $CTXS; do
    cool
    peak_kb=$( { /usr/bin/time -f "%M" llama-bench -m "$f" -t "$THREADS" -p "$c" -n 16 -r 1 \
                 -ctk "$KV" -ctv "$KV" $FA >/dev/null; } 2>&1 | tail -1 )
    echo "$(date -Iseconds),$name,$size_mb,$c,$KV,$THREADS,$((peak_kb/1024)),$(temp),$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)" >> "$MEM"
    echo "  ctx $c: peak RSS $((peak_kb/1024)) MB"
  done
done
echo "Saved: $SPEED  $MEM"
