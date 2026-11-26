#!/usr/bin/env bash
# EXP-001 Q6 (context length vs memory/speed) and Q2/Q3 (quant ladder: speed, perplexity, task score),
# plus the Pi re-run of the memory search benchmark.
#
# Resumable: every measurement is one checkpoint. A finished one leaves a marker in
# experiments/results/.done/, and is skipped next time. After a power cut, only the measurement
# that was running is lost. Run it again (or let the @reboot cron entry do it) and it carries on.
# It needs no PC: everything is logged on the Pi.
#
#   nohup bash bench/run_q6_q3.sh >> experiments/results/q6_q3.log 2>&1 &
#   bash bench/run_q6_q3.sh --status      # what's done / left, without running anything
#   bash bench/run_q6_q3.sh --install-boot / --remove-boot   # auto-resume at boot (user crontab)
#
# Needs GGUFs in ~/models (sudo bash scripts/export_ollama_gguf.sh) and llama.cpp tools.
set -uo pipefail
# cron starts us with PATH=/usr/bin:/bin; ollama lives in /usr/local/bin (its absence failed 2 steps once)
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$PWD
export PATH="$HOME/.local/bin:$PATH"
M=~/models
R=experiments/results
DONE=$R/.done
mkdir -p "$R" "$DONE"
COOL=${COOL:-65}
STEP_TIMEOUT=${STEP_TIMEOUT:-21600}  # 6 h: 16k-token runs of the 1.7B take hours; a truly hung step is still skipped (FAILED)
CTXS=${CTXS:-"512 2048 4096 8192 16384"}
Q6_MODELS=${Q6_MODELS:-"qwen3-0.6b-Q4_K_M qwen3-1.7b-Q4_K_M llama3.2-1b-Q4_K_M gemma3-1b-Q4_K_M"}
# kv_type:flash_attention. On the Pi's CPU, flash attention made decode at 2k context slower
# (6.3 vs 10.1 t/s for qwen3-0.6b), but a quantized KV cache needs it.
CONFIGS=${CONFIGS:-"f16:off q8_0:on"}
EXTRA=${EXTRA:-"qwen3-0.6b-Q4_K_M=f16:on"}
LADDER="llama3.2-1b-Q8_0 llama3.2-1b-Q4_K_M llama3.2-1b-req-Q4_K_M llama3.2-1b-req-Q3_K_S llama3.2-1b-req-Q2_K"
CTX_CSV=$R/llamacpp_ctx.csv
QUANT_CSV=$R/llamacpp_quant.csv
PPL_TEXT=research/eval_text_v2.txt
CRON_LINE="@reboot sleep 90 && cd $ROOT && bash bench/run_q6_q3.sh >> $R/q6_q3.log 2>&1"

temp() { vcgencmd measure_temp | grep -o '[0-9.]*'; }
cool() { local n=0; while awk "BEGIN{exit !($(temp) > $COOL)}" && [ $n -lt 60 ]; do sleep 5; n=$((n+1)); done; }
log() { echo "[$(date '+%F %T') $(temp)C] $*"; }

# --- the list of checkpoints, in order: "id|function args" ---
plan() {
  for name in $Q6_MODELS; do for cfg in $CONFIGS; do for c in $CTXS; do
    echo "q6_${name}_${cfg/:/_fa}_$c|q6_one $name ${cfg%%:*} ${cfg#*:} $c"; done; done; done
  for job in $EXTRA; do name=${job%%=*}; cfg=${job#*=}; for c in $CTXS; do
    echo "q6_${name}_${cfg/:/_fa}_$c|q6_one $name ${cfg%%:*} ${cfg#*:} $c"; done; done
  for q in Q4_K_M Q3_K_S Q2_K; do echo "quantize_$q|quantize_one $q"; done
  for name in $LADDER; do echo "ladder_$name|ladder_one $name"; done
  for q in Q3_K_S Q2_K; do echo "score_req_$q|score_one $q"; done
  echo "memory_search_v3|memory_search"
  echo "summary|summary"
}

q6_one() {  # model kv fa ctx -> one row in llamacpp_ctx.csv
  local name=$1 kv=$2 fa=$3 c=$4 f=$M/$1.gguf out tps rss
  [ -f "$f" ] || { log "missing $f"; return 1; }
  [ -f "$CTX_CSV" ] || echo "timestamp,model,kv_type,flash_attn,n_ctx,decode_tps,peak_rss_mb,max_temp_c,throttled" > "$CTX_CSV"
  cool
  out=$( { /usr/bin/time -f "RSS_KB %M" timeout "$STEP_TIMEOUT" llama-bench -m "$f" -t 4 -p 0 -n 32 -d "$c" -r 1 \
           -ctk "$kv" -ctv "$kv" -fa "$fa" -o csv; } 2>&1 )
  tps=$(echo "$out" | python3 -c "import csv,sys
rows=[r for r in csv.DictReader(l for l in sys.stdin if l.startswith(('build_commit','\"')))]
print(rows[-1]['avg_ts'] if rows else '')")
  rss=$(echo "$out" | awk '/RSS_KB/{print int($2/1024)}')
  [ -n "$tps" ] || { log "no result: $(echo "$out" | tail -3)"; return 1; }
  echo "$(date -Iseconds),$name,$kv,$fa,$c,$tps,$rss,$(temp),$(vcgencmd get_throttled | cut -d= -f2)" | tee -a "$CTX_CSV"
}

quantize_one() {  # requantize Llama-3.2-1B from Q8_0 (F16 isn't available offline); write to .tmp, then rename
  local q=$1 out=$M/llama3.2-1b-req-$1.gguf
  [ -f "$out" ] && return 0
  llama-quantize --allow-requantize "$M/llama3.2-1b-Q8_0.gguf" "$out.tmp" "$q" | tail -2 && mv "$out.tmp" "$out"
}

ladder_one() {  # speed (pp512/tg128) + perplexity + peak RAM -> one row in llamacpp_quant.csv
  local name=$1 f=$M/$1.gguf sp pp tg pl ppl rss
  [ -f "$f" ] || { log "missing $f"; return 1; }
  [ -f "$QUANT_CSV" ] || echo "timestamp,model,file_mb,pp512_tps,tg128_tps,perplexity,ppl_err,peak_rss_mb,throttled" > "$QUANT_CSV"
  cool
  sp=$(timeout "$STEP_TIMEOUT" llama-bench -m "$f" -t 4 -p 512 -n 128 -r 3 -o csv 2>/dev/null)
  pp=$(echo "$sp" | python3 -c "import csv,sys; r=list(csv.DictReader(sys.stdin)); print(r[0]['avg_ts'])")
  tg=$(echo "$sp" | python3 -c "import csv,sys; r=list(csv.DictReader(sys.stdin)); print(r[1]['avg_ts'])")
  cool
  pl=$( { /usr/bin/time -f "RSS_KB %M" timeout "$STEP_TIMEOUT" llama-perplexity -m "$f" -f "$PPL_TEXT" -c 256 -b 256 -t 4; } 2>&1 )
  ppl=$(echo "$pl" | grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' | awk '{print $5","$7}')
  rss=$(echo "$pl" | awk '/RSS_KB/{print int($2/1024)}')
  [ -n "$tg" ] && [ -n "$ppl" ] || { log "incomplete: pp=$pp tg=$tg ppl=$ppl"; return 1; }
  echo "$(date -Iseconds),$name,$(( $(stat -c %s "$f") / 1048576 )),$pp,$tg,$ppl,$rss,$(vcgencmd get_throttled | cut -d= -f2)" | tee -a "$QUANT_CSV"
}

score_one() {  # load a requantized file into Ollama with the official chat template, then the 20-question test
  local q=$1 tag="llama3.2-1b-req:${1,,}"
  until curl -s localhost:11434 >/dev/null; do sleep 5; done
  { echo "FROM $M/llama3.2-1b-req-$q.gguf"; ollama show --modelfile llama3.2:1b | grep -v '^FROM '; } > "/tmp/Modelfile.$q"
  ollama create "$tag" -f "/tmp/Modelfile.$q" >/dev/null || return 1
  cool
  timeout "$STEP_TIMEOUT" .venv/bin/python experiments/score.py --model "$tag"
}

memory_search() { cool; .venv/bin/python bench/memory_search.py; }
summary() { .venv/bin/python bench/summarize.py > "$R/summary_latest.md"; }

case "${1:-}" in
  --install-boot)
    (crontab -l 2>/dev/null | grep -v 'bench/run_q6_q3.sh'; echo "$CRON_LINE") | crontab -
    echo "Installed: $CRON_LINE"; exit 0 ;;
  --remove-boot)
    crontab -l 2>/dev/null | grep -v 'bench/run_q6_q3.sh' | crontab -; echo "Removed the @reboot entry"; exit 0 ;;
  --status)
    n=0; d=0
    while IFS='|' read -r id cmd; do n=$((n+1))
      if [ -f "$DONE/$id" ]; then d=$((d+1)); s=done; elif [ -f "$DONE/$id.FAILED" ]; then s=FAILED; else s=todo; fi
      printf '%-7s %s\n' "$s" "$id"; done < <(plan)
    echo "$d of $n checkpoints done"; exit 0 ;;
esac

# One copy at a time (a manual start and the boot entry must not both run)
exec 9>"/tmp/run_q6_q3.lock"
flock -n 9 || { log "already running"; exit 0; }

log "start (resuming: $(ls "$DONE" | grep -vc FAILED) checkpoints already done)"
while IFS='|' read -r id cmd; do
  [ -f "$DONE/$id" ] && continue
  [ -f "$DONE/$id.FAILED" ] && continue
  log "== $id"
  if $cmd < /dev/null; then touch "$DONE/$id"; else log "FAILED $id"; touch "$DONE/$id.FAILED"; fi
  sync   # flush results and markers to disk in case the power goes
done < <(plan)

log "all checkpoints done"
vcgencmd get_throttled
"$0" --remove-boot
echo Q6_Q3_DONE
