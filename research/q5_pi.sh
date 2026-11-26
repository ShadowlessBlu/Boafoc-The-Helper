#!/bin/bash
# EXP-001 Q5, the Pi's part (after research/finish_distill.sh copied the f16 GGUFs to ~/models):
#   bash research/q5_pi.sh create   quantize to Q4_K_M and create the Ollama models
#                                   q06-pipe-base (base Qwen3-0.6B through the same pipeline: the control)
#                                   q06-distill   (the student trained on qwen3:1.7b's answers)
#   bash research/q5_pi.sh score    score qwen3:0.6b, q06-pipe-base and q06-distill on test set v2,
#                                   thinking off, in the background (log research/data/q5_score.log)
#   bash research/q5_pi.sh result   pass count per model (its last complete attempt)
#   bash research/q5_pi.sh stop     stop scoring and remove the boot entry
# Each model inherits qwen3:0.6b's chat template and parameters, so only the weights differ.
# Survives power cuts: a quantize writes to a .tmp file first; scoring has a self-removing @reboot
# entry and per-model markers (research/data/q5/), and a model cut off mid-way is scored again.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
PY=.venv/bin/python
M="$HOME/models"
LOG="$ROOT/research/data/q5_score.log"
Q="$ROOT/research/data/q5"
MODELS="qwen3:0.6b q06-pipe-base q06-distill"
COOL=${COOL:-74}
TAG="# q5-score-resume"

wait_ollama() { until curl -s localhost:11434 >/dev/null; do sleep 5; done; }
remove_boot() { crontab -l 2>/dev/null | grep -v "$TAG" | crontab -; }

case "${1:-}" in
  create)
    wait_ollama
    tmpl=$(ollama show --modelfile qwen3:0.6b | grep -v '^FROM ')
    for name in base distill; do
      f16="$M/qwen3-0.6b-$name-f16.gguf"; q4="$M/qwen3-0.6b-$name-q4_k_m.gguf"
      [ -f "$f16" ] || { echo "missing $f16"; exit 1; }
      if [ ! -f "$q4" ] || [ "$f16" -nt "$q4" ]; then
        rm -f "$q4.tmp"
        nice llama-quantize "$f16" "$q4.tmp" Q4_K_M > "/tmp/quantize_$name.log" 2>&1 && mv "$q4.tmp" "$q4" \
          || { echo "quantize $name failed:"; tail -5 "/tmp/quantize_$name.log"; exit 1; }
        sync
      fi
      tag=$([ "$name" = base ] && echo q06-pipe-base || echo q06-distill)
      { echo "FROM $q4"; echo "$tmpl"; } > "/tmp/Modelfile.$tag"
      ollama create "$tag" -f "/tmp/Modelfile.$tag" > /dev/null || { echo "create $tag failed"; exit 1; }
      ls -la "$q4"
    done
    ollama list | grep -E "q06-|qwen3:0.6b"
    ;;
  score)
    mkdir -p "$Q"
    (crontab -l 2>/dev/null | grep -v "$TAG"; echo "@reboot sleep 150 && bash $ROOT/research/q5_pi.sh --run >> $LOG 2>&1 $TAG") | crontab -
    nohup bash "$0" --run >> "$LOG" 2>&1 &
    echo "scoring in the background, log $LOG"
    ;;
  --run)
    exec 9>"$Q/.lock"
    flock -n 9 || { echo "already running"; exit 0; }
    echo "=== $(date '+%F %T') start/resume"
    for m in $MODELS; do
      [ -f "$Q/$m.done" ] && continue
      wait_ollama
      date '+%Y-%m-%dT%H:%M:%S' > "$Q/$m.start"; sync   # rows before this belong to a cut-off attempt
      echo "--- $(date +%T) $m"
      if timeout 7200 $PY experiments/score.py --model "$m" --think off --set experiments/test_set_v2.json --cool-to "$COOL"; then
        touch "$Q/$m.done"; sync
      else
        echo "!!! $m failed"; exit 1
      fi
    done
    echo "=== $(date '+%F %T') scoring finished"
    remove_boot
    ;;
  result)
    # score.py writes each model's 60 rows at the end, to score_v2_r2.csv (score_v2.csv has older
    # columns); rows from before a model's last start belong to a cut-off attempt.
    $PY - "$Q" $MODELS <<'EOF'
import csv, sys, pathlib
q, models = pathlib.Path(sys.argv[1]), sys.argv[2:]
rows = list(csv.DictReader(open("experiments/results/score_v2_r2.csv", encoding="utf-8", errors="replace")))
for m in models:
    start = (q / f"{m}.start").read_text().strip() if (q / f"{m}.start").exists() else "9999"
    mine = [r for r in rows if r.get("model") in (m, f"{m} (think off)") and (r.get("timestamp") or "") >= start]
    state = "done" if (q / f"{m}.done").exists() else "not finished"
    if not mine:
        print(f"{m:14} not scored ({state})"); continue
    ok = sum(int(r["pass"]) for r in mine)
    secs = sum(float(r["seconds"]) for r in mine) / len(mine)
    cats = {}
    for r in mine:
        c = cats.setdefault(r["category"], [0, 0]); c[0] += int(r["pass"]); c[1] += 1
    print(f"{m:14} {ok:2}/{len(mine)}  {secs:.1f} s per answer  ({state})  " +
          "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(cats.items())))
EOF
    grep -a "scoring finished" "$LOG" 2>/dev/null | tail -1
    ;;
  stop) remove_boot; pkill -f "q5_pi.sh --run"; pkill -f "experiments/score.py"; echo stopped;;
  *) sed -n 2,12p "$0"; exit 1;;
esac
