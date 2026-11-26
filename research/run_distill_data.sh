#!/bin/bash
# EXP-001 Q5 step 1 on the Pi, unattended: qwen3:1.7b (thinking off, the assistant's own model and the
# best on the 60-question test, 54/60) writes 600 answers for the student to learn from.
# Survives power cuts: a user crontab @reboot entry starts it again 2 min after boot and it resumes;
# the entry removes itself when the data is complete.
#   bash research/run_distill_data.sh            run (or resume) now, in the background
#   bash research/run_distill_data.sh --status   progress
#   bash research/run_distill_data.sh --stop     stop and remove the boot entry
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/research/data/distill_v1.jsonl"
LOG="$ROOT/research/data/distill_v1.log"
TAG="# distill-data-resume"
cd "$ROOT" || exit 1
mkdir -p "$ROOT/research/data"

remove_boot() { crontab -l 2>/dev/null | grep -v "$TAG" | crontab -; }
case "${1:-}" in
  --status) echo "$(cat "$OUT.progress" 2>/dev/null || echo 0)/600 requests, $(wc -l < "$OUT" 2>/dev/null || echo 0) kept"; tail -2 "$LOG"; exit 0;;
  --stop) remove_boot; pkill -f "research/make_distill_data.py"; echo stopped; exit 0;;
  --run) ;;
  *) (crontab -l 2>/dev/null | grep -v "$TAG"; echo "@reboot sleep 120 && bash $ROOT/research/run_distill_data.sh --run >> $LOG 2>&1 $TAG") | crontab -
     nohup bash "$0" --run >> "$LOG" 2>&1 &
     echo "started (log: $LOG)"; exit 0;;
esac

exec 9>"$ROOT/research/data/.distill.lock"
flock -n 9 || { echo "already running"; exit 0; }
echo "=== $(date '+%F %T') start/resume"
if nice -n 10 "$ROOT/.venv/bin/python" research/make_distill_data.py --teacher qwen3:1.7b --think off \
     --n 600 --out "$OUT" --resume; then
  echo "=== $(date '+%F %T') finished"
  remove_boot
fi
