#!/bin/sh
# Train (or finish training) the wake words, skipping any that already have a report from a run
# with the real recordings ("REAL_" fields). Safe to run again after a power cut. Runs on cores
# 2-3 with the assistant paused (cooler, faster), then starts the assistant again.
#   setsid nohup sh training/resume_wake_training.sh uzi aiya boafo &
cd "$(dirname "$0")/.." || exit 1
W="$HOME/wake-train"
rm -f "$W/REAL_DONE"
pgrep -f "wake-train/powerlog[.]sh" > /dev/null || setsid nohup "$W/powerlog.sh" < /dev/null > /dev/null 2>&1 &
systemctl --user stop assistant
for w in "$@"; do
  if [ -f "$W/${w}_report.json" ] && grep -q REAL_ "$W/${w}_report.json"; then
    echo "$w: already done" >> "$W/resume.log"
    continue
  fi
  echo "$w: training $(date +%H:%M)" >> "$W/resume.log"
  taskset -c 2,3 .venv/bin/python -u training/train_wake_word.py "$w" > "$W/$w.log" 2>&1
done
echo finished > "$W/REAL_DONE"
systemctl --user start assistant
