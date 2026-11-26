#!/bin/bash
# Start the offline test (bench/offline_test.py) in the background, so it keeps running when the
# internet (and with it Tailscale and this SSH session) goes away. From the PC:
#   ssh pi5 "bash ~/assistant/bench/offline_test.sh"
# then unplug the internet at the router; the Pi says when to, and when it's done.
# Progress: tail -f ~/assistant/logs/offline_test.log     Report: experiments/results/offline_test_*.md
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
setsid nohup .venv/bin/python bench/offline_test.py "$@" >> logs/offline_test.log 2>&1 < /dev/null &
echo "offline test started (pid $!). Unplug the internet at the router when the Pi says so."
echo "Progress: tail -f ~/assistant/logs/offline_test.log"
