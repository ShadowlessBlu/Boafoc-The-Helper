#!/usr/bin/env bash
# EXP-001 Phase A in one go: predictions, speed, quality, red-team, memory search.
# Usage: bash bench/run_phase_a.sh "qwen3:0.6b gemma3:1b llama3.2:1b qwen3:1.7b" [redteam-model]
# Run it under nohup so it survives SSH drops; everything is logged to experiments/results/phase_a.log
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=${1:-"qwen3:0.6b gemma3:1b llama3.2:1b qwen3:1.7b"}
RT_MODEL=${2:-qwen3:1.7b}
PY=.venv/bin/python
R=experiments/results
mkdir -p "$R"

step() { echo; echo "===== $* ($(date +%H:%M:%S), $(vcgencmd measure_temp 2>/dev/null))"; }

step "kv_calc predictions"
$PY bench/kv_calc.py --model all > "$R/kv_calc.txt" 2>&1

step "speed: bench_ollama"
# shellcheck disable=SC2086
$PY bench/bench_ollama.py --models $MODELS --runs 3 --cool-to 65

for m in $MODELS; do
  step "quality: score $m"
  $PY experiments/score.py --model "$m"
done

step "red-team with $RT_MODEL"
$PY experiments/redteam.py --model "$RT_MODEL"

step "memory search"
$PY bench/memory_search.py

step "finished"
vcgencmd get_throttled
echo PHASE_A_DONE
