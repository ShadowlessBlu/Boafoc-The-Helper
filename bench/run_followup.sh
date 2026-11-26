#!/usr/bin/env bash
# Follow-up to bench/run_q6_q3.sh, for when the PC is away. Waits until that run has finished,
# then: installs the code staged in ~/assistant-next (memory-injection hardening, test set v2,
# scorer cool-down), runs the tests, scores 7 models on the 60-question test set v2, and re-runs
# the red-team to measure the new memory defence.
#
# Same safety as run_q6_q3.sh: one checkpoint per step (markers in experiments/results/.done/fu_*),
# a lock, and a user-crontab @reboot entry that removes itself when everything is done.
#   setsid nohup bash bench/run_followup.sh >> experiments/results/followup.log 2>&1 < /dev/null &
#   bash bench/run_followup.sh --status | --install-boot | --remove-boot
set -uo pipefail
# cron starts us with PATH=/usr/bin:/bin; ollama lives in /usr/local/bin (its absence failed 2 steps once)
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$PWD
STAGE=${STAGE:-$HOME/assistant-next}
R=experiments/results
DONE=$R/.done
mkdir -p "$DONE"
PY=.venv/bin/python
COOL=${COOL:-74}  # was 68: most of each question went on waiting; the Pi 5 only throttles from 80 degC
CRON_LINE="@reboot sleep 150 && cd $ROOT && bash bench/run_followup.sh >> $R/followup.log 2>&1"
# model|think (default = the model's own default)
V2_MODELS="qwen3:1.7b|off qwen3:0.6b|off qwen3:0.6b|default llama3.2:3b|default llama3.2:1b-instruct-q4_K_M|default gemma3:1b|default llama3.2:1b|default"
# the Llama-1B quant ladder (requantized from Q8_0 by run_q6_q3.sh), scored on v2 with the same code as the above.
# The requantized Q4_K_M is left out: its perplexity matched the official Q4 (43.9 vs 44.1), which is scored above.
V2_MODELS="$V2_MODELS llama3.2-1b-req:q3_k_s|default llama3.2-1b-req:q2_k|default"
RT_MODELS="qwen3:1.7b qwen3:0.6b"

log() { echo "[$(date '+%F %T') $(vcgencmd measure_temp | grep -o '[0-9.]*')C] $*"; }

plan() {
  echo "fu_deploy|deploy"
  echo "fu_tests|tests"
  echo "fu_create_req|create_req"
  for mt in $V2_MODELS; do echo "fu_score_v2_${mt/|/_think}|score_v2 ${mt%%|*} ${mt#*|}"; done
  for m in $RT_MODELS; do echo "fu_redteam_hardened_$m|redteam $m"; done
  echo "fu_voice_install|voice_install"
  echo "fu_summary|summary"
}

# Stage 4 software (Vosk, Piper, openWakeWord, whisper.cpp; ~300 MB). No MQTT part, so no sudo needed.
# Runs after the measurements so the whisper.cpp build doesn't disturb them. Safe to repeat.
voice_install() { timeout 7200 bash scripts/install_voice.sh --no-mqtt; }

deploy() {  # copy staged code over the project; results, data/, config.toml and the venv are not in the stage
  [ -d "$STAGE/src/assistant" ] || { log "no staged code in $STAGE"; return 1; }
  cp -r "$STAGE/." "$ROOT/" && log "deployed $(ls "$STAGE" | wc -l) items from $STAGE"
}
tests() {  # the whole output is searched: a test's buffered stdout is flushed after unittest's "OK" line
  local i out
  for i in 1 2; do  # retried once (a flaky API test was fixed on 2026-11-18, but just in case)
    out=$($PY -m unittest discover -s tests 2>&1)
    echo "$out" | grep -E '^(Ran|OK|FAILED|ERROR:|FAIL:)'
    echo "$out" | grep -qx 'OK' && return 0
  done
  return 1
}
create_req() {  # load the requantized Llama-1B files into Ollama with the official chat template
  local q tmpl
  wait_ollama
  tmpl=$(ollama show --modelfile llama3.2:1b | grep -v '^FROM ')
  for q in Q4_K_M Q3_K_S Q2_K; do
    { echo "FROM $HOME/models/llama3.2-1b-req-$q.gguf"; echo "$tmpl"; } > "/tmp/Modelfile.$q"
    ollama create "llama3.2-1b-req:${q,,}" -f "/tmp/Modelfile.$q" > /dev/null || return 1
  done
  ollama list | grep llama3.2-1b-req
}
wait_ollama() { until curl -s localhost:11434 >/dev/null; do sleep 5; done; }
score_v2() {
  wait_ollama
  timeout 21600 $PY experiments/score.py --model "$1" --think "$2" --set experiments/test_set_v2.json --cool-to "$COOL"
}
redteam() { wait_ollama; timeout 21600 $PY experiments/redteam.py --model "$1" --think off; }
summary() { $PY bench/summarize.py > "$R/summary_latest.md"; }

case "${1:-}" in
  --install-boot) (crontab -l 2>/dev/null | grep -v 'bench/run_followup.sh'; echo "$CRON_LINE") | crontab -; echo "Installed: $CRON_LINE"; exit 0 ;;
  --remove-boot) crontab -l 2>/dev/null | grep -v 'bench/run_followup.sh' | crontab -; echo "Removed"; exit 0 ;;
  --status)
    n=0; d=0
    while IFS='|' read -r id cmd; do n=$((n+1))
      if [ -f "$DONE/$id" ]; then d=$((d+1)); s=done; elif [ -f "$DONE/$id.FAILED" ]; then s=FAILED; else s=todo; fi
      printf '%-7s %s\n' "$s" "$id"; done < <(plan)
    echo "$d of $n follow-up checkpoints done"; exit 0 ;;
esac

exec 9>"/tmp/run_followup.lock"
flock -n 9 || { log "already running"; exit 0; }

# Wait for the main run: its last checkpoint is "summary", and it holds its own lock while running.
# children get "9>&-" so they never hold the lock after this script dies (a leftover "sleep" did once)
until [ -f "$DONE/summary" ]; do sleep 300 9>&-; done
flock /tmp/run_q6_q3.lock true

log "start (resuming: $(ls "$DONE" | grep -c '^fu_.*[^D]$') follow-up checkpoints done)"
while IFS='|' read -r id cmd; do
  [ -f "$DONE/$id" ] || [ -f "$DONE/$id.FAILED" ] && continue
  log "== $id"
  if $cmd < /dev/null 9>&-; then touch "$DONE/$id"; else log "FAILED $id"; touch "$DONE/$id.FAILED"; fi
  sync
  # the tests must pass before anything is measured with the new code
  [ "$id" = fu_tests ] && [ -f "$DONE/fu_tests.FAILED" ] && { log "tests failed: stopping"; break; }
done < <(plan)

log "follow-up finished"
"$0" --remove-boot
echo FOLLOWUP_DONE
