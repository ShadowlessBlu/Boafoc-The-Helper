#!/bin/bash
# Offline readiness: does anything the assistant uses need the internet? Run on the Pi while the
# assistant is running (it doesn't change anything). The real test is still to unplug the network
# and use it; this finds what would break first.
#   bash bench/offline_check.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
echo "== Hosts in config.toml (should all be local)"
grep -nE "^\s*(host|url|broker|llm_host)\s*=" config.toml 2>/dev/null || echo "(none set: defaults are 127.0.0.1)"

echo; echo "== Network connections of the assistant, screen, Ollama and Mosquitto to other machines"
pids=$( (pgrep -f "[p]ython -m assistant"; pgrep -x ollama; pgrep -f "[o]llama runner"; pgrep -x mosquitto) | sort -u | tr '\n' '|' | sed 's/|$//')
ss -tunpH state established 2>/dev/null | grep -E "pid=(${pids:-none})," \
  | awk '{print $1, $4, "->", $5, $NF}' \
  | grep -vE -- "-> (127\.|\[::1\]|\[::ffff:127\.|192\.168\.|10\.|100\.)" || echo "none (only local and LAN/Tailscale peers)"

echo; echo "== Files the voice engines load (must exist locally)"
# wake words, Whisper and Piper models, as set in config.toml
files=$(grep -E "^\s*(wake_word|whisper_model|piper_model)\s*=" config.toml \
  | sed -E 's/^[^"]*"([^"]*)".*/\1/' | tr ',' '\n' | sed 's/@.*//')
for f in $files; do
  [ -f "$f" ] && echo "ok  $f" || echo "MISSING  $f"
done
.venv/bin/python - <<'EOF'
import importlib.util, pathlib
spec = importlib.util.find_spec("openwakeword")
if spec:
    d = pathlib.Path(spec.origin).parent / "resources" / "models"
    need = ["melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx"]
    for n in need:
        print(("ok  " if (d / n).exists() else "MISSING  ") + f"openwakeword {n}")
EOF

echo; echo "== Ollama models present"
ollama list 2>/dev/null | awk 'NR>1{print "ok  " $1}' | grep -E "qwen3:1.7b|nomic-embed" || echo "MISSING qwen3:1.7b or nomic-embed-text"

echo; echo "== Clock (no internet = no time sync)"
timedatectl show -p NTPSynchronized -p TimeUSec 2>/dev/null
if [ -e /dev/rtc0 ]; then
  echo "RTC present: $(cat /sys/class/rtc/rtc0/name 2>/dev/null)"
  [ -r /sys/devices/platform/soc/soc:rpi_rtc/rtc/rtc0/charging_voltage ] \
    && echo "RTC battery charging voltage setting: $(cat /sys/devices/platform/soc/soc:rpi_rtc/rtc/rtc0/charging_voltage)"
fi
echo "Without a battery on the Pi 5's RTC connector, the clock stops while the Pi is off; offline,"
echo "it would come back with the time it had at the cut (wrong time answers)."
