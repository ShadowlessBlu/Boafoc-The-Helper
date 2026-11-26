#!/usr/bin/env bash
# Stage 4 + 6 setup on the Pi: Vosk, whisper.cpp, Piper, openWakeWord, Mosquitto.
# Usage: bash scripts/install_voice.sh [--no-whisper] [--no-mqtt]
# Downloads roughly 300 MB of models into ./models
set -euo pipefail
cd "$(dirname "$0")/.."
DO_WHISPER=1; DO_MQTT=1
for a in "$@"; do
  case "$a" in --no-whisper) DO_WHISPER=0 ;; --no-mqtt) DO_MQTT=0 ;; esac
done
[ -x .venv/bin/pip ] || { echo "Run scripts/setup_base.sh first"; exit 1; }
PIP=.venv/bin/pip
mkdir -p models

echo "==> Python voice packages"
$PIP install vosk piper-tts numpy paho-mqtt pygame-ce
if ! $PIP install openwakeword; then
  echo "   full openwakeword install failed (tflite-runtime has no wheel for this Python); using ONNX only"
  $PIP install --no-deps openwakeword
  $PIP install onnxruntime scipy scikit-learn tqdm requests
fi
.venv/bin/python -c "import openwakeword.utils as u; u.download_models()" || echo "   (wake word model download failed - check network)"

echo "==> Vosk small English model (~40 MB)"
if [ ! -d models/vosk-model-small-en-us-0.15 ]; then
  curl -fL -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  unzip -q /tmp/vosk.zip -d models && rm /tmp/vosk.zip
fi

echo "==> Piper voice en_US-lessac-medium (~60 MB)"
V=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
for f in en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json; do
  [ -f "models/$f" ] || curl -fL -o "models/$f" "$V/$f"
done
echo "   piper raw-output flag on this version:"
.venv/bin/piper --help 2>&1 | grep -i raw || echo "   (check 'piper --help' and set [voice] piper_cmd)"

if [ "$DO_WHISPER" = 1 ]; then
  echo "==> whisper.cpp + base.en model"
  mkdir -p ~/src
  [ -d ~/src/whisper.cpp ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp ~/src/whisper.cpp
  cmake -S ~/src/whisper.cpp -B ~/src/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build ~/src/whisper.cpp/build -j4 --config Release
  mkdir -p ~/.local/bin && ln -sf ~/src/whisper.cpp/build/bin/whisper-cli ~/.local/bin/whisper-cli
  for m in tiny.en base.en; do
    [ -f "models/ggml-$m.bin" ] || bash ~/src/whisper.cpp/models/download-ggml-model.sh "$m" models
  done
fi

if [ "$DO_MQTT" = 1 ]; then
  echo "==> Mosquitto broker (password login + per-user topic rules)"
  sudo apt install -y mosquitto mosquitto-clients
  sudo tee /etc/mosquitto/conf.d/assistant.conf >/dev/null <<'EOF'
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd
acl_file /etc/mosquitto/acl
EOF
  if [ ! -f /etc/mosquitto/acl ]; then
    sudo tee /etc/mosquitto/acl >/dev/null <<'EOF'
# The assistant (and the latency benchmark) may use everything under home/
user assistant
topic readwrite home/#
user latency-bench
topic readwrite home/#
EOF
  fi
  if ! sudo test -f /etc/mosquitto/passwd || ! sudo grep -q "^assistant:" /etc/mosquitto/passwd; then
    echo "   Choose a password for MQTT user 'assistant' (put it in ASSISTANT_MQTT_PASSWORD):"
    if sudo test -f /etc/mosquitto/passwd; then
      sudo mosquitto_passwd /etc/mosquitto/passwd assistant
    else
      sudo mosquitto_passwd -c /etc/mosquitto/passwd assistant
    fi
  fi
  sudo chown mosquitto: /etc/mosquitto/passwd /etc/mosquitto/acl 2>/dev/null || true
  sudo chmod 640 /etc/mosquitto/passwd /etc/mosquitto/acl
  sudo systemctl enable --now mosquitto
  sudo systemctl restart mosquitto
  echo "   Add each ESP32 with: bash scripts/add_mqtt_node.sh <node-id> <room_slug>"
fi

echo
echo "Done. Next: tune the mic with  .venv/bin/python -m assistant.voice.calibrate"
