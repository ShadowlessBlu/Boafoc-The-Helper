#!/usr/bin/env bash
# Copy Ollama's model files (they are plain GGUF) to ~/models so llama.cpp tools can use them.
# Ollama keeps them in /usr/share/ollama, readable only by the ollama user, so this needs sudo:
#   sudo bash scripts/export_ollama_gguf.sh [user] [model ...]
# Default models: the EXP-001 set (not the 3B, to save disk).
set -euo pipefail
USER_NAME=${1:-${SUDO_USER:-$(whoami)}}
shift || true
MODELS=${*:-"qwen3:0.6b qwen3:1.7b gemma3:1b llama3.2:1b llama3.2:1b-instruct-q4_K_M"}
DEST=$(getent passwd "$USER_NAME" | cut -d: -f6)/models
mkdir -p "$DEST"
for m in $MODELS; do
  blob=$(ollama show --modelfile "$m" | awk '/^FROM /{print $2; exit}')
  quant=$(ollama show "$m" | awk '/quantization/{print $2; exit}')
  name="$(echo "$m" | tr ':/' '--')-${quant:-unknown}.gguf"
  name=${name//-instruct-q4_K_M-Q4_K_M/-Q4_K_M}
  echo "$m -> $DEST/$name"
  cp "$blob" "$DEST/$name"
done
chown -R "$USER_NAME": "$DEST"
ls -la "$DEST"
