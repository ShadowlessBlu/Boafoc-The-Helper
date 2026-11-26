#!/usr/bin/env bash
# Stage 3 setup: Ollama, llama.cpp (built from source) and starter models.
# Usage: bash scripts/install_llm.sh [--no-llamacpp] [--no-models]
set -euo pipefail

DO_LLAMACPP=1; DO_MODELS=1
for a in "$@"; do
  case "$a" in
    --no-llamacpp) DO_LLAMACPP=0 ;;
    --no-models)   DO_MODELS=0 ;;
  esac
done

if ! command -v ollama >/dev/null; then
  echo "==> Installing Ollama (official install script from ollama.com)"
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "==> Ollama already installed: $(ollama --version)"
fi

if [ "$DO_MODELS" = 1 ]; then
  echo "==> Pulling starter models (several GB in total)"
  for m in qwen3:0.6b gemma3:1b llama3.2:1b qwen3:1.7b llama3.2:3b nomic-embed-text; do
    ollama pull "$m"
  done
fi

if [ "$DO_LLAMACPP" = 1 ]; then
  echo "==> Building llama.cpp"
  mkdir -p ~/src
  if [ -d ~/src/llama.cpp ]; then
    git -C ~/src/llama.cpp pull --ff-only
  else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
  fi
  cmake -S ~/src/llama.cpp -B ~/src/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build ~/src/llama.cpp/build --config Release -j4
  mkdir -p ~/.local/bin
  for b in llama-cli llama-bench llama-quantize llama-perplexity llama-server; do
    [ -f ~/src/llama.cpp/build/bin/$b ] && ln -sf ~/src/llama.cpp/build/bin/$b ~/.local/bin/$b
  done
  echo "llama.cpp tools linked into ~/.local/bin (make sure it is on PATH)"
fi

echo
ollama list || true
echo "Next: python3 bench/kv_calc.py --model all   and   python3 bench/bench_ollama.py"
