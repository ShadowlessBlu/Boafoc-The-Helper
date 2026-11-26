#!/usr/bin/env bash
# Stage 1 setup on the Pi: system update, base packages, project venv.
# Safe to run more than once. Needs sudo.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Updating system"
sudo apt update
sudo apt full-upgrade -y

echo "==> Installing base packages"
sudo apt install -y \
  git htop btop tmux build-essential cmake pkg-config \
  python3-venv python3-dev python3-pip \
  libopenblas-dev libcurl4-openssl-dev \
  sysbench time jq \
  alsa-utils sox \
  mosquitto-clients

echo "==> Adding $(whoami) to audio and gpio groups"
sudo usermod -aG audio,gpio "$(whoami)" || true

echo "==> Creating venv in $(pwd)/.venv"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

mkdir -p data experiments/results
[ -f config.toml ] || cp config.example.toml config.toml

echo
echo "Done. Log out and back in for the group changes to apply."
echo "Next: bash scripts/install_llm.sh"
