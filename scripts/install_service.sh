#!/usr/bin/env bash
# Install the assistant and nightly backup as systemd services (Stage 7).
# For the assistant itself, prefer scripts/install_user_service.sh: a system service cannot reach
# PipeWire, so it could not play to the Bluetooth speaker.
# Usage: bash scripts/install_service.sh
set -euo pipefail
cd "$(dirname "$0")/.."
U=$(whoami)
D=$(pwd)
for f in assistant.service assistant-backup.service assistant-backup.timer; do
  sed -e "s#/home/shadowlessblu/assistant#$D#g" -e "s#^User=shadowlessblu#User=$U#" "deploy/$f" \
    | sudo tee "/etc/systemd/system/$f" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now assistant-backup.timer
sudo systemctl enable --now assistant.service
sleep 3
systemctl --no-pager status assistant.service | head -12
echo
echo "Logs:    journalctl -u assistant -f"
echo "Health:  curl -s localhost:8765/health"
