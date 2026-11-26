#!/usr/bin/env bash
# Run the assistant as a systemd *user* service, started at boot.
#
# Why a user service and not a system one (scripts/install_service.sh): the Bluetooth speaker
# is reached through PipeWire, which runs in the user's session; a system service can't use it.
# No sudo needed. It starts at boot when "lingering" is on for the user (check:
# `loginctl show-user $USER -p Linger`; turn on once with `sudo loginctl enable-linger $USER`),
# otherwise at login (this Pi logs in automatically).
#
#   bash scripts/install_user_service.sh            API only: /ask, /face, /state (no microphone)
#   bash scripts/install_user_service.sh --voice    full assistant: API + wake word + voice loop
#   bash scripts/install_user_service.sh --remove   stop it and remove it
#
# Logs:    tail -f logs/assistant.log   (the user journal is not kept on this Pi)
# Status:  systemctl --user status assistant
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT="$HOME/.config/systemd/user/assistant.service"
MODE="--api"

case "${1:-}" in
  --voice) MODE="--service" ;;
  --remove)
    systemctl --user disable --now assistant.service 2>/dev/null || true
    rm -f "$UNIT" && systemctl --user daemon-reload && echo "removed"; exit 0 ;;
  "") ;;
  *) echo "unknown option $1"; exit 2 ;;
esac

mkdir -p "$(dirname "$UNIT")" "$ROOT/logs"
cat > "$UNIT" <<EOF
# Installed by scripts/install_user_service.sh. Mode: $MODE
[Unit]
Description=Personal AI Assistant ($MODE)
After=pipewire.service wireplumber.service
Wants=pipewire.service

[Service]
WorkingDirectory=$ROOT
Environment=PYTHONUNBUFFERED=1
# Ollama is a system service; if it isn't up yet the API still starts and /health shows it.
ExecStart=$ROOT/.venv/bin/python -m assistant $MODE
# The user journal isn't kept on this Pi, so output also goes to a file.
StandardOutput=append:$ROOT/logs/assistant.log
StandardError=append:$ROOT/logs/assistant.log
Restart=on-failure
RestartSec=5
MemoryMax=6G

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable assistant.service >/dev/null
systemctl --user restart assistant.service
sleep 4
systemctl --user --no-pager status assistant.service | head -6
echo "linger (start at boot without login): $(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo unknown)"
