#!/usr/bin/env bash
# Keep the Bluetooth speaker connected: after a reboot or power cut it doesn't come back by itself.
# Every 30 s, if it isn't connected, try once (it must be switched on and paired; it already is).
# Installed as a user service (no sudo):  bash scripts/speaker_reconnect.sh --install
#                                         bash scripts/speaker_reconnect.sh --remove
# Speaker address from SPEAKER_MAC, default the CALUS Blue P12.
set -u
MAC=${SPEAKER_MAC:-2B:BA:EB:6A:E4:B3}
UNIT="$HOME/.config/systemd/user/speaker-reconnect.service"

case "${1:-}" in
  --install)
    mkdir -p "$(dirname "$UNIT")"
    cat > "$UNIT" <<EOF
# Installed by scripts/speaker_reconnect.sh
[Unit]
Description=Reconnect the Bluetooth speaker ($MAC)
After=pipewire.service wireplumber.service

[Service]
Environment=SPEAKER_MAC=$MAC
ExecStart=/usr/bin/env bash $(cd "$(dirname "$0")" && pwd)/speaker_reconnect.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now speaker-reconnect.service
    echo "installed and started"; exit 0 ;;
  --remove)
    systemctl --user disable --now speaker-reconnect.service 2>/dev/null
    rm -f "$UNIT"; systemctl --user daemon-reload; echo "removed"; exit 0 ;;
esac

was=""
while true; do
  if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
    [ "$was" = yes ] || echo "$(date +%H:%M:%S) connected"
    was=yes
  else
    [ "$was" = no ] || echo "$(date +%H:%M:%S) not connected, retrying every 30 s"
    was=no
    timeout 20 bluetoothctl connect "$MAC" > /dev/null 2>&1 && echo "$(date +%H:%M:%S) reconnected"
  fi
  sleep 30
done
