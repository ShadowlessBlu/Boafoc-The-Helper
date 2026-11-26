#!/usr/bin/env bash
# Show the assistant's face full screen on the Pi's 7-inch DSI screen (Chromium kiosk).
# The assistant must be running (python -m assistant --api or --service) for live data;
# until it is, the page shows "Assistant offline" and keeps retrying.
#
#   bash scripts/face_kiosk.sh              open it now (works over SSH: uses the desktop session)
#   bash scripts/face_kiosk.sh --demo       cycle the four states, without the assistant
#   bash scripts/face_kiosk.sh --stop       close it
#   bash scripts/face_kiosk.sh --autostart     also open it at every login
#   bash scripts/face_kiosk.sh --no-autostart  stop opening it at login
#
# Autostart uses ~/.config/autostart/assistant-face.desktop, which the desktop session runs
# (lxsession-xdg-autostart) in addition to its own programs. A personal ~/.config/labwc/autostart
# could replace the system one and lose the panel, so it is not used.
set -euo pipefail
PORT=${PORT:-8765}
URL="http://127.0.0.1:${PORT}/face"
PROFILE="$HOME/.cache/assistant-face"
DESKTOP="$HOME/.config/autostart/assistant-face.desktop"
SELF="$(cd "$(dirname "$0")" && pwd)/face_kiosk.sh"

case "${1:-}" in
  --stop) pkill -f -- "--user-data-dir=$PROFILE" && echo "closed" || echo "not running"; exit 0 ;;
  --no-autostart) rm -f "$DESKTOP" && echo "removed $DESKTOP"; exit 0 ;;
  --demo) URL="$URL?demo" ;;
  --autostart)
    mkdir -p "$(dirname "$DESKTOP")"
    printf '[Desktop Entry]\nType=Application\nName=Assistant face\nExec=bash %s\nX-GNOME-Autostart-enabled=true\n' \
      "$SELF" > "$DESKTOP"
    echo "added $DESKTOP" ;;
esac

# Over SSH there is no display in the environment: borrow the desktop session's.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}

# The page comes from the assistant, so wait for it (at boot it starts later, or not yet at all).
# Once loaded, the page itself shows "Assistant offline" if the assistant stops, and recovers.
if ! curl -sf -o /dev/null --max-time 2 "${URL%%\?*}"; then
  echo "waiting for the assistant on port $PORT (start it with: .venv/bin/python -m assistant --api or --service)"
  until curl -sf -o /dev/null --max-time 2 "${URL%%\?*}"; do sleep 2; done
fi

pkill -f -- "--user-data-dir=$PROFILE" 2>/dev/null || true
sleep 1
# A separate profile, so it never touches the normal browser profile and can be closed on its own.
nohup chromium --kiosk --app="$URL" --user-data-dir="$PROFILE" --ozone-platform=wayland \
  --noerrdialogs --disable-infobars --no-first-run --password-store=basic \
  --disable-session-crashed-bubble --check-for-update-interval=31536000 \
  >/dev/null 2>&1 &
echo "face open on the screen: $URL"
