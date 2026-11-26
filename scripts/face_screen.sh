#!/usr/bin/env bash
# The assistant's face on the Pi's own 7-inch screen, as a Python (pygame) window.
# It reads the assistant's /state, so it can start before the assistant: it shows grey
# "Assistant offline" eyes until the assistant answers. Other devices use the web page
# instead (python -m assistant --view-link).
#
#   bash scripts/face_screen.sh                 open it now (works over SSH: uses the desktop session)
#   bash scripts/face_screen.sh --demo          cycle the four states, without the assistant
#   bash scripts/face_screen.sh --stop          close it
#   bash scripts/face_screen.sh --autostart     also open it at every login
#   bash scripts/face_screen.sh --no-autostart  stop opening it at login
#   bash scripts/face_screen.sh --desktop-icon [icon.png]  put an "Assistant" icon on the Pi desktop
#                                               that starts the assistant and this screen (after Settings > Exit)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP="$HOME/.config/autostart/assistant-screen.desktop"
ICON="$HOME/Desktop/assistant-screen.desktop"
PATTERN="python -m assistant.screen"
ARGS=()
ENTRY="[Desktop Entry]\nType=Application\nName=Assistant screen\nComment=Open the assistant's face and dashboard\nIcon=face-smile\nExec=bash $ROOT/scripts/face_screen.sh\nTerminal=false\nX-GNOME-Autostart-enabled=true\n"
ICON_IMG="${2:-$ROOT/assets/assistant-icon.png}"
[ -f "$ICON_IMG" ] || ICON_IMG=face-smile
LAUNCHER="[Desktop Entry]\nType=Application\nName=Assistant\nComment=Start the assistant and its screen\nIcon=$ICON_IMG\nExec=bash $ROOT/scripts/start_assistant.sh\nTerminal=false\n"

case "${1:-}" in
  --stop) pkill -f -- "$PATTERN" && echo "closed" || echo "not running"; exit 0 ;;
  --no-autostart) rm -f "$DESKTOP" && echo "removed $DESKTOP"; exit 0 ;;
  --demo) ARGS+=(--demo) ;;
  --autostart)
    mkdir -p "$(dirname "$DESKTOP")"
    printf "$ENTRY" > "$DESKTOP"
    echo "added $DESKTOP" ;;
  --desktop-icon)
    mkdir -p "$(dirname "$ICON")"
    printf "$LAUNCHER" > "$ICON"
    chmod +x "$ICON"
    gio set "$ICON" metadata::trusted true 2>/dev/null || true
    echo "added $ICON"; exit 0 ;;
esac

# Over SSH there is no display in the environment: borrow the desktop session's.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
export DISPLAY=${DISPLAY:-:0}   # fallback through Xwayland if SDL has no Wayland support

pkill -f -- "$PATTERN" 2>/dev/null || true
cd "$ROOT"
PYTHONPATH="$ROOT/src" nohup "$ROOT/.venv/bin/python" -m assistant.screen "${ARGS[@]}" \
  > "${XDG_RUNTIME_DIR}/assistant-screen.log" 2>&1 &
echo "screen open (log: ${XDG_RUNTIME_DIR}/assistant-screen.log)"
