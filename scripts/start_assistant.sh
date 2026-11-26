#!/usr/bin/env bash
# Start the whole assistant from the desktop icon: the voice/API service, then the screen.
# (At boot both start by themselves: the user service is enabled and the screen has an autostart
# entry. This is for after "Exit" in the screen's Settings, which stops both.)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
systemctl --user start assistant.service
exec bash "$ROOT/scripts/face_screen.sh"
