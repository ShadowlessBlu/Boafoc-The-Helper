#!/usr/bin/env bash
# Create an MQTT login for one ESP32 node, limited to its own topics.
# Usage: bash scripts/add_mqtt_node.sh livingroom-node living_room [password-file]
#        (password-file: read, set, then deleted; otherwise you are asked to type one)
set -euo pipefail
NODE=${1:?node id, e.g. livingroom-node}
ROOM=${2:?room slug, e.g. living_room}
[[ "$NODE" =~ ^[a-z0-9-]+$ && "$ROOM" =~ ^[a-z0-9_]+$ ]] || { echo "node: a-z0-9-   room: a-z0-9_"; exit 1; }

if sudo grep -q "^user $NODE\$" /etc/mosquitto/acl; then
  echo "ACL for $NODE already exists"
else
  sudo tee -a /etc/mosquitto/acl >/dev/null <<EOF

user $NODE
topic read home/$ROOM/light/set
topic write home/$ROOM/light/state
topic write home/$ROOM/temperature
topic write home/$NODE/status
topic read home/$NODE/ping
topic write home/$NODE/pong
EOF
fi
PWFILE=${3:-}
if [ -n "$PWFILE" ]; then
  # Password generated elsewhere (already in the node's secrets.h): read it, never print it, delete the file.
  sudo mosquitto_passwd -b /etc/mosquitto/passwd "$NODE" "$(cat "$PWFILE")"
  shred -u "$PWFILE" 2>/dev/null || rm -f "$PWFILE"
else
  echo "Choose a password for $NODE (goes in the node's secrets.h as MQTT_PASS):"
  sudo mosquitto_passwd /etc/mosquitto/passwd "$NODE"
fi
sudo chown root:mosquitto /etc/mosquitto/passwd /etc/mosquitto/acl   # what Mosquitto expects
sudo chmod 640 /etc/mosquitto/passwd /etc/mosquitto/acl
sudo systemctl restart mosquitto
echo "Done. Node '$NODE' may only use topics for room '$ROOM' and its own status/ping."
