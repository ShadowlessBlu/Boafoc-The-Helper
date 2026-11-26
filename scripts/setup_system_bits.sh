#!/usr/bin/env bash
# One-off system setup that needs sudo (asks for the password once):
#   1. microphone: Auto Gain Control off, saved so it survives reboots (alsactl store)
#   2. MQTT broker (Mosquitto): password login, per-user topic rules, enabled at boot.
#      The assistant's MQTT password is generated here and written into config.toml (mode 600);
#      it is never printed. Then the assistant is restarted with MQTT on and a test message is sent.
# Usage (on the Pi, in a terminal):  bash scripts/setup_system_bits.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MIC=${MIC_CARD:-Device}

echo "==> 1. Microphone: Auto Gain Control off, and saved"
amixer -q -c "$MIC" sset "Auto Gain Control" off
amixer -c "$MIC" sget "Auto Gain Control" | tail -1
sudo alsactl store
echo "   saved to /var/lib/alsa/asound.state (restored at every boot)"

echo "==> 2. MQTT broker"
sudo apt-get install -y mosquitto mosquitto-clients
sudo tee /etc/mosquitto/conf.d/assistant.conf >/dev/null <<'EOF'
listener 1883
allow_anonymous false
password_file /etc/mosquitto/passwd
acl_file /etc/mosquitto/acl
EOF
if [ ! -f /etc/mosquitto/acl ]; then
  sudo tee /etc/mosquitto/acl >/dev/null <<'EOF'
# The assistant (and the latency benchmark) may use everything under home/
user assistant
topic readwrite home/#
EOF
fi
PW=$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')
if sudo test -f /etc/mosquitto/passwd; then
  sudo mosquitto_passwd -b /etc/mosquitto/passwd assistant "$PW"
else
  sudo mosquitto_passwd -c -b /etc/mosquitto/passwd assistant "$PW"
fi
# Owner root, group mosquitto, mode 640: Mosquitto 2.0.21 warns otherwise, and future versions
# refuse a password file not owned by root.
sudo chown root:mosquitto /etc/mosquitto/passwd /etc/mosquitto/acl
sudo chmod 640 /etc/mosquitto/passwd /etc/mosquitto/acl
sudo systemctl enable --now mosquitto
sudo systemctl restart mosquitto

# Put the password into config.toml ([mqtt] enabled = true, password = ...) without printing it.
MQTT_PW="$PW" python3 - <<'EOF'
import os, re
from pathlib import Path
p = Path("config.toml"); s = p.read_text()
head, sep, rest = s.partition("[mqtt]")
nxt = rest.find("\n[")
sec, tail = (rest[:nxt], rest[nxt:]) if nxt != -1 else (rest, "")
sec = re.sub(r"(?m)^enabled = .*$", "enabled = true                # Mosquitto installed by scripts/setup_system_bits.sh", sec)
sec = re.sub(r"(?m)^password = .*$", 'password = "' + os.environ["MQTT_PW"] + '"', sec)
p.write_text(head + sep + sec + tail); p.chmod(0o600)
print("   config.toml: [mqtt] enabled, password written (not shown)")
EOF

echo "==> 3. Test: publish and receive one message with the assistant's login"
timeout 5 mosquitto_sub -h 127.0.0.1 -u assistant -P "$PW" -t home/test -C 1 > /tmp/mqtt_test.txt &
sleep 1
mosquitto_pub -h 127.0.0.1 -u assistant -P "$PW" -t home/test -m "hello from the assistant"
wait || true
if grep -q "hello" /tmp/mqtt_test.txt; then echo "   MQTT works"; else echo "   MQTT test FAILED"; fi
rm -f /tmp/mqtt_test.txt
unset PW

echo "==> 4. Restart the assistant with MQTT on"
systemctl --user restart assistant
sleep 25
curl -s -m 5 http://127.0.0.1:8765/health; echo
echo "Done. Add each ESP32 later with: bash scripts/add_mqtt_node.sh <node-id> <room_slug>"
