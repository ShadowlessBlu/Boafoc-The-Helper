#!/usr/bin/env bash
# Turn off SSH password login, so only SSH keys get in (admin and recovery plan, 2026-11-25).
# Run once, with sudo, from the PC (PowerShell):
#   ssh -t pi5 "sudo bash ~/assistant/scripts/ssh_key_only.sh"
# Undo:  ssh -t pi5 "sudo bash ~/assistant/scripts/ssh_key_only.sh --undo"
# If the PC's key is ever lost: log in on the Pi itself (keyboard and screen, the password still works
# there), run the undo line, then copy a new key with copy-ssh-key-to-pi.ps1.
#
# Safety: it refuses unless the user already has an authorized key, checks the new config with
# `sshd -t` before reloading, and a reload keeps open sessions, so this one stays connected.
set -euo pipefail
CONF=/etc/ssh/sshd_config.d/10-key-only.conf   # sorts before 50-cloud-init.conf: the first value wins
USER_NAME=${SUDO_USER:-shadowlessblu}
KEYS=$(getent passwd "$USER_NAME" | cut -d: -f6)/.ssh/authorized_keys

[ "$(id -u)" = 0 ] || { echo "Run it with sudo."; exit 1; }
if [ "${1:-}" = "--undo" ]; then
  rm -f "$CONF"
  sshd -t && systemctl reload ssh
  echo "Password login is allowed again."
  exit 0
fi
grep -qE '^(ssh-|ecdsa-)' "$KEYS" 2>/dev/null || { echo "No key in $KEYS: not changing anything."; exit 1; }
cat > "$CONF" <<'EOF'
# Key-only SSH (scripts/ssh_key_only.sh, 2026-11-25). Remove this file to allow passwords again.
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
if ! sshd -t; then
  rm -f "$CONF"
  echo "sshd rejected the config; nothing changed."
  exit 1
fi
systemctl reload ssh
echo "Done: SSH now accepts keys only ($(grep -cE '^(ssh-|ecdsa-)' "$KEYS") key(s) in $KEYS)."
sshd -T 2>/dev/null | grep -E '^(passwordauthentication|kbdinteractiveauthentication|pubkeyauthentication) '
