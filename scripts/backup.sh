#!/usr/bin/env bash
# Back up the assistant's data: the database (questions/replies, saved notes, audit log), config.toml,
# the screen's wake word settings and the experiment results. Keeps the newest 7. Each backup is
# checked right after it is made: the database copy is opened, checked for damage and counted, so a
# backup that would not restore is noticed the same night (in logs/backup.log).
#
#   bash scripts/backup.sh [backup dir]   back up now (default ~/backups/assistant)
#   bash scripts/backup.sh --install      run it every night at 03:30 as a user timer (no sudo; a night
#                                         missed because the Pi was off runs at the next boot)
#   bash scripts/backup.sh --check <file> check an existing backup (opens its database, counts rows)
#
# Restore (see the vault's Troubleshooting note):
#   systemctl --user stop assistant.service
#   tar -xzf ~/backups/assistant/assistant-<date>.tgz -C ~/assistant
#   systemctl --user start assistant.service
# The PC pulls the newest backup every week (scripts/pull_backup.ps1 on the PC).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

check() {  # <backup.tgz>: open the database inside it, run integrity_check, count rows
  local tmp; tmp=$(mktemp -d)
  tar -xzf "$1" -C "$tmp"
  "$PY" - "$tmp/data/assistant.db" <<'EOF'
import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
ok = db.execute("PRAGMA integrity_check").fetchone()[0]
counts = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("messages", "memories", "audit")}
print(f"check: database {ok}; " + ", ".join(f"{k} {v}" for k, v in counts.items()))
sys.exit(0 if ok == "ok" else 1)
EOF
  local rc=$?
  [ -f "$tmp/config.toml" ] && echo "check: config.toml present" || echo "check: no config.toml in it"
  rm -rf "$tmp"
  return $rc
}

if [ "${1:-}" = "--install" ]; then
  mkdir -p "$HOME/.config/systemd/user"
  cat > "$HOME/.config/systemd/user/assistant-backup.service" <<UNIT
[Unit]
Description=Back up the assistant's data (database, config, wake word settings)

[Service]
Type=oneshot
WorkingDirectory=$ROOT
ExecStart=/bin/sh -c 'bash $ROOT/scripts/backup.sh >> $ROOT/logs/backup.log 2>&1'
UNIT
  cat > "$HOME/.config/systemd/user/assistant-backup.timer" <<UNIT
[Unit]
Description=Nightly backup of the assistant's data

[Timer]
OnCalendar=*-*-* 03:30
Persistent=true

[Install]
WantedBy=timers.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable --now assistant-backup.timer
  systemctl --user list-timers assistant-backup.timer --no-pager
  exit 0
fi
if [ "${1:-}" = "--check" ]; then
  check "$2"
  exit $?
fi

DEST=${1:-$HOME/backups/assistant}
KEEP=${KEEP:-7}
mkdir -p "$DEST" logs
STAMP=$(date +%Y%m%d-%H%M%S)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Consistent copy of the live SQLite database (safe while the assistant is running)
if [ -f data/assistant.db ]; then
  mkdir -p "$TMP/data"
  "$PY" - "$TMP/data/assistant.db" <<'EOF'
import sqlite3, sys
src = sqlite3.connect("data/assistant.db")
dst = sqlite3.connect(sys.argv[1])
src.backup(dst)
dst.close(); src.close()
EOF
fi
for f in data/*.json; do [ -f "$f" ] && mkdir -p "$TMP/data" && cp "$f" "$TMP/data/"; done
[ -f config.toml ] && cp config.toml "$TMP/"
[ -d experiments/results ] && mkdir -p "$TMP/experiments" && cp -r experiments/results "$TMP/experiments/"

OUT="$DEST/assistant-$STAMP.tgz"
tar -czf "$OUT.part" -C "$TMP" .
sync "$OUT.part"
mv "$OUT.part" "$OUT"                     # a power cut mid-way leaves only a .part, never a broken .tgz
echo "$(date '+%F %T') backup: $OUT ($(du -h "$OUT" | cut -f1))"
if ! check "$OUT"; then
  echo "$(date '+%F %T') BACKUP CHECK FAILED: $OUT"
  exit 1
fi
rm -f "$DEST"/*.part
ls -1t "$DEST"/assistant-*.tgz | tail -n +$((KEEP + 1)) | xargs -r rm --
