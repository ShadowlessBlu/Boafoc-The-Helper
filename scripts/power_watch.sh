#!/bin/sh
# Power watch: why does the Pi keep switching off? (2026-11-20 and 2026-11-23: voltage normal right up
# to the end, then off.) Every 5 s it writes time, input voltage, throttle flags, temperature and load
# to logs/power/<date>.log, synced so the last line survives a sudden cut. At start it notes in
# logs/power/boots.log when this boot began and the last line written before it: a normal voltage
# there means the supply cut out (or the mains); a falling one means the supply sagged under load.
#
#   bash scripts/power_watch.sh            run in the foreground (the service runs this)
#   bash scripts/power_watch.sh --install  run it at every boot as a user service (no sudo)
#   bash scripts/power_watch.sh --report   boots and the last lines before each one
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/logs/power"
mkdir -p "$DIR"

if [ "${1:-}" = "--install" ]; then
  mkdir -p "$HOME/.config/systemd/user"
  cat > "$HOME/.config/systemd/user/power-watch.service" <<UNIT
[Unit]
Description=Log the Pi's input voltage, throttling and temperature (find the cause of power losses)

[Service]
ExecStart=/bin/sh $ROOT/scripts/power_watch.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable --now power-watch.service
  systemctl --user is-active power-watch.service
  exit 0
fi
if [ "${1:-}" = "--report" ]; then
  cat "$DIR/boots.log" 2>/dev/null || echo "no boots recorded yet"
  exit 0
fi

last=$(ls "$DIR"/20*.log 2>/dev/null | tail -1)
before=$( [ -n "$last" ] && tail -1 "$last" )
synced() { [ "$(timedatectl show -p NTPSynchronized --value 2>/dev/null)" = yes ]; }
# No RTC battery: until the clock syncs, the time is the one saved before the cut (a boot after a
# night off was logged as "yesterday 19:58"). Mark that, and add the real time once it syncs.
note=""; synced || note=" (clock not synced yet: times are from before the cut)"
echo "$(date '+%F %T') boot at $(uptime -s), up $(cut -d. -f1 /proc/uptime) s; last line before: ${before:-none} (${last##*/})$note" >> "$DIR/boots.log"
sync "$DIR/boots.log"
find "$DIR" -name '20*.log' -mtime +14 -delete 2>/dev/null

pending=${note:+1}
while true; do
  if [ -n "$pending" ] && synced; then
    echo "$(date '+%F %T') clock synced: that boot really began at $(uptime -s)" >> "$DIR/boots.log"
    sync "$DIR/boots.log"; pending=""
  fi
  f="$DIR/$(date +%F).log"
  v=$(vcgencmd pmic_read_adc EXT5V_V 2>/dev/null | sed 's/.*=//')
  th=$(vcgencmd get_throttled 2>/dev/null | sed 's/.*=//')
  t=$(vcgencmd measure_temp 2>/dev/null | sed "s/.*=//; s/'C//")
  echo "$(date +%T) ${v:-?} throttled=${th:-?} temp=${t:-?} load=$(cut -d' ' -f1 /proc/loadavg)" >> "$f"
  sync "$f"
  sleep 5
done
