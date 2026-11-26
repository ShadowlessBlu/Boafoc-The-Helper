#!/usr/bin/env bash
# Read-only system report for the Raspberry Pi. Changes nothing.
# Usage: bash scripts/baseline.sh > experiments/baseline-$(date +%F).md
set -u

have() { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n## %s\n\n```\n' "$1"; }
end() { printf '```\n'; }

echo "# Pi baseline — $(date '+%Y-%m-%d %H:%M')"

section "Board"
tr -d '\0' </proc/device-tree/model 2>/dev/null; echo
grep -m1 -i "revision" /proc/cpuinfo
echo "Cores: $(nproc)"
lscpu 2>/dev/null | grep -E "Model name|CPU max MHz|CPU min MHz"
end

section "OS"
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
uname -srm
echo "Hostname: $(hostname)  User: $(whoami)"
echo "Uptime: $(uptime -p)"
end

section "Memory"
free -h
grep -E "MemTotal|MemAvailable|SwapTotal" /proc/meminfo
swapon --show 2>/dev/null
[ -e /sys/block/zram0 ] && echo "zram0 present"
end

section "Storage"
df -h / /boot/firmware 2>/dev/null
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL 2>/dev/null
end

section "Thermals and power"
if have vcgencmd; then
  vcgencmd measure_temp
  vcgencmd measure_clock arm
  vcgencmd get_throttled
  echo "(get_throttled 0x0 = never throttled or under-volted since boot)"
  vcgencmd pmic_read_adc EXT5V_V 2>/dev/null
else
  echo "vcgencmd not found"
fi
ls /sys/class/hwmon/*/name 2>/dev/null | xargs -r cat
end

section "Network"
hostname -I
ip -br addr 2>/dev/null | grep -v "^lo"
end

section "Audio devices"
have arecord && arecord -l 2>&1 || echo "arecord not installed"
have aplay && aplay -l 2>&1 || echo "aplay not installed"
end

section "USB devices"
have lsusb && lsusb || echo "lsusb not installed"
end

section "Software"
python3 --version 2>&1
for c in git gcc cmake pip3 ollama llama-cli llama-bench docker mosquitto sysbench htop tmux; do
  if have "$c"; then printf '%-12s %s\n' "$c" "$(command -v "$c")"; else printf '%-12s %s\n' "$c" "-"; fi
done
have ollama && ollama --version 2>&1 && ollama list 2>&1
end

section "Groups for $(whoami)"
id
end

section "Load right now"
uptime
ps -eo pid,comm,%mem,%cpu --sort=-%mem | head -8
end
