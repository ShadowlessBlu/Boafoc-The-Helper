"""Put the Wi-Fi name and password into secrets.h without an editor (the password is typed hidden).

  python firmware/esp32_room_node/set_wifi.py [--suggest "Network name"] [--broker-ip 192.168.x.y]
"""
from __future__ import annotations

import argparse
import getpass
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def set_define(text: str, name: str, value: str) -> str:
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    return re.sub(rf'(?m)^(#define {name}\s+)"[^"\n]*"', lambda m: f'{m.group(1)}"{value}"', text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suggest", default="", help="network name offered as the default")
    ap.add_argument("--broker-ip", default="", help="the Pi's current address, used if its name isn't found")
    a = ap.parse_args()
    path = HERE / "secrets.h"
    text = path.read_text(encoding="utf-8")
    hint = f" [{a.suggest}]" if a.suggest else ""
    ssid = input(f"Wi-Fi network name (2.4 GHz){hint}: ").strip() or a.suggest
    if not ssid:
        raise SystemExit("no network name given; nothing changed")
    pw = getpass.getpass("Wi-Fi password (hidden): ")
    if not pw:
        raise SystemExit("no password given; nothing changed")
    text = set_define(text, "WIFI_SSID", ssid)
    text = set_define(text, "WIFI_PASS", pw)
    if a.broker_ip:
        text = set_define(text, "MQTT_URI", f"mqtt://{a.broker_ip}:1883")
    path.write_text(text, encoding="utf-8")
    print(f"saved: network {ssid!r}, password ({len(pw)} characters, not shown)")


if __name__ == "__main__":
    main()
