# ESP32 room node

A light relay and DHT22 temperature sensor, controlled by the assistant over MQTT.

| Pin | Use |
|---|---|
| GPIO 4 | Relay module IN (active-high; change `RELAY_ACTIVE_HIGH` if yours is active-low) |
| GPIO 5 | DHT22 data (10k pull-up to 3.3 V) |

## ESP32-S3 DevKit with its RGB LED (no relay)
With `LIGHT_RGB 1` (the default now) the board's RGB LED is the light: warm white = on, dark = off.
At boot: dim blue while joining Wi-Fi, a green flash when connected to MQTT, dim red if Wi-Fi failed.
Tested 2026-11-23 on an ESP32-S3 DevKit (CH343 USB-serial, COM5) as `office-node` / room `office`:
"Turn on the office light" → the assistant → MQTT → LED on, state back to the assistant.
Set `LIGHT_RGB 0` for a relay on `RELAY_PIN`, and `HAVE_DHT 1` for a DHT22.

The broker is found by name (`BROKER_HOST`, e.g. `shadowlessblu` = shadowlessblu.local over mDNS),
and looked up again after 20 s without MQTT, because a phone hotspot can give the Pi a new address.
`MQTT_URI` is only the fallback.

`set_wifi.py` writes the Wi-Fi name and password into `secrets.h` (password typed hidden).
The node's MQTT password can be generated and set without typing it:
`bash scripts/add_mqtt_node.sh office-node office <password-file>` (the file is deleted afterwards).

1. `copy secrets.example.h secrets.h` and fill in Wi-Fi and MQTT details.
2. Set `NODE_ID` and `ROOM` at the top of the sketch (`ROOM` must be one of `living_room`, `bedroom`, `kitchen`, `office`).
3. Build and flash with the Arduino IDE (board: your ESP32 variant), or:
   ```
   arduino-cli compile --fqbn esp32:esp32:esp32s3 esp32_room_node
   arduino-cli upload  --fqbn esp32:esp32:esp32s3 -p COM5 esp32_room_node
   ```
Temperature is only sent when the **DHT sensor library** and **Adafruit Unified Sensor** libraries are installed. Without them, the light and ping still work.

## Topics
| Topic | Direction | Payload |
|---|---|---|
| `home/<room>/light/set` | Pi → node | `on` / `off` (anything else is ignored) |
| `home/<room>/light/state` | node → Pi | `on` / `off` (retained) |
| `home/<room>/temperature` | node → Pi | `23.4` (retained, every 10 s) |
| `home/<node>/status` | node → Pi | `online` / `offline` (retained; `offline` is the last will) |
| `home/<node>/ping` → `/pong` | echo | anything (used by `bench/mqtt_latency.py`) |

## Safety
- The relay is switched **off** at every boot.
- Only the exact payloads `on` and `off` are accepted.
- Each node has its own MQTT username/password, and the broker ACL limits it to its own topics.
