"""Connects the Devices layer to real hardware over MQTT (needs `pip install paho-mqtt>=2`).

Topic scheme (matches firmware/esp32_room_node):
  <base>/<room_slug>/light/set      assistant -> device   "on" | "off"
  <base>/<room_slug>/light/state    device -> assistant   "on" | "off"   (retained)
  <base>/<room_slug>/temperature    device -> assistant   "23.4"         (retained)
  <base>/<node>/status              device -> assistant   "online" | "offline" (retained, last will)
"""
from __future__ import annotations

import logging
import threading

from .tools import ROOMS, Devices

log = logging.getLogger(__name__)

SLUG_TO_ROOM = {r.replace(" ", "_"): r for r in ROOMS}


def handle_message(devices: Devices, online: dict[str, str], base: str, topic: str, payload: bytes) -> str | None:
    """Apply one incoming message to the device state. Returns what changed, or None if ignored."""
    parts = topic.split("/")
    if len(parts) < 3 or parts[0] != base:
        return None
    text = payload.decode("utf-8", "replace").strip()
    if parts[-1] == "status" and len(parts) == 3:
        online[parts[1]] = text
        return f"{parts[1]} {text}"
    room = SLUG_TO_ROOM.get(parts[1])
    if room is None:
        return None
    if parts[2:] == ["temperature"]:
        try:
            value = float(text)
        except ValueError:
            return None
        if not -40 <= value <= 85:   # outside any real indoor sensor range
            return None
        devices.temps[room] = value
        return f"{room} temperature {value}"
    if parts[2:] == ["light", "state"] and text in ("on", "off"):
        devices.report_light(room, text)
        return f"{room} light {text}"
    return None


class MqttBridge:
    def __init__(self, devices: Devices, host: str, port: int = 1883, username: str = "",
                 password: str = "", base: str = "home", client_id: str = "pi5-assistant"):
        import paho.mqtt.client as mqtt  # optional dependency

        self.devices, self.base = devices, base
        self.online: dict[str, str] = {}
        self.connected = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        if username:
            self.client.username_pw_set(username, password or None)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = lambda *a: self.connected.clear()
        self.client.on_message = self._on_message
        # Retry every 1-5 s: at boot the broker may not be up yet, and with up to 30 s between tries the
        # assistant reached it only ~15 s after starting (a light tap then failed, 2026-11-23).
        self.client.reconnect_delay_set(1, 5)
        self.client.connect_async(host, port, keepalive=30)
        self.client.loop_start()
        devices.publish = self.publish
        devices.confirm_s = 6.0   # a light only counts once its device confirms; on the phone hotspot
                                  # most answers take 10-60 ms but one took 3.4 s (MQTT resends), 2026-11-23

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.connected.set()
            client.subscribe(f"{self.base}/#", qos=1)
            log.info("MQTT connected")
        else:
            log.warning("MQTT connect failed: %s", reason_code)

    def _on_message(self, client, userdata, msg):
        change = handle_message(self.devices, self.online, self.base, msg.topic, msg.payload)
        if change:
            log.debug("MQTT %s", change)

    CONNECT_WAIT_S = 3.0   # a command right after start-up or a broker restart waits this long for MQTT
    NOT_CONNECTED = "I'm not connected to the home devices right now. Try again in a few seconds."

    def publish(self, topic: str, payload: str) -> None:
        if not self.connected.wait(self.CONNECT_WAIT_S):
            raise ConnectionError(self.NOT_CONNECTED)
        info = self.client.publish(topic, payload, qos=1)
        info.wait_for_publish(timeout=3)

    def status(self) -> str:
        return "connected" if self.connected.is_set() else "disconnected"

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
