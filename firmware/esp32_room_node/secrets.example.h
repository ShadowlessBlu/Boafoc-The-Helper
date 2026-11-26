// Copy to secrets.h (git-ignored) and fill in.
#pragma once
#define WIFI_SSID "your-wifi"                   // 2.4 GHz (the ESP32 has no 5 GHz)
#define WIFI_PASS "your-wifi-password"
#define BROKER_HOST "shadowlessblu"             // the Pi's name: found as shadowlessblu.local
#define MQTT_URI  "mqtt://192.168.1.221:1883"   // fallback if the name isn't found
#define MQTT_USER "office-node"
#define MQTT_PASS "node-password"               // set with scripts/add_mqtt_node.sh
