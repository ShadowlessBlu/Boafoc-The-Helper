// ESP32 room node for the Pi5 AI Assistant.
// - Light controlled over MQTT:         home/<room>/light/set  ("on"/"off")  -> state on home/<room>/light/state
//   either a relay on RELAY_PIN, or (LIGHT_RGB 1) the board's RGB LED, e.g. on an ESP32-S3 DevKit
// - Temperature every 10 s (DHT22):     home/<room>/temperature
// - Online/offline with MQTT last will: home/<node>/status
// - Latency echo for the benchmark:     home/<node>/ping -> home/<node>/pong
// Uses the MQTT client built into the ESP32 Arduino core (ESP-IDF esp-mqtt), so no extra MQTT library.
// Copy secrets.example.h to secrets.h and fill it in before building.

#include <WiFi.h>
#include <ESPmDNS.h>
#include "mqtt_client.h"
#include "secrets.h"

// ---------- node settings ----------
#define NODE_ID   "office-node"
#define ROOM      "office"             // must match a room in the assistant (spaces -> _)
#define TEMP_PERIOD_MS 10000

// The light: 1 = the board's RGB LED (ESP32-S3 DevKit: RGB_BUILTIN, GPIO 48 on most boards);
//            0 = a relay module on RELAY_PIN
#define LIGHT_RGB 1
// GPIO 48 directly: through the RGB_BUILTIN alias (97 = 49 + 48) the LED stayed dark on this
// board, while the same calls on 48 work (firmware/led_finder, 2026-11-23).
#define RGB_PIN   48
#define RGB_ON_R 90                      // warm white, not full brightness
#define RGB_ON_G 60
#define RGB_ON_B 25
#define RELAY_PIN 4
#define RELAY_ACTIVE_HIGH true
#define DHT_PIN   5

// 1 = DHT22 temperature (needs "DHT sensor library" + "Adafruit Unified Sensor"); 0 = light only
#define HAVE_DHT 0

// The broker is found by name (mDNS, e.g. "shadowlessblu" = shadowlessblu.local): a phone hotspot
// or router may give the Pi a new address after a reboot. MQTT_URI in secrets.h is the fallback.
#ifndef BROKER_HOST
  #define BROKER_HOST ""
#endif
#if HAVE_DHT
  #include <Adafruit_Sensor.h>
  #include <DHT.h>
  DHT dht(DHT_PIN, DHT22);
#endif

static const char *T_SET    = "home/" ROOM "/light/set";
static const char *T_STATE  = "home/" ROOM "/light/state";
static const char *T_TEMP   = "home/" ROOM "/temperature";
static const char *T_STATUS = "home/" NODE_ID "/status";
static const char *T_PING   = "home/" NODE_ID "/ping";
static const char *T_PONG   = "home/" NODE_ID "/pong";

static esp_mqtt_client_handle_t mqtt = nullptr;
static volatile bool mqttUp = false;
static volatile uint32_t mqttDownSince = 0;
static bool lightOn = false;
static char brokerUri[96] = MQTT_URI;

static void rgb(uint8_t r, uint8_t g, uint8_t b) {
#if LIGHT_RGB
  rgbLedWrite(RGB_PIN, r, g, b);
#endif
}

static void showLight() {
#if LIGHT_RGB
  if (lightOn) rgb(RGB_ON_R, RGB_ON_G, RGB_ON_B); else rgb(0, 0, 0);
#else
  digitalWrite(RELAY_PIN, (lightOn == RELAY_ACTIVE_HIGH) ? HIGH : LOW);
#endif
}

static void setLight(bool on) {
  lightOn = on;
  showLight();
  Serial.printf("light %s\n", on ? "on" : "off");
  if (mqttUp) esp_mqtt_client_publish(mqtt, T_STATE, on ? "on" : "off", 0, 1, 1);
}

// Look the broker up by name; true if the URI changed.
static bool resolveBroker() {
  if (strlen(BROKER_HOST) == 0) return false;
  IPAddress ip = MDNS.queryHost(BROKER_HOST, 3000);
  if (ip == IPAddress(0, 0, 0, 0)) { Serial.printf("%s.local not found, using %s\n", BROKER_HOST, brokerUri); return false; }
  char uri[96];
  snprintf(uri, sizeof uri, "mqtt://%s:1883", ip.toString().c_str());
  if (strcmp(uri, brokerUri) == 0) return false;
  strncpy(brokerUri, uri, sizeof brokerUri - 1);
  Serial.printf("broker %s.local = %s\n", BROKER_HOST, brokerUri);
  return true;
}

static bool topicIs(const esp_mqtt_event_handle_t e, const char *t) {
  return e->topic_len == (int)strlen(t) && strncmp(e->topic, t, e->topic_len) == 0;
}

static void onMqtt(void *, esp_event_base_t, int32_t id, void *data) {
  auto e = (esp_mqtt_event_handle_t)data;
  switch ((esp_mqtt_event_id_t)id) {
    case MQTT_EVENT_CONNECTED:
      mqttUp = true;
      esp_mqtt_client_publish(mqtt, T_STATUS, "online", 0, 1, 1);
      esp_mqtt_client_subscribe(mqtt, T_SET, 1);
      esp_mqtt_client_subscribe(mqtt, T_PING, 0);
      esp_mqtt_client_publish(mqtt, T_STATE, lightOn ? "on" : "off", 0, 1, 1);
      Serial.println("MQTT connected");
      rgb(0, 40, 0); delay(250); showLight();   // green flash: connected
      break;
    case MQTT_EVENT_DISCONNECTED:
      if (mqttUp) mqttDownSince = millis();
      mqttUp = false;
      Serial.println("MQTT disconnected");
      break;
    case MQTT_EVENT_DATA:
      if (topicIs(e, T_PING)) {
        esp_mqtt_client_publish(mqtt, T_PONG, e->data, e->data_len, 0, 0);
      } else if (topicIs(e, T_SET)) {
        // Only accept exactly "on" or "off"; ignore anything else.
        if (e->data_len == 2 && strncmp(e->data, "on", 2) == 0) setLight(true);
        else if (e->data_len == 3 && strncmp(e->data, "off", 3) == 0) setLight(false);
        else Serial.println("ignored malformed light command");
      }
      break;
    default:
      break;
  }
}

static void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // no Wi-Fi power saving: on a phone hotspot the Pi lost the board after a while
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  rgb(0, 0, 25);                                 // dim blue: joining Wi-Fi
  for (int i = 0; i < 60 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print('.'); }
  Serial.printf(" %s\n", WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString().c_str() : "not connected yet");
  if (WiFi.status() != WL_CONNECTED) rgb(25, 0, 0);   // dim red: no Wi-Fi yet (keeps trying)
  else showLight();
}

void setup() {
  Serial.begin(115200);
#if !LIGHT_RGB
  pinMode(RELAY_PIN, OUTPUT);
#endif
  // Boot self-test: red, green, blue, so you can see the LED works.
  rgb(40, 0, 0); delay(250); rgb(0, 40, 0); delay(250); rgb(0, 0, 40); delay(250);
  setLight(false);                 // fail-safe: light off at boot
#if HAVE_DHT
  dht.begin();
#endif
  connectWiFi();
  MDNS.begin(NODE_ID);
  resolveBroker();

  esp_mqtt_client_config_t cfg = {};
  cfg.broker.address.uri = brokerUri;
  cfg.credentials.username = MQTT_USER;
  cfg.credentials.client_id = NODE_ID;
  cfg.credentials.authentication.password = MQTT_PASS;
  cfg.session.keepalive = 30;
  cfg.session.last_will.topic = T_STATUS;
  cfg.session.last_will.msg = "offline";
  cfg.session.last_will.qos = 1;
  cfg.session.last_will.retain = 1;
  mqtt = esp_mqtt_client_init(&cfg);
  esp_mqtt_client_register_event(mqtt, MQTT_EVENT_ANY, onMqtt, nullptr);
  esp_mqtt_client_start(mqtt);     // reconnects by itself
}

void loop() {
  static uint32_t last = 0, lastLookup = 0;
  // Disconnected for 20 s: the Pi may have a new address; look it up again (at most every 20 s).
  if (!mqttUp && millis() - lastLookup > 20000 && WiFi.status() == WL_CONNECTED) {
    lastLookup = millis();
    if (resolveBroker()) {
      esp_mqtt_client_set_uri(mqtt, brokerUri);
      esp_mqtt_client_reconnect(mqtt);
    }
  }
  if (millis() - last >= TEMP_PERIOD_MS) {
    last = millis();
#if HAVE_DHT
    float t = dht.readTemperature();
    if (!isnan(t) && mqttUp) {
      char buf[16];
      snprintf(buf, sizeof buf, "%.1f", t);
      esp_mqtt_client_publish(mqtt, T_TEMP, buf, 0, 0, 1);
    }
#endif
  }
  delay(20);
}
