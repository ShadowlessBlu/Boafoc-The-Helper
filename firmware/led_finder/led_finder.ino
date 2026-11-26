// Find which pin drives the board's RGB LED (WS2812): cycles red, green, blue on each candidate
// pin and prints what it is doing. ESP32-S3 DevKit and most copies use GPIO 48 (some: GPIO 38).
// Some copies (e.g. YD-ESP32-S3) only connect the LED after bridging the "RGB" solder pads.
// Only safe pins are tried: not the flash/PSRAM pins (26-37), USB (19/20) or the serial port (43/44).

const int PINS[] = {48, 38};
const char *NAMES[] = {"red", "green", "blue"};

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("LED finder: watch the board");
}

void loop() {
  for (int p : PINS) {
    for (int c = 0; c < 3; c++) {
      Serial.printf("GPIO %d: %s\n", p, NAMES[c]);
      rgbLedWrite(p, c == 0 ? 60 : 0, c == 1 ? 60 : 0, c == 2 ? 60 : 0);
      delay(2000);
    }
    rgbLedWrite(p, 0, 0, 0);
    Serial.printf("GPIO %d: off\n", p);
    delay(1500);
  }
}
