# Boafoc v2: Pi5 AI Assistant

**Boafoc** is an offline-first voice assistant for low-power edge devices. Version 2 runs entirely on a Raspberry Pi 5: wake words trained on real recordings (Uzi, Aiya, Boafo), whisper.cpp speech-to-text, a local LLM through Ollama (qwen3:1.7b or a distilled qwen3:0.6b), Piper text-to-speech, a Governor that checks every action, MQTT control of ESP32 devices, and a 7-inch touch dashboard. It works with the internet unplugged.

Version 1 (May 2026) explored African-language speech (Twi and Ewe). v2 is English-only for now; bringing those languages back onto this pipeline is planned.

Write-up: https://www.hackster.io/calvinsewornu123/boafoc-the-helper-multilingual-african-ai-assistant-7841a0

Not included here: trained wake-word models, training recordings, benchmark result files and `config.toml`/`secrets.h` (copy the `*.example.*` files). Licence: Apache 2.0.

> **Start here: [HANDOVER.md](HANDOVER.md)**: how it runs now (services, logs, everyday commands), what's still open, and the history.

An offline, modular personal AI assistant for a Raspberry Pi 5 (8 GB), plus the benchmark tools for experiment EXP-001 (small language models on the Pi).

The core code uses only the Python standard library (3.11+). It talks to a local [Ollama](https://github.com/ollama/ollama) server over HTTP.

## Layout

| Path | What it is |
|---|---|
| `src/assistant/` | The assistant: LLM client, Brain, Governor, tools, memory |
| `src/assistant/api.py` | HTTP API: `/health`, `/ask` (token), `/state`, `/history`, `/mute`, `/light`, `/face` (web face for other devices) |
| `src/assistant/screen.py` | Touch dashboard for the Pi's 7-inch screen (pygame): face, clock, history, home lights, Pi status, settings, wake words (on/off, thresholds, real/false marks) |
| `src/assistant/voice/wake_control.py` | Wake words switched and tuned from the screen; per-day event log in `logs/wake/` |
| `src/assistant/status.py` | What the assistant is doing + Pi readings, for the screen and `/state` |
| `src/assistant/mqtt_bridge.py` | Connects device tools to ESP32 nodes over MQTT |
| `src/assistant/voice/` | Mic, Silero/energy VAD, openWakeWord (trained Uzi/Aiya/Boafo), whisper.cpp, Piper, voice loop, transcript fixes, mic calibration |
| `firmware/esp32_room_node/` | ESP32 room node: light (relay or the S3's RGB LED), optional DHT22, MQTT with confirmation, mDNS broker lookup |
| `training/` | Wake-word training and the guided recording session (run on the PC) |
| `research/` | EXP-001 Q4/Q5: layer pruning, distillation data, LoRA training |
| `tests/` | Unit tests (`python -m unittest discover -s tests`); no Pi or LLM needed |
| `scripts/baseline.sh` | Read-only system report of the Pi (Markdown) |
| `scripts/setup_base.sh` | Base packages + venv on the Pi |
| `scripts/install_llm.sh` | Ollama, llama.cpp and starter models |
| `scripts/install_voice.sh` | Vosk, whisper.cpp, Piper, openWakeWord, Mosquitto (password + ACL) |
| `scripts/add_mqtt_node.sh` | MQTT login for one ESP32, limited to its own topics |
| `scripts/install_user_service.sh` | The assistant as a user service at boot (`--voice` for the full assistant); what the Pi uses |
| `scripts/install_service.sh` | System service + nightly backup timer (older option, no Bluetooth audio) |
| `scripts/face_screen.sh` | Open/close the touch screen, autostart, desktop icon |
| `scripts/start_assistant.sh` | What the desktop icon runs: service + screen |
| `scripts/setup_system_bits.sh` | sudo bundle: mic gain, Mosquitto, speaker reconnect |
| `scripts/power_watch.sh` | Logs voltage/throttling/temperature to find the cause of power losses |
| `bench/offline_test.py` | Full offline test with the internet unplugged (start with `bench/offline_test.sh`; `--now --quiet` = dry run) |
| `scripts/backup.sh` | Safe SQLite backup + config + wake settings + results, keeps 7, checks each one; `--install` = nightly user timer |
| `scripts/pull_backup.ps1` | (PC) copy the Pi's newest backup to the PC; `-Install` = weekly Task Scheduler job |
| `scripts/ssh_key_only.sh` | Turn off SSH password login (run once with sudo; `--undo`) |
| `scripts/sync_to_pi.ps1` | Copy this folder from the PC to the Pi |
| `bench/mqtt_latency.py` | MQTT round-trip latency to a node, QoS 0 vs 1 |
| `bench/kv_calc.py` | Predict weight RAM and KV cache size (EXP-001 Q1, Q6) |
| `bench/bench_ollama.py` | Speed / RAM / temperature benchmark via Ollama |
| `bench/bench_llamacpp.sh` | `llama-bench` runs + peak RAM per quant and context |
| `experiments/test_set_v1.json`, `test_set_v2.json` | 20- and 60-question assistant test sets |
| `experiments/score.py` | Runs the test set against a model and scores it |
| `prompts/` | Fixed benchmark prompts and the assistant's system prompt |
| `deploy/assistant.service` | systemd unit (system-service option) |
| `bench/ask.py` | Ask the running assistant a typed question through the API, with timing |

## Quick start (on the Pi)

```bash
# 1. from the PC (PowerShell):  .\scripts\sync_to_pi.ps1
# 2. on the Pi:
cd ~/assistant
bash scripts/baseline.sh > experiments/baseline-$(date +%F).md
bash scripts/setup_base.sh
bash scripts/install_llm.sh
python3 -m unittest discover -s tests
.venv/bin/python -m assistant                  # text chat
bash scripts/install_voice.sh                  # voice + MQTT (stage 4, 6)
.venv/bin/python -m assistant.voice.calibrate  # set vad_threshold in config.toml
.venv/bin/python -m assistant --service        # try voice + API in the foreground
bash scripts/install_service.sh                # start at boot (stage 7)
```

Modes: `python -m assistant` (text chat), `--api` (HTTP API only), `--service` (MQTT + API + voice).

## EXP-001 workflow

```bash
python3 bench/kv_calc.py --model all --ctx 2048 4096 8192 16384      # predictions
python3 bench/bench_ollama.py --models qwen3:0.6b llama3.2:1b gemma3:1b --runs 3
bash bench/bench_llamacpp.sh ~/models                                # per-quant + context sweep
python3 experiments/score.py --model qwen3:0.6b                      # quality /20
```

Results are appended to CSV files in `experiments/results/`.

## Config

Copy `config.example.toml` to `config.toml` and edit it. Every value has a default, so the file is optional.

## Safety design

The LLM never runs anything itself. It proposes a tool call, and `governor.py` checks it against an allow-list, an argument schema, risk tiers and rate limits. Each call is then allowed, sent back for spoken confirmation, or denied. Every decision is written to the audit table.
