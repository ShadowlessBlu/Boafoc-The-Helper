# Handover: how it runs now, and what's left

## Current state (2026-11-23)
The assistant runs on the Pi by itself from boot: wake word → speech → answer → spoken reply, the
7-inch touch screen, the HTTP API, and voice/touch control of a real light (an ESP32-S3 over MQTT).
161 unit tests pass on the PC and on the Pi.

**What runs, and where**
| Part | What it is | Started by |
|---|---|---|
| `assistant.service` (user service) | voice loop + API on port 8765 + MQTT bridge; log `logs/assistant.log` | boot (`install_user_service.sh --voice`) |
| Screen (`python -m assistant.screen`) | pygame dashboard on the 7-inch DSI screen; log `$XDG_RUNTIME_DIR/assistant-screen.log` | autostart at login (`face_screen.sh --autostart`) |
| `speaker-reconnect.service` | reconnects the Bluetooth speaker (CALUS P12) | boot |
| `power-watch.service` | input voltage, throttling, temperature every 5 s → `logs/power/` | boot (`power_watch.sh --install`) |
| Mosquitto 2.0.21 | MQTT broker with a login and ACL per device; files `root:mosquitto` 640 | boot (system service) |
| Ollama | `qwen3:1.7b`, thinking off, max 256 reply tokens | boot (system service) |

- **Wake words:** `models/wake/uzi.onnx@0.5`, `aiya.onnx@0.7`, `boafo.onnx@0.8` (trained on the PC with real
  recordings, `training/`); 3 frames in a row needed, and Silero must hear speech. Boafo still fires
  now and then during conversation.
- **Speech:** Whisper tiny.en with a short hint; `voice/fixups.py` corrects "<room> slide" → "<room> light".
  Piper `en_US-lessac-medium`, kept loaded. The mic drains for 1.6 s after each reply (the P12 lags ~1.5 s).
- **Brain:** plain light commands ("turn on the office light") skip the model (0.1 s); everything else goes
  to qwen3:1.7b. Every tool call passes the Governor; a reply that claims an action without a tool call
  is replaced by "Sorry, I didn't do that: I have no way to control it."
- **Devices:** a light only counts as switched when its device confirms (6 s window). The office light is
  the ESP32-S3 DevKit's RGB LED (pin 48), `firmware/esp32_room_node/`, node `office-node`.
- **Screen:** pages main, face, clock, history, home (tap a light to switch it), Pi status, settings
  (mic, dim, Exit = stop everything; the desktop "Assistant" icon starts it again).
  Settings > Wake words (2026-11-24): each word on/off and its threshold (live, kept in
  `data/wake_settings.json`, which wins over `config.toml`), with what it did today or in the last 7 days
  (woke, no speech, replied, near misses) from `logs/wake/<date>.csv`; Recent wakes: mark each answered
  wake Real or False, so each word gets a real false-wake rate. API: `GET/POST /wake`, `POST /wake/label`.
  Whisper's noise phrases ("Thank you.", ">>") now count as nothing heard: no reply (`voice/fixups.py`).

**Everyday commands (on the Pi, in `~/assistant`)**
```bash
systemctl --user status assistant.service          # running?
tail -f logs/assistant.log                         # what it heard and said, wake scores
systemctl --user restart assistant.service         # after a code change
bash scripts/start_assistant.sh                    # what the desktop icon runs (after Exit)
bash scripts/power_watch.sh --report               # each boot + the last reading before it
.venv/bin/python bench/ask.py "Turn on the office light."   # typed test through the API
curl -s localhost:8765/state                       # what the screen shows
```
Deploy from the PC: copy changed files to `~/assistant-next` first, run the tests there
(`~/assistant/.venv/bin/python -m unittest discover -s tests`), and only then to `~/assistant`.

**Still open**
- **Power (most urgent):** the Pi lost power 7 times on 2026-11-23 (about 12:44, 13:20, 14:22, 15:39, 15:50,
  15:53, 15:56). The firmware's reset reason is a power-on reset (`/proc/device-tree/chosen/bootloader/rsts`
  = 0x1000), not a crash or the watchdog. Voltage reads 4.9–5.1 V 5 s before each cut and nothing is ever
  flagged as throttled, so the supply cuts out suddenly rather than sagging. It negotiates only 3 A
  (`/proc/device-tree/chosen/power/max_current`). A cut also damaged the last line of a data file (NUL bytes).
  Check: the official 27 W 5 V / 5 A supply, the cable, and that it isn't a power bank running down; then a
  UPS or battery HAT. `bash scripts/power_watch.sh --report` lists every boot.
- **Boafo false wakes:** judge it on Settings > Wake words (switch it off or raise its threshold there).
- **Offline test: built, dry run 9/10 (2026-11-25), real run still to do.** `ssh pi5 "bash ~/assistant/bench/offline_test.sh"`,
  then unplug the internet at the router when the Pi says so (keep the home Wi-Fi on). It checks the service,
  model, time, memory + forget, speech round trip, wake words, the office light, telemetry and the screen, says
  the result, waits for the internet to come back and writes `experiments/results/offline_test_<date>.md`
  (`--now --quiet` = dry run). The dry run found two real problems, both fixed:
  - **ONNX Runtime sent telemetry to Microsoft** (`mobile.events.data.microsoft.com`) ~9 s after every start:
    on by default in its Linux builds. `ORT_DISABLE_TELEMETRY=1` is now set in `assistant/__init__.py`;
    a 50 s strace of the service then showed no outside connections. Its old device id and event store
    (`~/.cache/Microsoft/DeveloperTools/.onnxruntime/`) were deleted 2026-11-25 and didn't come back after a restart.
  - **The student saved a copy of a note on every recall question** ("Where is the spare key?" -> `remember`).
    `remember` now skips a note that's already there. Your 5 real notes had no copies.
- **Memory rules: done (2026-11-25).** Questions and replies are kept 30 days, the audit log 90 days
  (`[memory] history_days`, `audit_days`; pruned at start and every 6 h). Saved notes stay until you say
  "forget …": it names the note and waits for a spoken yes (never over the API). History page has
  Clear history (two taps; notes and audit stay). Deleted rows are overwritten in the file (secure_delete).
- **Admin/recovery: mostly done (2026-11-25).** Admin only over SSH. Nightly backup at 03:30 as a user timer
  (`bash scripts/backup.sh --install`; it had never been installed), each one checked after it's made
  (`logs/backup.log`), keeps 7 in `~/backups/assistant`; restore tested. The PC copies the newest one every
  Sunday 13:00 (Task Scheduler "Pi5 assistant backup pull", `scripts/pull_backup.ps1`, to
  `%USERPROFILE%\Backups\Pi5-AI-Assistant`, keeps 8). **Key-only SSH: on since 2026-11-25 17:12** (`scripts/ssh_key_only.sh`; password login is
  refused, keys work; undo with `--undo`). Not done: a full image of the NVMe drive.
- **EXP-001 Q5 distillation: done** (2026-11-24): student `q06-distill` 48/60 vs 46 for the base through the
  same pipeline, tool calls 16/16 vs 13, same speed and size (396 MB). **On trial as the assistant's model since 2026-11-24 18:37**
  (`config.toml` line 7; a test run answered in 2.4-3.0 s). It's below qwen3:1.7b's 54/60 on knowledge and
  reasoning. To go back: set `model = "qwen3:1.7b"` (or restore `config.toml.bak-20261124-distill-trial`) and
  `systemctl --user restart assistant.service`.
  How it was run: `research/README.md`.
- **False wakes in a busy room:** with nobody talking to it (2026-11-23 15:37–15:48, people or a TV nearby)
  it woke about every 20 s on all three words and answered out loud; aiya woke 12 times today with no real
  request. 2026-11-24 evening: 30 wakes in 35 min on all three words, nearly all room chatter.
  Now tunable per word on the screen (Settings > Wake words); mark Recent wakes for a few days, then switch
  off or raise the words with the most False marks.

## History (what was done, in order)
Everything below is the original step-by-step checklist, kept as a record.

### Already done on the PC
- Assistant code: Brain, Governor, memory, tools, HTTP API, MQTT bridge, voice pipeline — `src/assistant/`
- ESP32 firmware, compiles (76% flash) — `firmware/esp32_room_node/`
- All install/benchmark/backup scripts — `scripts/`, `bench/`
- EXP-001 predictions (`bench/kv_calc.py`) and first pruning results (see the vault: Model Pruning note)
- Pruned models in `research/models/` and a healed variant (see `experiments/results/eval_hf.csv`)
- 20-question test set + scorer, 20-prompt red-team set + runner — `experiments/`

### Steps 0–2: DONE on 2026-11-15
- The SSH key is installed, so `ssh shadowlessblu@<pi-ip>` logs in without a password.
- The project is copied to `~/assistant`. To refresh it later: `.\scripts\sync_to_pi.ps1` (defaults to `pi5`)
- The baseline is recorded in `experiments/baseline-2026-11-15.md`, and all 63 tests pass on the Pi.
- **Check before benchmarking:** the Pi idles near 70 °C, so fit or verify the active cooler. It also has no microphone; get a USB mic for step 6.
- **Tailscale (2026-11-18):** the Pi is `100.78.110.4` (name `shadowlessblu`) on any network, and `ssh pi5` works from the PC (alias in `C:\Users\DELL\.ssh\config`). Its local IP still changes: 192.168.1.220 at home, a new 192.168.x.165 each time on the phone hotspot.

### Steps 3–5: DONE on 2026-11-17/18
- Base setup, Ollama 0.40.16 and llama.cpp are installed; the Active Cooler works (it needed a reseat and a cold boot).
- **sudo tip:** run `ssh -t ...` commands that need the password from a normal PowerShell window; the app's terminal tab didn't pass the password through.
- Phase A ran on 6 models. Results: `experiments/results/pi-2026-11-18/summary.md`, and the vault's Model Comparison and Experiment Log notes.
- **Provisional model: `qwen3:1.7b` with `think = false`.** These are already the defaults in `config.py`; set them in `config.toml` only to change them. Thinking made answers 5–6× slower for the same score.
- Still open for EXP-001: Q5 distillation.

### Q6 + quant ladder: DONE on 2026-11-18/19 (55 of 57; the 2 failed score steps were redone on v2 below)
`bench/run_q6_q3.sh` runs on the Pi by itself (the PC can be off). It has 57 checkpoints: the context sweep (4 models × F16/Q8 KV × 512–16k tokens), the Q8→Q4→Q3→Q2 ladder for Llama 1B (speed, perplexity, 20-question score) and the memory search re-run.
- **Power cut:** each finished measurement leaves a marker in `experiments/results/.done/`. A user crontab `@reboot` entry restarts the script 90 s after boot and it skips what's done, so only the measurement that was running is lost. The entry removes itself when everything is finished.
- **Check progress:** `ssh pi5`, then `cd ~/assistant && bash bench/run_q6_q3.sh --status | tail -1` and `tail experiments/results/q6_q3.log`.
- **Results:** `experiments/results/llamacpp_ctx.csv`, `llamacpp_quant.csv`, `score_v1.csv`, `memory_search_v3.csv`, and `summary_latest.md` at the end. Bring them to the PC with `scp "pi5:assistant/experiments/results/*" experiments/results/pi-2026-11-18/`.
- **Stop it:** `pkill -f "^bash bench/run_q6_q3.sh"; pkill llama-bench; bash bench/run_q6_q3.sh --remove-boot`. A step that fails or hangs past 6 hours is marked `.FAILED` and skipped; delete that marker to retry it.
- **Tested for real:** the Pi lost power around 10:34 on 2026-11-18; it restarted by itself with "resuming: 4 checkpoints already done".

### Follow-up: DONE on 2026-11-19 12:14 (`bench/run_followup.sh`, 16 of 16 checkpoints)
Installed the code staged in `~/assistant-next` (memory-injection hardening, test set v2, scorer cool-down), ran the tests, scored 9 models on the 60-question **test set v2**, re-ran the red-team with the hardened memory prompt, and installed the voice software. Results (copied to `experiments/results/pi-2026-11-18/`): **qwen3:1.7b (think off) 54/60**, the clear pick; Q3_K_S 23/60 and Q2_K 10/60 with 0/16 tool use; red-team 0 breaches in 40. The scorer now uses a 74 °C cool-down and a 90 s `--timeout` (low-bit models sometimes never stop generating). The Pi runs much cooler with its case off. Same checkpoints, lock and self-removing `@reboot` entry. Progress: `bash bench/run_followup.sh --status | tail -1`; log in `experiments/results/followup.log`; results in `score_v2.csv`, `redteam_v1.csv` (newest rows) and `summary_latest.md`.

To repeat Phase A (it survives SSH drops, and logs to `experiments/results/phase_a.log`):
```bash
nohup bash bench/run_phase_a.sh "qwen3:0.6b qwen3:1.7b" qwen3:1.7b > experiments/results/phase_a.log 2>&1 &
.venv/bin/python bench/summarize.py                 # Markdown tables for the vault
.venv/bin/python experiments/score.py --model qwen3:1.7b --think off
```
Check the quantization behind an Ollama tag with `ollama show <model>`: `llama3.2:1b` is Q8_0, while `llama3.2:1b-instruct-q4_K_M` is Q4.

#### Original step 5 commands (for reference)
```bash
.venv/bin/python bench/kv_calc.py --model all          # predictions
.venv/bin/python bench/bench_ollama.py --models qwen3:0.6b gemma3:1b llama3.2:1b qwen3:1.7b llama3.2:3b --runs 3
bash bench/bench_llamacpp.sh ~/models "512 2048 4096 8192"   # if you put GGUFs in ~/models
.venv/bin/python experiments/score.py --model qwen3:0.6b     # repeat per model
.venv/bin/python experiments/redteam.py --model qwen3:1.7b   # Governor red-team
.venv/bin/python bench/memory_search.py                      # memory search on the Pi (has numpy)
```
Results land in `experiments/results/*.csv`. Copy the summary tables into the vault
(Experiment Log, Model Comparison, Databases). Pick the winner, set it as `model` in `config.toml`.

### Step 6 — voice and MQTT (30 min + ~300 MB)
**Status 2026-11-18:** USB mic fitted and working (`plughw:CARD=Device,DEV=0`, already set as `mic_device` in the Pi's `config.toml`). **2026-11-19: voice software installed** (`install_voice.sh --no-mqtt`: Vosk small-en, Piper `en_US-lessac-medium`, openWakeWord, whisper.cpp with tiny.en and base.en). A silent round trip (Piper speaks a sentence to a file, then each engine transcribes it) was word-perfect on Vosk (1.4 s), Whisper tiny (1.9 s) and base (4.5 s), times including model load; Piper took 1.15 s for 3.2 s of speech. **Your next step at the Pi:** calibrate the mic (below), then the text chat, then the voice loop. Speaker: CALUS Blue P12 over Bluetooth, set as `speaker_device = "pipewire"` (ALSA's default is HDMI on this Pi, so `aplay` can't reach Bluetooth). A future system service must run in the user session to reach PipeWire. The MQTT part needs sudo: run `bash scripts/install_voice.sh` in your own PowerShell window later.
```bash
bash scripts/install_voice.sh
.venv/bin/python -m assistant.voice.calibrate --device plughw:CARD=Device,DEV=0   # set vad_threshold in config.toml
.venv/bin/python -m assistant                    # text chat first
.venv/bin/python -m assistant --service          # voice loop in the foreground
```

### Step 7 — ESP32 node (an evening)
1. `copy firmware\esp32_room_node\secrets.example.h secrets.h`, fill in Wi-Fi + MQTT.
2. On the Pi: `bash scripts/add_mqtt_node.sh livingroom-node living_room` (makes its MQTT login).
3. Flash with Arduino IDE. Then: `.venv/bin/python bench/mqtt_latency.py --node livingroom-node`.
4. Set `[mqtt] enabled = true` in config.toml.

### Step 8 — run at boot (10 min)
```bash
sudo install -m 600 /dev/null /etc/assistant.env   # put ASSISTANT_MQTT_PASSWORD=... inside
bash scripts/install_service.sh
curl -s localhost:8765/health
```
Then `bash scripts/backup.sh --install` for the nightly backup at 03:30 (user timer, keeps 7, each one checked).

### Pruned/healed models → Pi (optional; probably skip)
PC results (2026-11-16) on the 20-question test set: base Qwen3-0.6B **15/20**, 4 layers pruned **3/20**,
pruned + LoRA-healed **5/20** (`experiments/results/score_hf_v1.csv`). Pruning a 0.6B model isn't
worth deploying. The steps below are only for repeating the experiment with a larger model (3B).

On the PC (or Colab), convert any model in `research/models/` to GGUF:
```bash
python llama.cpp/convert_hf_to_gguf.py research/models/qwen3-0.6b-pruned4mid-healed/merged --outtype f16 --outfile pruned-healed-f16.gguf
llama-quantize pruned-healed-f16.gguf pruned-healed-q4_k_m.gguf Q4_K_M
```
Copy to the Pi, `ollama create pruned-healed -f Modelfile` (Modelfile: `FROM ./pruned-healed-q4_k_m.gguf`),
then `experiments/score.py --model pruned-healed` and `bench_llamacpp.sh` for real speed.

### Where things are recorded
- Vault dashboard: `Journal\Projects\Personal AI Assistant.md` (links to everything)
- Every topic note: fill sections 6–8 (Results / Challenges / What I Learned) as you go — that's the point of the project.
