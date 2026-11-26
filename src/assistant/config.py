"""Settings with defaults, optionally overridden by config.toml.

Secrets (API token, MQTT password) can also come from environment variables
ASSISTANT_API_TOKEN and ASSISTANT_MQTT_PASSWORD so they stay out of the file.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class LLMConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:1.7b"
    num_ctx: int = 4096
    temperature: float = 0.3
    keep_alive: str = "30m"
    think: bool = False  # reasoning models (Qwen3): off = ~30x faster replies on the Pi
    num_predict: int = 256  # reply token cap; a runaway reply stops after ~30 s instead of 300 s


@dataclass
class MemoryConfig:
    db_path: str = "data/assistant.db"
    embed_model: str = "nomic-embed-text"
    top_k: int = 3
    min_score: float = 0.45
    history_days: float = 30   # questions and replies older than this are deleted (0 = keep forever)
    audit_days: float = 90     # the record of tool decisions; saved facts stay until you forget them


@dataclass
class GovernorConfig:
    default_rate_per_min: int = 10


@dataclass
class ApiConfig:
    enabled: bool = True
    host: str = "127.0.0.1"          # 0.0.0.0 to allow other machines (then set a token!)
    port: int = 8765
    token: str = ""                  # required for /ask when not empty
    view_token: str = ""             # read-only: lets other devices open /face#view=<it>; can't use /ask


@dataclass
class MqttConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    username: str = "assistant"
    password: str = ""
    base_topic: str = "home"


@dataclass
class VoiceConfig:
    sample_rate: int = 16000
    mic_device: str = "default"      # ALSA name, see `arecord -L`
    speaker_device: str = "default"
    wake_engine: str = "openwakeword"  # or "personal": words learned from your voice (voice/enroll.py)
    wake_word: str = "hey_jarvis"    # openWakeWord model; for "personal" one or more names: "uzi,aiya,boafo"
    wake_threshold: float = 0.5      # openWakeWord only; personal words use the threshold set at enrolment
    wake_patience: int = 3           # frames (80 ms) in a row above the threshold; 1 = wake on a single frame
    personal_wake_dir: str = "models/wake"
    wake_settings: str = "data/wake_settings.json"  # words on/off and thresholds set on the screen
    wake_log_dir: str = "logs/wake"  # what each wake word did, one CSV per day (voice/wake_control.py)
    vad_engine: str = "silero"       # "silero" (neural, ignores steady fan noise) or "energy" (loudness)
    silero_threshold: float = 0.5    # speech probability for "silero"
    vad_threshold: float = 500.0     # RMS level counted as speech for "energy" (tune with the mic)
    listen_timeout_s: float = 7.0    # after the beep, give up if nobody starts talking (P12 beeps ~1.5 s late)
    silence_ms: int = 800            # this much quiet ends an utterance
    max_utterance_s: float = 10.0
    stt_engine: str = "vosk"         # vosk | whisper
    vosk_model: str = "models/vosk-model-small-en-us-0.15"
    whisper_cmd: str = "whisper-cli -m {model} -f {wav} -nt -np -t 4"
    whisper_model: str = "models/ggml-tiny.en.bin"
    # Context for Whisper (its --prompt). On the Pi, tiny.en heard "once they cut the towel off the gun"
    # without it and "What is the capital of Ghana?" with it, at the same speed (2026-11-19).
    whisper_prompt: str = "A spoken question or request to a home voice assistant."
    whisper_fit_window: bool = False  # True: shorter encoder window (-ac), ~0.9 s faster but less accurate
    piper_cmd: str = "piper --model {model} --output_raw"
    piper_model: str = "models/en_US-lessac-medium.onnx"
    timing_log: str = "experiments/results/voice_timing.csv"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)


SECTIONS = ("llm", "memory", "governor", "api", "mqtt", "voice")


def _apply(obj, values: dict) -> None:
    names = {f.name for f in fields(obj)}
    for k, v in values.items():
        if k not in names:
            raise ValueError(f"unknown setting '{k}' in [{type(obj).__name__}]")
        setattr(obj, k, v)


def load_config(path: str | Path = "config.toml") -> Config:
    cfg = Config()
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        unknown = set(data) - set(SECTIONS)
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        for section in SECTIONS:
            if section in data:
                _apply(getattr(cfg, section), data[section])
    cfg.api.token = os.environ.get("ASSISTANT_API_TOKEN", cfg.api.token)
    cfg.api.view_token = os.environ.get("ASSISTANT_VIEW_TOKEN", cfg.api.view_token)
    cfg.mqtt.password = os.environ.get("ASSISTANT_MQTT_PASSWORD", cfg.mqtt.password)
    return cfg
