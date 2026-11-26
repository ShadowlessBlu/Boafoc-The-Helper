"""Run the assistant.

  python -m assistant             text chat in the terminal (commands: /memories /forget <id> /audit /quit)
  python -m assistant --api       HTTP API only (/health, /ask, /face, /state)
  python -m assistant --service   full service: MQTT + HTTP API + voice loop (what systemd runs)
  python -m assistant --view-link print the link other devices use to watch the face (has the view token)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import threading
import time

from .brain import Brain
from .config import Config, load_config
from .governor import Governor
from .llm import LLMError, OllamaClient
from .memory import Memory
from .tools import Devices, build_tools

log = logging.getLogger("assistant")


def build(cfg: Config) -> tuple[Brain, Devices]:
    llm = OllamaClient(cfg.llm.host, cfg.llm.model, cfg.llm.num_ctx, cfg.llm.temperature, cfg.llm.keep_alive,
                       think=cfg.llm.think, num_predict=cfg.llm.num_predict)
    embedder = None
    if cfg.memory.embed_model:
        embedder = lambda texts: llm.embed(texts, cfg.memory.embed_model)  # noqa: E731
    memory = Memory(cfg.memory.db_path, embedder)
    devices = Devices()
    governor = Governor(build_tools(memory, devices, cfg.memory.min_score), cfg.governor.default_rate_per_min)
    return Brain(llm, governor, memory, cfg.memory.top_k, cfg.memory.min_score), devices


def start_mqtt(cfg: Config, devices: Devices):
    if not cfg.mqtt.enabled:
        return None
    from .mqtt_bridge import MqttBridge
    bridge = MqttBridge(devices, cfg.mqtt.host, cfg.mqtt.port, cfg.mqtt.username, cfg.mqtt.password,
                        cfg.mqtt.base_topic)
    log.info("MQTT connecting to %s:%s", cfg.mqtt.host, cfg.mqtt.port)
    return bridge


def start_api(cfg: Config, brain: Brain, bridge, lock: threading.Lock, status=None, wake_control=None):
    from .api import AssistantAPI
    if cfg.api.host not in ("127.0.0.1", "localhost") and not cfg.api.token:
        raise SystemExit("Refusing to expose the API on the network without [api] token "
                         "(or ASSISTANT_API_TOKEN).")
    api = AssistantAPI(brain, cfg.api.token, cfg.llm.host, bridge.status if bridge else None, lock, status,
                       cfg.api.view_token, wake_control)
    server = api.make_server(cfg.api.host, cfg.api.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("API on http://%s:%s", cfg.api.host, cfg.api.port)
    return server


def start_pruning(cfg: Config, memory: Memory, lock: threading.Lock, every_s: float = 6 * 3600) -> None:
    """Memory rules: delete old questions/replies and audit rows at start, then every 6 hours."""
    def run():
        while True:
            try:
                with lock:
                    n_msg, n_audit = memory.prune(cfg.memory.history_days, cfg.memory.audit_days)
                if n_msg or n_audit:
                    log.info("memory rules: deleted %d messages older than %g days, %d audit rows older than %g days",
                             n_msg, cfg.memory.history_days, n_audit, cfg.memory.audit_days)
            except Exception:  # noqa: BLE001 - never stop the assistant over housekeeping
                log.exception("pruning old history failed")
            time.sleep(every_s)
    threading.Thread(target=run, daemon=True).start()


def view_link(cfg: Config) -> str:
    """The address other devices open to watch the face. The token goes after '#', so the browser
    never sends it in the request line (it stays out of server and proxy logs)."""
    import socket
    if cfg.api.host in ("127.0.0.1", "localhost"):
        return "The API only listens on the Pi: set [api] host = \"0.0.0.0\" and a token first."
    if not cfg.api.view_token:
        return "Set [api] view_token in config.toml first."
    if cfg.api.host != "0.0.0.0":
        hosts = [cfg.api.host]
    else:  # the name (Tailscale MagicDNS) and every IPv4 address (hotspot/LAN, Tailscale 100.x)
        hosts = [socket.gethostname()]
        try:
            out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
            hosts += [ip for ip in out.split() if ip.count(".") == 3]
        except (OSError, subprocess.SubprocessError):
            pass
    return "\n".join(f"http://{h}:{cfg.api.port}/face#view={cfg.api.view_token}" for h in hosts)


def text_chat(brain: Brain) -> None:
    print(f"Assistant ready ({brain.llm.model}). Type /quit to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/memories":
            for mid, t in brain.memory.list_memories():
                print(f"  #{mid}: {t}")
            continue
        if text.startswith("/forget "):
            try:
                print("  deleted" if brain.memory.forget(int(text.split()[1])) else "  no such memory")
            except ValueError:
                print("  usage: /forget <id>")
            continue
        if text == "/audit":
            for ts, tool, args, verdict, reason, outcome in brain.memory.audit_log():
                print(f"  {time.strftime('%H:%M:%S', time.localtime(ts))} {tool} {args} -> {verdict} ({reason}) {outcome}")
            continue
        t0 = time.perf_counter()
        try:
            reply = brain.handle(text, confirm=lambda q: input(f"  {q} [y/N] ").strip().lower() == "y")
        except LLMError as e:
            print(f"  LLM error: {e}")
            continue
        print(f"bot> {reply}   ({time.perf_counter() - t0:.1f}s)")


def no_telemetry() -> None:
    """ONNX Runtime telemetry is switched off by ORT_DISABLE_TELEMETRY in assistant/__init__.py
    (it must be set before ONNX Runtime loads); this also turns off its "non-essential" events."""
    try:
        import onnxruntime
        onnxruntime.disable_telemetry_events()
    except (ImportError, AttributeError):
        pass


def main() -> None:
    no_telemetry()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.toml")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--api", action="store_true")
    mode.add_argument("--service", action="store_true")
    mode.add_argument("--view-link", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(a.config)
    if a.view_link:
        print(view_link(cfg))
        return
    brain, devices = build(cfg)
    lock = threading.Lock()

    if not (a.api or a.service):
        text_chat(brain)
        return

    start_pruning(cfg, brain.memory, lock)
    from .status import Status
    status = Status(cfg.llm.model, home=lambda: {"lights": dict(devices.lights), "temps": dict(devices.temps),
                                              "devices": devices.connected()})
    status.set(wake_words=[w.strip() for w in cfg.voice.wake_word.split(",") if w.strip()])
    wake_control = None
    if a.service and cfg.voice.wake_engine == "openwakeword" and cfg.voice.wake_word:
        from .voice.wake_control import WakeControl
        wake_control = WakeControl(cfg.voice.wake_word, cfg.voice.wake_threshold, cfg.voice.wake_settings,
                                   cfg.voice.wake_log_dir)
        status.set(wake_words=wake_control.entries())
    bridge = start_mqtt(cfg, devices) if a.service else None
    if cfg.api.enabled or a.api:
        start_api(cfg, brain, bridge, lock, status, wake_control)

    if a.api:
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            return

    from .voice.loop import VoiceLoop, build_voice
    try:
        wake, stt, tts = build_voice(cfg.voice, wake_control)
    except (ImportError, FileNotFoundError) as e:
        log.error("voice engines not available: %s (run scripts/install_voice.sh)", e)
        sys.exit(1)
    log.info("voice loop ready (wake word: %s, STT: %s, VAD: %s)",
             ", ".join(wake_control.entries()) if wake_control else (cfg.voice.wake_word or "Enter"),
             cfg.voice.stt_engine, cfg.voice.vad_engine)
    VoiceLoop(brain, wake, stt, tts, cfg.voice, lock, status, wake_control).run_forever()


if __name__ == "__main__":
    main()
