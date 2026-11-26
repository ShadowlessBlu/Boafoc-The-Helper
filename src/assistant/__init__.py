"""Offline modular personal AI assistant for Raspberry Pi 5."""
import os

# ONNX Runtime (wake word, voice detection, Piper) has Microsoft's 1DS telemetry switched on in its
# official Linux builds: the service contacted mobile.events.data.microsoft.com about 9 s after every
# start (found by bench/offline_test.py, 2026-11-25). onnxruntime.disable_telemetry_events() only stops
# "non-essential" events; this variable, set before ONNX Runtime loads, turns the uploader, the events
# and the persistent device identifier off for the whole process (onnxruntime/Privacy.md).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"
