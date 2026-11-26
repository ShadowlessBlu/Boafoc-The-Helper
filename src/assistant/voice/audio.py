"""Audio in/out through ALSA command-line tools (arecord / aplay), plus speech detection (VAD):
a simple loudness one, and Silero (a small neural model) that isn't fooled by steady fan noise.

Using arecord avoids PortAudio and keeps the dependencies small on the Pi.
Audio format everywhere: 16-bit signed little-endian mono PCM.
"""
from __future__ import annotations

import array
import math
import subprocess
from typing import Iterator

BYTES_PER_SAMPLE = 2


def rms(chunk: bytes) -> float:
    """Root-mean-square level of 16-bit PCM (0 ... 32768)."""
    if len(chunk) < 2:
        return 0.0
    a = array.array("h")
    a.frombytes(chunk[: len(chunk) // 2 * 2])
    return math.sqrt(sum(s * s for s in a) / len(a))


class EnergyVAD:
    """Collects one utterance: starts when the level passes `threshold`, ends after
    `silence_ms` of quiet or after `max_s`. Keeps `pre_ms` of audio from before the start
    so the first syllable isn't cut off."""

    def __init__(self, threshold: float, sample_rate: int = 16000, chunk_ms: int = 30,
                 silence_ms: int = 800, max_s: float = 10.0, pre_ms: int = 300, start_timeout_s: float = 5.0):
        self.threshold = threshold
        self.chunk_ms = chunk_ms
        self.silence_chunks = max(1, silence_ms // chunk_ms)
        self.max_chunks = int(max_s * 1000 / chunk_ms)
        self.pre_chunks = max(0, pre_ms // chunk_ms)
        self.start_timeout_chunks = int(start_timeout_s * 1000 / chunk_ms)
        self.chunk_bytes = sample_rate * chunk_ms // 1000 * BYTES_PER_SAMPLE

    def is_speech(self, chunk: bytes) -> bool:
        return rms(chunk) >= self.threshold

    def reset(self) -> None:
        """Called at the start of each utterance (models with memory clear it here)."""

    def collect(self, chunks: Iterator[bytes]) -> bytes:
        """Return the utterance audio, or b"" if nobody spoke before the start timeout."""
        self.reset()
        pre: list[bytes] = []
        voiced: list[bytes] = []
        quiet = 0
        waited = 0
        for c in chunks:
            loud = self.is_speech(c)
            if not voiced:
                if loud:
                    voiced = pre + [c]
                    quiet = 0
                else:
                    pre = (pre + [c])[-self.pre_chunks:] if self.pre_chunks else []
                    waited += 1
                    if waited >= self.start_timeout_chunks:
                        return b""
                continue
            voiced.append(c)
            quiet = 0 if loud else quiet + 1
            if quiet >= self.silence_chunks or len(voiced) >= self.max_chunks:
                break
        return b"".join(voiced)


class SileroVAD(EnergyVAD):
    """Same rules as EnergyVAD, but "is this speech?" is answered by Silero VAD (the model that
    ships with openWakeWord) instead of loudness. On the Pi, a question at RMS 370-700 against fan
    noise at 290-430 was cut off after 1.2 s by the loudness rule (2026-11-19); the model looks
    at the sound's shape, so steady noise doesn't count as speech and quiet syllables still do.

    `threshold` here is a speech probability (0-1). `model` is for tests; normally it's loaded.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000, model=None, **kw):
        super().__init__(threshold, sample_rate, **kw)
        import numpy as np
        self.np = np
        if model is None:
            from openwakeword.vad import VAD
            model = VAD()
        self.model = model

    def reset(self) -> None:
        self.model.reset_states()

    def speech_probability(self, chunk: bytes) -> float:
        x = self.np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype=self.np.int16)
        return float(self.model.predict(x, frame_size=len(x))) if len(x) else 0.0

    def is_speech(self, chunk: bytes) -> bool:
        return self.speech_probability(chunk) >= self.threshold


def make_vad(engine: str, energy_threshold: float, silero_threshold: float, **kw) -> EnergyVAD:
    """The configured detector; falls back to loudness if Silero can't be loaded."""
    if engine == "silero":
        try:
            return SileroVAD(silero_threshold, **kw)
        except (ImportError, OSError) as e:  # no openwakeword/onnxruntime, or no model file
            import logging
            logging.getLogger(__name__).warning("Silero VAD unavailable (%s): using loudness", e)
    elif engine != "energy":
        raise ValueError(f"unknown vad_engine {engine!r}")
    return EnergyVAD(energy_threshold, **kw)


class Microphone:
    def __init__(self, device: str = "default", sample_rate: int = 16000, chunk_ms: int = 30):
        self.cmd = ["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw"]
        self.chunk_bytes = sample_rate * chunk_ms // 1000 * BYTES_PER_SAMPLE
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Microphone":
        # bufsize=0: no hidden buffer in Python, so drain() can really empty what's waiting
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=2)

    def chunks(self) -> Iterator[bytes]:
        """Fixed-size chunks (an unbuffered read may return less than asked for)."""
        assert self.proc and self.proc.stdout, "use 'with Microphone() as mic'"
        buf = b""
        while True:
            data = self.proc.stdout.read(self.chunk_bytes - len(buf))
            if not data:
                return
            buf += data
            if len(buf) >= self.chunk_bytes:
                yield buf[:self.chunk_bytes]
                buf = buf[self.chunk_bytes:]

    def drain(self) -> int:
        """Throw away audio recorded while nobody was reading it (while the assistant thought and
        spoke). Without this, the wake word detector heard the assistant's own reply from the buffer
        and woke itself ("uzi 0.97-1.00" right after each reply, 2026-11-23). Returns bytes dropped."""
        if not (self.proc and self.proc.stdout):
            return 0
        import os
        import select
        fd, dropped = self.proc.stdout.fileno(), 0
        try:
            while select.select([fd], [], [], 0)[0]:
                data = os.read(fd, 65536)
                if not data:
                    break
                dropped += len(data)
        except (OSError, ValueError):  # e.g. select on pipes isn't supported on Windows
            pass
        return dropped


def play_command(sample_rate: int, device: str = "default") -> list[str]:
    """The player for raw 16-bit mono PCM on stdin.

    device "pipewire" (or "pipewire:<node name>") plays through PipeWire with pw-cat, which reaches
    Bluetooth speakers; on this Pi, ALSA's "default" goes to HDMI and fails (error 524) because the
    pipewire-alsa bridge isn't installed. Anything else is an ALSA device for aplay.
    """
    if device == "pipewire" or device.startswith("pipewire:"):
        cmd = ["pw-cat", "--playback", "--raw", "--format", "s16", "--rate", str(sample_rate), "--channels", "1"]
        target = device.partition(":")[2]
        return cmd + (["--target", target] if target else []) + ["-"]
    return ["aplay", "-q", "-D", device, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw"]


def play_pcm(pcm: bytes, sample_rate: int, device: str = "default") -> None:
    subprocess.run(play_command(sample_rate, device), input=pcm, check=False)


def beep(sample_rate: int = 16000, freq: float = 880, ms: int = 120, device: str = "default") -> None:
    n = sample_rate * ms // 1000
    a = array.array("h", (int(8000 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)))
    play_pcm(a.tobytes(), sample_rate, device)


def write_wav(path: str, pcm: bytes, sample_rate: int = 16000) -> None:
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(BYTES_PER_SAMPLE)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
