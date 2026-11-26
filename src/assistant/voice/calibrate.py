"""Measure microphone levels to pick [voice] vad_threshold.

  PYTHONPATH=src python3 -m assistant.voice.calibrate [--device default] [--seconds 3]
Stay quiet for the first part, then speak normally for the second part.

  python3 -m assistant.voice.calibrate --speak [--config config.toml] [--save clip.wav] [--set]
No keyboard needed (for a Pi with no screen): the speaker says when to be quiet and when to talk,
using the mic, speaker and Piper voice from config.toml. --save keeps the speech part as a WAV,
--set writes the suggested vad_threshold into config.toml.
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import time
from pathlib import Path

from .audio import Microphone, play_pcm, rms, write_wav


def measure(device: str, seconds: float, rate: int = 16000, keep: list | None = None) -> list[float]:
    levels = []
    with Microphone(device, rate) as mic:
        for i, c in enumerate(mic.chunks()):
            levels.append(rms(c))
            if keep is not None:
                keep.append(c)
            if (i + 1) * 30 >= seconds * 1000:
                break
    return levels


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[int(p * (len(s) - 1))]


def suggest(quiet: list[float], speech: list[float]) -> tuple[dict, int | None]:
    """Stats for both parts and the suggested threshold (None if speech isn't clearly louder).

    Speech is measured by its loud frames (80th percentile), not the median: counting "one, two, ..."
    is about half pauses, so the median measured the room (2026-11-19 on the Pi: median 262 vs
    80th percentile ~600). The threshold is the geometric midpoint, since loudness is logarithmic.
    """
    q90, s80 = pct(quiet, 0.90), pct(speech, 0.80)
    stats = {"quiet_median": statistics.median(quiet), "quiet_p90": q90, "speech_p80": s80,
             "speech_max": max(speech), "snr_db": 20 * math.log10(max(s80, 1) / max(q90, 1))}
    return stats, (round(math.sqrt(q90 * s80)) if s80 > 1.5 * q90 else None)


def set_threshold(config_path: Path, value: int) -> None:
    """Replace the number on the `vad_threshold = ...` line, keeping the rest of the file as it is."""
    text = config_path.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^(vad_threshold\s*=\s*)[0-9.]+", lambda m: f"{m.group(1)}{value:.1f}", text)
    if n != 1:
        raise SystemExit(f"no single 'vad_threshold =' line in {config_path}")
    config_path.write_text(new, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="ALSA mic (default: config's mic_device, else 'default')")
    ap.add_argument("--seconds", type=float, default=3)
    ap.add_argument("--speak", action="store_true", help="spoken prompts instead of pressing Enter")
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--save", help="write the speech part to this WAV file")
    ap.add_argument("--set", action="store_true", help="write the suggested vad_threshold into --config")
    a = ap.parse_args()

    cfg = None
    if a.speak or a.device is None or a.set:
        from ..config import load_config
        cfg = load_config(a.config).voice if a.config.exists() else None
    device = a.device or (cfg.mic_device if cfg else "default")

    if a.speak:
        from .engines import PiperTTS
        if cfg is None:
            raise SystemExit(f"--speak needs {a.config} for the speaker and Piper voice")
        tts = PiperTTS(cfg.piper_cmd, cfg.piper_model)

        def prompt(text: str) -> None:
            print(text, flush=True)
            play_pcm(tts.synthesize(text), tts.sample_rate, cfg.speaker_device)
            time.sleep(2.5)  # Bluetooth plays on ~1.5 s after pw-cat returns (measured on the Pi)
        prompt(f"Microphone check. Please stay quiet for {a.seconds:.0f} seconds, starting now.")
    else:
        input(f"Stay QUIET for {a.seconds:.0f} s. Press Enter to start...")
    quiet = measure(device, a.seconds)
    if a.speak:
        prompt(f"Thank you. Now please talk normally for {a.seconds:.0f} seconds. "
               "For example, count from one to ten.")
    else:
        input(f"Now SPEAK normally for {a.seconds:.0f} s. Press Enter to start...")
    clip: list[bytes] = []
    speech = measure(device, a.seconds, keep=clip)
    if a.save:
        write_wav(a.save, b"".join(clip))

    stats, value = suggest(quiet, speech)
    print(f"quiet: median {stats['quiet_median']:.0f}, 90th pct {stats['quiet_p90']:.0f}")
    print(f"speech: 80th pct {stats['speech_p80']:.0f}, max {stats['speech_max']:.0f}")
    if value is None:
        print("Speech is not louder than the background: move the mic closer or raise the gain (alsamixer).")
        if a.speak:
            prompt("I couldn't hear you clearly. Please move the microphone closer and try again.")
        return
    print(f"Suggested vad_threshold = {value}")
    print(f"SNR estimate: {stats['snr_db']:.1f} dB")
    if a.set:
        set_threshold(a.config, value)
        print(f"Set vad_threshold = {value} in {a.config}")
    if a.speak:
        prompt("Calibration done. Thank you.")


if __name__ == "__main__":
    main()
