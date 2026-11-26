"""Speaker-only voice test: typed question -> brain -> Piper -> speaker, timed like the voice loop.

No microphone needed. Uses the model, speaker and Piper voice from config.toml, but a throw-away
memory database, so the test chat isn't saved into the assistant's real memory.

  .venv/bin/python bench/speak_test.py [--config config.toml] [--no-beep] [questions ...]

Plays the wake beep, then speaks one reply per question, streamed like the assistant (the first
sentence is spoken while the model writes the rest; --no-stream: wait for the whole reply).
Appends to experiments/results/speak_test.csv:
  brain_s            question in -> first sentence ready (streamed) or whole reply ready
  tts_first_audio_s  -> that sentence synthesised and handed to the speaker
  first_audio_s      the sum: what the user waits after they stop talking, minus speech-to-text
A Bluetooth speaker adds its own delay on top (it plays ~1.5 s behind on the CALUS P12).
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.__main__ import build  # noqa: E402
from assistant.config import load_config  # noqa: E402
from assistant.voice.audio import beep  # noqa: E402
from assistant.voice.engines import PiperTTS, split_sentences  # noqa: E402
from assistant.voice.loop import Speaker  # noqa: E402

QUESTIONS = ["What is the capital of Ghana?",
             "Give me one short tip for staying focused.",
             "What time is it?"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="*", default=QUESTIONS)
    ap.add_argument("--config", type=Path, default=ROOT / "config.toml")
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--no-stream", action="store_true", help="old behaviour: whole reply, then speak")
    a = ap.parse_args()

    cfg = load_config(a.config)
    tmp = tempfile.TemporaryDirectory()
    cfg.memory.db_path = str(Path(tmp.name) / "speak_test.db")
    brain, _ = build(cfg)
    tts = PiperTTS(cfg.voice.piper_cmd, cfg.voice.piper_model)
    speaker = Speaker(tts, cfg.voice.speaker_device)

    t = time.perf_counter()
    brain.llm.chat([{"role": "user", "content": "Say OK."}])  # load the model, as the running assistant keeps it
    print(f"model {cfg.llm.model} ready in {time.perf_counter() - t:.1f} s (not counted)")
    t = time.perf_counter()
    tts.synthesize("Ready.")
    print(f"Piper warm-up {time.perf_counter() - t:.2f} s (not counted)")

    if not a.no_beep:
        print("beep")
        beep(cfg.voice.sample_rate, device=cfg.voice.speaker_device)
        time.sleep(1.5)

    out = ROOT / "experiments" / "results" / "speak_test.csv"
    if out.exists() and "mode" not in out.read_text(encoding="utf-8").splitlines()[0]:
        out = out.with_name("speak_test_v2.csv")  # older file without the mode column
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "model", "question", "reply", "brain_s",
                                          "tts_first_audio_s", "first_audio_s", "spoken_s", "mode"])
        if new:
            w.writeheader()
        for q in a.questions:
            t0 = time.perf_counter()
            if a.no_stream:
                reply = brain.handle(q)  # confirm defaults to "no", so nothing sensitive can run
                brain_s = time.perf_counter() - t0
                first = speaker.say(reply, split_sentences(reply))
            else:
                ready: list[float] = []

                def timed(sentences):
                    for sentence in sentences:
                        if not ready:
                            ready.append(time.perf_counter() - t0)
                        yield sentence
                first_total = speaker.say_iter(timed(brain.respond(q)))
                reply = brain.last_reply
                brain_s = ready[0] if ready else first_total
                first = max(0.0, first_total - brain_s)
            spoken_s = time.perf_counter() - t0 - brain_s - first
            print(f"Q: {q}\nA: {reply}\n   brain {brain_s:.2f} s + first audio {first:.2f} s"
                  f" = {brain_s + first:.2f} s; speaking took {spoken_s:.1f} s")
            w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "model": cfg.llm.model,
                        "question": q, "reply": reply, "brain_s": round(brain_s, 2),
                        "tts_first_audio_s": round(first, 2), "first_audio_s": round(brain_s + first, 2),
                        "spoken_s": round(spoken_s, 1), "mode": "whole" if a.no_stream else "stream"})
            time.sleep(1.5)  # let the Bluetooth speaker finish
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
