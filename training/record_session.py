"""Guided recording session on the Pi's 7-inch screen, for wake-word training.

The screen says what to do and when; no speaker needed. Steps:
  1. talk normally for 4 minutes without the wake words (what the models must ignore)
  2-4. each wake word: "Say: UZI" appears for 2.5 s, then a short pause, until 20 were heard
       (max 28 cues). After each cue the screen shows whether it heard you.
A live mic meter and a "hearing speech" light show whether you're close enough.

Saves to <out>/: talk.wav, word_<w>.wav and word_<w>.json (each cue's start/end in 30 ms chunks,
and whether speech was heard), after each step, so a power cut only loses the step in progress.
The assistant must be stopped first (it holds the mic); stop the face screen too (this takes it).

  .venv/bin/python training/record_session.py [--out ~/wake-train/real2] [--talk 240] [--per-word 20]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from assistant.config import load_config  # noqa: E402
from assistant.screen import ACCENT, BG, DIM, FONTS, LINE, OFF, TEXT, WARN  # noqa: E402
from assistant.voice.audio import Microphone, SileroVAD, rms, write_wav  # noqa: E402

W, H = 800, 480
GOOD = (127, 227, 208)
WORDS = ["uzi", "aiya", "boafo"]
SAY_S, PAUSE_S = 2.5, 1.6


class Recorder(threading.Thread):
    """Reads the mic all the time; keeps chunks only while `recording`. Live level and speech."""

    def __init__(self, device: str):
        super().__init__(daemon=True)
        self.device = device
        self.vad = SileroVAD(0.5)
        self.lock = threading.Lock()
        self.chunks: list[bytes] = []
        self.speech: list[bool] = []
        self.recording = False
        self.level = 0.0
        self.last_speech = 0.0
        self.error = ""

    def run(self) -> None:
        try:
            with Microphone(self.device, 16000) as mic:
                for c in mic.chunks():
                    sp = self.vad.is_speech(c)
                    with self.lock:
                        self.level = 0.6 * self.level + 0.4 * rms(c)
                        if sp:
                            self.last_speech = time.monotonic()
                        if self.recording:
                            self.chunks.append(c)
                            self.speech.append(sp)
        except Exception as e:  # noqa: BLE001 - show it on screen
            self.error = str(e)

    def count(self) -> int:
        with self.lock:
            return len(self.chunks)

    def heard_between(self, a: int, b: int) -> bool:
        with self.lock:
            return any(self.speech[a:b])

    def take(self) -> tuple[bytes, list[bool]]:
        with self.lock:
            data, sp = b"".join(self.chunks), list(self.speech)
            self.chunks, self.speech = [], []
            return data, sp


class Screen:
    def __init__(self, pg, window: bool):
        self.pg = pg
        flags = 0 if window else pg.FULLSCREEN | pg.SCALED
        self.s = pg.display.set_mode((W, H), flags)
        pg.display.set_caption("Recording")
        pg.mouse.set_visible(False)
        self.fonts: dict = {}

    def font(self, size: int):
        if size not in self.fonts:
            self.fonts[size] = self.pg.font.SysFont(FONTS, size)
        return self.fonts[size]

    def text(self, s: str, size: int, color, center=None, topleft=None) -> None:
        img = self.font(size).render(s, True, color)
        r = img.get_rect(center=center) if center else img.get_rect(topleft=topleft)
        self.s.blit(img, r)

    def frame(self, step: str, title: str, lines: list[tuple[str, int, tuple]], rec: Recorder,
              bar: float | None = None, footer: str = "") -> None:
        pg, s = self.pg, self.s
        s.fill(BG)
        self.text(step, 18, DIM, topleft=(28, 20))
        self.text(title, 46, TEXT, center=(W // 2, 120))
        y = 200
        for line, size, color in lines:
            self.text(line, size, color, center=(W // 2, y))
            y += size + 22
        # mic meter: log scale, 100 (silence) .. 10000 (loud)
        lvl = min(1.0, max(0.0, (math.log10(max(rec.level, 1)) - 2) / 2))
        hearing = time.monotonic() - rec.last_speech < 0.4
        pg.draw.rect(s, LINE, (28, 420, 560, 18), border_radius=9)
        pg.draw.rect(s, GOOD if hearing else OFF, (28, 420, int(560 * lvl), 18), border_radius=9)
        self.text("mic", 14, DIM, topleft=(28, 398))
        pg.draw.circle(s, GOOD if hearing else OFF, (620, 429), 10)
        self.text("hearing speech" if hearing else "no speech", 16, GOOD if hearing else DIM, topleft=(638, 418))
        if bar is not None:
            pg.draw.rect(s, LINE, (28, 380, 744, 6), border_radius=3)
            pg.draw.rect(s, ACCENT, (28, 380, int(744 * min(1.0, bar)), 6), border_radius=3)
        if footer:
            self.text(footer, 16, DIM, topleft=(28, 452))
        if rec.error:
            self.text(f"Microphone problem: {rec.error[:60]}", 18, WARN, center=(W // 2, 360))
        pg.display.flip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path.home() / "wake-train" / "real2")
    ap.add_argument("--talk", type=float, default=240)
    ap.add_argument("--per-word", type=int, default=20)
    ap.add_argument("--words", nargs="+", default=WORDS)
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--intro", type=float, default=60, help="seconds before it starts by itself (touch = now)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.toml").voice

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame as pg
    pg.init()
    scr = Screen(pg, a.window)
    rec = Recorder(cfg.mic_device)
    rec.start()
    clock = pg.time.Clock()
    n_steps = 1 + len(a.words)

    def pump() -> bool:
        """Handle events; True if the screen was touched or a key pressed."""
        touched = False
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                raise SystemExit(0)
            if e.type in (pg.MOUSEBUTTONDOWN, pg.FINGERDOWN, pg.KEYDOWN):
                touched = True
        return touched

    def wait(seconds: float, draw, skippable: bool = False) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if pump() and skippable:
                return
            draw(end - time.monotonic())
            clock.tick(20)

    # --- intro
    wait(a.intro, lambda left: scr.frame(
        "Recording session", "Wake-word recordings",
        [(f"1. Talk normally for {a.talk / 60:.0f} minutes (no wake words)", 24, TEXT),
         ("2-4. Say each word when it appears: " + ", ".join(w.upper() for w in a.words), 24, TEXT),
         ("Sit where you'd talk to the assistant, facing the mic.", 20, DIM),
         (f"Starting in {left:.0f} s  -  touch the screen to start now", 22, ACCENT)], rec), skippable=True)

    # --- 1: talk
    rec.take()
    rec.recording = True
    t0 = time.monotonic()
    wait(a.talk, lambda left: scr.frame(
        f"Step 1 of {n_steps}", "Talk normally",
        [("Chat, read aloud, take a call - anything.", 26, TEXT),
         ("Don't say " + ", ".join(w.capitalize() for w in a.words) + ".", 26, WARN),
         (f"{int(left // 60)}:{int(left % 60):02d} left", 40, ACCENT)], rec,
        bar=(time.monotonic() - t0) / a.talk, footer="If the light stays grey while you talk, move closer to the mic."))
    rec.recording = False
    data, sp = rec.take()
    write_wav(str(a.out / "talk.wav"), data, 16000)
    (a.out / "talk.json").write_text(json.dumps({"seconds": len(data) / 32000, "speech_fraction": sum(sp) / max(1, len(sp))}))

    # --- 2..: words
    for k, word in enumerate(a.words, start=2):
        wait(8, lambda left, w=word, k=k: scr.frame(
            f"Step {k} of {n_steps}", f"Next: {w.upper()}",
            [(f"Say \"{w.capitalize()}\" each time it appears in big letters.", 24, TEXT),
             ("Once per cue, the way you'd call the assistant. Vary it a little.", 20, DIM),
             (f"Starting in {left:.0f} s", 26, ACCENT)], rec))
        rec.take()
        rec.recording = True
        cues, heard = [], 0
        while heard < a.per_word and len(cues) < a.per_word + 8:
            start = rec.count()
            wait(SAY_S, lambda left, w=word: scr.frame(
                f"Step {k} of {n_steps}   -   {heard} of {a.per_word} heard", "",
                [(f"Say: {w.upper()}", 92, ACCENT)], rec, bar=heard / a.per_word))
            end = rec.count()
            wait(0.5, lambda left: None)  # the end of the word may spill past the cue
            ok = rec.heard_between(start, rec.count())
            heard += ok
            cues.append({"start": start, "end": rec.count(), "cue_end": end, "heard": ok})
            msg = ("heard", GOOD) if ok else ("didn't hear that - a bit louder or closer", WARN)
            wait(PAUSE_S - 0.5, lambda left, m=msg: scr.frame(
                f"Step {k} of {n_steps}   -   {heard} of {a.per_word} heard", "",
                [("...", 60, DIM), (m[0], 26, m[1])], rec, bar=heard / a.per_word))
        rec.recording = False
        data, _ = rec.take()
        write_wav(str(a.out / f"word_{word}.wav"), data, 16000)
        (a.out / f"word_{word}.json").write_text(json.dumps({"chunk_ms": 30, "cues": cues, "heard": heard}))

    wait(8, lambda left: scr.frame("Done", "All done - thank you!",
                                   [("The recordings are saved.", 26, TEXT)], rec))
    (a.out / "DONE").write_text("ok")
    pg.quit()


if __name__ == "__main__":
    main()
