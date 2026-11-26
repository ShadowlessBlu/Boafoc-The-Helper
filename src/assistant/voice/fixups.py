"""Small, targeted corrections for words Whisper keeps mishearing in this house.

Only exact "<room> <misheard word>" pairs are changed, so nothing new can be invented. (A Whisper
hint with example commands fixed "office slide" but made it hallucinate "turn off the kitchen
light" on unclear speech, 2026-11-23, so that route is closed.)
"""
from __future__ import annotations

import re

from ..tools import ROOMS

# Heard instead of "light" in real tests: "Turn on the office slide." (5 times in a row)
_MISHEARD_LIGHT = ("slide", "slides", "slight", "flight", "lie", "life", "lied")
_ROOM_LIGHT = re.compile(
    r"\b(" + "|".join(re.escape(r) for r in ROOMS) + r")\s+(" + "|".join(_MISHEARD_LIGHT) + r")\b",
    re.I)


# What Whisper writes when it hears only noise or far-off talk after a false wake. On 2026-11-24
# five of 19 "questions" in 35 minutes were "Thank you." or ">> >>", and each got a reply.
_PHANTOMS = {"thank you", "thank you very much", "thanks", "thanks for watching", "thank you for watching",
             "you", "bye", "bye bye"}


def is_phantom(text: str) -> bool:
    """True for text Whisper makes up from noise: one of the phrases above, or no words at all."""
    words = re.sub(r"[^a-z' ]+", " ", text.lower()).split()
    return not words or " ".join(words) in _PHANTOMS


def fix_transcript(text: str) -> str:
    def repl(m: re.Match) -> str:
        plural = m.group(2).lower().endswith("s") and m.group(2).lower() != "lies"
        return f"{m.group(1)} light{'s' if plural else ''}"
    return _ROOM_LIGHT.sub(repl, text)
