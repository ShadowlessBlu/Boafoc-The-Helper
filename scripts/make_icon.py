"""Draw the desktop launcher icon: hexagon core with three arms ending in open chevrons, in the
face's purple, on a dark rounded tile (an original design in the style the user picked).

  python scripts/make_icon.py [--size 256]   -> assets/assistant-icon.png and assets/assistant-icon.svg
Needs pygame-ce for the PNG.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PURPLE = (185, 166, 255)
TILE = (11, 10, 16)


def hex_points(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    return [(cx + math.cos(math.pi / 3 * i + math.pi / 6) * r, cy + math.sin(math.pi / 3 * i + math.pi / 6) * r)
            for i in range(6)]


def arms() -> list[dict]:
    """For each arm (in a 120x120 box): the line and the chevron's three points."""
    out = []
    for k in range(3):
        a = math.radians(-90 + k * 120)
        ca, sa = math.cos(a), math.sin(a)
        px, py = -sa, ca

        def at(d: float, o: float = 0.0) -> tuple[float, float]:
            return 60 + ca * d + px * o, 60 + sa * d + py * o
        out.append({"line": (at(23), at(42)), "chevron": [at(44, -9), at(54), at(44, 9)]})
    return out


def svg() -> str:
    col = "#B9A6FF"
    pts = lambda ps: " ".join(f"{x:.1f},{y:.1f}" for x, y in ps)  # noqa: E731
    body = (f'<polygon points="{pts(hex_points(60, 60, 17))}" fill="none" stroke="{col}" stroke-width="5" '
            f'stroke-linejoin="round"/><polygon points="{pts(hex_points(60, 60, 6.5))}" fill="{col}"/>')
    for arm in arms():
        (x1, y1), (x2, y2) = arm["line"]
        body += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="6" '
                 f'stroke-linecap="round"/><polyline points="{pts(arm["chevron"])}" fill="none" stroke="{col}" '
                 f'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">'
            f'<rect width="120" height="120" rx="24" fill="#0b0a10"/>{body}</svg>\n')


def png(path: Path, size: int) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as pg
    pg.init()
    ss = 4                                   # draw 4x larger, then smooth down: clean edges
    big = size * ss
    k = big / 120
    surf = pg.Surface((big, big), pg.SRCALPHA)
    pg.draw.rect(surf, TILE, (0, 0, big, big), border_radius=int(24 * k))

    def thick_line(a, b, width):
        a, b = (a[0] * k, a[1] * k), (b[0] * k, b[1] * k)
        pg.draw.line(surf, PURPLE, a, b, int(width * k))
        for p in (a, b):                      # round caps and joins
            pg.draw.circle(surf, PURPLE, p, width * k / 2)

    outer = hex_points(60, 60, 17)
    for i in range(6):
        thick_line(outer[i], outer[(i + 1) % 6], 5)
    pg.draw.polygon(surf, PURPLE, [(x * k, y * k) for x, y in hex_points(60, 60, 6.5)])
    for arm in arms():
        thick_line(*arm["line"], 6)
        c = arm["chevron"]
        thick_line(c[0], c[1], 5)
        thick_line(c[1], c[2], 5)
    path.parent.mkdir(parents=True, exist_ok=True)
    pg.image.save(pg.transform.smoothscale(surf, (size, size)), str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    out = ROOT / "assets"
    (out / "assistant-icon.svg").parent.mkdir(parents=True, exist_ok=True)
    (out / "assistant-icon.svg").write_text(svg(), encoding="utf-8")
    png(out / "assistant-icon.png", a.size)
    print(f"saved {out / 'assistant-icon.png'} and .svg")


if __name__ == "__main__":
    main()
