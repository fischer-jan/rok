"""A colour picker beside a swatch: saturation and value, hue, ready colours
(your favourites, then this file's colours) and a button that puts the
colour back. The shader panel and the View Settings panel use it.

The picker works in sRGB 0..255, as the swatches show colours.
"""

from __future__ import annotations

import colorsys

import numpy as np

from .gfx import pen
from .object import document
from .paint import round_outline, round_rect, text

PICK_W = 236
PICK_PAD = 10
PICK_SV_H = 120
PICK_HUE_W = 14
PICK_CELL = 22
PICK_PER_ROW = 9
PICK_ROW = 24

PLATE = (0.075, 0.07, 0.066, 1.0)
EDGE = (0.228, 0.212, 0.196, 1.0)
EMBER = (0.910, 0.365, 0.227, 1.0)
BUTTON_BG = (1.0, 1.0, 1.0, 0.07)
BUTTON_HOVER = (0.208, 0.192, 0.180, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)


def quick_colours() -> list[tuple[int, int, int]]:
    """The picker's ready colours: your favourites, then the colours this file uses."""
    from . import palette_ui

    out = list(palette_ui.favourites())
    scene = document.scene
    if scene is not None:
        used = np.flatnonzero(document.used_counts()[1:]) + 1
        for i in used[:18]:
            c = tuple(int(v) for v in scene.palette[int(i), :3])
            if c not in out:
                out.append(c)
    return out[:27]


def layout(swatch, region, scale: float, quick: int) -> dict:
    """The picker's parts beside a swatch (region pixels, origin bottom left):
    "PANEL", "SV", "HUE", "Q:<k>" for the ready colours and "RESET"."""
    s = scale
    w = PICK_W * s
    pad = PICK_PAD * s
    inner = w - 2 * pad
    rows = -(-quick // PICK_PER_ROW) if quick else 0
    h = pad + PICK_SV_H * s + pad + rows * (PICK_CELL + 4) * s + PICK_ROW * s + pad
    x = min(max(4 * s, swatch[0] + swatch[2] / 2 - w / 2), region.width - w - 4 * s)
    y = swatch[1] + swatch[3] + 8 * s
    if y + h > region.height - 4 * s:
        y = swatch[1] - 8 * s - h
    parts = {"PANEL": (x, y, w, h)}
    top = y + h - pad
    sv_w = inner - PICK_HUE_W * s - 8 * s
    parts["SV"] = (x + pad, top - PICK_SV_H * s, sv_w, PICK_SV_H * s)
    parts["HUE"] = (x + pad + sv_w + 8 * s, top - PICK_SV_H * s, PICK_HUE_W * s, PICK_SV_H * s)
    top -= PICK_SV_H * s + pad
    cell = PICK_CELL * s
    gap = (inner - PICK_PER_ROW * cell) / max(1, PICK_PER_ROW - 1)
    for k in range(quick):
        r, c = divmod(k, PICK_PER_ROW)
        parts[f"Q:{k}"] = (x + pad + c * (cell + gap), top - (r + 1) * cell - r * 4 * s, cell, cell)
    top -= rows * (PICK_CELL + 4) * s
    parts["RESET"] = (x + pad, y + pad, inner, PICK_ROW * s)
    return parts


def hsv_of(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def rgb_of(hsv) -> tuple[int, int, int]:
    rgb = colorsys.hsv_to_rgb(*hsv)
    return tuple(int(v) for v in np.clip(np.round(np.asarray(rgb) * 255), 0, 255))


def dragged(hsv, part: str, rect, px: float, py: float) -> tuple[float, float, float]:
    """The colour after a press or drag at (px, py) on the "SV" square or the "HUE" bar."""
    h, sat, val = hsv
    x, y, w, hh = rect
    if part == "SV":
        sat = min(1.0, max(0.0, (px - x) / w))
        val = min(1.0, max(0.0, (py - y) / hh))
    else:
        h = min(0.999, max(0.0, (py - y) / hh))
    return (h, sat, val)


def part_at(parts: dict, px: float, py: float) -> str | None:
    return next((n for n, r in parts.items() if n != "PANEL" and _inside(r, px, py)), None)


def _inside(rect, px: float, py: float) -> bool:
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def draw(parts: dict, hsv, quick, current, hover: str | None, reset_text: str, scale: float) -> None:
    """The picker. `current` is the colour now (sRGB 0..255), ringed among the ready ones."""
    from rok.core import hud

    from . import palette_ui

    s = scale
    # The plate as triangles: a layer draws its triangles before its rounded
    # rects, and the gradients are triangles too; a rect plate would cover them.
    pen.fan(pen.to_3d(hud.rounded_rect(*parts["PANEL"], 10 * s)), PLATE)
    round_outline(parts["PANEL"], 10 * s, EDGE, 1.0 * s)
    h, sat, val = hsv
    palette_ui._sv_square(parts["SV"], h)
    palette_ui._hue_bar(parts["HUE"])
    sv = parts["SV"]
    mx, my = sv[0] + sat * sv[2], sv[1] + val * sv[3]
    round_outline((mx - 6 * s, my - 6 * s, 12 * s, 12 * s), 6 * s, (0.0, 0.0, 0.0, 0.8), 3.0 * s)
    round_outline((mx - 5 * s, my - 5 * s, 10 * s, 10 * s), 5 * s, (1.0, 1.0, 1.0, 1.0), 1.5 * s)
    hue = parts["HUE"]
    hy = hue[1] + h * hue[3]
    round_outline((hue[0] - 2 * s, hy - 3 * s, hue[2] + 4 * s, 6 * s), 2 * s, (0.0, 0.0, 0.0, 0.8), 2.5 * s)
    for k, c in enumerate(quick):
        rect = parts.get(f"Q:{k}")
        if rect is None:
            break
        round_rect(rect, 4 * s, tuple(v / 255.0 for v in c) + (1.0,))
        lit = c == current or hover == f"Q:{k}"
        round_outline(rect, 4 * s, EMBER if c == current else (TEXT if lit else EDGE), (2.0 if lit else 1.0) * s)
    reset = parts["RESET"]
    on = hover == "RESET"
    round_rect(reset, reset[3] / 2, BUTTON_HOVER if on else BUTTON_BG)
    text(reset_text, reset[0] + reset[2] / 2, reset[1] + reset[3] / 2, 11 * s, TEXT if on else TEXT_DIM, "centre")
