"""Small drawing helpers for Rok's own UI: rounded plates, outlines and text.

The star, the welcome screen and the colour
popup use them. Positions are region pixels with the origin at the bottom
left; call them inside a `gfx.pixels` pass. Rounded plates use the host's own
rounded rectangles, so their edges are smooth.
"""

from __future__ import annotations

from . import gfx
from .gfx import fonts, current
from .opkit import CONTEXT

FONT = 0


def _rgba(colour) -> tuple:
    c = tuple(float(v) for v in colour)
    return c if len(c) == 4 else c + (1.0,)


def _flip_y(y: float, h: float) -> float:
    return float(CONTEXT.region.height) - float(y) - float(h)


def _placed(rect, radius: float, outline: bool):
    """The rect moved by gfx.shifted and cut by gfx.clipped, or None when
    nothing of it shows. A cut outline is left out: it would draw a false edge."""
    x, y, w, h = (float(v) for v in rect)
    if w <= 0 or h <= 0:
        return None
    dx, dy = gfx._t.off2
    x, y = x + dx, y + dy
    c = gfx._t.clip
    if c is not None:
        x0, y0 = max(x, c[0]), max(y, c[1])
        x1, y1 = min(x + w, c[0] + c[2]), min(y + h, c[1] + c[3])
        if x1 - x0 <= 0.5 or y1 - y0 <= 0.5:
            return None
        if (x0, y0, x1, y1) != (x, y, x + w, y + h):
            if outline:
                return None
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            radius = min(float(radius), w / 2.0, h / 2.0)
    return x, y, w, h, float(radius)


def _faded(colour) -> tuple:
    c = _rgba(colour)
    return c[:3] + (c[3] * gfx._t.alpha,)


def round_rect(rect, radius: float, colour) -> None:
    """A filled rectangle with round corners."""
    placed = _placed(rect, radius, False)
    if placed is None:
        return
    x, y, w, h, r = placed
    current().rect(x, _flip_y(y, h), w, h, _faded(colour), radius=r)


def round_outline(rect, radius: float, colour, width: float) -> None:
    """The outline of a rectangle with round corners, `width` pixels wide."""
    placed = _placed(rect, radius, True)
    if placed is None:
        return
    x, y, w, h, r = placed
    current().rect(x, _flip_y(y, h), w, h, _faded(colour), radius=r, stroke=float(width))


def text(value: str, x: float, cy: float, size: float, colour, align: str = "left") -> float:
    """`value` with its middle on `cy`; returns its width. `align` says where x
    is: "left", "centre" or "right"."""
    fonts.size(FONT, size)
    tw, _th = fonts.dimensions(FONT, value)
    _w, cap = fonts.dimensions(FONT, "H")
    if align == "centre":
        x -= tw / 2.0
    elif align == "right":
        x -= tw
    fonts.color(FONT, *colour)
    fonts.position(FONT, float(round(x)), float(round(cy - cap / 2.0)), 0)
    fonts.draw(FONT, value)
    return tw


def text_width(value: str, size: float) -> float:
    fonts.size(FONT, size)
    return fonts.dimensions(FONT, value)[0]


def clip_text(value: str, width: float, size: float) -> str:
    """`value`, cut with an ellipsis to fit `width`."""
    if text_width(value, size) <= width:
        return value
    cut = value
    while cut and text_width(cut + "…", size) > width:
        cut = cut[:-1]
    return cut + "…" if cut else ""
