"""The welcome screen's layout: the logo and the name on top, then New and Open on
the left and the recent files on the right."""

from __future__ import annotations

import os
from dataclasses import dataclass

WIDTH = 820  # the panel's width at scale 1
HEADER_H = 112  # the part with the logo and the name
LOWER_H = 196  # the part under the header
PAD = 26
MARGIN = 24  # between the panel and the view's edge
BUTTON_H = 58
BUTTON_GAP = 12
ROW_H = 30
MAX_RECENT = 5
LOGO = 84  # the logo's side, at scale 1
LOGO_GAP = 14  # above and below the logo, and between the logo and the name

Rect = tuple[float, float, float, float]

NEW = "NEW"
OPEN = "OPEN"


@dataclass(frozen=True)
class RecentRow:
    path: str
    name: str  # the file's name
    folder: str  # the folder's name, for the grey text after it
    rect: Rect


@dataclass(frozen=True)
class WelcomeLayout:
    rect: Rect
    header: Rect
    logo: Rect  # left in the header, left of the wordmark
    title: tuple[float, float]  # the wordmark's left edge and its middle line
    new_button: Rect
    open_button: Rect
    start_caption: Rect
    recent_caption: Rect
    rows: tuple[RecentRow, ...]
    hint: Rect
    scale: float


def _inside(rect: Rect, x: float, y: float) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def welcome_layout(scale: float, region_width: int, region_height: int, recent) -> WelcomeLayout:
    """The panel for a region of this size, centred. `recent` are file paths,
    newest first; the first MAX_RECENT show."""
    s = float(scale)
    margin, pad = MARGIN * s, PAD * s
    width = min(WIDTH * s, region_width - 2 * margin)
    # Too low a region: the header gets lower, never the buttons.
    header_h = min(HEADER_H * s, region_height - 2 * margin - LOWER_H * s)
    header_h = max(header_h, 60 * s)
    height = header_h + LOWER_H * s
    x = (region_width - width) / 2.0
    y = (region_height - height) / 2.0
    rect = (x, y, width, height)
    header = (x, y + LOWER_H * s, width, header_h)
    middle = y + LOWER_H * s + header_h / 2.0
    logo_side = min(LOGO * s, header_h - 2 * LOGO_GAP * s)
    logo = (x + pad, middle - logo_side / 2.0, logo_side, logo_side)
    title = (x + pad + logo_side + LOGO_GAP * s, middle)

    lower_top = y + LOWER_H * s - 14 * s
    col_w = (width - 3 * pad) / 2.0
    left = x + pad
    right = x + 2 * pad + col_w
    caption_h = 22 * s
    start_caption = (left, lower_top - caption_h, col_w, caption_h)
    recent_caption = (right, lower_top - caption_h, col_w, caption_h)
    by = lower_top - caption_h - 6 * s
    new_button = (left, by - BUTTON_H * s, col_w, BUTTON_H * s)
    open_button = (left, by - 2 * BUTTON_H * s - BUTTON_GAP * s, col_w, BUTTON_H * s)
    rows = []
    for k, path in enumerate(list(recent)[:MAX_RECENT]):
        ry = by - (k + 1) * ROW_H * s
        rows.append(RecentRow(path, os.path.basename(path), os.path.basename(os.path.dirname(path)),
                              (right, ry, col_w, ROW_H * s)))
    hint = (x + pad, y, width - 2 * pad, 26 * s)
    return WelcomeLayout(rect, header, logo, title, new_button, open_button, start_caption, recent_caption,
                         tuple(rows), hint, s)


def target_at(layout: WelcomeLayout, x: float, y: float):
    """What a click at (x, y) does: NEW, OPEN, a recent row, or None."""
    if _inside(layout.new_button, x, y):
        return NEW
    if _inside(layout.open_button, x, y):
        return OPEN
    return next((row for row in layout.rows if _inside(row.rect, x, y)), None)


def in_panel(layout: WelcomeLayout, x: float, y: float) -> bool:
    return _inside(layout.rect, x, y)
