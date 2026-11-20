"""The shader settings panel's layout and slider arithmetic.

The panel opens when a shader is picked (the gallery, Cmd+K). On top: the
shader's picture, its name and what it makes. Below: a slider per argument,
bracketed to the range the shader's header gives and snapped to its step
(the old dialog let a 0-40 argument go to 1000), then small
steppers for Colours and Iterations, a note line, and Back / Run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geom2d import Rect, in_rect
from .shader_scope import FACES, FEATHER_MAX, ROOM_MAX, SCOPES, SHAPES

WIDTH = 560
PAD = 20
THUMB = 120
HEADER_GAP = 16
ROW_H = 34
LABEL_W = 150
VALUE_W = 70
TRACK_H = 6
STEP_BUTTON = 24
NOTE_H = 22
BUTTON_W = 110
BUTTON_H = 34
MARGIN = 24
SIDE_WIDTH = 420  # beside the model, so the live preview shows
SIDE_THUMB = 88
SIDE_BUTTON_W = 86  # three buttons beside the model: Back, Brush, Run
SIDE_BUTTON_GAP = 10
SIDE_BUTTON_W4 = 70  # four: Back, Brush, Live, Run
SIDE_BUTTON_GAP4 = 8
SIDE_LABEL_W = 110
PREVIEW_W, PREVIEW_H = 78, 22  # the Preview switch in the header
CHIP_GAP = 4
ONLY = 18  # a colour of the Only colours row
ONLY_GAP = 5
SWATCH = 26  # a colour of the shader's colour row (pick them)
SWATCH_GAP = 6


@dataclass(frozen=True)
class Control:
    """One row: an argument's slider, or a stepper (Colours, Iterations)."""

    key: str  # "a0" .. for an argument, "colour_count" or "iterations" for a stepper
    label: str
    low: float
    high: float
    step: float
    precision: int
    stepper: bool
    row: Rect
    label_rect: Rect
    track: Rect  # the slider's bar; for a stepper the value between its buttons
    value_rect: Rect
    minus: Rect | None = None
    plus: Rect | None = None
    tip: str = ""


@dataclass(frozen=True)
class PanelLayout:
    rect: Rect
    thumb: Rect
    title: Rect
    about: Rect
    controls: tuple[Control, ...]
    notes: tuple[Rect, ...]
    back: Rect
    run: Rect
    divider: float  # y of the line under the header
    scale: float
    swatches: tuple[Rect, ...] = ()  # the shader's own colours, to pick
    colours_label: Rect | None = None
    # Where it runs: a row that opens into the rows below.
    where_row: Rect | None = None
    scope_chips: tuple[tuple[str, Rect], ...] = ()
    face_chips: tuple[tuple[str, Rect], ...] = ()
    only_swatches: tuple[Rect, ...] = ()
    where_labels: tuple[tuple[str, Rect], ...] = ()  # "Runs on", "Only on", "Only colours"
    brush: Rect | None = None  # the shader brush, beside the model only
    shape_chips: tuple[tuple[str, Rect], ...] = ()  # Box, Sphere: with a volume only
    live: Rect | None = None  # run as a live shader, between Brush and Run
    preview_switch: Rect | None = None  # beside the model: the live preview on or off

    def control(self, key: str) -> Control | None:
        return next((c for c in self.controls if c.key == key), None)


@dataclass(frozen=True)
class Spec:
    """What a row shows: an argument (a slider) or a stepper."""

    key: str
    label: str
    low: float
    high: float
    step: float = 1.0
    precision: int = 0
    stepper: bool = False
    tip: str = ""  # shown when the mouse rests on the label


def colour_rows(colours: int, scale: float, width: float, label_w: float = LABEL_W,
                size: float = SWATCH, gap: float = SWATCH_GAP) -> tuple[int, int]:
    """(swatches per row, rows) for `colours` swatches in a panel `width` wide."""
    if colours <= 0:
        return 0, 0
    s = float(scale)
    room = width - 2 * PAD * s - label_w * s
    per_row = max(1, int((room + gap * s) // ((size + gap) * s)))
    return per_row, -(-colours // per_row)


def _chips(keys, x: float, y: float, w: float, h: float, gap: float) -> tuple[tuple[str, Rect], ...]:
    """Equal chips side by side in (x, y, w, h)."""
    n = len(keys)
    cw = (w - gap * (n - 1)) / n
    return tuple((k, (x + i * (cw + gap), y, cw, h)) for i, k in enumerate(keys))


def panel_layout(scale: float, region_width: int, region_height: int, specs, notes: int = 1,
                 colours: int = 0, side_top: float | None = None, where_open: bool | None = None,
                 where_colours: int = 0, shape_row: bool = False, live_button: bool = False) -> PanelLayout:
    """The panel in the middle of the region for these rows, the shader's
    `colours` (a row of swatches) and note lines. With `side_top` it stands
    at the right edge instead, its top at `side_top` or as near under it as
    the region allows. `where_open` adds the Where row (None: none), open
    into Runs on, Only on, Only colours (`where_colours` swatches), Room and
    Feather. `live_button` adds Live beside the model."""
    s = float(scale)
    side = side_top is not None
    label_w = SIDE_LABEL_W if side else LABEL_W
    width = min((WIDTH if side_top is None else SIDE_WIDTH) * s, region_width - 2 * MARGIN * s)
    pad = PAD * s
    header = (SIDE_THUMB if side else THUMB) * s
    rows = len(specs)
    per_row, colour_lines = colour_rows(colours, s, width, label_w)
    only_per_row, only_lines = colour_rows(where_colours, s, width, label_w, ONLY, ONLY_GAP)
    where_rows = 0 if where_open is None else (1 + (4 + max(1, only_lines) + int(shape_row) if where_open else 0))
    height = (pad + header + HEADER_GAP * s + rows * ROW_H * s + colour_lines * ROW_H * s
              + where_rows * ROW_H * s + notes * NOTE_H * s + pad + BUTTON_H * s + pad)
    if side_top is None:
        x = (region_width - width) / 2.0
        y = max(MARGIN * s, (region_height - height) / 2.0)
    else:
        x = max(MARGIN * s, region_width - width - MARGIN * s)
        y = max(MARGIN * s, min(side_top, region_height - MARGIN * s) - height)
    top = y + height
    thumb = (x + pad, top - pad - header, header, header)
    text_x = thumb[0] + header + pad
    text_w = x + width - pad - text_x
    switch_w, switch_h = PREVIEW_W * s, PREVIEW_H * s
    preview_switch = (x + width - pad - switch_w, top - pad - switch_h, switch_w, switch_h) if side else None
    title = (text_x, top - pad - 30 * s, text_w - ((switch_w + 8 * s) if side else 0), 30 * s)
    about = (text_x, top - pad - header, text_w, header - 34 * s)
    divider = top - pad - header - HEADER_GAP * s / 2
    left = x + pad  # where a row's label starts
    inner = left + label_w * s  # where its control starts
    inner_w = x + width - pad - inner

    def control(spec: Spec, row_y: float) -> Control:
        row = (left, row_y, width - 2 * pad, ROW_H * s)
        label_rect = (left, row_y, label_w * s, ROW_H * s)
        value_rect = (x + width - pad - VALUE_W * s, row_y, VALUE_W * s, ROW_H * s)
        track_w = value_rect[0] - 12 * s - inner
        if spec.stepper:
            b = STEP_BUTTON * s
            cy = row_y + (ROW_H * s - b) / 2
            minus = (inner, cy, b, b)
            track = (inner + b, row_y, 48 * s, ROW_H * s)
            plus = (track[0] + track[2], cy, b, b)
            return Control(spec.key, spec.label, spec.low, spec.high, spec.step, spec.precision, True,
                           row, label_rect, track, value_rect, minus, plus, spec.tip)
        track = (inner, row_y + (ROW_H * s - TRACK_H * s) / 2, track_w, TRACK_H * s)
        return Control(spec.key, spec.label, spec.low, spec.high, spec.step, spec.precision, False,
                       row, label_rect, track, value_rect, tip=spec.tip)

    controls = []
    row_y = top - pad - header - HEADER_GAP * s
    for spec in specs:
        row_y -= ROW_H * s
        controls.append(control(spec, row_y))
    swatches = []
    colours_label = None
    if colours > 0:
        top_row = row_y
        for k in range(colours):
            line, col = divmod(k, per_row)
            sy = top_row - (line + 1) * ROW_H * s + (ROW_H * s - SWATCH * s) / 2
            swatches.append((inner + col * (SWATCH + SWATCH_GAP) * s, sy, SWATCH * s, SWATCH * s))
        colours_label = (left, top_row - ROW_H * s, label_w * s, ROW_H * s)
        row_y = top_row - colour_lines * ROW_H * s
    where_row = None
    scope_chips: tuple = ()
    shape_chips: tuple = ()
    face_chips: tuple = ()
    only = []
    labels = []
    if where_open is not None:
        row_y -= ROW_H * s
        where_row = (left, row_y, width - 2 * pad, ROW_H * s)
        if where_open:
            chip_h = 24 * s
            row_y -= ROW_H * s
            labels.append(("Runs on", (left, row_y, label_w * s, ROW_H * s)))
            scope_chips = _chips(SCOPES, inner, row_y + (ROW_H * s - chip_h) / 2, inner_w, chip_h, CHIP_GAP * s)
            if shape_row:
                row_y -= ROW_H * s
                labels.append(("Shape", (left, row_y, label_w * s, ROW_H * s)))
                shape_chips = _chips(SHAPES, inner, row_y + (ROW_H * s - chip_h) / 2, inner_w * 0.5, chip_h,
                                     CHIP_GAP * s)
            row_y -= ROW_H * s
            labels.append(("Only on", (left, row_y, label_w * s, ROW_H * s)))
            face_chips = _chips(FACES, inner, row_y + (ROW_H * s - chip_h) / 2, inner_w, chip_h, CHIP_GAP * s)
            top_row = row_y
            labels.append(("Only colours", (left, top_row - ROW_H * s, label_w * s, ROW_H * s)))
            for k in range(where_colours):
                line, col = divmod(k, only_per_row)
                sy = top_row - (line + 1) * ROW_H * s + (ROW_H * s - ONLY * s) / 2
                only.append((inner + col * (ONLY + ONLY_GAP) * s, sy, ONLY * s, ONLY * s))
            row_y = top_row - max(1, only_lines) * ROW_H * s
            row_y -= ROW_H * s
            controls.append(control(Spec("room", "Room", 0, ROOM_MAX, 1, 0, tip=ROOM_TIP), row_y))
            row_y -= ROW_H * s
            controls.append(control(Spec("feather", "Feather", 0, FEATHER_MAX, 1, 0, tip=FEATHER_TIP), row_y))
    note_rects = []
    for _k in range(notes):
        row_y -= NOTE_H * s
        note_rects.append((left, row_y, width - 2 * pad, NOTE_H * s))
    four = side and live_button
    bw, gap = ((SIDE_BUTTON_W4, SIDE_BUTTON_GAP4) if four else (SIDE_BUTTON_W, SIDE_BUTTON_GAP) if side
               else (BUTTON_W, 12))
    run = (x + width - pad - bw * s, y + pad, bw * s, BUTTON_H * s)
    live = (run[0] - gap * s - bw * s, y + pad, bw * s, BUTTON_H * s) if four else None
    brush = ((live or run)[0] - gap * s - bw * s, y + pad, bw * s, BUTTON_H * s) if side else None
    back_x = (brush[0] if brush is not None else run[0]) - gap * s - bw * s
    back = (back_x, y + pad, bw * s, BUTTON_H * s)
    return PanelLayout((x, y, width, height), thumb, title, about, tuple(controls), tuple(note_rects),
                       back, run, divider, s, tuple(swatches), colours_label,
                       where_row, scope_chips, face_chips, tuple(only), tuple(labels), brush, shape_chips, live,
                       preview_switch)


def snap(value: float, low: float, high: float, step: float, precision: int) -> float:
    """The value inside [low, high], on the step grid from `low`, rounded to
    `precision` places (a whole number at 0)."""
    value = min(max(float(value), low), high)
    if step > 0:
        value = low + round((value - low) / step) * step
        value = min(max(value, low), high)
    value = round(value, max(0, int(precision)))
    return float(int(value)) if precision <= 0 else value


def value_at(control: Control, x: float) -> float:
    """The slider's value under a pointer at x, snapped."""
    tx, _ty, tw, _th = control.track
    t = 0.0 if tw <= 0 else (x - tx) / tw
    return snap(control.low + t * (control.high - control.low), control.low, control.high, control.step,
                control.precision)


def x_of(control: Control, value: float) -> float:
    """Where the knob sits for a value."""
    tx, _ty, tw, _th = control.track
    span = control.high - control.low
    t = 0.0 if span == 0 else (min(max(value, control.low), control.high) - control.low) / span
    return tx + t * tw


def nudged(control: Control, value: float, steps: int) -> float:
    """The value `steps` steps on (a wheel click, a stepper button)."""
    step = control.step if control.step > 0 else 10.0 ** -max(0, control.precision)
    return snap(value + steps * step, control.low, control.high, control.step, control.precision)


def fine(control: Control, start_value: float, start_x: float, x: float) -> float:
    """A Shift drag: a tenth of the pointer's way, from where it started."""
    tx, _ty, tw, _th = control.track
    span = control.high - control.low
    delta = 0.0 if tw <= 0 else (x - start_x) / tw * span / 10.0
    return snap(start_value + delta, control.low, control.high, control.step, control.precision)


def text_of(value: float, precision: int) -> str:
    return f"{int(round(value))}" if precision <= 0 else f"{value:.{precision}f}"


def hit(layout: PanelLayout, x: float, y: float) -> tuple[str, str] | None:
    """What is under a point: ("RUN", ""), ("BACK", ""), ("BRUSH", ""), ("LIVE", ""), ("COLOUR", k), ("SCOPE", scope),
    ("FACE", face), ("ONLY", k) for an Only colours swatch, ("WHERE", "") for the
    Where row, ("MINUS", key),
    ("PLUS", key), ("TRACK", key) for a slider row, ("LABEL", key) for a row's name, ("PANEL", "") elsewhere
    in the panel, or None outside it."""
    if layout.preview_switch is not None and in_rect(layout.preview_switch, x, y):
        return ("PREVIEW", "")
    if in_rect(layout.run, x, y):
        return ("RUN", "")
    if in_rect(layout.back, x, y):
        return ("BACK", "")
    if layout.brush is not None and in_rect(layout.brush, x, y):
        return ("BRUSH", "")
    if layout.live is not None and in_rect(layout.live, x, y):
        return ("LIVE", "")
    for k, rect in enumerate(layout.swatches):
        if in_rect(rect, x, y):
            return ("COLOUR", str(k))
    for kind, chips in (("SCOPE", layout.scope_chips), ("SHAPE", layout.shape_chips), ("FACE", layout.face_chips)):
        for key, rect in chips:
            if in_rect(rect, x, y):
                return (kind, key)
    for k, rect in enumerate(layout.only_swatches):
        if in_rect(rect, x, y):
            return ("ONLY", str(k))
    if layout.where_row is not None and in_rect(layout.where_row, x, y):
        return ("WHERE", "")
    for c in layout.controls:
        if in_rect(c.label_rect, x, y):
            return ("LABEL", c.key)
        if c.stepper:
            if in_rect(c.minus, x, y):
                return ("MINUS", c.key)
            if in_rect(c.plus, x, y):
                return ("PLUS", c.key)
        elif in_rect(c.row, x, y) and x >= c.track[0] - 8 * layout.scale:
            return ("TRACK", c.key)
    if in_rect(layout.rect, x, y):
        return ("PANEL", "")
    return None


TIP_W = 300  # a row's tooltip box, in points
TIP_PAD = 10
TIP_LINE = 16
TIP_LINES = 24  # the most lines a tooltip shows
ROOM_TIP = ("How many empty cells above the allowed voxels a shader may fill. Only on and Only colours "
            "allow filled voxels only, so a shader that adds (grass, trees, snow) needs Room to add "
            "anything. 0: it changes only voxels that are there.")
FEATHER_TIP = ("A soft border, in cells: the shader's change fades out over this distance from the edge "
               "of the allowed area instead of stopping at a hard line. 0: a hard edge.")


def tip_rect(layout: PanelLayout, control: Control, lines: int, region_width: float) -> Rect:
    """Where a row's tooltip of `lines` lines goes: left of the panel, its top at
    the row's top, pushed inside the region."""
    s = layout.scale
    w = TIP_W * s
    h = (2 * TIP_PAD + lines * TIP_LINE) * s
    x = layout.rect[0] - 8 * s - w
    if x < 0:
        x = min(layout.rect[0] + layout.rect[2] + 8 * s, max(0.0, region_width - w))
    top = control.row[1] + control.row[3]
    y = max(0.0, top - h)
    return (x, y, w, h)


def wrap(text: str, width_of, width: float, lines: int = 2) -> list[str]:
    """`text` in at most `lines` lines that fit `width` (width_of(s) measures);
    the last one cut with an ellipsis when it does not fit."""
    words = text.split()
    out: list[str] = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if width_of(trial) <= width or not line:
            line = trial
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    if len(out) > lines:
        out = out[:lines]
        last = out[-1]
        while last and width_of(last + "…") > width:
            last = last[:-1]
        out[-1] = last + "…"
    return out


def span_steps(low: float, high: float, step: float) -> int:
    """How many steps a range holds (for a slider's tick marks)."""
    if step <= 0:
        return 0
    return int(math.floor((high - low) / step + 1e-9))
