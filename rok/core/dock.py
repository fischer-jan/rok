"""The timeline dock's geometry and hit testing.

The dock runs along the bottom of the view in Animate mode, as one card:

    ┌──────────────────────────────────────────────────────────────────────┐
    │ ⏮ ▶ Loop   12 / 24   1.0 s  24 fps     Hold Linear Smooth In Out  Voxel │  the bar
    │ PARTS            0     4     8     12    16    20    24                 │  the ruler
    │ ▾ Propeller      ◆──────────────────────────────────◆                   │  a part
    │     Turn Z  90°  ◆──────────────────────────────────◆                   │  its channels
    └──────────────────────────────────────────────────────────────────────┘

The bar holds the transport (to the start, play), the loop mode, the frame
readout, the clip's length and frame rate, the ease of the picked keys and
the Voxel look switch. The left column names the parts; an open part lists
its channels with the value at the play head. Right of the column the keys
sit on their rows, and the play head crosses every row.

Measurements are in points; `dock_layout` multiplies by the UI scale.
Region pixels have y up: a rect is (x, bottom, width, height). numpy-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

from . import anim

Rect = tuple[float, float, float, float]

MARGIN = 12  # from the view's edges
PAD = 10
TAB_H = 26  # the clip tabs over the bar
BAR_H = 34
RULER_H = 22
ROW_H = 24
NAME_W = 170
BUTTON_H = 24
GAP = 6
KEY_R = 5.5  # a key's half size
INDENT = 14  # a child part's name sits this much right of its parent's
MAX_ROWS = 8

BUTTONS = ("START", "PLAY", "LOOP")
MARKER_W = 9  # a marker's triangle, points
EASE_IDS = tuple(f"EASE_{e}" for e in anim.EASES)
EASE_SHORT = {anim.HOLD: "Hold", anim.LINEAR: "Linear", anim.SMOOTH: "Smooth", anim.EASE_IN: "In",
              anim.EASE_OUT: "Out"}
LOOP_LABELS = {anim.LOOP: "Loop", anim.ONCE: "Once", anim.PING_PONG: "Ping-pong"}

# Hit kinds.
BUTTON, RULER, KEY, ROW, OPENER, CARD, MARKER = "BUTTON", "RULER", "KEY", "ROW", "OPENER", "CARD", "MARKER"

PART_ROW, CHANNEL_ROW, FRAMES_ROW, COLOUR_ROW = "PART", "CHANNEL", "FRAMES", "COLOUR"
MARKER_ROW = "MARKERS"  # the channel of a ROW hit in the markers row (uid -1)


@dataclass(frozen=True)
class Button:
    id: str
    rect: Rect
    label: str
    on: bool = False  # drawn lit: the playing Play, Voxel look on, the picked keys' ease


@dataclass(frozen=True)
class Row:
    kind: str  # PART_ROW or CHANNEL_ROW
    uid: int
    channel: str | None  # None on a part's own row
    label: str
    rect: Rect  # the whole row
    keys: tuple[tuple[int, str], ...]  # (frame, ease); a part's row: every channel's frames
    open: bool = False
    picked: bool = False  # the picked part's own row
    value: str = ""  # a channel row: its value at the play head
    depth: int = 0  # how far the part sits under its parents (indent)
    values: tuple = ()  # a Frames row: which frame each key shows (1-based), in key order


@dataclass(frozen=True)
class Hit:
    kind: str
    id: str = ""  # BUTTON
    uid: int = 0  # KEY, ROW, OPENER
    channel: str | None = None  # KEY, ROW: None on a part's row
    frame: int = 0  # KEY, RULER


@dataclass(frozen=True)
class DockLayout:
    rect: Rect
    bar: Rect
    tabs: Rect  # the clip tabs' strip over the bar
    ruler: Rect
    lane: tuple[float, float]  # x of frame 0 and of frame `length`
    rows: tuple[Row, ...]
    buttons: tuple[Button, ...]
    readout: Rect  # the frame readout's place in the bar
    length: int
    scale: float
    hint: str = ""  # a line in the rows' place when there is nothing to show
    markers_row: Rect = (0.0, 0.0, 0.0, 0.0)  # the Markers row under the parts
    markers: tuple[tuple[int, str], ...] = ()  # (frame, name)

    def x_of(self, frame: float) -> float:
        x0, x1 = self.lane
        return x0 + (x1 - x0) * float(frame) / max(1, self.length)

    def frame_at(self, x: float) -> int:
        """The whole frame nearest x, inside the clip (0 .. length)."""
        x0, x1 = self.lane
        f = round((x - x0) / max(1e-6, x1 - x0) * self.length)
        return int(min(max(f, 0), self.length))

    def button(self, ident: str) -> Button | None:
        return next((b for b in self.buttons if b.id == ident), None)


def _in(rect: Rect, x: float, y: float) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def height(rows: int, scale: float) -> float:
    shown = max(1, min(MAX_ROWS, rows))
    return round((PAD + TAB_H + BAR_H + RULER_H + (shown + 1) * ROW_H + PAD) * scale)  # + the Markers row


def tab_id(index: int) -> str:
    return f"TAB_{index}"


TAB_ADD = "TAB_ADD"


def value_text(name: str, value) -> str:
    """A channel's value as the dock shows it: 90°, +2, 100 %."""
    if name in anim.TURN:
        return f"{float(value):.0f}°" if abs(value - round(value)) < 0.05 else f"{float(value):.1f}°"
    if name in anim.MOVE:
        v = float(value)
        text = f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
        return ("+" + text) if v > 0 else text
    if name == anim.SCALE:
        return f"{float(value[0]):.0f} %"
    if name == anim.VISIBLE:
        return "shown" if value else "hidden"
    return str(value)


def dock_layout(scale: float, left: float, width: float, clip: anim.Clip, parts, picked_uid: int,
                opened=frozenset(), frame: int = 0, playing: bool = False, ease: str | None = None,
                clips=None, active_clip: int = 0, onion: bool = False, frames=None) -> DockLayout:
    """The dock between x = `left` and `left + width`, MARGIN above the view's
    bottom. `parts` are (uid, name) or (uid, name, depth) of the scene's parts
    in tree order: every part gets a row, indented by its depth. `opened`
    holds the uids whose channel rows show. `ease` lights the picked keys'
    ease button. `clips` are the clips' names for the tabs (default: the one
    clip's); `active_clip` is the lit tab."""
    s = float(scale)
    if clips is None:
        clips = [clip.name]
    frames = frames or {}  # uid -> how many frames the part has; a part with more than one gets a Frames row

    def px(v: float) -> float:
        return round(v * s)

    tracks = {t.target.uid: t for t in clip.tracks if t.target.kind == anim.PART}
    row_specs: list[tuple] = []
    for entry in parts:
        uid, name = entry[0], entry[1]
        depth = int(entry[2]) if len(entry) > 2 else 0
        track = tracks.get(uid)
        has = track is not None and any(p.keys for p in track.properties.values())
        open_ = uid in opened and has
        keys: dict[int, str] = {}
        if has:
            for prop in track.properties.values():
                for k in prop.keys:
                    keys.setdefault(k.time, k.ease)
        row_specs.append((PART_ROW, uid, None, name, tuple(sorted(keys.items())), open_, uid == picked_uid, "", depth))
        if frames.get(uid, 1) > 1:
            state_prop = track.properties.get(anim.STATE) if track is not None else None
            state_keys = tuple((k.time, k.ease) for k in state_prop.keys) if state_prop is not None else ()
            values = tuple(int(k.value) + 1 for k in state_prop.keys) if state_prop is not None else ()
            row_specs.append((FRAMES_ROW, uid, anim.STATE, "Frames", state_keys, False, False,
                              f"{frames[uid]}", depth, values))
        if open_:
            for channel in anim.PROPERTIES_OF[anim.PART]:
                prop = track.properties.get(channel)
                if prop is None or not prop.keys:
                    continue
                value = value_text(channel, anim.value_at(prop, clip.time_at(frame)))
                if channel == anim.STATE and frames.get(uid, 1) > 1:
                    continue  # the Frames row shows it
                row_specs.append((CHANNEL_ROW, uid, channel, anim.CHANNEL_LABELS[channel],
                                  tuple((k.time, k.ease) for k in prop.keys), False, False, value, depth))
    for track in clip.tracks:  # colour motions: a row each, read-only
        if track.target.kind != anim.COLOUR:
            continue
        keys = {}
        for prop in track.properties.values():
            for k in prop.keys:
                keys.setdefault(k.time, k.ease)
        if not keys:
            continue
        label = track.name or "Colour " + ", ".join(str(s) for s in track.target.slots[:3])
        row_specs.append((COLOUR_ROW, -1, COLOUR_ROW, label, tuple(sorted(keys.items())), False, False, "", 0))
    h = height(len(row_specs), s)
    rect = (float(left), float(px(MARGIN)), float(width), float(h))
    x, y, w, _h = rect
    top = y + h - px(PAD)
    tabs = (x + px(PAD), top - px(TAB_H), w - 2 * px(PAD), px(TAB_H))
    bar = (x + px(PAD), tabs[1] - px(BAR_H), w - 2 * px(PAD), px(BAR_H))
    ruler = (x + px(PAD), bar[1] - px(RULER_H), w - 2 * px(PAD), px(RULER_H))
    lane = (x + px(PAD) + px(NAME_W), x + w - px(PAD) - px(KEY_R) * 2)

    # The clip tabs, then the bar, left to right, then the right group right to left.
    buttons: list[Button] = []
    tx = tabs[0]
    ty = tabs[1] + (tabs[3] - px(BUTTON_H)) / 2
    for i, name in enumerate(clips):
        tw = px(min(160, max(56, 14 + 7 * len(name))))
        buttons.append(Button(tab_id(i), (tx, ty, tw, px(BUTTON_H)), name, on=i == active_clip))
        tx += tw + px(2)
    buttons.append(Button(TAB_ADD, (tx + px(GAP), ty, px(26), px(BUTTON_H)), "+"))
    bx = bar[0]
    by = bar[1] + (bar[3] - px(BUTTON_H)) / 2
    for ident, label, bw in (("START", "⏮", 30), ("PLAY", "■" if playing else "▶", 30),
                             ("LOOP", LOOP_LABELS[clip.loop_mode], 78)):
        buttons.append(Button(ident, (bx, by, px(bw), px(BUTTON_H)), label, on=ident == "PLAY" and playing))
        bx += px(bw) + px(GAP)
    readout = (bx + px(GAP), by, px(78), px(BUTTON_H))
    bx = readout[0] + readout[2] + px(GAP)
    for ident, label, bw in (("LENGTH", f"{clip.seconds:.2g} s", 56), ("FPS", f"{clip.fps} fps", 56)):
        buttons.append(Button(ident, (bx, by, px(bw), px(BUTTON_H)), label))
        bx += px(bw) + px(GAP)
    rx = bar[0] + bar[2]
    look_w = px(92)
    rx -= look_w
    buttons.append(Button("VOXEL_LOOK", (rx, by, look_w, px(BUTTON_H)), "Voxel look", on=clip.voxel_look))
    rx -= px(GAP)
    onion_w = px(62)
    rx -= onion_w
    buttons.append(Button("ONION", (rx, by, onion_w, px(BUTTON_H)), "Onion", on=onion))
    rx -= px(GAP) * 3
    for e in reversed(anim.EASES):
        bw = px(58 if e == anim.SMOOTH else 50)
        rx -= bw
        buttons.append(Button(f"EASE_{e}", (rx, by, bw, px(BUTTON_H)), EASE_SHORT[e], on=ease == e))
        rx -= px(2)

    rows: list[Row] = []
    ry = ruler[1]
    for spec in row_specs[:MAX_ROWS]:
        kind, uid, channel, label, keys, open_, picked, value, depth = spec[:9]
        values = spec[9] if len(spec) > 9 else ()
        ry -= px(ROW_H)
        rows.append(Row(kind, uid, channel, label, (x + px(PAD), ry, w - 2 * px(PAD), px(ROW_H)), keys,
                        open_, picked, value, depth, values))
    hint = ""
    if not rows:
        hint = "Pick a part: click it in the view"
        ry -= px(ROW_H)
    ry -= px(ROW_H)
    markers_row = (x + px(PAD), ry, w - 2 * px(PAD), px(ROW_H))
    markers = tuple(sorted((int(m.time), str(m.name)) for m in clip.markers))
    return DockLayout(rect, bar, tabs, ruler, lane, tuple(rows), tuple(buttons), readout, max(1, int(clip.length)),
                      s, hint, markers_row, markers)


def hit(layout: DockLayout, x: float, y: float) -> Hit | None:
    """What a point in the dock hits, or None outside it. A point inside that
    hits nothing is Hit("DOCK")."""
    if not _in(layout.rect, x, y):
        return None
    for b in layout.buttons:
        if _in(b.rect, x, y):
            return Hit(BUTTON, id=b.id)
    lx0, lx1 = layout.lane
    r = KEY_R * layout.scale + 2 * layout.scale
    if _in(layout.ruler, x, y) and x >= lx0 - r:
        return Hit(RULER, frame=layout.frame_at(x))
    for row in layout.rows:
        if not _in(row.rect, x, y):
            continue
        if x >= lx0 - r:
            near = [(abs(layout.x_of(t) - x), t) for t, _e in row.keys]
            near = [n for n in near if n[0] <= r]
            if near:
                return Hit(KEY, uid=row.uid, channel=row.channel, frame=min(near)[1])
            return Hit(ROW, uid=row.uid, channel=row.channel, frame=layout.frame_at(x))
        if row.kind == PART_ROW and x < row.rect[0] + (18 + INDENT * row.depth) * layout.scale:
            return Hit(OPENER, uid=row.uid)
        return Hit(ROW, uid=row.uid, channel=row.channel, frame=-1)
    if _in(layout.markers_row, x, y):
        if x >= lx0 - r:
            near = [(abs(layout.x_of(t) - x), t) for t, _n in layout.markers]
            near = [n for n in near if n[0] <= MARKER_W * layout.scale]
            if near:
                return Hit(MARKER, uid=-1, channel=MARKER_ROW, frame=min(near)[1])
            return Hit(ROW, uid=-1, channel=MARKER_ROW, frame=layout.frame_at(x))
        return Hit(ROW, uid=-1, channel=MARKER_ROW, frame=-1)
    return Hit("DOCK")


def key_shape(ease: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """A key's outline by its ease: a square holds, a diamond blends evenly,
    a rounded diamond is smooth, a half-flat one eases in or out."""
    if ease == anim.HOLD:
        return [(cx - r * 0.82, cy - r * 0.82), (cx + r * 0.82, cy - r * 0.82), (cx + r * 0.82, cy + r * 0.82),
                (cx - r * 0.82, cy + r * 0.82)]
    if ease == anim.LINEAR:
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if ease == anim.EASE_IN:
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r * 0.6, cy + r * 0.6), (cx - r * 0.6, cy - r * 0.6)]
    if ease == anim.EASE_OUT:
        return [(cx, cy - r), (cx + r * 0.6, cy - r * 0.6), (cx + r * 0.6, cy + r * 0.6), (cx, cy + r), (cx - r, cy)]
    return [(cx + r * math.cos(a) * (1.0 if k % 2 == 0 else 0.86), cy + r * math.sin(a) * (1.0 if k % 2 == 0 else 0.86))
            for k, a in enumerate(math.radians(d) for d in range(-90, 270, 45))]


def ruler_ticks(layout: DockLayout, fps: int) -> list[tuple[int, bool]]:
    """(frame, labelled) for the ruler: a labelled tick every few frames, chosen
    so labels stay about 48 points apart."""
    lx0, lx1 = layout.lane
    per_frame = (lx1 - lx0) / max(1, layout.length)
    steps = [1, 2, 4, 5, 6, 8, 10, 12, 20, 24, 30, 48, 60, 120, 240]
    label_every = next((st for st in steps if st * per_frame >= 48 * layout.scale), steps[-1])
    small = next((st for st in steps if st * per_frame >= 10 * layout.scale), label_every)
    out = []
    for f in range(0, layout.length + 1, small):
        out.append((f, f % label_every == 0))
    return out
