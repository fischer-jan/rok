"""The View Settings panel's layout: where every part stands. Pure.

The panel stands at the right edge of the view, from the top, so the model
stays in sight and every change shows in it at once. It is two panels in
one place: View Settings (the editing view's look) and Render Settings (the
file's light). Render Settings has a preset row
and four sections (Lights, Sky and Ground, Finish, Quality); one is open at a
time, so the panel stays short enough for a small window.

`panel_layout` returns `Item`s: what a part is (`kind`), which value it
changes (`key`), and its rect in region pixels (origin bottom left). The app
draws them by kind and `hit` finds the one under the mouse. A slider item
carries a shader-panel `Control`, so the slider arithmetic is shared
(rok.core.shader_panel: value_at, fine, nudged, x_of).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom2d import Rect, in_rect
from .shader_panel import Control

WIDTH = 380
PAD = 14
MARGIN = 16
ROW = 28
HEADER = 40
FOOT = 46
CHIP_H = 24
GAP = 4
LABEL_W = 96
VALUE_W = 54
TRACK_H = 6
SWATCH = 22
DOME = 176  # the light dome's height, its disc and the turn ring
MENU_W = 210
MENU_ROW = 28

LOOK, RENDER = "LOOK", "RENDER"
TITLES = {LOOK: "View Settings", RENDER: "Render Settings"}
LIGHTS, SKY, FINISH, QUALITY = "LIGHTS", "SKY", "FINISH", "QUALITY"
SECTIONS = ((LIGHTS, "Lights"), (SKY, "Sky and Ground"), (FINISH, "Finish"), (QUALITY, "Quality"))
BACKDROPS = (("SKY", "Sky"), ("COLOUR", "Colour"), ("CLEAR", "Clear"))
GROUNDS = (("OFF", "None"), ("ON", "Ground"))
KINDS = (("SUN", "Sun"), ("LAMP", "Lamp"))
SAMPLES = ((128, "128"), (512, "512"), (2048, "2048"), (8192, "8192"))
PICTURE_SAMPLES = ((0, "Preset"), (128, "128"), (512, "512"), (2048, "2048"))


@dataclass(frozen=True)
class Slider:
    key: str
    label: str
    low: float
    high: float
    step: float
    precision: int
    tip: str = ""


# The sliders of Render Settings: key -> Slider. A light's keys start with "light.".
SLIDERS = {s.key: s for s in (
    Slider("light.power", "Strength", 0.0, 4.0, 0.05, 2, "How bright the light is"),
    Slider("light.softness_sun", "Softness", 0.0, 20.0, 0.5, 1,
           "How wide the sun is, in degrees: a wider sun gives softer shadow edges"),
    Slider("light.softness_lamp", "Size", 0.05, 1.5, 0.05, 2,
           "How big the lamp is, in scene radii: a bigger lamp gives softer shadows"),
    Slider("light.azimuth", "Turn", -180.0, 180.0, 1.0, 0,
           "Where the light stands around the model, in degrees"),
    Slider("light.elevation", "Height", -10.0, 90.0, 1.0, 0,
           "How high the light stands, in degrees: 0 is level with the ground, 90 straight above"),
    Slider("light.distance", "Distance", 1.0, 8.0, 0.1, 1,
           "How far the lamp is from the model, in scene radii; the model stays as bright"),
    Slider("turn", "Turn All", -180.0, 180.0, 1.0, 0, "Turns every light around the model together"),
    Slider("sky_strength", "Sky", 0.0, 3.0, 0.05, 2, "How much light the sky gives"),
    Slider("exposure", "Exposure", -4.0, 4.0, 0.25, 2, "Brighter or darker, in stops"),
    Slider("bloom", "Bloom", 0.0, 1.0, 0.05, 2, "A glow around bright lights and glowing colours"),
    Slider("outline", "Outline", 0.0, 3.0, 0.05, 2,
           "Dark ink lines along the edges of shapes; 1 is black, more makes the lines thicker"),
    Slider("voxel_grid", "Voxel Grid", 0.0, 2.0, 0.02, 2,
           "A darker line around every voxel face; 1 is black, more makes the lines thicker"),
)}
TONES = (("CLIP", "Clipped"), ("FILM", "Film"), ("FILM_PUNCHY", "Punchy"), ("FILM_HIGH", "High"))


@dataclass(frozen=True)
class State:
    """What the layout depends on."""

    tab: str = LOOK
    section: str | None = LIGHTS  # the open section of Render Settings
    looks: tuple[tuple[str, str], ...] = ()  # (key, label) of the viewport looks
    lights: tuple[str, ...] = ("SUN",)  # each light's kind
    picked: int = 0  # the light whose card shows
    backdrop: str = "SKY"  # "SKY", "COLOUR" or "CLEAR"
    ground: bool = True
    changed: bool = False  # the light differs from its preset: Reset shows
    menu: bool = False  # the preset list is open
    presets: tuple[tuple[str, str], ...] = ()  # (id, label)
    dome: bool = True


@dataclass(frozen=True)
class Item:
    kind: str  # TAB CLOSE LABEL CHIP TOGGLE SLIDER SWATCH SECTION BUTTON LIGHT ADD MENU NOTE DOME PRESET
    key: str
    rect: Rect
    text: str = ""
    value: object = None  # a chip's value, a light's index
    control: Control | None = None
    tip: str = ""


@dataclass(frozen=True)
class PanelLayout:
    rect: Rect
    scale: float
    items: tuple[Item, ...]
    menu: Rect | None = None  # the preset list, left of the panel
    dome: Rect | None = None

    def find(self, kind: str, key: str = "", value=None) -> Item | None:
        return next((i for i in self.items if i.kind == kind and (not key or i.key == key)
                     and (value is None or i.value == value)), None)

    def control(self, key: str) -> Control | None:
        item = self.find("SLIDER", key)
        return item.control if item is not None else None


@dataclass
class _Builder:
    s: float
    x: float
    width: float
    y: float = 0.0  # the top of the next row, from the panel's top (0) down
    items: list = field(default_factory=list)

    @property
    def left(self) -> float:
        return self.x + PAD * self.s

    @property
    def inner(self) -> float:
        return self.left + LABEL_W * self.s

    @property
    def right(self) -> float:
        return self.x + self.width - PAD * self.s

    def row(self, h: float = ROW) -> float:
        """Take a row `h` points high; returns its bottom."""
        self.y -= h * self.s
        return self.y

    def add(self, kind: str, key: str, rect: Rect, text: str = "", value=None, control=None, tip: str = ""):
        self.items.append(Item(kind, key, rect, text, value, control, tip))

    def chips(self, kind_key: str, choices, x: float, w: float, y: float, h: float, columns: int = 0) -> None:
        """Equal chips side by side in (x, w); `columns` wraps them into rows
        (the caller took the rows)."""
        s = self.s
        n = len(choices)
        cols = columns or n
        cw = (w - GAP * s * (cols - 1)) / cols
        for k, (value, text) in enumerate(choices):
            r, c = divmod(k, cols)
            self.add("CHIP", kind_key, (x + c * (cw + GAP * s), y - r * (h + GAP * s), cw, h), text, value)

    def labelled_chips(self, label: str, key: str, choices, columns: int = 0, swatch: str | None = None,
                       swatch_shown: bool = False) -> None:
        """A label and a row of chips; `swatch` keeps room at the row's end for
        a colour swatch, shown when `swatch_shown`."""
        s = self.s
        cols = columns or len(choices)
        rows = -(-len(choices) // cols)
        first = self.row()
        for _ in range(rows - 1):
            self.row(CHIP_H + GAP)
        self.add("LABEL", key, (self.left, first, LABEL_W * s, ROW * s), label)
        cy = first + (ROW - CHIP_H) * s / 2
        right = self.right
        if swatch is not None:
            right -= (SWATCH + 8) * s
            if swatch_shown:
                self.add("SWATCH", swatch, (self.right - SWATCH * s, first + (ROW - SWATCH) * s / 2, SWATCH * s,
                                            SWATCH * s))
        self.chips(key, choices, self.inner, right - self.inner, cy, CHIP_H * s, cols)

    def slider(self, key: str, label: str | None = None) -> None:
        spec = SLIDERS[key]
        s = self.s
        y = self.row()
        row = (self.left, y, self.right - self.left, ROW * s)
        label_rect = (self.left, y, LABEL_W * s, ROW * s)
        value_rect = (self.right - VALUE_W * s, y, VALUE_W * s, ROW * s)
        track = (self.inner, y + (ROW - TRACK_H) * s / 2, value_rect[0] - 10 * s - self.inner, TRACK_H * s)
        control = Control(key, label or spec.label, spec.low, spec.high, spec.step, spec.precision, False,
                          row, label_rect, track, value_rect, tip=spec.tip)
        self.add("SLIDER", key, row, label or spec.label, control=control, tip=spec.tip)

    def swatches(self, label: str, keys) -> None:
        """A row of colour swatches, each with a word after it."""
        s = self.s
        y = self.row()
        self.add("LABEL", keys[0][0], (self.left, y, LABEL_W * s, ROW * s), label)
        x = self.inner
        for key, word in keys:
            rect = (x, y + (ROW - SWATCH) * s / 2, SWATCH * s, SWATCH * s)
            self.add("SWATCH", key, rect, word)
            x += (SWATCH + 6) * s + (len(word) * 7 + 14) * s if word else (SWATCH + 10) * s


def panel_layout(scale: float, region_width: float, region_height: float, state: State) -> PanelLayout:
    s = float(scale)
    width = min(WIDTH * s, region_width - 2 * MARGIN * s)
    x = max(MARGIN * s, region_width - width - MARGIN * s)
    b = _Builder(s, x, width)
    # The header: the panel's title and the close button.
    b.row(PAD)
    y = b.row(HEADER - PAD)
    b.add("TITLE", state.tab, (b.left, y + 2 * s, b.right - b.left - CHIP_H * s - GAP * s, CHIP_H * s),
          TITLES[state.tab])
    b.add("CLOSE", "", (b.right - CHIP_H * s, y + 2 * s, CHIP_H * s, CHIP_H * s), "×")
    b.row(6)
    dome = None
    if state.tab == LOOK:
        b.labelled_chips("Look", "look", state.looks, columns=3)
        b.row(6)
        for key, text in (("occlusion", "Ambient Occlusion"), ("smooth", "Smooth Edges"), ("box", "Model Box")):
            y = b.row()
            b.add("TOGGLE", key, (b.left, y, b.right - b.left, ROW * s), text)
        b.row(6)
        y = b.row(22)
        b.add("NOTE", "", (b.left, y, b.right - b.left, 22 * s), "How the editing view looks, in every file.")
    else:
        y = b.row()
        b.add("LABEL", "preset", (b.left, y, LABEL_W * s, ROW * s), "Preset")
        reset_w = 64 * s
        preset_w = b.right - b.inner - (reset_w + GAP * s if state.changed else 0)
        b.add("PRESET", "", (b.inner, y + (ROW - CHIP_H) * s / 2, preset_w, CHIP_H * s))
        if state.changed:
            b.add("BUTTON", "reset", (b.right - reset_w, y + (ROW - CHIP_H) * s / 2, reset_w, CHIP_H * s), "Reset")
        b.row(4)
        for key, text in SECTIONS:
            y = b.row()
            b.add("SECTION", key, (b.left, y, b.right - b.left, ROW * s), text, value=state.section == key)
            if state.section != key:
                continue
            if key == LIGHTS:
                dome = _lights(b, state)
            elif key == SKY:
                b.swatches("Sky", (("sky_top", "Top"), ("sky_horizon", "Horizon")))
                b.slider("sky_strength", "Sky Light")
                b.labelled_chips("Background", "backdrop", BACKDROPS, swatch="backdrop_colour",
                                 swatch_shown=state.backdrop == "COLOUR")
                b.labelled_chips("Ground", "ground", GROUNDS, swatch="ground_colour", swatch_shown=state.ground)
                if state.backdrop == "CLEAR":
                    y = b.row(20)
                    b.add("NOTE", "", (b.left, y, b.right - b.left, 20 * s),
                          "Clear: pictures only; the view shows the sky.")
            elif key == FINISH:
                b.slider("exposure")
                b.labelled_chips("Curve", "tone", TONES)
                b.slider("bloom")
                b.slider("outline")
                b.slider("voxel_grid")
            elif key == QUALITY:
                b.labelled_chips("View", "samples", SAMPLES)
                b.labelled_chips("Picture", "picture_samples", PICTURE_SAMPLES)
                y = b.row(20)
                b.add("NOTE", "", (b.left, y, b.right - b.left, 20 * s), "Samples are yours, in every file.")
            b.row(4)
    # The foot: a note, Revert and Done.
    b.row(8)
    y = b.row(FOOT - 8 - PAD)
    bw, bh = 76 * s, (FOOT - 8 - PAD) * s
    b.add("BUTTON", "done", (b.right - bw, y, bw, bh), "Done")
    b.add("BUTTON", "revert", (b.right - 2 * bw - 8 * s, y, bw, bh), "Revert")
    if state.tab == RENDER:
        b.add("NOTE", "", (b.left, y, b.right - 2 * bw - 16 * s - b.left, bh), "Saved with the file")
    b.row(PAD)
    height = -b.y
    top = region_height - MARGIN * s
    bottom = top - height
    shift = top
    items = tuple(Item(i.kind, i.key, _moved(i.rect, shift), i.text, i.value, _moved_control(i.control, shift), i.tip)
                  for i in b.items)
    rect = (x, bottom, width, height)
    if dome is not None:
        dome = _moved(dome, shift)
    menu = None
    if state.tab == RENDER and state.menu and state.presets:
        preset = next(i for i in items if i.kind == "PRESET")
        h = (len(state.presets) * MENU_ROW + 2 * 6) * s
        mx = x - 8 * s - MENU_W * s
        my = max(MARGIN * s, preset.rect[1] + preset.rect[3] - h)
        menu = (mx, my, MENU_W * s, h)
        rows = []
        for k, (pid, label) in enumerate(state.presets):
            ry = my + h - 6 * s - (k + 1) * MENU_ROW * s
            rows.append(Item("MENU", pid, (mx + 6 * s, ry, (MENU_W - 12) * s, MENU_ROW * s), label, pid))
        items = tuple(rows) + items
    return PanelLayout(rect, s, items, menu, dome)


def _lights(b: _Builder, state: State) -> Rect | None:
    """The Lights section: the dome, a chip per light and "+", and the picked
    light's card. Returns the dome's rect (from the panel's top)."""
    s = b.s
    dome = None
    if state.dome:
        b.row(4)
        y = b.row(DOME)
        dome = (b.left, y, b.right - b.left, DOME * s)
        b.add("DOME", "", dome)
    y = b.row()
    n = len(state.lights)
    cw = 64 * s
    cy = y + (ROW - CHIP_H) * s / 2
    for k, kind in enumerate(state.lights):
        b.add("LIGHT", str(k), (b.left + k * (cw + GAP * s), cy, cw, CHIP_H * s),
              f"{'Sun' if kind == 'SUN' else 'Lamp'} {k + 1}", k)
    if n < 4:
        b.add("ADD", "", (b.left + n * (cw + GAP * s), cy, CHIP_H * s, CHIP_H * s), "+")
    if not n:
        return dome
    picked = min(max(state.picked, 0), n - 1)
    kind = state.lights[picked]
    y = b.row()
    b.add("LABEL", "kind", (b.left, y, LABEL_W * s, ROW * s), "Kind")
    cy = y + (ROW - CHIP_H) * s / 2
    chips_w = 132 * s
    b.chips("kind", KINDS, b.inner, chips_w, cy, CHIP_H * s)
    b.add("SWATCH", "light.colour", (b.inner + chips_w + 10 * s, y + (ROW - SWATCH) * s / 2, SWATCH * s, SWATCH * s))
    bw = 30 * s
    b.add("BUTTON", "remove", (b.right - bw, cy, bw, CHIP_H * s), "×")
    b.add("BUTTON", "mute", (b.right - 2 * bw - GAP * s, cy, bw, CHIP_H * s), "on")
    b.slider("light.power")
    b.slider("light.softness_sun" if kind == "SUN" else "light.softness_lamp")
    b.slider("light.azimuth")
    b.slider("light.elevation")
    if kind == "LAMP":
        b.slider("light.distance")
    if n > 1:
        b.slider("turn")
    return dome


def _moved(rect: Rect | None, dy: float) -> Rect | None:
    if rect is None:
        return None
    x, y, w, h = rect
    return (x, y + dy, w, h)


def _moved_control(c: Control | None, dy: float) -> Control | None:
    if c is None:
        return None
    from dataclasses import replace

    return replace(c, row=_moved(c.row, dy), label_rect=_moved(c.label_rect, dy), track=_moved(c.track, dy),
                   value_rect=_moved(c.value_rect, dy))


def hit(layout: PanelLayout, x: float, y: float) -> Item | None:
    """The item under a point (a slider row is the slider, its label the
    LABEL), a ("PANEL") item for the panel elsewhere, or None outside."""
    if layout.menu is not None and in_rect(layout.menu, x, y):
        return next((i for i in layout.items if i.kind == "MENU" and in_rect(i.rect, x, y)),
                    Item("MENU_BG", "", layout.menu))
    for item in layout.items:
        if item.kind in ("LABEL", "NOTE", "MENU", "TITLE"):
            continue
        if item.kind == "SLIDER":
            c = item.control
            if in_rect(c.label_rect, x, y):
                return Item("SLIDER_LABEL", item.key, c.label_rect, item.text, control=c, tip=item.tip)
            if in_rect(c.row, x, y) and x >= c.track[0] - 8 * layout.scale:
                return item
            continue
        if in_rect(item.rect, x, y):
            return item
    if in_rect(layout.rect, x, y):
        return Item("PANEL", "", layout.rect)
    return None
