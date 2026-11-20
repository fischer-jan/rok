"""The welcome screen: the logo and the name, with New, Open and the recent files
under them. The recent files come from `load_recent` and open with
`rok.open_recent`.

It opens once when Rok starts without a file, and again from File > Welcome
Screen. The logo is assets/icons/welcome/logo.png. A click on New opens
the New dialog, on Open the file dialog, on a recent file that file. Esc, a
right-click or a click outside the panel closes it and leaves the empty
model. The layout lives in rok.core.welcome.
"""

from __future__ import annotations

import os

import numpy as np

from rok.core import png
from rok.core import welcome as wl

from . import object as voxel_object
from . import opkit, recent
from .gfx import ICONS, pen
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, StringProp, register
from .paint import clip_text, round_outline, round_rect, text

PICTURE_FOLDER = "welcome"
LOGO = "logo"

PANEL_BG = (0.060, 0.056, 0.053, 0.98)
PANEL_EDGE = (0.228, 0.212, 0.196, 1.0)
BUTTON_BG = (1.0, 1.0, 1.0, 0.06)
BUTTON_HOVER = (0.208, 0.192, 0.180, 1.0)
EMBER = (0.910, 0.365, 0.227, 1.0)
EMBER_HOVER = (0.960, 0.470, 0.330, 1.0)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
TEXT_FAINT = (0.450, 0.430, 0.410, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
SCRIM = (0.0, 0.0, 0.0, 0.45)  # over the view behind the welcome screen
OPEN_AFTER_S = 0.6  # the screen opens this long after the window is up

_session: "_Session | None" = None
_shown_at_start = False
_skip_at_start = False  # the restore question shows instead (recovery.py)
_tried: set[str] = set()


class _Session:
    __slots__ = ("hover", "recent")

    def __init__(self) -> None:
        self.hover = None  # wl.NEW, wl.OPEN or a recent row's path
        self.recent = load_recent()  # read once when the screen opens


def active() -> bool:
    return _session is not None


def load_recent() -> list[str]:
    """The remembered paths that still exist, most recent first."""
    return [p for p in recent.load() if os.path.isfile(p)]


def _layout(region):
    entries = _session.recent if _session is not None else []
    return wl.welcome_layout(1.0, region.width, region.height, entries)


def _hover_key(target):
    return target.path if isinstance(target, wl.RecentRow) else target


@register
class WelcomeAction(Action):
    command_id = "rok.welcome"
    title = "Welcome"
    tooltip = "Show the welcome screen: start a new model, open a file or pick a recent one"
    @classmethod
    def poll(cls, context):
        return context.region is not None

    def invoke(self, context, event):
        global _session
        if _session is not None:
            _close()
            return {"CANCELLED"}
        _session = _Session()
        _load_logo()  # now, not while drawing
        context.prefs.add_modal(self)
        voxel_object.redraw_3d_views()
        return {"RUNNING"}

    def modal(self, context, event):
        session = _session
        if session is None:
            return {"CANCELLED"}
        region = context.region
        kind, value = event.type, event.value
        mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
        layout = _layout(region)
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}:
            hover = _hover_key(wl.target_at(layout, mx, my))
            if hover != session.hover:
                session.hover = hover
                region.tag_redraw()
            return {"RUNNING"}
        if kind == "LEFT_MOUSE" and value == "PRESS":
            target = wl.target_at(layout, mx, my)
            if target is not None:
                _act(_hover_key(target))
                return {"FINISHED"}
            if not wl.in_panel(layout, mx, my):
                _close()
                return {"CANCELLED"}
            return {"RUNNING"}
        if value == "PRESS" and kind == "N" and not (event.cmd_key or event.ctrl):
            _act(wl.NEW)
            return {"FINISHED"}
        if value == "PRESS" and kind == "O" and not (event.cmd_key or event.ctrl):
            _act(wl.OPEN)
            return {"FINISHED"}
        if value == "PRESS" and kind in {"ESCAPE", "RIGHT_MOUSE", "ENTER", "NUMPAD_ENTER", "SPACE"}:
            _close()
            return {"CANCELLED"}
        # Losing focus does not close it: Rok may start behind another app.
        return {"RUNNING"}

    def cancel(self, context):
        _close()


def _act(key) -> None:
    """Close the screen, then do what was picked."""
    _close()
    if key == wl.NEW:
        if opkit.ops.rok.new.poll():
            opkit.ops.rok.new("INVOKE")
    elif key == wl.OPEN:
        if opkit.ops.rok.open.poll():
            opkit.ops.rok.open("INVOKE")
    elif key:
        opkit.ops.rok.open_recent("INVOKE", filepath=str(key))


def skip_at_start() -> None:
    """Do not open the screen at start (a question shows instead)."""
    global _skip_at_start
    _skip_at_start = True


def close() -> None:
    """Close the screen if it is open."""
    if _session is not None:
        _close()


def _close() -> None:
    global _session
    _session = None
    voxel_object.redraw_3d_views()


@register
class OpenRecentAction(Action):
    command_id = "rok.open_recent"
    title = "Open Recent"
    tooltip = "Open this file"
    filepath = StringProp(name="File", description="The .vox or .rok file to open", default="")

    def invoke(self, context, event):
        if document.modified:
            return context.prefs.ask_confirm(
                self, event, title="Discard unsaved changes?", confirm_text="Discard"
            )
        return self.execute(context)

    def execute(self, context):
        from . import io_ops  # late: io_ops imports this module

        return io_ops.open_path(self, context, self.filepath)


# --- the start ----------------------------------------------------------------------


class _Opener(Action):
    """Waits for its timer, then opens the screen. The screen's modal is then
    the newest one, so it sees every event before the tools do."""

    command_id = "rok.welcome_opener"
    title = "Welcome Opener"

    def modal(self, context, event):
        if event.type != "TIMER" or self._timer.time_duration < OPEN_AFTER_S - 0.01:
            return {"PASS"}
        context.prefs.remove_timer(self._timer)
        if _session is None and not _skip_at_start and opkit.ops.rok.welcome.poll():
            opkit.ops.rok.welcome("INVOKE")
        return {"FINISHED", "PASS"}


def show_at_start(context=None) -> None:
    """Open the screen once per session, a moment after the window is up. Only
    when Rok started without a file, and not when ROK_NO_WELCOME is set."""
    global _shown_at_start
    context = context or CONTEXT
    app = context.app
    if _shown_at_start or app is None or app.start_path or os.environ.get("ROK_NO_WELCOME"):
        return
    _shown_at_start = True
    opener = _Opener()
    opener._timer = context.prefs.add_timer(OPEN_AFTER_S)
    context.prefs.add_modal(opener)


# --- drawing ------------------------------------------------------------------------


def _load_logo() -> bool:
    """Put the logo into the host's icon atlas once. The icon loader takes only
    small images, so the logo is read here with a larger limit."""
    app = CONTEXT.app
    if app is None:
        return False
    key = f"{PICTURE_FOLDER}/{LOGO}"
    if key in app.icons.loaded:
        return True
    if LOGO in _tried:
        return False
    _tried.add(LOGO)
    path = ICONS / PICTURE_FOLDER / f"{LOGO}.png"
    try:
        _w, _h, rgba = png.decode(path.read_bytes(), max_pixels=1024 * 1024)
    except (OSError, png.PngError):
        return False
    app.host.add_image(key, np.ascontiguousarray(rgba, dtype=np.uint8))
    app.icons.loaded.add(key)
    return True


def _button(rect, label: str, sub: str, hovered: bool, primary: bool, s: float) -> None:
    x, y, w, h = rect
    if primary:
        round_rect(rect, 10 * s, EMBER_HOVER if hovered else EMBER)
        ink, dim = WHITE, (1.0, 1.0, 1.0, 0.8)
    else:
        round_rect(rect, 10 * s, BUTTON_HOVER if hovered else BUTTON_BG)
        if hovered:
            round_outline(rect, 10 * s, EMBER, 1.5 * s)
        ink, dim = TEXT, TEXT_DIM
    text(label, x + 16 * s, y + h * 0.63, 17 * s, ink)
    text(sub, x + 16 * s, y + h * 0.29, 11 * s, dim)


def draw_content(layout: wl.WelcomeLayout, hover, entries) -> None:
    """The panel and everything on it, for a layout."""
    s = layout.scale
    round_rect(layout.rect, 8 * s, PANEL_BG)
    # The logo and the wordmark, top left.
    tx, ty = layout.title
    text("Rok", tx, ty, 54 * s, TEXT)
    if _load_logo():
        CONTEXT.app.icons.draw(LOGO, layout.logo, (1, 1, 1, 1), folder=PICTURE_FOLDER, image=True)

    text("START", layout.start_caption[0] + 2 * s, layout.start_caption[1] + layout.start_caption[3] / 2,
         10 * s, TEXT_FAINT)
    _button(layout.new_button, "New…", "Pick a grid size", hover == wl.NEW, True, s)
    _button(layout.open_button, "Open…", "A .vox file or a Rok file", hover == wl.OPEN, False, s)

    text("RECENT", layout.recent_caption[0] + 2 * s, layout.recent_caption[1] + layout.recent_caption[3] / 2,
         10 * s, TEXT_FAINT)
    if not layout.rows:
        rx, ry, _rw, _rh = layout.recent_caption
        text("Nothing opened yet", rx + 2 * s, ry - 22 * s, 12 * s, TEXT_DIM)
    for row in layout.rows:
        x, y, w, h = row.rect
        hovered = hover == row.path
        if hovered:
            round_rect(row.rect, 6 * s, BUTTON_HOVER)
        name = clip_text(row.name, w * 0.6, 13 * s)
        used = text(name, x + 8 * s, y + h / 2, 13 * s, TEXT if hovered else TEXT_DIM)
        folder = clip_text(row.folder, w - used - 24 * s, 11 * s)
        text(folder, x + 16 * s + used, y + h / 2, 11 * s, TEXT_FAINT)
    hx, hy, _hw, hh = layout.hint
    text("N new · O open · Esc starts with an empty model", hx, hy + hh / 2, 10.5 * s, TEXT_FAINT)


def draw(region, scale: float = 1.0) -> None:
    """The screen, over everything else in the view. Reads `_session`."""
    session = _session
    if session is None:
        return
    layout = _layout(region)
    pen.rect_fill((0, 0, region.width, region.height), SCRIM)
    draw_content(layout, session.hover, session.recent)


def _draw(context) -> None:
    draw(context.region)


def _draw_edge(context) -> None:
    """The panel's edge, in a layer of its own, over everything on the panel."""
    if _session is not None:
        layout = _layout(context.region)
        round_outline(layout.rect, 8 * layout.scale, PANEL_EDGE, 1.0 * layout.scale)


registry.add_draw("welcome", 800, "PIXEL", _draw)
registry.add_draw("welcome_edge", 801, "PIXEL", _draw_edge)
registry.startup.append(show_at_start)
