"""Cmd+K / Ctrl+K command palette: every action by name.

Type to filter (printable ASCII), arrows or the mouse to highlight, Enter or a
click to run. The list and the fuzzy match live in `rok.core.chrome`.

The palette lists only the commands whose action Rok has: the animation
rows and the UI switch are parked or gone, and a command whose action is
not registered does not show.
"""

from __future__ import annotations

import sys
import time

from rok.core import chrome as kit
from rok.core.geom2d import in_rect, rect_tris

from . import object as voxel_object
from . import opkit
from .gfx import fonts, pen
from .handlers import registry
from .opkit import Action, register
from .ops_common import _late, redraw

_session: "_Session | None" = None

PANEL_BG = (0.060, 0.056, 0.053, 0.96)
PANEL_EDGE = (0.300, 0.285, 0.272, 1.0)
ROW_HOVER = (0.235, 0.220, 0.210, 1.0)
ROW_ON = (0.910, 0.460, 0.160, 0.35)
TEXT = (0.905, 0.885, 0.855, 1.0)
TEXT_DIM = (0.620, 0.600, 0.575, 1.0)
LINE_PX = 1.0
FONT = 0
# A second call this soon after the palette opened is the same key seen twice
# (the menu bar's Cmd+K and a keymap item): it must not close the palette.
REOPEN_GUARD_S = 0.15

# Commands of parked features: never listed.
PARKED_IDS = frozenset({"ui_switch"})


class _Session:
    __slots__ = ("query", "index", "opened")

    def __init__(self):
        self.query = ""
        self.index = 0
        self.opened = time.monotonic()


def active() -> bool:
    return _session is not None


def status_hint() -> str | None:
    return kit.COMMAND_STATUS if _session is not None else None


def available(cmd: kit.Command) -> bool:
    """True when Rok can run `cmd`: its operator is registered and not parked.
    Saving as a .blend file is gone with the animation."""
    if cmd.id in PARKED_IDS or ("format", "BLEND") in cmd.props:
        return False
    return f"rok.{cmd.id}" in opkit.REGISTRY


def _hits(session: _Session) -> tuple[kit.Command, ...]:
    commands = kit.COMMANDS + tuple(_late("xs_ops", "shader_commands")() or ())
    return kit.command_match(session.query, tuple(c for c in commands if available(c)))


def _layout_of(session: _Session, region) -> kit.CommandLayout:
    return kit.command_layout(
        1.0,
        region.width,
        region.height,
        session.query,
        _hits(session),
        session.index,
    )


def _key_label(key: str) -> str:
    if not key:
        return ""
    if sys.platform == "darwin":
        return key.replace("Ctrl+", "Cmd+")
    return key


def _custom_key(cmd) -> str:
    """A key the user gave the command in View > Customize Rail Menu and Keys."""
    if cmd.props:
        return ""
    from . import model_menu

    return model_menu.label(model_menu.keys().get(cmd.id, ""))


@register
class CommandSearchAction(Action):
    command_id = "rok.command_search"
    title = "Search Commands"
    tooltip = "Search any action by name"

    @classmethod
    def poll(cls, context):
        return context.region is not None

    def invoke(self, context, event):
        global _session
        if _session is not None:
            if time.monotonic() - _session.opened < REOPEN_GUARD_S:
                return {"CANCELLED"}
            self._close(context)
            return {"CANCELLED"}
        _session = _Session()
        context.prefs.add_modal(self)
        redraw()
        return {"RUNNING"}

    def modal(self, context, event):
        if _session is None:
            return {"CANCELLED"}
        region = context.region
        kind = event.type
        if kind == "MOUSE_MOVE":
            layout = _layout_of(_session, region)
            row = kit.command_hit(layout, float(event.mouse_region_x), float(event.mouse_region_y))
            if row is not None and row != _session.index:
                _session.index = row
                region.tag_redraw()
            return {"RUNNING"}
        if kind == "LEFT_MOUSE" and event.value == "PRESS":
            layout = _layout_of(_session, region)
            mx, my = float(event.mouse_region_x), float(event.mouse_region_y)
            row = kit.command_hit(layout, mx, my)
            if row is None:
                if not in_rect(layout.rect, mx, my):
                    self._close(context)
                    return {"CANCELLED"}
                return {"RUNNING"}
            _session.index = row
            return self._run(context)
        if kind in {"ENTER", "NUMPAD_ENTER"} and event.value == "PRESS":
            return self._run(context)
        if kind in {"ESCAPE", "RIGHT_MOUSE"} and event.value == "PRESS":
            self._close(context)
            return {"CANCELLED"}
        if kind == "TAB" and event.value == "PRESS":
            self._close(context)
            return {"CANCELLED"}
        if kind == "FOCUS_LOST":
            self._close(context)
            return {"CANCELLED"}
        if kind in {"UP_ARROW", "DOWN_ARROW"} and event.value == "PRESS":
            hits = _hits(_session)
            if hits:
                step = -1 if kind == "UP_ARROW" else 1
                _session.index = (_session.index + step) % min(len(hits), kit.COMMAND_MAX)
                voxel_object.redraw_3d_views()
            return {"RUNNING"}
        if kind == "BACKSPACE" and event.value == "PRESS":
            _session.query = _session.query[:-1]
            _session.index = 0
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        ch = event.unicode
        if event.value == "PRESS" and ch and ch.isprintable() and ch.isascii():
            if event.ctrl or event.cmd_key:
                return {"RUNNING"}
            _session.query += ch
            _session.index = 0
            voxel_object.redraw_3d_views()
            return {"RUNNING"}
        return {"RUNNING"}

    def cancel(self, context):
        self._close(context)

    def _run(self, context):
        session = _session
        hits = _hits(session) if session is not None else ()
        shown = hits[: kit.COMMAND_MAX] if session is not None else ()
        index = session.index if session is not None else 0
        self._close(context)
        if not shown:
            return {"CANCELLED"}
        cmd = shown[min(max(0, index), len(shown) - 1)]
        op = getattr(opkit.ops.rok, cmd.id)
        if op.poll():
            op("INVOKE", **dict(cmd.props))
        return {"FINISHED"}

    def _close(self, context) -> None:
        global _session
        _session = None
        redraw()


def draw(region, scale: float = 1.0) -> None:
    """The search panel. Reads `_session`."""
    session = _session
    if session is None:
        return
    layout = _layout_of(session, region)
    pen.tris(pen.to_3d(rect_tris(layout.rect)), PANEL_BG)
    pen.lines(pen.to_3d(kit.rect_lines(layout.rect)), PANEL_EDGE, LINE_PX * scale)
    prompt = layout.query if layout.query else "Search commands…"
    _draw_row_text(prompt, layout.query_rect, 12 * scale, TEXT if layout.query else TEXT_DIM, scale)
    if layout.query:
        # The caret after the typed text.
        fonts.size(FONT, 12 * scale)
        tw, _th = fonts.dimensions(FONT, layout.query)
        x, y, _w, h = layout.query_rect
        cx = x + round(6 * scale) + tw + 2
        pen.lines(pen.to_3d([(cx, y + 4 * scale), (cx, y + h - 4 * scale)]), TEXT, LINE_PX * scale)
    for i, (cmd, rect) in enumerate(zip(layout.hits, layout.rows)):
        if i == layout.index:
            pen.tris(pen.to_3d(rect_tris(rect)), ROW_ON)
        _draw_row_text(cmd.name, rect, 11 * scale, TEXT, scale)
        key = _custom_key(cmd) or _key_label(cmd.key)
        if key:
            _draw_row_text(key, rect, 11 * scale, TEXT_DIM, scale, align="right")
    if not layout.hits:
        qx, qy, qw, qh = layout.query_rect
        empty = (qx, qy - round(kit.COMMAND_PAD * scale) - qh, qw, qh)
        _draw_row_text("No command found", empty, 11 * scale, TEXT_DIM, scale)


def _draw_row_text(text: str, rect, size: float, colour, scale: float, align: str = "left") -> None:
    x, y, w, h = rect
    fonts.size(FONT, size)
    tw, _th = fonts.dimensions(FONT, text)
    _cw, cap = fonts.dimensions(FONT, "H")  # one baseline for every row, descenders or not
    pad = round(6 * scale)
    tx = x + w - tw - pad if align == "right" else x + pad
    fonts.color(FONT, *colour)
    fonts.position(FONT, float(tx), float(y + (h - cap) / 2.0), 0)
    fonts.draw(FONT, text)


def _draw(context) -> None:
    draw(context.region)


registry.add_draw("command", 700, "PIXEL", _draw)
registry.keymap.extend([
    ("rok.command_search", {"type": "K", "value": "PRESS", "ctrl": True}, None),
    ("rok.command_search", {"type": "K", "value": "PRESS", "cmd_key": True}, None),
])
