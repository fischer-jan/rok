"""The key cursor in the app: the arrow keys step an outlined cell of the
active model one cell at a time on the floor, Shift with Up and Down a layer
up and down (rok.core.key_cursor). Return acts there as a click would;
Option+Return as an Option-click.

The first step starts at the cell under the mouse (the one a click would
change). The ghost of the brush, or of the brick in Brick mode, shows at the
cursor until the mouse really moves (a camera move keeps it); then the
mouse leads again. Esc puts the cursor away.

In Model level Return runs the Voxel-mode brush of Attach, Erase or Paint
with the brush size, Mirror, the mask and the selection as a click does. In
Brick mode it runs the brick tool.
"""

from __future__ import annotations

import numpy as np

from rok.core import edit
from rok.core import key_cursor as core
from rok.core.chrome import LEVEL_BRICK, LEVEL_MODEL
from rok.core.raycast import Hit
from rok.core.undo import ModelRecord, StrokeRecorder, naming

from . import object as voxel_object
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, BoolProp, EnumProp, register
from .ops_common import commit, redraw
from .state import Hover, state

LET_GO_PX = 4.0  # the mouse moved this far from where the cursor took over: it leads again
CURSOR_INK = (1.0, 1.0, 1.0, 0.9)

_cursor: dict = {"uid": None, "cell": None, "mouse": None}


def active() -> bool:
    """The cursor shows: on the active model, inside it or in Brick mode."""
    return (_cursor["cell"] is not None and _cursor["uid"] == document.active_uid
            and state.level in (LEVEL_MODEL, LEVEL_BRICK) and not state.animating)


def cell():
    return _cursor["cell"] if active() else None


def clear() -> None:
    if _cursor["cell"] is None:
        return
    _cursor.update(uid=None, cell=None, mouse=None)
    redraw()


def _voxel_tool(alt: bool) -> str:
    wm = CONTEXT.prefs
    return edit.effective_tool(getattr(wm, "rok_tool", edit.ATTACH), alt, False)


def _start_cell():
    """Where the first step starts: the cell a click would change now, else
    the middle of the floor."""
    grid = document.grid
    if state.level == LEVEL_BRICK:
        from . import brick_mode

        found = brick_mode.last_cell()
        if found is not None:
            return found
    elif state.hover is not None and state.hover.target is not None:
        return tuple(int(v) for v in state.hover.target)
    sx, sy, _sz = grid.size
    return (sx // 2, sy // 2, 0)


def hit_at(c, filled_cell: bool):
    """A made-up ray hit for the cursor cell: on the cell itself (Erase,
    Paint, Pull, Pick) or on the top of the cell under it (Attach, Place),
    so the tools read it as they read the mouse's hit."""
    x, y, z = (int(v) for v in c)
    if filled_cell:
        return Hit((x, y, z), (0, 0, 1), 0.0)
    return Hit((x, y, z - 1), (0, 0, 1), 0.0)


def step(key: str) -> None:
    grid = document.grid
    view_state = CONTEXT.view_state
    if grid is None or view_state is None:
        return
    if not active():
        _cursor.update(uid=document.active_uid, cell=_start_cell())
    else:
        _cursor["cell"] = core.moved(_cursor["cell"], key, view_state.view_direction(), grid.size)
    _cursor["mouse"] = None  # the next hover check notes where the mouse rests
    show()


def took_over(mx: float, my: float) -> bool:
    """mode.update_hover's check: True while the cursor leads (it showed the
    ghost there); False once the mouse moved away, which puts it away."""
    if not active():
        return False
    at = _cursor["mouse"]
    if at is None:
        _cursor["mouse"] = (mx, my)
        at = _cursor["mouse"]
    if abs(mx - at[0]) > LET_GO_PX or abs(my - at[1]) > LET_GO_PX:
        clear()
        return False
    show()
    return True


def show() -> None:
    """The ghost at the cursor: the brick in Brick mode, else the voxel brush."""
    c = cell()
    if c is None:
        return
    if state.level == LEVEL_BRICK:
        from . import brick_mode

        brick_mode.show_at(hit_at(c, state.brick_tool != "PLACE" or state.alt))
    else:
        tool = _voxel_tool(state.alt)
        hit = hit_at(c, tool != edit.ATTACH)
        target = edit.target_cell(tool, hit, document.grid.size)
        state.hover = Hover(hit, tool, target, getattr(CONTEXT.prefs, "rok_mode", edit.VOXEL))
    redraw()


def act(alt: bool) -> str | None:
    """Return at the cursor: a click there. None, or why nothing happened."""
    c = cell()
    if c is None:
        return None
    if state.level == LEVEL_BRICK:
        from . import brick_mode

        return brick_mode.act_at(hit_at(c, state.brick_tool != "PLACE" or alt), alt)
    return _brush_at(c, alt)


def _brush_at(c, alt: bool) -> str | None:
    """The Voxel-mode brush of Attach, Erase or Paint at cell `c`, with its
    size and Mirror; Paint and Erase keep to the selection when there is one."""
    from rok.core.brick import LOCKED_MESSAGE

    grid = document.grid
    if grid.bricks is not None:
        return LOCKED_MESSAGE
    wm = CONTEXT.prefs
    if getattr(wm, "rok_mode", edit.VOXEL) != edit.VOXEL:
        return "Return at the key cursor works in Voxel mode"
    tool = _voxel_tool(alt)
    if tool not in (edit.ATTACH, edit.ERASE, edit.PAINT):
        return "Return at the key cursor works with Attach, Erase and Paint"
    size = int(getattr(wm, "rok_brush_size", 1))
    cells = edit.sphere_offsets(size) + np.asarray(c, dtype=np.int64)
    cells = cells[edit.in_bounds_mask(cells, grid.size)]
    mirror = tuple(bool(v) for v in getattr(wm, "rok_mirror", (False, False, False)))
    if any(mirror):
        cells = edit.mirror_cells(cells, grid.size, mirror)
    now = grid.voxels[cells[:, 0], cells[:, 1], cells[:, 2]]
    colour = int(document.colour)
    if tool == edit.ATTACH:
        keep = now == 0
    elif tool == edit.ERASE:
        keep = now != 0
    else:
        keep = (now != 0) & (now != colour)
    sel = document.selection
    if tool != edit.ATTACH and sel is not None and not sel.is_empty():
        keep &= sel.mask[cells[:, 0], cells[:, 1], cells[:, 2]]
    cells = cells[keep]
    if len(cells) == 0:
        return None
    recorder = StrokeRecorder(grid)
    recorder.set_many(cells, 0 if tool == edit.ERASE else colour)
    record = recorder.finish()
    if record is None:
        return None
    with naming(f"{tool.title()} Voxel"):
        commit(ModelRecord(document.active_uid, record), serial=True)
    voxel_object.flush_dirty()
    return None


def _say(message: str) -> None:
    app = CONTEXT.app
    if app is not None:
        app.say(message, 4.0)


def _can_move() -> bool:
    return (document.grid is not None and not state.stroke_active and not state.animating
            and state.level in (LEVEL_MODEL, LEVEL_BRICK))


@register
class KeyCursorMoveAction(Action):
    """An arrow key (Shift with Up or Down: a layer): the key cursor one cell on."""

    command_id = "rok.key_cursor_move"
    title = "Move the Key Cursor"
    tooltip = "Step the key cursor one cell: the arrows on the floor, Shift with Up and Down a layer up and down"
    key = EnumProp(name="Key", items=tuple((k, k, "") for k in core.KEYS), default="W")

    @classmethod
    def poll(cls, context):
        return _can_move()

    def execute(self, context):
        step(self.key)
        return {"FINISHED"}


@register
class KeyCursorActAction(Action):
    """Return: a click at the key cursor (Option+Return: an Option-click)."""

    command_id = "rok.key_cursor_act"
    title = "Click at the Key Cursor"
    tooltip = "Run the tool at the key cursor, as a click there would"
    alt = BoolProp(name="Option", default=False)

    @classmethod
    def poll(cls, context):
        return _can_move() and active()

    def execute(self, context):
        why = act(bool(self.alt))
        if why:
            _say(why)
        show()
        return {"FINISHED"}


@register
class KeyCursorClearAction(Action):
    """Esc puts the key cursor away; the mouse leads again."""

    command_id = "rok.key_cursor_clear"
    title = "Put the Key Cursor Away"
    tooltip = "The mouse leads again"

    @classmethod
    def poll(cls, context):
        return active()

    def execute(self, context):
        clear()
        from . import mode

        mode.rehover()
        return {"FINISHED"}


def _draw(context) -> None:
    """The cursor cell's outline, in the world."""
    from rok.core import hud

    from .gfx import pen

    c = cell()
    if c is None or document.model is None:
        return
    origin = np.asarray(voxel_object.model_origin(document.model), dtype=np.float32)
    pen.lines(hud.box_edges((1, 1, 1)) + np.asarray(c, dtype=np.float32) + origin, CURSOR_INK, 2.0)


registry.add_draw("key_cursor", 691, "VIEW", _draw)
