"""Brick mode: build a model from bricks taken out of a bin.

Edit > Brick Mode turns it on and off; it works inside the active model, and
the level switch keeps its Model cell lit. A bin on the right edge holds the
bricks; a click takes one into the hand (the keys 1 to 9 too), it hangs
under the mouse as a ghost where it would land, and a click sets it. Every
brick is real voxels at once, and the model's brick table
(rok.core.brick) keeps which cells belong together.

A brick model is locked: only this level and the two Convert commands write
its cells. Model level refuses every tool on it, the scene only moves it and
turns it by quarters, Animate only poses it. Convert to Bricks makes every
voxel a 1×1; Convert to Voxels drops the table for good (undo brings it
back).
"""

from __future__ import annotations

import random

import numpy as np

from rok.core import brick, brick_bin
from rok.core import link as link_core
from rok.core.chrome import LEVEL_BRICK, LEVEL_MODEL, LEVEL_SCENE
from rok.core.live_shader import LiveRecord
from rok.core.undo import BrickRecord, naming, stacked_record

from . import object as voxel_object
from .handlers import registry
from .object import document
from .opkit import CONTEXT, Action, BoolProp, EnumProp, IntProp, register
from .ops_common import _late, can_touch_model, commit, is_brick_model, redraw
from .paint import clip_text, round_outline, round_rect, text
from .state import state

TOOLS = ("PLACE", "PULL", "PAINT", "PICK")
TOOL_NAMES = {"PLACE": "Place", "PULL": "Pull", "PAINT": "Paint", "PICK": "Pick"}
NOT_BRICKS = "Convert this model to bricks to build on it, or add a model"
EMPTY_HAND = "Take a brick from the bin (1 to 9)"
SCENE_LOCKED = "A brick model only moves here, and turns by quarters with R. Convert it to voxels first."
SIDEWAYS = "A linked copy of this model is turned on its side: make it unique first"
SAY_S = 4.0


def active() -> bool:
    """Brick mode is on (and not under Animate mode)."""
    return state.level == LEVEL_BRICK and not state.animating


def can_build() -> bool:
    """The active model takes bricks: a brick model, or an empty voxel model
    (it becomes a brick model with its first brick)."""
    grid = document.grid
    return grid is not None and (grid.bricks is not None or not grid.voxels.any())


def say(message: str) -> None:
    app = CONTEXT.app
    if app is not None:
        app.say(message, SAY_S)
    voxel_object.redraw_statusbar()


def status() -> str:
    """The status line's hint in Brick mode."""
    if not can_build():
        return NOT_BRICKS
    if state.brick_tool == "PLACE" and state.brick_hand is None:
        return EMPTY_HAND
    if _ghost["reason"]:
        return f"The brick does not fit here: {_ghost['reason']}"
    hints = {
        "PLACE": "Click: set the brick · R, Shift R: turn it · Option click: pull · Esc: empty the hand",
        "PULL": "Click a brick: take it out",
        "PAINT": "Click a brick: give it the current colour",
        "PICK": "Click a brick: take its shape and colour",
    }
    return hints[state.brick_tool]


def readout() -> tuple[str, str, str]:
    """(title, tip, glyph) of the tool readout in Brick mode: the tool, and
    the brick in the hand."""
    tool = state.brick_tool
    title = TOOL_NAMES[tool]
    hand = state.brick_hand
    if tool == "PLACE" and hand is not None:
        a, b = sorted(hand[:2])
        title = f"Place {a}×{b}"
    tip = NOT_BRICKS if not can_build() else {
        "PLACE": "Click: set it · R: turn it · ⌥ click: pull" if hand is not None else EMPTY_HAND,
        "PULL": "Click a brick: take it out",
        "PAINT": "Click a brick: the current colour",
        "PICK": "Click a brick: its shape and colour",
    }[tool]
    return title, tip, f"BRICK_{tool}"


# --- the hand and the tools ------------------------------------------------------------


def set_tool(name: str) -> None:
    if name not in TOOLS:
        return
    state.brick_tool = name
    _changed()


def take(size) -> None:
    """Put a w × d brick in the hand; Place is the tool then."""
    state.brick_hand = (int(size[0]), int(size[1]))
    state.brick_tool = "PLACE"
    _changed()


def empty_hand() -> None:
    state.brick_hand = None
    _changed()


def _changed() -> None:
    _late("mode", "rehover")()
    redraw()


def level_changed(old: str, new: str) -> None:
    """level_ops.set_level's hook: the bin opens with the level, the hand
    starts empty, and View Settings makes room."""
    if new == LEVEL_BRICK:
        state.brick_hand = None
        state.brick_turn = 0
        _last_hit[0] = None  # an old model's hit: the key cursor starts on the floor instead
        app = CONTEXT.app
        view_panel = _late_module("view_panel")
        if view_panel is not None and app is not None:
            view_panel._close(app, keep_view=True)
        if not can_build():
            say(NOT_BRICKS)
    elif old == LEVEL_BRICK:
        state.brick_hand = None
        clear_ghost()
    app = CONTEXT.app
    if app is not None and getattr(app, "_started", False):
        app.set_menu()  # Edit > Brick Mode's check mark


def toggle() -> bool:
    """Edit > Brick Mode: into the active model with the bricks, or back to
    Model level. True when the mode changed."""
    from . import anim_mode, level_ops

    if active():
        return level_ops.set_level(LEVEL_MODEL)
    if document.model is None:
        return False
    anim_mode.leave()
    if state.level == LEVEL_SCENE:
        if document.scene.is_hidden(document.model):
            say("The picked model is hidden: show its layer first")
            return False
        level_ops.step_into(document.model.uid)
    return level_ops.set_level(LEVEL_BRICK)


# --- random colours ------------------------------------------------------------------

_rng = random.Random()


def set_random(on: bool) -> None:
    """The bin's Random colours switch."""
    state.brick_random = bool(on)
    if on:
        state.brick_next = brick.next_random(None, _rng)
    _changed()


def _random_rgba() -> tuple:
    r, g, b = brick.RANDOM_COLOURS[state.brick_next % len(brick.RANDOM_COLOURS)][1]
    return (r, g, b, 255)


def ghost_colour() -> int:
    """The palette slot the ghost shows: the current colour, or with Random
    colours the slot nearest the next brick's colour (its own slot is claimed
    when the brick is set)."""
    if not state.brick_random or document.scene is None:
        return int(document.colour)
    from rok.core.palette import nearest

    candidates = np.ones(256, dtype=bool)
    candidates[0] = False
    return nearest(document.scene.palette, _random_rgba(), candidates)


def place_colour() -> tuple[int, str | None]:
    """The palette slot the next brick takes: the current colour, or with
    Random colours a slot holding exactly the next random colour (a free one
    takes it; the change goes into the brick's undo step). (slot, note)."""
    if not state.brick_random:
        return int(document.colour), None
    from . import xs_ops

    rgba = _random_rgba()
    try:
        slot = xs_ops.store_slots()(rgba)
    except xs_ops.PaletteFull:
        return int(document.colour), "No free colour slot for Random colours: the brick takes the current colour"
    if tuple(int(c) for c in document.scene.palette[slot]) != rgba:
        voxel_object.write_colours({slot: rgba})
    return int(slot), None


def _late_module(name: str):
    import importlib

    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ImportError:
        return None


# --- the ghost ---------------------------------------------------------------------------

# What the view shows for the mouse: the host model carrying the ghost bricks,
# and the cages drawn as lines (grid-space (low corner, dims) boxes) with their kind.
_ghost: dict = {"idx": None, "cages": [], "kind": None, "reason": None}
_last_hit: list = [None]  # the mouse's last hit in Brick mode: where the key cursor starts
CAGE_INK = {"REFUSED": (0.94, 0.26, 0.35, 0.9), "PULL": (0.94, 0.26, 0.35, 0.9), "PAINT": (1.0, 1.0, 1.0, 0.8),
            "PICK": (1.0, 1.0, 1.0, 0.8)}


def clear_ghost() -> None:
    """Take the brick ghost and the cages off the view."""
    app = CONTEXT.app
    if _ghost["idx"] is not None and app is not None and hasattr(app.host, "set_ghost"):
        app.host.set_ghost(_ghost["idx"], [])
    _ghost.update(idx=None, cages=[], kind=None, reason=None)


def _host_index(uid: int):
    sync = getattr(CONTEXT.app, "sync", None)
    if sync is None or uid not in sync.uids:
        return None
    return sync.uids.index(uid)


def _show_bricks(boxes, w: int, d: int, colour: int) -> None:
    """Real ghost voxels where the bricks would land."""
    app = CONTEXT.app
    idx = _host_index(document.active_uid)
    if idx is None or app is None or not hasattr(app.host, "set_ghost"):
        return
    box = np.full((w, d, 1), np.uint8(colour), dtype=np.uint8)
    app.host.set_ghost(idx, [((int(x), int(y), int(z)), box) for x, y, z in boxes])
    _ghost["idx"] = idx


def _show_cages(boxes, kind: str) -> None:
    _ghost["cages"] = [((int(x), int(y), int(z)), (int(w), int(d), 1)) for x, y, z, w, d in boxes]
    _ghost["kind"] = kind


def start_of(hit):
    """(x, y, z) the brick's first cell and start height take from a ray hit,
    or None off the grid. A top face starts the brick on it, a side face
    beside it at its height, the floor (a wall hit from above) at z = 0."""
    if hit is None:
        return None
    if hit.wall:
        if tuple(hit.normal) != (0, 0, 1):
            return None
        x, y, _z = hit.voxel
        return (int(x), int(y), 0)
    tx, ty, tz = hit.target
    return (int(tx), int(ty), int(tz))


def placing(hit, grid=None):
    """(boxes, w, d, reason) of a Place click at `hit` with the brick in the
    hand, its turn and Mirror; reason None when it fits. None without a
    brick or off the grid."""
    grid = document.grid if grid is None else grid
    hand = state.brick_hand
    start = start_of(hit)
    if hand is None or start is None or grid is None:
        return None
    x, y, z = start
    sx, sy, _sz = grid.size
    if not (0 <= x < sx and 0 <= y < sy):
        return None
    ox, oy, w, d = brick.turned(hand[0], hand[1], state.brick_turn)
    mirror = tuple(bool(v) for v in getattr(CONTEXT.prefs, "rok_mirror", (False, False, False)))
    boxes, reason = brick.plan(grid, x + ox, y + oy, z, w, d, mirror)
    return boxes, w, d, reason


def _hit(region, view_state, mx: float, my: float):
    from rok.core.raycast import raycast

    from . import mode

    origin, direction = mode.mouse_ray(region, view_state, mx, my)
    return raycast(document.grid.visible, origin, direction)


def target_brick(hit):
    """(cells, index or None) of the brick under a hit: its cells, or the
    loose cell alone. None for a hit on no voxel."""
    grid = document.grid
    if hit is None or hit.wall or grid is None:
        return None
    cell = tuple(int(v) for v in hit.voxel)
    table = grid.bricks
    i = None if table is None else table.at(cell)
    if i is None:
        return np.asarray([cell], dtype=np.int64), None
    return table.cells(i), i


# --- the scene's lock ----------------------------------------------------------------


def picked_models() -> list:
    """The models a scene command acts on: the picked group, or the active model."""
    scene_tools = _late_module("scene_tools")
    if scene_tools is not None:
        members = scene_tools.group_models()
        if members:
            return list(members)
    return [document.model] if document.model is not None else []


def group_has_bricks() -> bool:
    return any(m.grid.bricks is not None for m in picked_models())


def refuse_in_scene() -> bool:
    """True, with the reason in the status line, when a picked model is
    built from bricks: the caller does not run."""
    if not group_has_bricks():
        return False
    say(SCENE_LOCKED)
    return True


def turn_table(model, kind: str, axis: int, turns: int, old_size):
    """A whole-model turn or flip in the scene took `model`'s voxels from a
    grid of `old_size`: its brick table follows. Returns the BrickRecord, or
    None for a voxel model."""
    table = model.grid.bricks
    if table is None:
        return None
    moved = table.transformed(link_core.transform_matrix(kind, axis, turns), old_size)
    model.grid.bricks = moved
    return BrickRecord(model.uid, table.copy(), moved.copy())


def turn_allowed(models, kind: str, axis: int) -> bool:
    """A brick model turns only by quarters about the upright axis. False,
    with the reason in the status line, for any other change of one."""
    if kind == "ROTATE" and int(axis) == 2:
        return True
    if any(m.grid.bricks is not None for m in models):
        say(SCENE_LOCKED)
        return False
    return True


# --- presses ---------------------------------------------------------------------------


def _bin_layout(region) -> brick_bin.BinLayout:
    return brick_bin.bin_layout(1.0, region.width, region.height)


def bin_shown() -> bool:
    view_panel = _late_module("view_panel")
    return active() and not (view_panel is not None and view_panel.active())


def press(p) -> set:
    """A press in the view in Brick mode: the bin, or the tool in the view.
    Never a voxel stroke."""
    region = p.region
    if bin_shown() and region is not None:
        layout = _bin_layout(region)
        if brick_bin.contains(layout, p.mx, p.my):
            if brick_bin.hit_switch(layout, p.mx, p.my):
                if p.left:
                    set_random(not state.brick_random)
                return {"FINISHED"}
            item = brick_bin.hit(layout, p.mx, p.my)
            if item is not None and p.left:
                if can_build():
                    take(item.value)
                else:
                    say(NOT_BRICKS)
            return {"FINISHED"}
    if not p.left:
        empty_hand()  # a right-click empties the hand
        return {"FINISHED"}
    if not can_build():
        say(NOT_BRICKS)
        return {"FINISHED"}
    if state.brick_tool == "PLACE" and state.brick_hand is None and not p.event.alt:
        say(EMPTY_HAND)
        return {"FINISHED"}
    return click(p)


def hover(region, view_state, mx: float, my: float) -> None:
    """The mouse moved in Brick mode: the brick in the hand is a ghost where
    it would land (a red cage where it cannot), and Pull, Paint and Pick
    outline the brick they would take."""
    if document.grid is None or not can_build() or view_state is None:
        clear_ghost()
        redraw()
        return
    hit = _hit(region, view_state, mx, my)
    _last_hit[0] = hit
    show_at(hit)


def last_cell():
    """The cell the key cursor starts at: where the brick in the hand would
    start, or the brick under the mouse for the other tools; None off the grid."""
    hit = _last_hit[0]
    if hit is None:
        return None
    if state.brick_tool == "PLACE":
        start = start_of(hit)
        if start is None or document.grid is None:
            return None
        size = document.grid.size
        return start if all(0 <= start[k] < size[k] for k in range(3)) else None
    return None if hit.wall else tuple(int(v) for v in hit.voxel)


def act_at(hit, alt: bool) -> str | None:
    """A click at `hit` with the brick tool (Option: Pull in Place)."""
    if not can_build():
        return NOT_BRICKS
    tool = "PULL" if state.brick_tool == "PLACE" and alt else state.brick_tool
    return {"PLACE": place_at, "PULL": pull_at, "PAINT": paint_at, "PICK": pick_at}[tool](hit)


def show_at(hit) -> None:
    """The ghost or the outline for a hit (the mouse's, or the key cursor's)."""
    clear_ghost()
    if document.grid is None or not can_build():
        redraw()
        return
    tool = "PULL" if state.brick_tool == "PLACE" and state.alt else state.brick_tool
    if tool == "PLACE":
        found = placing(hit)
        if found is not None:
            boxes, w, d, reason = found
            if reason is None:
                _show_bricks(boxes, w, d, ghost_colour())
            else:
                _show_cages([(x, y, z, w, d) for x, y, z in boxes], "REFUSED")
                _ghost["reason"] = reason
    else:
        target = target_brick(hit)
        if target is not None:
            cells, _i = target
            lo, hi = cells.min(axis=0), cells.max(axis=0)
            _show_cages([(lo[0], lo[1], lo[2], hi[0] - lo[0] + 1, hi[1] - lo[1] + 1)], tool)
    redraw()


def click(p) -> set:
    """The active tool where the mouse is: Place sets the brick in the hand
    (Option: Pull), Pull takes a brick out, Paint recolours it, Pick takes
    its shape and colour. One undo step."""
    if p.view_state is None or document.grid is None:
        return {"FINISHED"}
    hit = _hit(p.region, p.view_state, p.mx, p.my)
    tool = "PULL" if state.brick_tool == "PLACE" and p.event.alt else state.brick_tool
    why = {"PLACE": place_at, "PULL": pull_at, "PAINT": paint_at, "PICK": pick_at}[tool](hit)
    if why:
        say(why)
    hover(p.region, p.view_state, p.mx, p.my)
    return {"FINISHED"}


def _commit(uid: int, recorder, before, label: str) -> None:
    from rok.core.undo import ModelRecord

    grid = document.grid
    after = None if grid.bricks is None else grid.bricks.copy()
    stroke = recorder.finish()
    record = stacked_record(None if stroke is None else ModelRecord(uid, stroke), BrickRecord(uid, before, after))
    with naming(label):
        commit(record, serial=True, hover=True)
    voxel_object.flush_dirty()


def place_at(hit) -> str | None:
    """Set the brick in the hand where `hit` lands it. None, or why not."""
    from rok.core.undo import StrokeRecorder

    if state.brick_hand is None:
        return EMPTY_HAND
    found = placing(hit)
    if found is None:
        return "Point at the floor or at a brick"
    boxes, w, d, reason = found
    if reason is not None:
        return f"The brick does not fit: {reason}"
    grid = document.grid
    colour, note = place_colour()
    before = None if grid.bricks is None else grid.bricks.copy()
    recorder = StrokeRecorder(grid)
    placed = brick.put(grid, boxes, w, d, colour, write=recorder.set_many)
    _commit(document.active_uid, recorder, before, f"Brick {placed[0].name()}")
    if state.brick_random:
        state.brick_next = brick.next_random(state.brick_next, _rng)
    return note


def _brick_cells(hit):
    target = target_brick(hit)
    if target is None:
        return None, None, "Point at a brick"
    cells, i = target
    if document.grid.sections and document.grid.is_cut(cells).any():
        return None, None, "Part of that brick is in a hidden slice"
    return cells, i, None


def pull_at(hit) -> str | None:
    """Take the brick under `hit` out; a loose cell goes as a 1×1."""
    from rok.core.undo import StrokeRecorder

    cells, i, why = _brick_cells(hit)
    if why:
        return why
    grid = document.grid
    before = None if grid.bricks is None else grid.bricks.copy()
    recorder = StrokeRecorder(grid)
    with brick.unlocked(grid):
        recorder.set_many(cells, 0, ignore_lock=True)
    name = "1×1"
    if i is not None:
        name = grid.bricks.remove(i).name()
    _commit(document.active_uid, recorder, before, f"Pull {name}")
    return None


def paint_at(hit) -> str | None:
    """Give the brick under `hit` the current colour, its cells and its record."""
    from dataclasses import replace

    from rok.core.undo import StrokeRecorder

    cells, i, why = _brick_cells(hit)
    if why:
        return why
    grid = document.grid
    colour = int(document.colour)
    before = None if grid.bricks is None else grid.bricks.copy()
    recorder = StrokeRecorder(grid)
    with brick.unlocked(grid):
        recorder.set_many(cells, colour, ignore_lock=True)
    if i is not None:
        grid.bricks.bricks[i] = replace(grid.bricks.bricks[i], colour=colour)
    if recorder.finish() is None and (before is None or before == grid.bricks):
        return None  # the same colour already: no step
    _commit(document.active_uid, recorder, before, "Paint Brick")
    return None


def pick_at(hit) -> str | None:
    """The brick under `hit` goes into the hand, as it lies, and its colour
    becomes the current colour."""
    target = target_brick(hit)
    if target is None:
        return "Point at a brick"
    cells, i = target
    if i is None:
        size, colour = (1, 1), int(document.grid.voxels[tuple(cells[0])])
    else:
        b = document.grid.bricks.bricks[i]
        size, colour = (b.w, b.d), b.colour
    document.colour = colour
    take(size)
    state.brick_turn = 0
    return None


# --- converting ------------------------------------------------------------------------


def _sideways(model) -> bool:
    members = link_core.members(document.scene, model.link) or [model]
    return any(not brick.upright(m.orient) for m in members)


def convert_to_bricks(model) -> str | None:
    """Make `model` a brick model of 1×1s as one undo step: its mask goes and
    its live shaders are baked. None, or why it cannot."""
    scene = document.scene
    if model.grid.bricks is not None:
        return "This model is built from bricks already"
    if _sideways(model):
        return SIDEWAYS
    from . import select_ops

    parts = [select_ops.set_mask(None, model.uid)]
    shaders = scene.live.get(model.uid)
    if shaders:
        scene.live.pop(model.uid, None)
        parts.append(LiveRecord(model.uid, tuple(shaders), ()))
    table = brick.convert_to_bricks(model.grid)
    parts.insert(0, BrickRecord(model.uid, None, table.copy()))
    with naming("Convert to Bricks"):
        commit(stacked_record(*parts), serial=True, sync=True, hover=True)
    voxel_object.flush_dirty()
    return None


def convert_to_voxels(model) -> str | None:
    """Make `model` plain voxels again as one undo step; the bricks are gone."""
    table = model.grid.bricks
    if table is None:
        return "This model is plain voxels already"
    brick.convert_to_voxels(model.grid)
    with naming("Convert to Voxels"):
        commit(BrickRecord(model.uid, table.copy(), None), serial=True, sync=True, hover=True)
    voxel_object.flush_dirty()
    return None


# --- actions ---------------------------------------------------------------------------


@register
class BrickModeAction(Action):
    """Edit > Brick Mode: build the active model from bricks out of a bin."""

    command_id = "rok.brick_mode"
    title = "Brick Mode"
    tooltip = "Build the model from bricks out of a bin; again to go back to the voxel tools"

    @classmethod
    def poll(cls, context):
        return document.model is not None and not state.stroke_active

    def execute(self, context):
        toggle()
        return {"FINISHED"}


@register
class BrickToolAction(Action):
    command_id = "rok.brick_tool"
    title = "Brick Tool"
    tooltip = "Pick Brick mode's tool: Place (B), Pull (U), Paint (G) or Pick (I)"
    tool = EnumProp(name="Tool", items=tuple((t, TOOL_NAMES[t], "") for t in TOOLS), default="PLACE")

    @classmethod
    def poll(cls, context):
        return active()

    def execute(self, context):
        set_tool(self.tool)
        return {"FINISHED"}


@register
class BrickTakeAction(Action):
    command_id = "rok.brick_take"
    title = "Take a Brick"
    tooltip = "Take a brick from the bin into the hand: 1 to 9 take the first nine"
    number = IntProp(name="Number", description="The brick's place in the bin, from 1", default=1, min=1, max=99)

    @classmethod
    def poll(cls, context):
        return active()

    def execute(self, context):
        every = brick_bin.bricks()
        if not 1 <= self.number <= len(every):
            return {"CANCELLED"}
        if not can_build():
            self.report({"WARNING"}, NOT_BRICKS)
            return {"CANCELLED"}
        take(every[self.number - 1])
        return {"FINISHED"}


@register
class BrickTurnAction(Action):
    """R turns the brick in the hand a quarter turn clockwise about its first
    cell, Shift+R back."""

    command_id = "rok.brick_turn"
    title = "Turn the Brick"
    tooltip = "Turn the brick in the hand a quarter turn (R; Shift+R the other way)"
    back = BoolProp(name="Back", default=False)

    @classmethod
    def poll(cls, context):
        return active() and state.brick_hand is not None

    def execute(self, context):
        state.brick_turn = (state.brick_turn + (-1 if self.back else 1)) % 4
        _changed()
        return {"FINISHED"}


@register
class BrickEscapeAction(Action):
    """Esc with a brick in the hand empties the hand; the next Esc goes to the scene."""

    command_id = "rok.brick_escape"
    title = "Empty the Hand"
    tooltip = "Put the brick in the hand back"

    @classmethod
    def poll(cls, context):
        return active() and state.brick_hand is not None

    def execute(self, context):
        empty_hand()
        return {"FINISHED"}


@register
class ConvertToBricksAction(Action):
    command_id = "rok.convert_to_bricks"
    title = "Convert to Bricks"
    tooltip = "Every voxel of the model becomes a 1×1 brick; then only Brick mode builds on it"

    @classmethod
    def poll(cls, context):
        return can_touch_model() and not is_brick_model()

    def execute(self, context):
        why = convert_to_bricks(document.model)
        if why is not None:
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        self.report({"INFO"}, f"{len(document.grid.bricks)} bricks: build on them in Brick mode")
        return {"FINISHED"}


@register
class ConvertToVoxelsAction(Action):
    command_id = "rok.convert_to_voxels"
    title = "Convert to Voxels"
    tooltip = "The model becomes plain voxels again; the bricks are gone for good (only undo brings them back)"

    @classmethod
    def poll(cls, context):
        return can_touch_model() and is_brick_model()

    def execute(self, context):
        why = convert_to_voxels(document.model)
        if why is not None:
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        self.report({"INFO"}, "Plain voxels now: every tool works on the model")
        return {"FINISHED"}


# --- drawing the bin --------------------------------------------------------------------


def _colour_rgba() -> tuple:
    grid = document.grid
    index = int(getattr(document, "colour", 1))
    if grid is None:
        return (0.8, 0.3, 0.2, 1.0)
    r, g, b, _a = (int(v) for v in grid.palette[index])
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def draw(region) -> None:
    from . import view_panel as vpanel

    layout = _bin_layout(region)
    s = layout.scale
    round_rect(layout.rect, 14 * s, vpanel.PANEL_BG)
    round_outline(layout.rect, 14 * s, vpanel.PANEL_EDGE, 1.0 * s)
    usable = can_build()
    colour = _colour_rgba()
    if not usable:
        colour = colour[:3] + (0.3,)
    line_ink = tuple(c * 0.62 for c in colour[:3]) + (colour[3],)
    hand = state.brick_hand
    for item in layout.items:
        if item.kind == "TITLE":
            text(item.text, item.rect[0], item.rect[1] + item.rect[3] / 2, 14 * s, vpanel.TEXT)
        elif item.kind == "RANDOM":
            on = state.brick_random
            box = (item.rect[0], item.rect[1] + (item.rect[3] - 16 * s) / 2, 16 * s, 16 * s)
            round_rect(box, 4 * s, vpanel.EMBER if on else vpanel.BUTTON_BG)
            round_outline(box, 4 * s, vpanel.PANEL_EDGE, 1.0 * s)
            if on:
                from .gfx import pen

                pts = [(box[0] + 4 * s, box[1] + 8 * s), (box[0] + 7 * s, box[1] + 4.5 * s),
                       (box[0] + 12.5 * s, box[1] + 11.5 * s)]
                pen.lines(pen.to_3d([pts[0], pts[1], pts[1], pts[2]]), vpanel.EMBER_DARK, 2.0 * s)
            text(item.text, box[0] + 24 * s, item.rect[1] + item.rect[3] / 2, 12 * s,
                 vpanel.TEXT if on else vpanel.TEXT_DIM)
        elif item.kind == "ROW":
            text(item.text, item.rect[0], item.rect[1] + item.rect[3] / 2, 11 * s, vpanel.TEXT_DIM)
        elif item.kind == "FOOT":
            words = item.text if usable else NOT_BRICKS
            text(clip_text(words, item.rect[2], 11 * s), item.rect[0], item.rect[1] + item.rect[3] / 2, 11 * s,
                 vpanel.TEXT_FAINT)
        elif item.kind == "BRICK":
            taken = hand is not None and usable and tuple(sorted(hand[:2])) == tuple(sorted(item.value))
            round_rect(item.rect, 6 * s, vpanel.BUTTON_HOVER if taken else vpanel.BUTTON_BG)
            w, d = max(item.value), min(item.value)
            cell = layout.cell
            x0 = item.rect[0] + (item.rect[2] - w * cell) / 2
            y0 = item.rect[1] + (item.rect[3] - d * cell) / 2
            if state.brick_random and usable:
                # Random colours: each brick of the bin in one of the colours.
                r, g, b = brick.RANDOM_COLOURS[brick_bin.bricks().index(item.value) % len(brick.RANDOM_COLOURS)][1]
                colour = (r / 255.0, g / 255.0, b / 255.0, 1.0)
                line_ink = tuple(c * 0.62 for c in colour[:3]) + (1.0,)
            round_rect((x0, y0, w * cell, d * cell), 2 * s, colour)
            thick = max(1.0, 1.0 * s)
            for ax, ay, bx, by in brick_bin.divisions(item, cell):
                if ax == bx:
                    round_rect((ax - thick / 2, ay, thick, by - ay), 0, line_ink)
                else:
                    round_rect((ax, ay - thick / 2, bx - ax, thick), 0, line_ink)
            if taken:
                round_outline(item.rect, 6 * s, vpanel.EMBER, 2.0 * s)
            if item.key:
                text(str(item.key), item.rect[0] + 3 * s, item.rect[1] + item.rect[3] - 7 * s, 9 * s,
                     vpanel.TEXT_FAINT)


def _draw_cages(context) -> None:
    """The red cage of a refused brick, and the outline of the brick Pull,
    Paint or Pick would take: lines in the world, round the grid boxes."""
    from rok.core import hud

    from .gfx import pen

    if not active() or not _ghost["cages"] or document.model is None:
        return
    origin = np.asarray(voxel_object.model_origin(document.model), dtype=np.float32)
    ink = CAGE_INK.get(_ghost["kind"], CAGE_INK["REFUSED"])
    for lo, dims in _ghost["cages"]:
        pen.lines(hud.box_edges(dims) + np.asarray(lo, dtype=np.float32) + origin, ink, 2.0)


registry.add_draw("brick_cages", 690, "VIEW", _draw_cages)


def _draw_handler(context) -> None:
    region = context.region
    if region is None or not bin_shown() or document.grid is None:
        return
    draw(region)


registry.add_draw("brick_bin", 764, "PIXEL", _draw_handler)
