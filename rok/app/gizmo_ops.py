"""The transform gizmo's state and the operator that applies a drag.

`refresh()` decides whether the gizmo shows and where. A drag on a handle runs
inside `rok.stroke`; for a model it calls `begin_model_drag`, `update_model_drag`
and `end_model_drag` here, and on release `rok.transform` applies the result to
the document. The overlay draws the gizmo from `state.gizmo`.

The gizmo lives in the active model's grid space for both targets: `pivot2` is
in doubled grid units and `origin` is the model origin. A model transform
converts the pivot to doubled world cells for grid_ops.

During a drag, Rok shows a model at its new place with a display-only
transform in `state.turn_preview` (uid -> 4 x 4, grid units),
which the app sends to the host on every flush.

Animate mode is parked: a MOVE or ROTATE on a model with an animation track
edits the rest pose, as it does on any other model.

Imports object, state and the core modules only, never tools or mode.
"""

from __future__ import annotations

import numpy as np

from rok.core import brick as brick_core
from rok.core import free_rotate, gizmo, grid_ops, hud, select
from rok.core import link as link_core
from rok.core.edit import SELECT
from rok.core.select import Selection
from rok.core.undo import (
    GridRecord,
    ModelRecord,
    Record,
    StrokeRecorder,
    changes_saved_data,
    selection_record,
    stacked_record,
)

from . import cage_ops
from . import object as voxel_object
from .handlers import registry
from .object import document
from .opkit import (
    CONTEXT, BoolProp, EnumProp, FloatProp, IntProp, IntVectorProp, Action, register,
)
from .ops_common import commit
from .state import GizmoState, state

GIZMO_LENGTH_PX = 90.0
UI_SCALE = 1.0  # Rok works in logical pixels

_TARGET_ITEMS = (
    (gizmo.SELECTION, "Selection", "Act on the selected voxels"),
    (gizmo.MODEL, "Model", "Act on the whole model"),
)
_KIND_ITEMS = (
    (gizmo.MOVE, "Move", "Shift by whole cells"),
    (gizmo.ROTATE, "Rotate", "Turn a quarter turn at a time"),
    (gizmo.FLIP, "Flip", "Mirror across the pivot"),
)

# (uid, selection version) the pivot override was set for; it drops when they change.
_override_key: tuple[int, int] | None = None

# The model drag in progress: (uid, position at the press), else None.
_drag: tuple[int, tuple[int, int, int]] | None = None

# The uids a ring drag turns on the GPU (state.turn_preview), to clear at the end:
# only models with animation states, which cannot take a free angle.
_turned: set[int] = set()

# The ring drag in progress: scene_tools._Rotation, which turns the real voxels.
_rotation = None


# --- state ------------------------------------------------------------------------


def _active_tool() -> str:
    return getattr(CONTEXT.prefs, "rok_tool", "")


def _pivot_from_override(size) -> tuple[int, int, int] | None:
    """The override as doubled grid units, clamped to the grid box, or None."""
    override = document.pivot_override
    if override is None:
        return None
    p = np.rint(2.0 * np.asarray(override, dtype=np.float64)).astype(np.int64)
    p = np.clip(p, 0, 2 * np.asarray(size, dtype=np.int64))
    return (int(p[0]), int(p[1]), int(p[2]))


def _drop_stale_override(model) -> None:
    global _override_key
    if document.pivot_override is None:
        _override_key = None
        return
    key = (model.uid, model.selection.version)
    if _override_key != key:
        document.pivot_override = None
        _override_key = None


def selection_pivot2(model) -> tuple[int, int, int] | None:
    """The gizmo pivot for the selection, doubled grid units. None when empty."""
    bounds = model.selection.bounds()
    if bounds is None:
        return None
    override = _pivot_from_override(model.grid.size)
    if override is not None:
        return override
    lo, hi = bounds
    return (lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1)


def model_pivot2(model) -> tuple[int, int, int]:
    """The gizmo pivot for the whole model: the box centre, doubled grid units."""
    override = _pivot_from_override(model.grid.size)
    if override is not None:
        return override
    sx, sy, sz = model.grid.size
    return (int(sx), int(sy), int(sz))


def _to_world2(model, pivot2) -> tuple[int, int, int]:
    """Doubled grid units of `model` to doubled world cells."""
    px, py, pz = model.position
    return (int(pivot2[0]) + 2 * px, int(pivot2[1]) + 2 * py, int(pivot2[2]) + 2 * pz)


def _scene_group():
    """scene_tools' (group_models, group_box), or None before it is imported."""
    try:
        from . import scene_tools  # late: scene_tools imports this module
    except ImportError:
        return None
    models = getattr(scene_tools, "group_models", None)
    box = getattr(scene_tools, "group_box", None)
    if models is None or box is None:
        return None
    return models, box


def refresh() -> None:
    """Recompute state.gizmo from the action, the selection and the scene.
    The lattice cage is checked here too: its rest goes stale on the same events."""
    cage_ops.refresh()
    model = document.model
    scene = document.scene
    in_scene = state.level == "SCENE"
    if (model is None or scene is None or (_active_tool() != SELECT and not in_scene)
            or state.level == "BRICK" or (not in_scene and model.grid.bricks is not None)):
        state.gizmo = None
        return
    if in_scene and (scene.is_hidden(model) or state.scene_tool == "CUT"):
        state.gizmo = None
        return
    if state.animating and state.anim_paint:
        state.gizmo = None  # a paint tool is on: the click paints the part's frame, no handle in the way
        return
    _drop_stale_override(model)
    pivot2 = None if in_scene else selection_pivot2(model)
    if in_scene:
        # The scene: the gizmo moves and turns the picked model, and its cubes
        # stretch it (scene_tools); it has no centre square (GIZMO_SKIP_IN_SCENE).
        target = gizmo.MODEL
        pivot2 = model_pivot2(model)
        group = _scene_group()
        if group is not None and len(group[0]()) > 1:
            # A group turns about its common centre: the gizmo sits there.
            lo, hi = group[1]()
            pivot2 = tuple(lo[i] + hi[i] - 2 * int(model.position[i]) for i in range(3))
    elif pivot2 is not None:
        target = gizmo.SELECTION
    elif len(scene.models) >= 2:
        target = gizmo.MODEL
        pivot2 = model_pivot2(model)
    else:
        state.gizmo = None
        return
    origin = voxel_object.model_origin(model)
    if state.animating:
        # Animate mode: the part's own pivot, where the pose at the play head has it.
        from . import anim_mode

        target = gizmo.MODEL
        pivot2, origin = anim_mode.gizmo_place(model)
    state.gizmo = GizmoState(
        target=target,
        pivot2=pivot2,
        origin=origin,
        length_px=GIZMO_LENGTH_PX * UI_SCALE,
        uid=model.uid,
    )


def set_pivot(pivot2) -> None:
    """Shift+drag on the centre: move the pivot. Not recorded."""
    global _override_key
    model = document.model
    if model is None:
        return
    if state.animating:
        from . import anim_mode
        from rok.core import gizmo as gizmo_core

        anim_mode.gizmo_preview(gizmo_core.GizmoResult(gizmo_core.PIVOT, pivot2=tuple(pivot2)))
        return
    document.pivot_override = (pivot2[0] / 2.0, pivot2[1] / 2.0, pivot2[2] / 2.0)
    _override_key = (model.uid, model.selection.version)
    if state.gizmo is not None:
        state.gizmo.pivot2 = (int(pivot2[0]), int(pivot2[1]), int(pivot2[2]))
    voxel_object.redraw_3d_views()


def save_pivot() -> tuple:
    """The pivot state at the press of a pivot drag, for Esc."""
    return (document.pivot_override, _override_key)


def restore_pivot(saved: tuple) -> None:
    """Esc on a pivot drag: put the pivot back where the press found it."""
    global _override_key
    document.pivot_override, _override_key = saved
    refresh()
    voxel_object.redraw_3d_views()


# --- the model drag ------------------------------------------------------------------


def _place_empty(uid: int, position) -> None:
    """Show model `uid` at `position` without moving it in the scene. A position equal to the model's own drops the preview."""
    scene = document.scene
    if scene is None:
        return
    try:
        model = scene.by_uid(uid)
    except ValueError:
        return
    delta = [float(position[i]) - float(model.position[i]) for i in range(3)]
    if delta == [0.0, 0.0, 0.0]:
        state.turn_preview.pop(uid, None)
    else:
        m = np.eye(4)
        m[:3, 3] = delta
        state.turn_preview[uid] = m
    voxel_object.redraw_3d_views()


def begin_model_drag(gesture=None) -> None:
    """Remember the model's position; the model moves live during a move drag.
    In Animate mode the drag poses the part instead (anim_mode)."""
    global _drag
    if state.animating:
        from . import anim_mode

        anim_mode.gizmo_begin(gesture)
        _drag = None
        return
    model = document.model
    if model is None:
        _drag = None
        return
    _drag = (model.uid, tuple(int(p) for p in model.position))
    state.gizmo_ghost = None
    state.model_drag_offset = (0, 0, 0)


def update_model_drag(result: gizmo.GizmoResult) -> None:
    """Show a MODEL result: the model at the offset, or a ghost of the turned model.

    The overlay adds `state.model_drag_offset` to the active model's origin, so the
    box, the mirror planes and the gizmo move with the model.
    """
    if state.animating:
        from . import anim_mode

        anim_mode.gizmo_preview(result)
        return
    if _drag is None:
        return
    uid, position = _drag
    model = document.model
    if model is None or model.uid != uid:
        return
    state.model_drag_offset = (0, 0, 0)
    if result.kind == gizmo.MOVE:
        state.gizmo_ghost = None
        offset = tuple(int(v) for v in result.offset)
        state.model_drag_offset = (offset[0], offset[1], offset[2])
        _place_empty(uid, tuple(position[i] + offset[i] for i in range(3)))
    elif result.kind == gizmo.ROTATE:
        state.gizmo_ghost = None
        _turn_live(model, result, position)
    elif result.kind == gizmo.FLIP:
        _clear_turn()
        _place_empty(uid, position)
        _update_ghost(model, result)
    elif result.kind == gizmo.PIVOT:
        set_pivot(result.pivot2)
    else:
        state.gizmo_ghost = None
        _clear_turn()
        _rest()
        _place_empty(uid, position)
    voxel_object.redraw_3d_views()


def turning_models(model) -> list:
    """The models a ring drag on `model`'s gizmo turns: the picked group, or the model."""
    group = _scene_group()
    if group is not None and state.level == "SCENE":
        members = list(group[0]())
        if len(members) > 1:
            return members
    return [model]


def _turn_live(model, result: gizmo.GizmoResult, press_position) -> None:
    """Turn the real voxels of the models by the drag's angle, from the voxels
    they had at the press, so turning back and forth never wears them down.
    The renderer marches the voxels directly: a turn costs a texture upload,
    no mesh rebuild."""
    global _rotation
    from . import scene_tools  # late: scene_tools imports this module

    # The pivot in world cells, from the place at the press: the models move as they turn.
    pivot2 = tuple(int(result.pivot2[i]) + 2 * int(press_position[i]) for i in range(3))
    if _rotation is None:
        models = turning_models(model)
        if any(m.states for m in models):
            _turn_on_gpu(models, result, pivot2)  # a free angle would break the animation
            return
        _rotation = scene_tools._Rotation(models, result.axis)
    _rotation.pivot2 = pivot2
    if not _rotation.apply(result.degrees):
        _rotation.preview(result.degrees)  # too big for the grid: show it, keep nothing


def _turn_on_gpu(models, result: gizmo.GizmoResult, pivot2) -> None:
    from rok.core import free_rotate

    items = [(m.grid.voxels, tuple(int(p) for p in m.position)) for m in models]
    matrices = free_rotate.preview_matrices(items, result.axis, result.degrees, pivot2)
    _clear_turn()
    for m, matrix in zip(models, matrices):
        state.turn_preview[m.uid] = matrix
        _turned.add(m.uid)


def _rest() -> None:
    """The turning models back as they were at the press; the rotation stays
    ready for the next angle."""
    if _rotation is not None and _rotation.applied:
        _rotation.restore()


def take_rotation():
    """The ring drag's rotation, for the release to keep or undo; None if none."""
    global _rotation
    rotation, _rotation = _rotation, None
    return rotation


def _clear_turn() -> None:
    for uid in _turned:
        state.turn_preview.pop(uid, None)
    _turned.clear()


def _update_ghost(model, result: gizmo.GizmoResult) -> None:
    """The exposed surface of the transformed model at its new place, cached per turns."""
    pivot2 = _to_world2(model, result.pivot2)
    key = (model.uid, result.kind, result.axis, result.turns, pivot2, document.edit_serial)
    if state.gizmo_ghost is not None and state.gizmo_ghost[0] == key:
        return
    voxels, position = _transformed(model, result.kind, result.axis, result.turns, pivot2)
    ox, oy, oz = document.view_offset
    origin = (ox + position[0], oy + position[1], oz + position[2])
    size = np.asarray(voxels.shape, dtype=np.int64)
    tris, lines = hud.mask_surface(voxels != 0, (0, 0, 0), size - 1)
    state.gizmo_ghost = (key, (origin, tris, lines))


def end_model_drag(cancel: bool) -> None:
    """Drop the ghost and show the model where the scene says it is."""
    global _drag
    if state.animating:
        from . import anim_mode

        anim_mode.gizmo_end()
    if _drag is not None:
        uid, position = _drag
        model = None
        if document.scene is not None:
            try:
                model = document.scene.by_uid(uid)
            except ValueError:
                model = None
        _clear_turn()
        rotation = take_rotation()
        if rotation is not None and cancel:
            rotation.restore()
        _place_empty(uid, position if cancel or model is None else model.position)
    _drag = None
    state.gizmo_ghost = None
    state.model_drag_offset = (0, 0, 0)
    voxel_object.redraw_3d_views()


def _transformed(model, kind: str, axis: int, turns: int, pivot2_world, voxels=None):
    """(voxels, position) of the model after a turn or a flip about a world
    pivot; `voxels` instead of the model's own (its mask)."""
    grid = model.grid
    source = grid.voxels if voxels is None else voxels
    if kind != gizmo.ROTATE:
        return grid_ops.flip_model(source, model.position, axis, pivot2_world)
    voxels, position = grid_ops.rotate_model(source, model.position, axis, turns, pivot2_world)
    if document.pivot_override is None:
        # The plain box centre can sit on a half cell, and snap_pivot then pushes the
        # model one cell every turn. Keep t = position + size // 2 instead, as
        # rok.rotate does: exact for any size, so four turns restore
        # the position. A user-moved pivot keeps the snap, which the ghost shows.
        size = grid.size
        t = tuple(int(model.position[i]) + size[i] // 2 for i in range(3))
        position = tuple(t[i] - voxels.shape[i] // 2 for i in range(3))
    return voxels, position


# --- the operators -------------------------------------------------------------------


@register
class RotateDegreesAction(Action):
    """Turn the selection, or the whole model, by a typed angle: the gizmo's rings turn live, which is slow on a very big
    model; this dialog turns once. In the rail's Model menu, off by default."""

    command_id = "rok.rotate_degrees"
    title = "Rotate by Degrees"
    tooltip = "Turn the selection about its centre, or the whole model, by an angle you type"
    # Z, about one pivot (several axes at once). A quarter
    # turn loses nothing.
    x = FloatProp(name="X", description="The turn about X, the axis from left to right (the red ring)",
                      min=-180.0, max=180.0, default=0.0, step=0.1, precision=1)
    y = FloatProp(name="Y", description="The turn about Y, the axis from front to back (the green ring)",
                      min=-180.0, max=180.0, default=0.0, step=0.1, precision=1)
    z = FloatProp(name="Z", description="The turn about Z, the axis that points up (the blue ring)",
                      min=-180.0, max=180.0, default=45.0, step=0.1, precision=1)

    @classmethod
    def poll(cls, context):
        return document.model is not None and state.level == "MODEL" and document.grid.bricks is None

    def invoke(self, context, event):
        return context.prefs.open_dialog(self, title="Rotate by Degrees", confirm_text="Rotate")

    def draw(self, context):
        from .overlay import AXIS_COLOURS

        layout = self.layout
        # Each axis in its ring's colour, named by where it points (show which axis is which).
        layout.prop(self, "x", text="X · left to right", colour=AXIS_COLOURS[0])
        layout.prop(self, "y", text="Y · front to back", colour=AXIS_COLOURS[1])
        layout.prop(self, "z", text="Z · up", colour=AXIS_COLOURS[2])
        has_selection = document.selection is not None and not document.selection.is_empty()
        layout.label(text="Turns the selection about its centre" if has_selection else "Turns the whole model")
        layout.label(text="Several angles turn X, then Y, then Z · voxels that leave the grid are lost")

    def steps(self) -> list[tuple[int, float]]:
        """The turns to make, (axis, degrees) in order, the zero ones left out."""
        out = []
        for axis, value in enumerate((self.x, self.y, self.z)):
            degrees = free_rotate.snap(float(value))
            if degrees != 0.0:
                out.append((axis, degrees))
        return out

    def execute(self, context):
        model = document.model
        if model is None:
            return {"CANCELLED"}
        steps = self.steps()
        if not steps:
            return {"CANCELLED"}
        sel = model.selection
        if sel is not None and not sel.is_empty():
            return self._turn_selection(model, steps)
        from . import scene_tools

        rotation = scene_tools._Rotation([model], steps[0][0])
        if not rotation.apply_steps(steps):
            self.report({"ERROR"}, rotation.error)
            return {"CANCELLED"}
        rotation.commit()
        return {"FINISHED"}

    def _turn_selection(self, model, steps):
        """The selected voxels turned about the selection's centre, as the
        gizmo's ring turns them, one axis after the other, in one undo step."""
        from . import select_ops

        grid = model.grid
        sel = model.selection
        pivot2 = selection_pivot2(model)
        before = sel.mask.copy()
        kept = before & ~select_ops.unmasked(before, model.uid)  # masked voxels stay where they are
        moving = Selection(grid.size)
        moving.replace(before & ~kept)
        recorder = StrokeRecorder(grid)
        for axis, degrees in steps:
            quarter = free_rotate.quarter_turns(degrees)
            if quarter is not None:
                new = select.rotate_cells(grid.visible, moving, axis, quarter, gizmo.snap_pivot(pivot2, axis),
                                          recorder.set_many)
            else:
                new = gizmo.free_rotate_cells(grid.visible, moving, axis, degrees, pivot2, recorder.set_many)
            moving.replace(new)
        new = moving.mask
        sel.replace(new | kept)
        voxel_object.flush_dirty()
        record = recorder.finish()
        sel.prune(grid.visible)
        sel_record = selection_record(model.uid, before, sel.mask)
        record = sel_record if record is None else stacked_record(ModelRecord(model.uid, record), sel_record)
        commit(record, modified=changes_saved_data(record), serial=True, sync=True)
        return {"FINISHED"}


@register
class TransformAction(Action):
    """Apply a gizmo result to the selection or the active model."""

    command_id = "rok.transform"
    title = "Transform"
    tooltip = "Move, turn or flip the selection or the whole model"
    target = EnumProp(
        name="Target",
        description="What to act on",
        items=_TARGET_ITEMS,
        default=gizmo.SELECTION,
    )
    kind = EnumProp(name="Kind", description="What to do", items=_KIND_ITEMS, default=gizmo.MOVE)
    axis = IntProp(name="Axis", description="0 is X, 1 is Y, 2 is Z", min=0, max=2, default=0)
    turns = IntProp(name="Turns", description="How many quarter turns", min=0, max=3, default=1)
    offset = IntVectorProp(name="Offset", description="How far to move, in cells", size=3, default=(0, 0, 0))
    copy = BoolProp(name="Copy", description="Leave the voxels behind and move a copy", default=False)

    @classmethod
    def poll(cls, context):
        return document.model is not None

    def execute(self, context):
        model = document.model
        if model is None:
            return {"CANCELLED"}
        if model.grid.bricks is not None and (self.target == gizmo.SELECTION or (
                self.kind != gizmo.MOVE and not (self.kind == gizmo.ROTATE and int(self.axis) == 2))):
            # A brick model only moves, and turns by quarters about the upright axis.
            from .brick_mode import SCENE_LOCKED

            self.report({"WARNING"}, SCENE_LOCKED)
            return {"CANCELLED"}
        if self.target == gizmo.SELECTION:
            record = self._apply_selection(model)
        else:
            record = self._apply_model(model)
        if record is None:
            return {"CANCELLED"}
        # A flip of a symmetric selection writes no voxel: it leaves the file saved.
        commit(record, modified=changes_saved_data(record), serial=True, sync=True)
        return {"FINISHED"}

    def _apply_selection(self, model):
        grid = model.grid
        sel = model.selection
        pivot2 = selection_pivot2(model)
        if pivot2 is None:
            return None
        from . import select_ops

        before = sel.mask.copy()
        # Masked voxels stay where they are: the move takes the rest.
        kept = before & ~select_ops.unmasked(before, model.uid)
        if kept.any():
            moving = Selection(grid.size)
            moving.replace(before & ~kept)
            sel = moving
        recorder = StrokeRecorder(grid)
        write = recorder.set_many
        axis = int(self.axis)
        if self.kind == gizmo.MOVE:
            offset = tuple(int(v) for v in self.offset)
            if offset == (0, 0, 0):
                return None
            new = select.move_cells(grid.visible, sel, offset, self.copy, write)
        elif self.kind == gizmo.ROTATE:
            turns = int(self.turns) % 4
            if turns == 0:
                return None
            if turns % 2:  # a half turn keeps cells on cells for any pivot
                pivot2 = gizmo.snap_pivot(pivot2, axis)
            new = select.rotate_cells(grid.visible, sel, axis, turns, pivot2, write)
        else:
            new = select.flip_cells(grid.visible, sel, axis, pivot2, write)
        sel = model.selection
        sel.replace(new | kept)
        voxel_object.flush_dirty()
        record = recorder.finish()
        sel.prune(grid.visible)
        sel_record = selection_record(model.uid, before, sel.mask)
        if record is None:
            return sel_record
        return stacked_record(ModelRecord(model.uid, record), sel_record)

    def _apply_model(self, model) -> Record | None:
        grid = model.grid
        before = tuple(int(p) for p in model.position)
        if self.kind == gizmo.MOVE:
            offset = tuple(int(v) for v in self.offset)
            if offset == (0, 0, 0):
                return None
            after = (before[0] + offset[0], before[1] + offset[1], before[2] + offset[2])
            model.position = after
            return ModelRecord(model.uid, None, before, after)
        axis = int(self.axis)
        turns = int(self.turns) % 4
        if self.kind == gizmo.ROTATE and turns == 0:
            return None
        pivot2 = _to_world2(model, model_pivot2(model))
        if self.kind == gizmo.ROTATE and turns % 2:
            # Only for a moved pivot: with the default one _transformed throws the
            # turned position away and keeps the box centre instead.
            pivot2 = gizmo.snap_pivot(pivot2, axis)
        voxels, after = _transformed(model, self.kind, axis, turns, pivot2)
        from . import select_ops

        mask_record = select_ops.carried_mask(
            model, lambda v: _transformed(model, self.kind, axis, turns, pivot2, voxels=v)[0])
        inner = GridRecord(before=grid.voxels.copy(), after=voxels)
        old_size = grid.size
        with brick_core.unlocked(grid):
            grid.replace_voxels(voxels)
        from .brick_mode import turn_table

        bricks = turn_table(model, self.kind, axis, turns, old_size)  # the bricks turn with the voxels
        model.position = tuple(int(p) for p in after)
        # The whole model turned in the scene: a linked copy turns alone (link.py).
        orient_before = model.orient
        model.orient = link_core.turned(orient_before, link_core.transform_matrix(self.kind, axis, turns))
        # The array turned: its cells no longer line up with the selection or the
        # pivot. The dropped selection goes on the record, so undo brings it back.
        sel_before = None if model.selection.is_empty() else model.selection.mask.copy()
        model.selection = Selection(grid.size)
        document.pivot_override = None
        record = ModelRecord(model.uid, inner, before, model.position,
                             orient_before=orient_before, orient_after=model.orient)
        record = stacked_record(mask_record, record, bricks)  # the mask and the bricks turn with the voxels
        if sel_before is None:
            return record
        return stacked_record(record, selection_record(model.uid, sel_before, model.selection.mask))


def _after_load(context) -> None:
    """New or Open: the old scene's gizmo and drag must not survive."""
    global _drag, _override_key
    _drag = None
    _override_key = None
    state.gizmo_ghost = None
    state.model_drag_offset = (0, 0, 0)
    refresh()


registry.load_post.append(_after_load)
