"""The scene's tools for whole models: pick one or a group, drag to move,
Shift-drag to leave copies behind, Option-drag to move up and down, arrow
keys to nudge, G to drop onto what is below, Cmd+J to merge, the gizmo's
cubes to stretch (tools.py starts rok.scene_stretch), and the Cut tool to
slice a model in two.

The group: the picked model (document.model, which carries the gizmo) and
the models Shift-click added to it (state.group). A drag, a nudge, a drop
and the gizmo's arrows move the whole group; turning, stretching, flipping
and cutting act on the picked model alone.

A press on a model in the scene picks it (tools.py); once the mouse moves
DRAG_START_PX, rok.scene_drag takes over. The models slide on the floor at
the height where the press grabbed; with Option they move straight up and
down on an upright plane facing the view. Shift, as everywhere in Rok,
copies. The Empties move live; the scene changes once, on release, as one
undo step. Esc or a right click puts them back.
"""

from __future__ import annotations

import copy
import importlib

import numpy as np

from rok.core import brick as brick_core
from rok.core import cut as cut_core
from rok.core import link as link_core
from rok.core import hud, scene_move, shape
from rok.core import stretch as stretch_core
from rok.core.anim import MASK, PART
from rok.core.grid import VoxelGrid
from rok.core.select import Selection
from rok.core.undo import BrickRecord, GridRecord, ModelRecord, selection_record, stacked_record
from . import object as voxel_object
from . import select_ops
from .gfx import pen
from .object import document
from .opkit import BoolProp, EnumProp, IntProp, Action, StringProp, ops, register
from .ops_common import commit, redraw
from .state import state


def _m(name: str):
    """A sibling module looked up when used: tools, mode, overlay, gizmo_ops and
    edge_wrap import this module, or are imported alongside it."""
    return importlib.import_module(f"{__package__}.{name}")


def _in_scene() -> bool:
    """The scene tools may run: in the scene, not in Animate mode (it poses parts)."""
    return (
        state.level == "SCENE"
        and document.scene is not None
        and document.model is not None
        and not state.stroke_active
        and not state.animating
    )


# --- the group ----------------------------------------------------------------------


def group_models() -> list:
    """The picked model and the models Shift-click added, in scene order;
    models that are gone or on a hidden layer drop out."""
    scene = document.scene
    if scene is None or document.model is None:
        return []
    # state.group holds every member, the leader too, so an undo that picks
    # another member keeps the group. A leader outside it means the group is
    # stale (a plain click picked another model): the leader stands alone.
    uids = set(state.group) if document.active_uid in state.group else {document.active_uid}
    return [m for m in scene.models if m.uid in uids and (m.uid == document.active_uid or not scene.is_hidden(m))]


def _bricks_picked() -> bool:
    """A picked model is built from bricks: it only moves, and turns by quarters."""
    members = group_models()
    return any(m.grid.bricks is not None for m in members)


def _late_brick(name: str):
    from . import brick_mode

    return getattr(brick_mode, name)


def group_uids() -> set[int]:
    return {m.uid for m in group_models()}


def clear_group() -> None:
    state.group.clear()


def toggle_in_group(uid: int) -> None:
    """Shift-click: a model outside the group joins it and leads it (picked);
    one inside leaves it, and when it led, the next one leads."""
    from .tools import StrokeAction

    members = group_uids()
    if uid not in members:
        state.group.clear()
        state.group |= members | {uid}
        StrokeAction._activate(uid)
        return
    if len(members) == 1:
        return  # the last one stays picked
    if uid == document.active_uid:
        lead = next(m.uid for m in group_models() if m.uid != uid)
        state.group.discard(uid)
        StrokeAction._activate(lead)
    else:
        state.group.discard(uid)
    voxel_object.redraw_3d_views()


def _commit_moves(moves) -> None:
    """Move each (model, before, after) as one undo step."""

    records = []
    for model, before, after in moves:
        if tuple(before) == tuple(after):
            continue
        model.position = after
        records.append(ModelRecord(model.uid, None, position_before=before, position_after=after))
    record = stacked_record(*records)
    if record is None:
        return
    commit(record, serial=True, sync=True, hover=True)


def _commit_move(model, before, after, copy: bool) -> None:
    """Move `model` from `before` to `after` as one undo step; with `copy`, a
    copy stays at `before`."""
    if not copy:
        _commit_moves([(model, before, after)])
        return
    token = begin_copy([model])
    finish_copy(token, [(model, before, after)])


def begin_copy(models):
    """Copies of `models` appear at their places as a copy-drag starts, so the
    originals never seem to vanish while they move away. Not recorded yet:
    finish_copy records them with the moves, cancel_copy takes them away."""
    scene = document.scene
    before = tuple(scene.models)
    active = document.active_uid
    copies = []
    for model in models:
        grid = brick_core.carry(model.grid, VoxelGrid(model.grid.size, model.grid.voxels, scene.palette))
        name = scene.unique_name(model.display_name(scene.models.index(model)))
        copy = scene.add(grid, name, tuple(model.position), model.layer)
        # A linked copy: it shares the shape (core/link.py). The original's
        # new link id is not recorded; alone in its group it means nothing.
        if not model.link:
            model.link = scene.new_link()
        copy.link, copy.orient = model.link, model.orient
        copies.append(copy.uid)
    document.set_active(active)
    voxel_object.sync_scene()
    return (before, active, copies)


def finish_copy(token, moves) -> None:
    """Each (model, before, after) moved and its copy at `before`, as one undo
    step. The timeline is untouched here, but both snapshots go on the record
    like every other ListRecord push (section 6.1)."""
    from . import scene_ops

    models_before, active_before, _uids = token
    timeline_before = copy.deepcopy(document.scene.timeline)
    records = []
    for model, before, after in moves:
        model.position = after
        records.append(ModelRecord(model.uid, None, position_before=before, position_after=after))
    timeline_after = copy.deepcopy(document.scene.timeline)
    commit(
        scene_ops.ListRecord(
            models_before, tuple(document.scene.models),
            active_before, document.active_uid, stacked_record(*records),
            timeline_before, timeline_after,
        ),
        serial=True, sync=True, hover=True,
    )


def cancel_copy(token) -> None:
    """Take a copy-drag's copies away again (Esc, or a drag that moved nothing)."""
    scene = document.scene
    for uid in token[2]:
        try:
            scene.remove(scene.index_of(uid))
        except ValueError:
            pass
    voxel_object.sync_scene()
    voxel_object.redraw_3d_views()


def show_group_offset(offset) -> None:
    """While a drag moves the picked model's Empty, move the rest of the
    group's Empties with it."""
    for model in group_models():
        if model.uid != document.active_uid:
            _m("gizmo_ops")._place_empty(model.uid, tuple(p + o for p, o in zip(model.position, offset)))


def reset_group_empties() -> None:
    for model in group_models():
        _m("gizmo_ops")._place_empty(model.uid, tuple(model.position))


@register
class SceneDragAction(Action):
    """Drag the picked model, or its whole group: on the floor, or up and down
    with Option. Started from a press on a model in the scene, not bound to a
    key.

    `toggle`: the press was a Shift-press; a release without moving adds the
    pressed model to the group or takes it out, and a drag copies. `cycle`: a
    release without moving picks the next model along the line of sight.
    `single`: a plain press on one model of a group; a release without
    moving picks it alone."""

    command_id = "rok.scene_drag"
    title = "Move Model"
    tooltip = "Drag the picked models; Shift leaves copies behind; Option moves them up and down"
    copy = BoolProp(name="Copy", default=False, options={"SKIP_SAVE"})
    cycle = BoolProp(name="Cycle", default=False, options={"SKIP_SAVE"})
    toggle = BoolProp(name="Toggle", default=False, options={"SKIP_SAVE"})
    single = BoolProp(name="Single", default=False, options={"SKIP_SAVE"})
    # Ctrl-drag: let go over another model and the dragged one carves its
    # shape out of it (scene_arrange.carve_drop); `keep` leaves the cutter.
    carve = BoolProp(name="Carve", default=False, options={"SKIP_SAVE"})
    keep = BoolProp(name="Keep cutter", default=False, options={"SKIP_SAVE"})
    # A press on a gizmo arrow: the model moves along that axis only (-1: free).
    axis = IntProp(name="Axis", default=-1, options={"SKIP_SAVE"})
    pressed = IntProp(name="Pressed model", default=-1, options={"SKIP_SAVE"})
    grab_x = IntProp(default=0, options={"SKIP_SAVE"})
    grab_y = IntProp(default=0, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _in_scene() and context.view_state is not None

    def invoke(self, context, event):
        from . import mode

        region, view_state = context.region, context.view_state
        try:
            self._grabbed = document.scene.by_uid(self.pressed) if self.pressed >= 0 else document.model
        except ValueError:
            return {"CANCELLED"}
        self._press = (event.mouse_region_x, event.mouse_region_y)
        origin, direction = mode.world_ray(region, view_state, self.grab_x, self.grab_y)
        if self.axis >= 0:
            # The arrow's line runs through the model's centre.
            self._grab = tuple(float(p) + s / 2.0 for p, s in zip(self._grabbed.position, self._grabbed.grid.size))
            self._s0 = self._along(origin, direction)
            if self._s0 is None:
                return {"CANCELLED"}
        else:
            hit_t = self._hit_t(region, view_state)  # the grabbed model's own voxel, even behind another
            if hit_t is None:
                return {"CANCELLED"}
            self._grab = tuple(o + d * hit_t for o, d in zip(origin, direction))
        self._view_dir = mode.view_dir(view_state)
        self._moving = False
        self._offset = (0, 0, 0)
        self._models = []
        self._copy = None
        self._wrap = None  # set when the press becomes a drag: it goes on past the view's edge
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def _along(self, origin, direction):
        """How far along the drag's axis the ray passes closest, or None."""
        unit = [0.0, 0.0, 0.0]
        unit[self.axis] = 1.0
        return shape.axis_drag_param(self._grab, unit, origin, direction)

    def _hit_t(self, region, view_state):
        from . import mode
        from rok.core.raycast import raycast

        from rok.core.raycast import box_enter

        origin, direction = mode.mouse_ray(region, view_state, self.grab_x, self.grab_y, self._grabbed)
        hit = raycast(self._grabbed.grid.tool_voxels, origin, direction, walls=False)
        if hit is not None and not hit.wall:
            return hit.t
        # No voxel of it there (an empty model, or a gap): its box holds the grab.
        return box_enter(origin, direction, self._grabbed.grid.size)

    def _start_moving(self, context) -> None:
        """The press became a drag: settle what moves, show the copies."""
        from .tools import StrokeAction

        if self._grabbed.uid not in group_uids() or self.carve:
            # A Shift-drag on a model outside the group copies that model alone.
            clear_group()
            StrokeAction._activate(self._grabbed.uid)
        self._models = group_models()
        self._before = {m.uid: tuple(int(p) for p in m.position) for m in self._models}
        lo = [min(self._before[m.uid][i] for m in self._models) for i in range(3)]
        hi = [max(self._before[m.uid][i] + m.grid.size[i] for m in self._models) for i in range(3)]
        self._box = (tuple(lo), tuple(hi))
        self._moving = True
        state.stroke_active = True
        self._copy = begin_copy(self._models) if self.copy else None
        # What a drag may snap to: every shown model outside the group, the
        # copies just made included (a row of copies lines up by itself).
        scene = document.scene
        moving = {m.uid for m in self._models}
        self._others = [(m.uid, (tuple(int(p) for p in m.position),
                                 tuple(int(p) + s for p, s in zip(m.position, m.grid.size))))
                        for m in scene.models if m.uid not in moving and not scene.is_hidden(m)]
        _m("gizmo_ops").begin_model_drag()
        context.window.cursor_modal_set("SCROLL_XY")

    def modal(self, context, event):
        from . import mode

        region, view_state = context.region, context.view_state
        if event.type in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN", "LEFT_ALT", "RIGHT_ALT", "CMD"}:
            mx, my = event.mouse_region_x, event.mouse_region_y
            if not self._moving:
                dx, dy = mx - self._press[0], my - self._press[1]
                if dx * dx + dy * dy < scene_move.DRAG_START_PX ** 2:
                    return {"RUNNING"}
                self._start_moving(context)
                self._wrap = _m("edge_wrap").MouseWrap()
            mx, my = self._wrap.pos(context, event)
            origin, direction = mode.world_ray(region, view_state, mx, my)
            if self.axis >= 0:
                s = self._along(origin, direction)
                if s is not None:
                    offset = [0, 0, 0]
                    offset[self.axis] = int(round(s - self._s0))
                    self._offset = tuple(offset)
                    self._show()
                return {"RUNNING"}
            if event.alt:
                point = scene_move.ray_plane_facing(origin, direction, self._grab, self._view_dir)
            else:
                point = scene_move.ray_plane_z(origin, direction, self._grab[2])
            if point is not None:
                offset = scene_move.drag_offset(self._grab, point, vertical=bool(event.alt))
                state.snap_uids = set()
                if not event.cmd_key and not self.carve:  # Cmd: no snapping; a carve goes into the other model
                    axes = (2,) if event.alt else (0, 1)
                    offset, hit = scene_move.snap_offset(self._box, offset, [b for _u, b in self._others], axes)
                    state.snap_uids = {self._others[k][0] for k in hit}
                self._offset = offset
                self._show()
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            if not self._moving and not self.carve:
                self._click(context)
            return self._finish(context, cancel=not self._moving or self._offset == (0, 0, 0))
        if event.type in {"ESCAPE", "RIGHT_MOUSE"} and event.value == "PRESS":
            return self._finish(context, cancel=True)
        if event.type == "FOCUS_LOST":
            return self._finish(context, cancel=not self._moving)
        return {"RUNNING"}

    def _click(self, context) -> None:
        """A release without moving: Shift adds or takes out, a click on one of a
        group picks it alone, a click on the picked model cycles."""
        from . import mode
        from .tools import StrokeAction

        if self.toggle:
            toggle_in_group(self._grabbed.uid)
            return
        if self.single:
            clear_group()
            StrokeAction._activate(self._grabbed.uid)
            return
        if self.cycle:
            under = mode.models_under(context.region, context.view_state, self.grab_x, self.grab_y)
            nxt = scene_move.next_in_cycle(under, document.active_uid)
            if nxt is not None:
                clear_group()
                StrokeAction._activate(nxt)
                # The double-click window stays on the model picked at the
                # press: a second quick click steps into it, not into `nxt`.

    def _show(self) -> None:
        from rok.core import gizmo

        _m("gizmo_ops").update_model_drag(gizmo.GizmoResult(kind=gizmo.MOVE, offset=self._offset))
        show_group_offset(self._offset)

    def _finish(self, context, cancel: bool):
        state.snap_uids = set()
        if self._moving:
            context.window.cursor_modal_restore()
            _m("gizmo_ops").end_model_drag(cancel=True)  # the Empties back; the commit moves them for real
            reset_group_empties()
            state.stroke_active = False
        moves = [(m, self._before[m.uid], tuple(b + o for b, o in zip(self._before[m.uid], self._offset)))
                 for m in self._models]
        if self.carve and not cancel:
            from . import scene_arrange

            self.report({"INFO"}, scene_arrange.carve_drop(self._grabbed, moves[0][2], keep=bool(self.keep)))
        elif not cancel:
            if self._copy is not None:
                finish_copy(self._copy, moves)
            else:
                _commit_moves(moves)
        elif self._copy is not None:
            cancel_copy(self._copy)
        redraw()
        return {"CANCELLED"} if cancel else {"FINISHED"}


_DIRECTIONS = (
    ("RIGHT", "Right", ""), ("LEFT", "Left", ""), ("UP", "Away", ""), ("DOWN", "Toward", ""),
    ("RAISE", "Up", ""), ("LOWER", "Down", ""),
)


@register
class SceneNudgeAction(Action):
    """The arrow keys in the scene: move the picked models one cell (Shift:
    NUDGE_BIG) along the floor axis closest to that screen direction; Cmd+Up
    and Cmd+Down move them up and down."""

    command_id = "rok.scene_nudge"
    title = "Nudge Model"
    tooltip = "Move the picked models one cell; Shift: ten"
    direction = EnumProp(name="Direction", items=_DIRECTIONS, default="RIGHT")
    big = BoolProp(name="Ten", default=False)

    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):
        from . import mode

        step = scene_move.NUDGE_BIG if self.big else 1
        if self.direction in ("RAISE", "LOWER"):
            axis = (0, 0, 1 if self.direction == "RAISE" else -1)
        else:
            view_state = context.view_state or _first_view_state(context)
            if view_state is None:
                return {"CANCELLED"}
            right = view_state.axes_matrix()[:, 0]
            forward = mode.view_dir(view_state)
            ax_right, ax_up = scene_move.nudge_axes(tuple(right), forward)
            axis = {
                "RIGHT": ax_right, "LEFT": tuple(-v for v in ax_right),
                "UP": ax_up, "DOWN": tuple(-v for v in ax_up),
            }[self.direction]
        moves = []
        for model in group_models():
            before = tuple(int(p) for p in model.position)
            moves.append((model, before, tuple(b + a * step for b, a in zip(before, axis))))
        _commit_moves(moves)
        return {"FINISHED"}


def _first_view_state(context):
    return context.view_state  # Rok has one view


@register
class SceneDropAction(Action):
    """G in the scene, and Tools > Drop to Floor: the picked models move
    straight down, together, until the lowest of them sits on the floor grid.
    Models below are not in the way; a model sunk into the
    floor comes up onto it."""

    command_id = "rok.scene_drop"
    title = "Drop to Floor"
    tooltip = "Put the picked models on the floor grid"
    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):
        members = group_models()
        # The floor grid is z = 0, and the scene's cells are the world's.
        dz = scene_move.floor_offset([(m.grid.voxels != 0, tuple(m.position)) for m in members])
        if dz == 0:
            self.report({"INFO"}, "Already on the floor")
            return {"CANCELLED"}
        moves = []
        for model in members:
            before = tuple(int(p) for p in model.position)
            moves.append((model, before, (before[0], before[1], before[2] + dz)))
        _commit_moves(moves)
        return {"FINISHED"}


@register
class SceneDropOntoAction(Action):
    """Shift G in the scene, and Tools > Drop: the picked models move straight
    down, together, until one of them rests on another model's voxel, or on
    the floor grid when nothing is below."""

    command_id = "rok.scene_drop_onto"
    title = "Drop"
    tooltip = "Move the picked models down until they rest on a voxel"
    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):
        members = group_models()
        uids = {m.uid for m in members}
        scene = document.scene
        others = [(m.grid.voxels != 0, tuple(m.position)) for m in scene.models
                  if m.uid not in uids and not scene.is_hidden(m)]
        dz = scene_move.drop_onto_offset([(m.grid.voxels != 0, tuple(m.position)) for m in members], others)
        if dz == 0:
            self.report({"INFO"}, "Already resting")
            return {"CANCELLED"}
        moves = []
        for model in members:
            before = tuple(int(p) for p in model.position)
            moves.append((model, before, (before[0], before[1], before[2] + dz)))
        _commit_moves(moves)
        return {"FINISHED"}


@register
class SceneFlipAction(Action):
    """Flip in the scene's Tools: the picked models mirror across one axis,
    as one piece about their common centre."""

    command_id = "rok.scene_flip"
    title = "Flip Models"
    tooltip = "Mirror the picked models across X, Y or Z"
    axis = IntProp(name="Axis", min=0, max=2, default=0)

    @classmethod
    def poll(cls, context):
        return _in_scene() and ops.rok.transform.poll() and not _bricks_picked()

    def execute(self, context):
        transform_group("FLIP", int(self.axis))
        return {"FINISHED"}


def group_box(members=None) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The box around the picked models, in the scene's cells: (low, high)."""
    members = group_models() if members is None else members
    lo = tuple(min(int(m.position[i]) for m in members) for i in range(3))
    hi = tuple(max(int(m.position[i]) + int(m.grid.size[i]) for m in members) for i in range(3))
    return lo, hi


def transform_group(kind: str, axis: int, turns: int = 1) -> set:
    """Turn or flip the picked models as one piece about their common centre,
    as one undo step. One picked model goes through rok.transform as before."""
    from rok.core import gizmo, grid_ops

    members = group_models()
    if not _late_brick("turn_allowed")(members or [document.model], kind, axis):
        return {"CANCELLED"}
    if len(members) < 2:
        return ops.rok.transform(target="MODEL", kind=kind, axis=axis, turns=turns)
    lo, hi = group_box(members)
    pivot2 = tuple(lo[i] + hi[i] for i in range(3))  # doubled: the group's centre
    if kind == gizmo.ROTATE and turns % 2:
        pivot2 = gizmo.snap_pivot(pivot2, axis)
    records = []
    for model in members:
        before = tuple(int(p) for p in model.position)
        def change(v, before=before):
            if kind == gizmo.ROTATE:
                return grid_ops.rotate_model(v, before, axis, turns, pivot2)
            return grid_ops.flip_model(v, before, axis, pivot2)

        voxels, after = change(model.grid.voxels)
        after = tuple(int(p) for p in after)
        records.append(select_ops.carried_mask(model, lambda v, change=change: change(v)[0]))
        orient = link_core.turned(model.orient, link_core.transform_matrix(kind, axis, turns))
        records.append(ModelRecord(model.uid, GridRecord(before=model.grid.voxels.copy(), after=voxels),
                                   position_before=before, position_after=after,
                                   orient_before=model.orient, orient_after=orient))
        sel_before = None if model.selection.is_empty() else model.selection.mask.copy()
        old_size = model.grid.size
        with brick_core.unlocked(model.grid):
            model.grid.replace_voxels(voxels)
        records.append(_late_brick("turn_table")(model, kind, axis, turns, old_size))
        model.selection = Selection(model.grid.size)
        model.position = after
        model.orient = orient
        if sel_before is not None:  # the dropped selection goes on the record for undo
            records.append(selection_record(model.uid, sel_before, model.selection.mask))
    document.pivot_override = None
    commit(stacked_record(*records), serial=True, sync=True, hover=True)
    return {"FINISHED"}


@register
class SceneFitAction(Action):
    """Fit in the scene: every picked model's grid shrinks to its voxels, which
    keep their place. One undo step."""

    command_id = "rok.scene_fit"
    title = "Fit Picked Models"
    tooltip = "Shrink the picked models' grids to their voxels"
    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):

        records = []
        for model in group_models():
            voxels, offset = cut_core.fitted(model.grid.voxels)
            if voxels.shape == model.grid.voxels.shape:
                continue
            before = tuple(int(p) for p in model.position)
            after = tuple(p + o for p, o in zip(before, offset))
            box = tuple(slice(int(o), int(o) + n) for o, n in zip(offset, voxels.shape))
            records.append(select_ops.carried_mask(model, lambda v, box=box: v[box]))
            records.append(ModelRecord(model.uid, GridRecord(before=model.grid.voxels.copy(), after=voxels.copy()),
                                       position_before=before, position_after=after))
            with brick_core.unlocked(model.grid):
                model.grid.replace_voxels(voxels)
            table = model.grid.bricks
            if table is not None:  # the bricks keep their place: the table shifts with the crop
                model.grid.bricks = table.shifted(offset)
                records.append(BrickRecord(model.uid, table, model.grid.bricks.copy()))
            model.selection = Selection(voxels.shape)
            model.position = after
        record = stacked_record(*records)
        if record is None:
            self.report({"INFO"}, "Already fitted")
            return {"CANCELLED"}
        commit(record, serial=True, sync=True, hover=True)
        return {"FINISHED"}


@register
class SceneDeleteAction(Action):
    """Delete in the scene: the picked models leave the scene. One undo step.
    A scene keeps at least one model."""

    command_id = "rok.scene_delete"
    title = "Delete Picked Models"
    tooltip = "Remove the picked models from the scene"
    @classmethod
    def poll(cls, context):
        # A key picked in the timeline strip takes Delete (rok.key_delete).
        return _in_scene() and not getattr(state, "selected_keys", None)

    def execute(self, context):
        from . import scene_ops

        scene = document.scene
        members = group_models()
        if len(members) >= len(scene.models):
            self.report({"WARNING"}, "A scene needs one model")
            return {"CANCELLED"}
        gone = {m.uid for m in members}
        # The nearest model above the picked one in the list is picked next.
        above = [m for m in scene.models[:scene.index_of(document.active_uid)] if m.uid not in gone]

        def delete():
            for model in members:
                scene.models.remove(model)
            scene.timeline.tracks = [
                t for t in scene.timeline.tracks
                if not (t.target.kind in (PART, MASK) and t.target.uid in gone)
            ]
            document.set_active((above[-1] if above else scene.models[0]).uid)

        scene_ops._scene_change(delete)
        clear_group()
        self.report({"INFO"}, "Deleted 1 model" if len(members) == 1 else f"Deleted {len(members)} models")
        return {"FINISHED"}


@register
class SceneMergeAction(Action):
    """Cmd+J in the scene: the picked models become one, the picked (leading)
    one, which grows to hold them; later models in the list on top."""

    command_id = "rok.scene_merge"
    title = "Merge Picked Models"
    tooltip = "Join the picked models into one"
    @classmethod
    def poll(cls, context):
        # The picked model takes the others' voxels: not a brick model.
        return _in_scene() and len(group_models()) > 1 and document.model.grid.bricks is None

    def execute(self, context):
        from . import scene_ops

        scene = document.scene
        members = group_models()
        lead = document.model
        voxels, position = scene_ops._merged_or_report(self, members)
        if voxels is None:
            return {"CANCELLED"}
        voxels, lock = scene_ops.merged_masks(members, voxels, position)

        def merge():
            unlinked = scene_ops.unlink_record(lead)  # the merge is this model's alone
            grid_before = lead.grid.voxels.copy()
            position_before = lead.position
            for other in members:
                if other.uid != lead.uid:
                    scene.models.remove(other)
            masked = select_ops.set_mask(lock, lead.uid)  # before the grid changes size
            lead.grid.replace_voxels(voxels)
            lead.position = position
            lead.selection = Selection(voxels.shape)
            return stacked_record(unlinked, masked, ModelRecord(lead.uid, GridRecord(grid_before, voxels.copy()),
                                                                position_before=position_before,
                                                                position_after=position))

        scene_ops._scene_change(merge)
        clear_group()
        self.report({"INFO"}, f"Merged {len(members)} models")
        return {"FINISHED"}


# --- stretching -------------------------------------------------------------------


stretch_label: tuple[str, tuple[float, float]] | None = None  # (text, region pixel) while a stretch runs


def _to_region(region, view_state, cell_point):
    """A point in the scene's cells on the screen, or None behind the view."""
    from .view3d import world_to_region

    ox, oy, oz = document.view_offset
    p = world_to_region(region, view_state, (cell_point[0] + ox, cell_point[1] + oy, cell_point[2] + oz))
    return None if p is None else (float(p[0]), float(p[1]))


def draw_scene_extras(region, scale: float) -> None:
    """The stretch's size label at the mouse, and the Cut tool's plane."""
    from . import overlay
    from .opkit import CONTEXT

    view_state = CONTEXT.view_state
    if view_state is None:
        return
    if stretch_label is not None:
        text, (x, y) = stretch_label
        overlay.draw_label(text, x + 16 * scale, y + 16 * scale, scale)
    _draw_cut_plane(region, view_state, scale)


def _show_model(uid: int, shown: bool) -> None:
    """Hide or show the chunk objects of model `uid` (the ghost stands in for
    it while a stretch runs)."""
    if shown:
        voxel_object.hidden_preview.discard(uid)
    else:
        voxel_object.hidden_preview.add(uid)
    voxel_object.flush_dirty()


@register
class SceneStretchAction(Action):
    """Drag a gizmo cube of the picked model: the voxels stretch or squeeze
    along that axis, the opposite face stays put. Option grows or crops the
    box instead; Shift scales all three axes together.

    The preview is the model's surface, built once when the drag starts and
    scaled on every move (crop shows it unmoved: cropped voxels keep their
    place), so a big model stretches as smoothly as a small one. The voxels
    are resampled once, on release."""

    command_id = "rok.scene_stretch"
    title = "Stretch Model"
    tooltip = "Stretch or squeeze the picked model; Option: grow or crop its box; Shift: all axes"
    handle = StringProp(name="Handle", default="SH_X+", options={"SKIP_SAVE"})
    uniform = BoolProp(name="All axes", default=False, options={"SKIP_SAVE"})  # the centre sphere

    @classmethod
    def poll(cls, context):
        return _in_scene() and context.view_state is not None and not _bricks_picked()

    def invoke(self, context, event):
        self._handle = stretch_core.handle_from_id(self.handle)
        if self._handle is None:
            return {"CANCELLED"}
        region, view_state = context.region, context.view_state
        model = self._model = document.model
        # Several picked models stretch as one box; each keeps its place in it.
        self._members = group_models() if len(group_models()) > 1 else [model]
        self._group = len(self._members) > 1
        if self._group:
            lo, hi = group_box(self._members)
            self._position, self._size = lo, tuple(h - l for h, l in zip(hi, lo))
        else:
            self._size = tuple(int(v) for v in model.grid.size)
            self._position = tuple(int(p) for p in model.position)
        face = stretch_core.face_centre(self._position, self._size, self._handle)
        out = list(face)
        out[self._handle.axis] += self._handle.side
        a, b = _to_region(region, view_state, face), _to_region(region, view_state, out)
        if a is None or b is None:
            return {"CANCELLED"}
        self._axis_px = (b[0] - a[0], b[1] - a[1])
        self._start = (event.mouse_region_x, event.mouse_region_y)
        self._wrap = _m("edge_wrap").MouseWrap()  # a stretch goes on past the view's edge
        self._key = None
        self._target = self._size
        self._new_position = self._position
        tris, lines = [], []
        for member in self._members:
            voxels = member.grid.voxels
            t, li = hud.mask_surface(voxels != 0, (0, 0, 0), np.asarray(voxels.shape) - 1)
            shift = np.asarray([int(p) - b for p, b in zip(member.position, self._position)], dtype=np.float32)
            tris.append(t + shift)
            lines.append(li + shift)
        self._surface = (np.concatenate(tris), np.concatenate(lines))
        state.stroke_active = True
        state.hover_handle = self._handle.id
        for member in self._members:
            _show_model(member.uid, False)
        self._update(context, event)
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def _update(self, context, event) -> None:
        global stretch_label
        now = self._wrap.pos(context, event)
        cells = stretch_core.cells_along(self._start, now, self._axis_px)
        uniform = bool(event.shift) or bool(self.uniform)
        crop = bool(event.alt) and not self._group  # a group always stretches
        limit = 4 * stretch_core.MAX_SIZE if self._group else stretch_core.MAX_SIZE
        target = stretch_core.new_size(self._size, self._handle, cells, uniform, limit)
        key = (target, uniform, crop)
        cursor = (event.mouse_region_x, event.mouse_region_y)  # the label stays by the real cursor
        stretch_label = (f"{target[0]} x {target[1]} x {target[2]}" + ("  crop" if crop else ""), cursor)
        if key != self._key:
            self._key = key
            self._target = target
            self._new_position = stretch_core.new_position(self._position, self._size, target, self._handle, uniform)
            tris, lines = self._surface
            ox, oy, oz = document.view_offset
            if crop:
                # Cropped or grown, the voxels keep their place and size.
                base = self._position
            else:
                factor = np.asarray(target, dtype=np.float32) / np.asarray(self._size, dtype=np.float32)
                tris, lines = tris * factor, lines * factor
                base = self._new_position
            origin = (ox + base[0], oy + base[1], oz + base[2])
            box_origin = (ox + self._new_position[0], oy + self._new_position[1], oz + self._new_position[2])
            state.gizmo_ghost = (("STRETCH", key, box_origin), (origin, tris, lines))
        voxel_object.redraw_3d_views()

    def modal(self, context, event):
        if event.type in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN", "LEFT_SHIFT", "RIGHT_SHIFT",
                          "LEFT_ALT", "RIGHT_ALT"}:
            self._update(context, event)
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            return self._finish(context, cancel=False)
        if event.type in {"ESCAPE", "RIGHT_MOUSE"} and event.value == "PRESS":
            return self._finish(context, cancel=True)
        if event.type == "FOCUS_LOST":
            return self._finish(context, cancel=True)
        return {"RUNNING"}

    def _finish(self, context, cancel: bool):
        global stretch_label
        from . import model_ops

        stretch_label = None
        state.gizmo_ghost = None
        state.stroke_active = False
        for member in self._members:
            _show_model(member.uid, True)
        changed = self._key is not None and tuple(self._target) != tuple(self._size)
        if not cancel and changed and self._group:
            self._finish_group()
        elif not cancel and changed:
            uniform, crop = self._key[1], self._key[2]
            voxels = self._model.grid.voxels
            target, handle = tuple(self._target), self._handle

            def change(v):
                return (stretch_core.crop_or_grow(v, target, handle, uniform) if crop
                        else stretch_core.stretch(v, target))

            if crop and model_ops._loses_masked(change):
                self.report({"ERROR"}, model_ops.MASK_LOSS)
                cancel = True
            else:
                model_ops._apply_grid_op(context, change(voxels), self._new_position, mask_op=change)
        voxel_object.apply_layer_visibility()  # a model on a hidden layer stays hidden
        redraw()
        return {"CANCELLED"} if cancel or not changed else {"FINISHED"}

    def _finish_group(self) -> None:
        """Every picked model stretched to its share of the new box, one undo step."""
        stretch_group(self._members, self._position, self._size, self._target, self._new_position)


def stretch_group(members, box_lo, box_size, target, new_lo) -> None:
    """Stretch the group's box (`box_lo`, `box_size`) to `target` cells from
    `new_lo`: every model to its share of it, one undo step."""

    parts = [(tuple(int(p) for p in m.position), tuple(int(v) for v in m.grid.size)) for m in members]
    placed = stretch_core.group_parts(parts, box_lo, box_size, target, new_lo)
    records = []
    for member, (before, _size), (after, size) in zip(members, parts, placed):
        voxels = stretch_core.stretch(member.grid.voxels, size)
        records.append(select_ops.carried_mask(member, lambda v, size=size: stretch_core.stretch(v, size)))
        records.append(ModelRecord(member.uid, GridRecord(before=member.grid.voxels.copy(), after=voxels),
                                   position_before=before, position_after=after))
        sel_before = None if member.selection.is_empty() else member.selection.mask.copy()
        member.grid.replace_voxels(voxels)
        member.selection = Selection(member.grid.size)
        member.position = after
        if sel_before is not None:
            records.append(selection_record(member.uid, sel_before, member.selection.mask))
    document.pivot_override = None
    commit(stacked_record(*records), serial=True, sync=True, hover=True)


# --- cutting ----------------------------------------------------------------------


def aim_cut(region, view_state, rx: float, ry: float) -> tuple[int, int, int] | None:
    """The cutting plane under the mouse: (uid, axis, boundary), or None off
    every model or at a model's edge."""
    from . import mode
    from rok.core.raycast import raycast

    uid = mode.model_at(region, view_state, rx, ry)
    if uid is None:
        return None
    model = document.scene.by_uid(uid)
    origin, direction = mode.mouse_ray(region, view_state, rx, ry, model)
    hit = raycast(model.grid.tool_voxels, origin, direction, walls=False)
    if hit is None or hit.wall:
        return None
    point = tuple(o + d * hit.t for o, d in zip(origin, direction))
    here = [p + q for p, q in zip(model.position, point)]
    screen = {}
    base = _to_region(region, view_state, here)
    for axis in range(3):
        step = list(here)
        step[axis] += 1.0
        tip = _to_region(region, view_state, step)
        screen[axis] = (0.0, 0.0) if base is None or tip is None else (tip[0] - base[0], tip[1] - base[1])
    axis = cut_core.pick_axis(hit.normal, screen, other=state.cut_other)
    k = cut_core.boundary(point, axis, model.grid.size)
    return None if k is None else (uid, axis, k)


CUT_FILL = (0.91, 0.365, 0.227, 0.28)
CUT_EDGE = (1.0, 0.55, 0.35, 1.0)


def _draw_cut_plane(region, view_state, scale: float) -> None:
    """The plane the Cut tool would cut along, over the model under the mouse."""
    aim = state.cut_aim
    if state.scene_tool != "CUT" or aim is None or document.scene is None:
        return
    try:
        model = document.scene.by_uid(aim[0])
    except ValueError:
        return
    position, size, k = model.position, model.grid.size, aim[2]
    members = group_models()
    if len(members) > 1 and model.uid in group_uids():
        # The whole group is cut: the plane spans the group's box.
        lo, hi = group_box(members)
        k += int(position[aim[1]]) - lo[aim[1]]
        position, size = lo, tuple(h - l for h, l in zip(hi, lo))
    corners = [_to_region(region, view_state, c) for c in cut_core.plane_corners(position, size, aim[1], k)]
    if any(c is None for c in corners):
        return
    quad = np.asarray(corners, dtype=np.float32)
    pen.fan(pen.to_3d(quad), CUT_FILL)
    edges = np.asarray([corners[i // 2 % 4] if i % 2 == 0 else corners[(i // 2 + 1) % 4] for i in range(8)], np.float32)
    pen.lines(pen.to_3d(edges), CUT_EDGE, 2.0 * scale)


@register
class SceneCutAction(Action):
    """Cut a model in two along a plane: the part from the plane on becomes a
    model of its own, in place. One undo step."""

    command_id = "rok.scene_cut"
    title = "Cut Model"
    tooltip = "Cut the model in two along the plane"
    uid = IntProp(name="Model", default=0)
    axis = IntProp(name="Axis", default=0, min=0, max=2)
    at = IntProp(name="Boundary", default=1, min=1)

    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):
        from . import scene_ops

        scene = document.scene
        try:
            source = scene.by_uid(self.uid)
        except ValueError:
            return {"CANCELLED"}
        if source.grid.bricks is not None:
            self.report({"WARNING"}, _late_brick("SCENE_LOCKED"))
            return {"CANCELLED"}
        if not 0 < self.at < source.grid.size[self.axis]:
            return {"CANCELLED"}
        # A model of a picked group cuts the whole group along the same plane.
        members = group_models() if source.uid in group_uids() else []
        members = members if len(members) > 1 else [source]
        world = int(source.position[self.axis]) + self.at
        cuts = []
        for model in members:
            k = world - int(model.position[self.axis])
            if not 0 < k < model.grid.size[self.axis]:
                continue
            low, high = cut_core.split_at(model.grid.voxels, self.axis, k)
            if low.any() and high.any():
                cuts.append((model, k, low, high))
        if not cuts:
            self.report({"INFO"}, "Nothing on one side of the cut")
            return {"CANCELLED"}

        def cut_one(model, k, low, high):
            unlinked = scene_ops.unlink_record(model)  # the cut is this copy's alone
            name = scene.part_name(model.display_name(scene.index_of(model.uid)))
            before = model.grid.voxels.copy()
            position_before = tuple(int(p) for p in model.position)
            # Each part's box hugs its voxels; the voxels keep their place.
            low, low_at = cut_core.fitted(low)
            high, high_at = cut_core.fitted(high)
            low_position = tuple(p + o for p, o in zip(position_before, low_at))
            high_position = list(p + o for p, o in zip(position_before, high_at))
            high_position[self.axis] += k
            # The mask splits as the voxels do.
            mask = select_ops.mask_of(model.uid)
            low_mask = high_mask = None
            if mask is not None:
                mask_low, mask_high = cut_core.split_at(mask.astype(np.uint8), self.axis, k)
                low_mask = mask_low[tuple(slice(o, o + n) for o, n in zip(low_at, low.shape))] != 0
                high_mask = mask_high[tuple(slice(o, o + n) for o, n in zip(high_at, high.shape))] != 0
            masked = select_ops.set_mask(low_mask, model.uid) if mask is not None else None
            model.grid.replace_voxels(low)
            model.selection = Selection(model.grid.size)
            model.position = low_position
            grid = VoxelGrid(high.shape, high, scene.palette)
            grid.lock = high_mask if high_mask is not None and high_mask.any() else None
            scene.add(grid, name, tuple(high_position), model.layer)
            return stacked_record(unlinked, masked, ModelRecord(model.uid, GridRecord(before=before, after=low.copy()),
                                                                position_before=position_before,
                                                                position_after=low_position))

        def cut():
            records = [cut_one(*c) for c in cuts]
            document.set_active(source.uid)
            return records[0] if len(records) == 1 else stacked_record(*records)

        scene_ops._scene_change(cut)
        state.cut_aim = None
        return {"FINISHED"}


@register
class SceneToolAction(Action):
    """Pick the scene's tool: Pick (move and stretch) or Cut. C and Esc."""

    command_id = "rok.scene_tool"
    title = "Scene Tool"
    tooltip = "Pick moves and stretches models; Cut slices them in two"
    tool = EnumProp(name="Tool", items=(("PICK", "Pick", ""), ("CUT", "Cut", ""), ("TOGGLE_AXIS", "Turn the plane", "")))

    @classmethod
    def poll(cls, context):
        return state.level == "SCENE" and not state.stroke_active

    def execute(self, context):
        if self.tool == "TOGGLE_AXIS":
            if state.scene_tool != "CUT":
                return {"PASS"}  # Tab opens the wheel as always
            state.cut_other = not state.cut_other
        elif self.tool == "PICK" and state.scene_tool == "PICK":
            return {"PASS"}  # Esc with Pick on: nothing to leave
        else:
            state.scene_tool = self.tool
            state.cut_aim = None
            _m("gizmo_ops").refresh()  # the Cut tool has no gizmo
        from . import mode

        mode.rehover()
        redraw()
        return {"FINISHED"}


@register
class SceneTurnAction(Action):
    """R in the scene: a quarter turn of the picked model about the upright
    axis, in place (Shift+R: the other way). One undo step. The gizmo's rings
    turn by any angle."""

    command_id = "rok.scene_turn"
    title = "Turn Model"
    tooltip = "Turn the picked model a quarter about the upright axis"
    back = BoolProp(name="The other way", default=False)

    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):
        return transform_group("ROTATE", 2, 3 if self.back else 1)


# --- rotate by any angle (the gizmo's rings) ---------------------------------------


class _Rotation:
    """The picked models turned by an angle, from the voxels they had when the
    tool started. `preview` turns the models' Empties (their meshes follow on the
    GPU, no voxel changes: a mouse move costs nothing whatever the model size);
    `apply` bakes an angle into the voxels (the models change, nothing goes on
    the undo stack); `restore` puts them back; `commit` records the turn as one
    undo step. Every angle turns from the original voxels, so moving the mouse
    back and forth never wears the model down."""

    def __init__(self, models, axis: int, pivot2=None) -> None:
        self.axis = axis
        self.pivot2 = pivot2  # doubled world cells; None: the middle of the models' box
        self.error = ""
        self.applied = None  # the angle in the voxels now; None: the voxels from the start
        self.items = []
        for model in models:
            sel = None if model.selection.is_empty() else model.selection.mask.copy()
            self.items.append({
                "model": model,
                "voxels": model.grid.voxels.copy(),
                "position": tuple(int(p) for p in model.position),
                "orient": model.orient,
                "link": model.link,
                "selection": sel,
                "lock": select_ops.mask_of(model.uid),
            })

    def _items(self) -> list:
        return [(i["voxels"], i["position"]) for i in self.items]

    def fits(self, degrees: float) -> bool:
        """True when every model turned by `degrees` fits the size limit."""
        from rok.core import free_rotate

        return all(max(free_rotate.turned_size(v.shape, self.axis, degrees)) <= free_rotate.MAX_SIZE
                   for v, _p in self._items())

    def preview(self, degrees: float) -> None:
        """Turn the models' Empties by `degrees`; the voxels stay as they are.
        The overlay draws the boxes through the same matrices."""
        from rok.core import free_rotate

        matrices = free_rotate.preview_matrices(self._items(), self.axis, degrees, self.pivot2)
        state.turn_preview = {}
        for item, m in zip(self.items, matrices):
            # The app sends these to the host with every flush: the GPU turns the models.
            state.turn_preview[item["model"].uid] = m
        voxel_object.redraw_3d_views()

    def clear_preview(self) -> None:
        """The models back at rest."""
        state.turn_preview = {}

    def apply(self, degrees: float) -> bool:
        """Bake `degrees` about this rotation's axis into the models' voxels.
        False (and .error) when the turned model would be too big, or when
        masked voxels would be cut away. The models then keep the last good angle."""
        return self.apply_steps([(self.axis, degrees)], applied=degrees)

    def apply_steps(self, steps, applied=None) -> bool:
        """Bake a chain of turns, each (axis, degrees) about the pivot, in
        order, from the press voxels (Rotate by Degrees turns
        about several axes at once). False (and .error) when a turned model
        would be too big, or when masked voxels would be cut away."""
        from rok.core import free_rotate

        from .model_ops import MASK_LOSS, _mask_kept

        self.clear_preview()
        try:
            results = [(i["voxels"], i["position"]) for i in self.items]
            locks = [None if i["lock"] is None else (i["lock"].astype(np.uint8), i["position"]) for i in self.items]
            for axis, degrees in steps:
                results = free_rotate.rotate_models(results, axis, degrees, self.pivot2)
                locks = [None if lk is None else free_rotate.rotate_models([lk], axis, degrees, self.pivot2)[0]
                         for lk in locks]
        except ValueError as exc:
            self.error = str(exc)
            return False
        locks = [None if lk is None else lk[0] != 0 for lk in locks]
        for item, (voxels, _position), lock in zip(self.items, results, locks):
            if not _mask_kept(item["voxels"], item["lock"], voxels, lock):
                self.error = MASK_LOSS
                return False
        self.error = ""
        self.applied = applied
        for item, (voxels, position), lock in zip(self.items, results, locks):
            model = item["model"]
            model.grid.lock = lock  # the mask turns with the voxels
            model.grid.replace_voxels(voxels)
            model.selection = Selection(model.grid.size)
            model.position = position
        voxel_object.sync_scene()
        return True

    def restore(self) -> None:
        self.clear_preview()
        self.applied = None
        for item in self.items:
            model = item["model"]
            model.grid.replace_voxels(item["voxels"])
            model.grid.lock = item["lock"]
            model.position = item["position"]
            model.selection = Selection(model.grid.size)
            if item["selection"] is not None:
                model.selection.mask[...] = item["selection"]
        voxel_object.sync_scene()

    def commit(self) -> None:
        """Record the models as they show now. A model that was a linked copy
        leaves its group: a link holds only quarter turns."""
        records = []
        for item in self.items:
            model = item["model"]
            record = ModelRecord(
                model.uid, GridRecord(before=item["voxels"], after=model.grid.voxels.copy()),
                position_before=item["position"], position_after=tuple(int(p) for p in model.position))
            if item["link"]:
                record.link_before, record.link_after = item["link"], 0
                record.orient_before, record.orient_after = item["orient"], link_core.IDENTITY
                model.link, model.orient = 0, link_core.IDENTITY
            records.append(record)
            if item["lock"] is not None:
                from rok.core.undo import mask_record

                records.append(mask_record(model.uid, item["lock"], model.grid.lock, model.grid.size))
            if item["selection"] is not None:
                records.append(selection_record(model.uid, item["selection"], model.selection.mask))
        document.pivot_override = None
        commit(stacked_record(*records), serial=True, sync=True, hover=True)


@register
class SceneFrameAction(Action):
    """F in the scene: frame the picked model."""

    command_id = "rok.scene_frame"
    title = "Frame Model"
    tooltip = "Fit the picked model into the view"
    @classmethod
    def poll(cls, context):
        return state.level == "SCENE" and document.model is not None and not state.stroke_active

    def execute(self, context):
        found = _first_view_state(context)
        if found is None:
            return {"CANCELLED"}
        voxel_object.frame_models(context, group_models())
        voxel_object.redraw_3d_views()
        return {"FINISHED"}


@register
class ScenePlaceAction(Action):
    """A press on a shape in New's grid. A click makes a model of that shape
    next to the scene; a drag carries the new model into the view, on the
    floor under the mouse, snapping to its neighbours (Cmd: no snap), and the
    release puts it there. One undo step either way."""

    command_id = "rok.scene_place"
    title = "Place New Shape"
    tooltip = "Click: a new model of this shape; drag: place it in the view"
    solid = StringProp(name="Shape", default="BOX", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _in_scene() and context.view_state is not None

    def invoke(self, context, event):
        self._press = (event.mouse_region_x, event.mouse_region_y)
        self._model = None
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def _place(self, context, event) -> None:
        """The new model on the floor, centred under the mouse."""
        from . import mode

        region, view_state = context.region, context.view_state
        origin, direction = mode.world_ray(region, view_state, event.mouse_region_x, event.mouse_region_y)
        ox, oy, oz = document.view_offset
        point = scene_move.ray_plane_z(origin, direction, oz)
        if point is None:
            return
        size = tuple(int(v) for v in self._model.grid.size)
        pos = (int(round(point[0] - ox - size[0] / 2)), int(round(point[1] - oy - size[1] / 2)), 0)
        state.snap_uids = set()
        if not event.cmd_key:
            box = ((0, 0, 0), size)
            pos, hit = scene_move.snap_offset(box, pos, [b for _u, b in self._others], (0, 1))
            state.snap_uids = {self._others[k][0] for k in hit}
        if tuple(self._model.position) != tuple(pos):
            self._model.position = pos
            voxel_object.sync_scene()
        voxel_object.redraw_3d_views()

    def modal(self, context, event):
        from . import scene_ops

        if event.type in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN", "CMD"}:
            if self._model is None:
                dx, dy = event.mouse_region_x - self._press[0], event.mouse_region_y - self._press[1]
                if dx * dx + dy * dy < scene_move.DRAG_START_PX ** 2:
                    return {"RUNNING"}
                scene = document.scene
                self._others = [(m.uid, (tuple(int(p) for p in m.position),
                                         tuple(int(p) + s for p, s in zip(m.position, m.grid.size))))
                                for m in scene.models if not scene.is_hidden(m)]
                layer = int(document.model.layer) if document.model is not None else 0
                # Shown while it is carried; the release records it (below).
                self._model = scene.add(scene_ops.new_grid(self.solid), scene.new_model_name(), (0, 0, 0), layer)
                state.stroke_active = True
                context.window.cursor_modal_set("SCROLL_XY")
            self._place(context, event)
            return {"RUNNING"}
        if event.type == "LEFT_MOUSE" and event.value == "RELEASE":
            if self._model is None:
                return ops.rok.model_add(solid=self.solid)  # a click: next to the scene
            return self._finish(context, keep=True)
        if event.type in {"ESCAPE", "RIGHT_MOUSE", "FOCUS_LOST"} and event.value in {"PRESS", "NOTHING"}:
            return self._finish(context, keep=False)
        return {"RUNNING"}

    def _finish(self, context, keep: bool):
        from . import scene_ops

        state.snap_uids = set()
        state.stroke_active = False
        if self._model is None:
            return {"CANCELLED"}
        context.window.cursor_modal_restore()
        scene, model = document.scene, self._model
        scene.models.remove(model)
        if not keep:
            voxel_object.sync_scene()
            voxel_object.redraw_3d_views()
            return {"CANCELLED"}

        def add():
            scene.models.append(model)
            clear_group()
            document.set_active(model.uid)

        scene_ops._scene_change(add)
        return {"FINISHED"}
