"""The scene's Tools for several models: Pick Linked Copies, Align, Array and
Carve with Model (the maths is rok/core/arrange.py).

Align and Array open a card that shows the result in the scene while its
values change; OK makes it one undo step and Cancel puts the scene back.
Run without the card (Cmd+K's last values, a script), they act at once.
Carve has no card: the picked model cuts the models it overlaps, then waits
on a hidden layer named "Cutters", so it can cut again.
"""

from __future__ import annotations

import copy

import numpy as np

from rok.core import arrange
from rok.core import brick as brick_core
from rok.core import layers as layers_core
from rok.core import link as link_core
from rok.core.grid import VoxelGrid
from rok.core.undo import GridRecord, LayerRecord, ModelRecord, stacked_record
from . import object as voxel_object
from . import opkit, select_ops
from .object import document
from .opkit import BoolProp, EnumProp, IntProp, Action, register
from .ops_common import commit
from .scene_tools import _commit_moves, _in_scene, clear_group, group_models
from .state import state

CUTTERS = "Cutters"  # the hidden layer a cutter waits on after a carve
AXES = (("0", "X", "Along X"), ("1", "Y", "Along Y"), ("2", "Z", "Along Z"))


def _place(uid: int, position) -> None:
    from . import gizmo_ops

    gizmo_ops._place_empty(uid, tuple(int(p) for p in position))


def _show() -> None:
    voxel_object.sync_scene()
    voxel_object.redraw_3d_views()


# --- Pick Linked Copies -------------------------------------------------------------------


@register
class ScenePickLinkedAction(Action):
    command_id = "rok.scene_pick_linked"
    title = "Pick Linked Copies"
    tooltip = "Pick every linked copy of the picked model, to move or delete them together"

    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):
        scene = document.scene
        model = document.model
        copies = [m for m in link_core.members(scene, model.link) if not scene.is_hidden(m)]
        if len(copies) < 2:
            self.report({"INFO"}, "Not a linked copy")
            return {"CANCELLED"}
        state.group.clear()
        state.group |= {m.uid for m in copies}
        _show()
        self.report({"INFO"}, f"Picked {len(copies)} linked copies")
        return {"FINISHED"}


# --- Align ------------------------------------------------------------------------------------


def _align_changed(op, context) -> None:
    op.live()


@register
class SceneAlignAction(Action):
    """Line the picked models up on the picked (leading) one, or spread them
    with equal gaps. The models move live while the card is open."""

    command_id = "rok.scene_align"
    title = "Align Models"
    tooltip = "Line the picked models up on the picked one along X, Y or Z, or spread them evenly"
    axis = EnumProp(name="Axis", items=AXES, default="0", update=_align_changed)
    how = EnumProp(
        name="Align",
        items=((arrange.MIN, "Min", "The low sides meet the picked model's"),
               (arrange.CENTRE, "Centre", "The centres meet the picked model's"),
               (arrange.MAX, "Max", "The high sides meet the picked model's"),
               (arrange.SPREAD, "Spread", "Equal gaps between the models; the outer two stay")),
        default=arrange.MIN, update=_align_changed,
    )

    @classmethod
    def poll(cls, context):
        return _in_scene() and len(group_models()) >= 2

    def _moves(self):
        members = group_models()
        boxes = [(tuple(int(p) for p in m.position), tuple(int(n) for n in m.grid.size)) for m in members]
        axis = int(self.axis)
        if self.how == arrange.SPREAD:
            after = arrange.spread(boxes, axis)
        else:
            lead = next(k for k, m in enumerate(members) if m.uid == document.active_uid)
            after = arrange.aligned(boxes, lead, axis, self.how)
        return [(m, b[0], a) for m, b, a in zip(members, boxes, after)]

    def invoke(self, context, event):
        object.__setattr__(self, "_open", True)
        self.live()
        return context.prefs.open_dialog(self, title="Align Models", confirm_text="Align")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "axis", expand=True)
        layout.prop(self, "how", expand=True)
        layout.label(text="Lines up on the picked model; Spread needs three models", small=True)

    def live(self) -> None:
        if not getattr(self, "_open", False) or not _in_scene():
            return
        for model, _before, after in self._moves():
            _place(model.uid, after)
        voxel_object.redraw_3d_views()

    def _drop_preview(self) -> None:
        for model in group_models():
            _place(model.uid, model.position)
        object.__setattr__(self, "_open", False)

    def cancel(self, context) -> None:
        self._drop_preview()
        voxel_object.redraw_3d_views()

    def execute(self, context):
        self._drop_preview()
        if self.how == arrange.SPREAD and len(group_models()) < 3:
            self.report({"INFO"}, "Spread needs three picked models")
            return {"CANCELLED"}
        moves = [(m, b, a) for m, b, a in self._moves() if tuple(b) != tuple(a)]
        if not moves:
            self.report({"INFO"}, "The models are lined up already")
            return {"CANCELLED"}
        _commit_moves(moves)
        return {"FINISHED"}


# --- Array ----------------------------------------------------------------------------------


def _array_changed(op, context) -> None:
    op.live()


@register
class SceneArrayAction(Action):
    """Repeat the picked model (or the picked group, as one block) along an
    axis. The copies show live while the card is open."""

    command_id = "rok.scene_array"
    title = "Array"
    tooltip = "Repeat the picked models in a row along X, Y or Z, as linked copies"
    count = IntProp(name="Count", description="How many in the row, the original counted",
                    min=2, max=arrange.ARRAY_MAX, default=3, update=_array_changed)
    axis = EnumProp(name="Axis", items=AXES, default="0", update=_array_changed)
    direction = EnumProp(name="Direction", items=(("1", "+", "Toward higher cells"), ("-1", "−", "Toward lower cells")),
                         default="1", update=_array_changed)
    gap = IntProp(name="Gap", description="Empty cells between two copies; below 0 they overlap",
                  min=-63, max=256, default=0, update=_array_changed)
    linked = BoolProp(name="Linked", description="The copies share the original's shape: an edit to one changes all",
                      default=True, update=_array_changed)

    @classmethod
    def poll(cls, context):
        return _in_scene()

    def _offsets(self, members):
        lo = [min(int(m.position[i]) for m in members) for i in range(3)]
        hi = [max(int(m.position[i]) + int(m.grid.size[i]) for m in members) for i in range(3)]
        size = tuple(h - low for low, h in zip(lo, hi))
        return arrange.array_offsets(size, int(self.count), int(self.axis), int(self.direction), int(self.gap))

    def _add_copies(self, members) -> list[int]:
        """The copies as models in the scene, not recorded."""
        scene = document.scene
        uids = []
        for d in self._offsets(members):
            for model in members:
                grid = brick_core.carry(model.grid, VoxelGrid(model.grid.size, model.grid.voxels, scene.palette))
                name = scene.unique_name(model.display_name(scene.models.index(model)))
                position = tuple(int(p) + o for p, o in zip(model.position, d))
                made = scene.add(grid, name, position, model.layer)
                if self.linked:
                    if not model.link:
                        model.link = scene.new_link()
                    made.link, made.orient = model.link, model.orient
                uids.append(made.uid)
        return uids

    def _remove(self, uids) -> None:
        scene = document.scene
        for uid in uids:
            try:
                scene.remove(scene.index_of(uid))
            except ValueError:
                pass

    def invoke(self, context, event):
        object.__setattr__(self, "_shown", [])
        object.__setattr__(self, "_members", group_models())
        object.__setattr__(self, "_active", document.active_uid)
        self.live()
        return context.prefs.open_dialog(self, title="Array", confirm_text="Make Copies")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "count")
        layout.prop(self, "axis", expand=True)
        layout.prop(self, "direction", expand=True)
        layout.prop(self, "gap")
        layout.prop(self, "linked")
        layout.label(text="A group repeats as one block", small=True)

    def live(self) -> None:
        members = getattr(self, "_members", None)
        if members is None or document.scene is None:
            return
        self._remove(getattr(self, "_shown", []))
        object.__setattr__(self, "_shown", self._add_copies(members))
        document.set_active(self._active)
        _show()

    def _drop_preview(self) -> None:
        if getattr(self, "_members", None) is not None:
            self._remove(getattr(self, "_shown", []))
            document.set_active(self._active)
            object.__setattr__(self, "_members", None)
            object.__setattr__(self, "_shown", [])
            _show()

    def cancel(self, context) -> None:
        self._drop_preview()

    def execute(self, context):
        from . import scene_ops

        self._drop_preview()
        members = group_models()
        scene = document.scene
        before, active = tuple(scene.models), document.active_uid
        timeline_before = copy.deepcopy(scene.timeline)
        links_before = [(m.uid, m.link) for m in members]
        uids = self._add_copies(members)
        # A model that was alone gets a link id for its copies; it goes on the record.
        relinked = [ModelRecord(uid, None, link_before=old, link_after=scene.by_uid(uid).link)
                    for uid, old in links_before if scene.by_uid(uid).link != old]
        commit(scene_ops.ListRecord(before, tuple(scene.models), active, active, stacked_record(*relinked),
                                    timeline_before, copy.deepcopy(scene.timeline)),
               serial=True, sync=True, hover=True)
        self.report({"INFO"}, f"Made {len(uids)} {'linked ' if self.linked else ''}copies")
        return {"FINISHED"}


# --- Carve with Model ---------------------------------------------------------------------


def _cutters_layer(scene, cutter) -> int | None:
    """The layer the cutter waits on: one named Cutters, else an empty one.
    None when every layer holds a model."""
    used = {int(m.layer) for m in scene.models if m is not cutter}
    infos = layers_core.infos(scene.layers)
    named = [i.id for i in infos if i.name == CUTTERS and i.id not in used]
    empty = [i.id for i in infos if i.id not in used]
    return (named or empty or [None])[0]


def carve(cutter) -> tuple[list, list[int]]:
    """Cut the cutter's filled cells, where it stands, out of every shown model
    it overlaps; a linked target's copies take the hole too. A cutter that
    overlaps one of its own linked copies is made unique first, so it keeps
    its shape. Returns the undo records and the uids of the models that
    changed (none: nothing changed)."""
    from .scene_ops import unlink_record

    scene = document.scene

    def mask_for(target):
        return arrange.carve_mask(target.grid.voxels, target.position, cutter.grid.voxels, cutter.position,
                                  select_ops.mask_of(target.uid))

    # A brick model is locked: the cutter passes through it.
    shown = [m for m in scene.models if m is not cutter and not scene.is_hidden(m) and m.grid.bricks is None]
    unlinked = None
    own = [m for m in link_core.members(scene, cutter.link) if m is not cutter]
    if any(m in shown and mask_for(m) is not None for m in own):
        unlinked = unlink_record(cutter)
    same = {m.uid for m in link_core.members(scene, cutter.link)} | {cutter.uid}
    before = {}
    for target in [m for m in shown if m.uid not in same]:
        mask = mask_for(target)
        if mask is None:
            continue
        for member in [target] + [m for m in link_core.members(scene, target.link) if m is not target]:
            before.setdefault(member.uid, member.grid.voxels.copy())
        target.grid.set_voxels(np.argwhere(mask), 0, ignore_cut=True, ignore_lock=True)
        link_core.sync(scene, target.uid)  # its copies take the hole before the next target is cut
    changed = [uid for uid, v in before.items() if (scene.by_uid(uid).grid.voxels != v).any()]
    if not changed and unlinked is not None:
        unlinked.undo(scene)  # nothing was cut: the cutter stays linked
        unlinked = None
    records = [ModelRecord(uid, GridRecord(before=before[uid], after=scene.by_uid(uid).grid.voxels.copy()))
               for uid in changed]
    return ([unlinked] if unlinked is not None else []) + records, changed


def carve_drop(cutter, position, keep: bool) -> str:
    """A Ctrl-drag let go: the cutter carves where it was dropped. Without
    `keep` it is used up (it leaves the scene); with `keep` it stays where the
    drag started. One undo step. Returns the notice to show."""
    from . import scene_ops

    scene = document.scene
    start = tuple(int(p) for p in cutter.position)
    cutter.position = tuple(int(p) for p in position)
    records, changed = carve(cutter)
    cutter.position = start
    if not changed:
        return "Drop the model onto another one to carve it"
    clear_group()
    if keep:
        document.set_active(changed[0])
        commit(stacked_record(*records), serial=True, sync=True, hover=True)
    else:
        def use_up():
            scene.models.remove(cutter)
            document.set_active(changed[0])
            return stacked_record(*records)

        scene_ops._scene_change(use_up)
    n = len(changed)
    return f"Carved {n} model{'s' if n != 1 else ''}" + ("" if keep else "; the cutter is used up")


@register
class SceneCarveDragAction(Action):
    """Ctrl+left press on a model in the scene: a drag carves where it is let
    go (the cutter is used up); with Shift the cutter stays where it was.
    On a gizmo arrow the picked model carves along that axis. A press beside
    every model passes, so Ctrl+left drag orbits there."""

    command_id = "rok.scene_carve_drag"
    title = "Carve by Dragging"
    tooltip = "Ctrl-drag a model onto another to cut its shape out; Shift keeps the cutter"
    keep = BoolProp(name="Keep cutter", default=False, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _in_scene() and context.view_state is not None and state.scene_tool == "PICK"

    def invoke(self, context, event):
        from rok.core import gizmo, scene_move
        from . import mode
        from .tools import StrokeAction

        mx, my = event.mouse_region_x, event.mouse_region_y
        handle = StrokeAction.handle_at(context.region, context.view_state, mx, my)
        if handle is not None:
            if not handle.startswith("MOVE_") or document.model is None:
                return {"PASS"}  # another handle keeps its own drag
            # An arrow: the picked model carves along that axis.
            clear_group()
            StrokeAction._activate(document.active_uid)
            return opkit.call("rok.scene_drag", "INVOKE", carve=True, keep=bool(self.keep),
                              pressed=document.active_uid, axis=gizmo.handle_axis(handle),
                              grab_x=int(mx), grab_y=int(my))
        under, solid = mode.models_hit(context.region, context.view_state, mx, my)
        if not under:
            return {"PASS"}  # beside every model: Ctrl+drag orbits
        grabbed = scene_move.choose_grab(under, solid, {document.active_uid})
        clear_group()
        StrokeAction._activate(grabbed)
        return opkit.call("rok.scene_drag", "INVOKE", carve=True, keep=bool(self.keep), pressed=grabbed,
                          grab_x=int(mx), grab_y=int(my))


@register
class SceneCarveAction(Action):
    """The picked model cuts the cells it fills out of every shown model it
    overlaps. A linked target is carved with all its copies. The cutter moves
    to a hidden layer named Cutters, so it can be shown and used again."""

    command_id = "rok.scene_carve"
    title = "Carve with Model"
    tooltip = "Cut the picked model's shape out of the models it overlaps; the cutter waits on a hidden layer"

    @classmethod
    def poll(cls, context):
        return _in_scene()

    def execute(self, context):
        scene = document.scene
        cutter = document.model
        records, changed = carve(cutter)
        if not changed:
            self.report({"INFO"}, "The cutter touches no model")
            return {"CANCELLED"}
        layer = _cutters_layer(scene, cutter)
        if layer is not None:
            layers_before = list(scene.layers)
            layers = layers_core.set_name(layers_before, layer, CUTTERS)
            layers = layers_core.set_hidden(layers, layer, True)
            moves = ((cutter.uid, int(cutter.layer), layer),) if int(cutter.layer) != layer else ()
            scene.layers = layers
            cutter.layer = layer
            records.append(LayerRecord(layers_before, list(scene.layers), moves))
        clear_group()
        document.set_active(changed[0])
        commit(stacked_record(*records), serial=True, sync=True, hover=True)
        n = len(changed)
        where = f"; the cutter is on the hidden layer {CUTTERS}" if layer is not None else ""
        self.report({"INFO"}, f"Carved {n} model{'s' if n != 1 else ''}{where}")
        return {"FINISHED"}
