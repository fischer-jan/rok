"""Whole-model operations: Flip, Rotate 90 degrees, Resize Grid, Grow, Shrink,
Hollow, Smooth, Mirror Half, and the Whole Model submenu (Fill All, Fill Volume, Clear
All, Inverse, Fit Grid, Loop, Scale, Repeat, Diagonal). The rail and Cmd+K run them; the menu bar
has no Model menu.

Flip, rotate, resize and the whole-grid commands below replace the whole array
(a GridRecord). Grow, shrink, hollow and mirror half write the changed cells
through a StrokeRecorder, so a cell beyond a section plane is never overwritten.
Grow, shrink and hollow read `grid.visible`, so a cut voxel neither grows nor
shields its neighbours. Mirror half reads the real array: a cut voxel still
mirrors onto its partner.
"""

from __future__ import annotations

import numpy as np

from rok.core import edit, grid_ops, morph, region, reshape, shape, size_entry
from rok.core.select import Selection
from rok.core.undo import GridRecord, ModelRecord, StrokeRecorder, stacked_record

from . import object as voxel_object
from . import opkit
from .opkit import BoolProp, EnumProp, IntProp, IntVectorProp, Action, register
from .handlers import registry
from .object import document
from .ops_common import can_edit_model, can_edit_voxels, redraw, refresh_views
from .state import state

_AXIS_ITEMS = (("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", ""))


def _cursor(context, kind: str) -> None:
    window = getattr(context, "window", None)
    if window is not None:
        window.cursor_set(kind)


def _moved_mask(mask_op) -> np.ndarray | None:
    """The active model's mask taken through a whole-grid change `mask_op`
    (voxels -> voxels), or None when nothing is masked."""
    from .select_ops import mask_of

    mask = mask_of()
    if mask is None:
        return None
    return np.asarray(mask_op(mask.astype(np.uint8))) != 0


def _loses_masked(mask_op) -> bool:
    """True when a whole-grid change would cut masked voxels away."""
    from .select_ops import mask_of

    mask = mask_of()
    moved = _moved_mask(mask_op)
    return mask is not None and int(moved.sum()) < int(mask.sum())


def _mask_kept(old_voxels, old_mask, new_voxels, new_mask) -> bool:
    """True when every colour under `old_mask` is still under `new_mask`,
    on at least as many cells. No mask counts as kept. More copies of the
    same colours still count as kept."""
    if old_mask is None:
        return True
    old_mask = np.asarray(old_mask, dtype=bool)
    if not old_mask.any():
        return True
    if new_mask is None:
        return False
    new_mask = np.asarray(new_mask, dtype=bool)
    new_voxels = np.asarray(new_voxels)
    if new_mask.shape != new_voxels.shape or int(new_mask.sum()) < int(old_mask.sum()):
        return False
    old = {int(c) for c in np.asarray(old_voxels)[old_mask]}
    return {int(c) for c in new_voxels[new_mask]} == old


def _spoils_mask(new_voxels, mask_op) -> bool:
    """True when `mask_op` would drop a masked colour or cover fewer cells."""
    from .select_ops import mask_of

    mask = mask_of()
    if mask is None:
        return False
    return not _mask_kept(document.grid.voxels, mask, new_voxels, _moved_mask(mask_op))


MASK_LOSS = "Masked voxels would be cut away: clear the mask first (Option+M)"


def _apply_grid_op(context, new_voxels, position=None, mask_op=None) -> None:
    """Replace the active model's voxels, push one ModelRecord and bring the scene in line.

    The mask holds: `mask_op`, the same change for any
    voxels, takes the mask along with the voxels it moves; without it the
    change keeps the cells where they are and masked cells keep their value."""
    from .select_ops import mask_of, set_mask

    model = document.model
    grid = model.grid
    mask = mask_of()
    new_mask = None
    if mask is not None:
        if mask_op is not None:
            new_mask = _moved_mask(mask_op)
        elif new_voxels.shape == grid.voxels.shape:
            new_voxels = np.where(mask, grid.voxels, new_voxels).astype(np.uint8)
            new_mask = mask
    before = tuple(int(p) for p in model.position)
    after = before if position is None else tuple(int(p) for p in position)
    if after == before and np.array_equal(grid.voxels, new_voxels):
        return  # nothing changed: no record, no lost selection, and the file stays saved
    masked = set_mask(new_mask) if mask is not None else None  # before the grid changes size
    inner = GridRecord(before=grid.voxels.copy(), after=new_voxels)
    grid.replace_voxels(new_voxels)
    model.position = after
    document.undo.push(stacked_record(masked, ModelRecord(model.uid, inner, before, after)))
    document.modified = True
    # A whole-grid operation drops the selection; its cells no longer line up.
    model.selection = Selection(grid.size)
    document.pivot_override = None
    voxel_object.sync_scene()
    refresh_views(serial=True, hover=True)


def _limit_mask():
    """The selection mask when there is a selection, else None (the whole model)."""
    selection = document.selection
    if selection is None or selection.is_empty():
        return None
    return selection.mask


def _apply_cells(new_voxels, before=None, palette_record=None):
    """Write the cells where `before` differs from `new_voxels`. Return the cells.

    A `palette_record` (a shader that set its colours first) goes on the undo
    stack with the voxel change as one step.

    `before` defaults to `grid.visible`, where cut cells look empty; set_voxels
    refuses any write into them. Mirror Half passes the real array instead, so a
    cut voxel does not copy emptiness over its visible mirror partner.
    """
    grid = document.grid
    if before is None:
        before = grid.visible
    cells, values = morph.changed_cells(before, new_voxels)
    recorder = StrokeRecorder(grid)
    if len(cells):
        recorder.set_many(cells, values)
    record = recorder.finish()
    model_record = None if record is None else ModelRecord(document.active_uid, record)
    stacked = stacked_record(palette_record, model_record)
    if stacked is not None:
        document.undo.push(stacked)
        document.modified = True
    voxel_object.flush_dirty()
    return cells


def _scope_label(layout) -> None:
    if _limit_mask() is None:
        layout.label(text="Applies to the whole model")
    else:
        layout.label(text="Applies to the selection")


@register
class FlipAction(Action):
    command_id = "rok.flip"
    title = "Flip"
    tooltip = "Mirror the model along one axis"
    axis = EnumProp(
        name="Axis", description="The axis to work on", items=_AXIS_ITEMS, default="X"
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        axis = "XYZ".index(self.axis)
        _apply_grid_op(context, grid_ops.flip(document.grid.voxels, axis),
                       mask_op=lambda v: grid_ops.flip(v, axis))
        return {"FINISHED"}


@register
class RotateAction(Action):
    command_id = "rok.rotate"
    title = "Rotate 90°"
    tooltip = "Turn the model a quarter turn around one axis"
    axis = EnumProp(
        name="Axis", description="The axis to turn around", items=_AXIS_ITEMS, default="Z"
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        axis = "XYZ".index(self.axis)
        model = document.model
        new_voxels = grid_ops.rotate90(model.grid.voxels, axis)
        # Keep the box centre t = position + size // 2 in place: four turns restore the position.
        size = model.grid.size
        t = tuple(int(model.position[i]) + size[i] // 2 for i in range(3))
        position = tuple(t[i] - new_voxels.shape[i] // 2 for i in range(3))
        _apply_grid_op(context, new_voxels, position, mask_op=lambda v: grid_ops.rotate90(v, axis))
        return {"FINISHED"}


@register
class ResizeAction(Action):
    command_id = "rok.resize"
    title = "Resize Grid"
    tooltip = "Change the grid size, keeping the model at the origin corner"
    size = IntVectorProp(
        name="Size",
        description="The new grid size in voxels",
        size=3,
        min=grid_ops.MIN_SIZE,
        max=grid_ops.MAX_SIZE,
        default=(32, 32, 32),
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def invoke(self, context, event):
        self.size = document.grid.size
        return context.prefs.open_dialog(
            self, title="Resize Grid", confirm_text="Resize"
        )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "size", text="Size")
        layout.label(text="Voxels outside the new size are removed")

    def execute(self, context):
        size = tuple(self.size)
        try:
            new_voxels = grid_ops.resize(document.grid.voxels, size)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if _loses_masked(lambda v: grid_ops.resize(v, size)):
            self.report({"ERROR"}, MASK_LOSS)
            return {"CANCELLED"}
        _apply_grid_op(context, new_voxels, mask_op=lambda v: grid_ops.resize(v, size))
        voxel_object.frame_scene(context)  # as Recenter does: the new size fits the view
        return {"FINISHED"}


_NUMPAD = {f"NUMPAD_{d}": str(d) for d in range(10)}


@register
class GridSizeTypeAction(Action):
    """Type a new grid size into the label at the lower left: the field opens with the size selected, the first key replaces it, Enter
    resizes, Esc or a click elsewhere leaves the size as it was."""

    command_id = "rok.grid_size_type"
    title = "Type Grid Size"
    tooltip = "Type a new grid size: 40 40 40, or 40 for a cube; Enter resizes"
    @classmethod
    def poll(cls, context):
        return can_edit_model() and state.size_edit is None

    def invoke(self, context, event):
        state.size_edit = size_entry.start_text(document.grid.size)
        state.size_edit_fresh = True
        context.prefs.add_modal(self)
        _redraw_all()
        return {"RUNNING"}

    def modal(self, context, event):
        if state.size_edit is None or document.grid is None:
            return self._end(cancelled=True)  # a file load or New closed it
        kind, value = event.type, event.value
        if kind in {"MOUSE_MOVE", "MOUSE_MOVE_BETWEEN", "TIMER", "TIMER_REPORT", "WHEEL_UP",
                    "WHEEL_DOWN", "TRACKPAD_PAN", "TRACKPAD_ZOOM", "FOCUS_LOST"}:
            return {"PASS"}
        if value != "PRESS":
            return {"RUNNING"}
        if kind in {"ENTER", "NUMPAD_ENTER"}:
            size = size_entry.parse_size(state.size_edit)
            if size is None:
                self.report({"WARNING"}, "Type a size like 40 40 40, or 40 for a cube (1 to 256)")
                return {"RUNNING"}
            self._end()
            if tuple(size) != tuple(document.grid.size):
                opkit.ops.rok.resize("EXEC", size=size)
            return {"FINISHED"}
        if kind == "ESCAPE":
            return self._end(cancelled=True)
        if kind in {"LEFT_MOUSE", "RIGHT_MOUSE", "MIDDLE_MOUSE"}:
            self._end(cancelled=True)
            return {"CANCELLED", "PASS"}  # the click still does what it does
        if kind == "BACKSPACE":
            state.size_edit, state.size_edit_fresh = size_entry.backspace(state.size_edit, state.size_edit_fresh)
        else:
            char = _NUMPAD.get(kind) or event.unicode
            if char:
                state.size_edit, state.size_edit_fresh = size_entry.type_char(
                    state.size_edit, state.size_edit_fresh, char
                )
        _redraw_all()
        return {"RUNNING"}  # every other key is eaten: T or R must not switch the tool

    def cancel(self, context):
        self._end(cancelled=True)

    def _end(self, cancelled: bool = False):
        state.size_edit = None
        state.size_edit_fresh = False
        _redraw_all()
        return {"CANCELLED"} if cancelled else {"FINISHED"}


def _redraw_all() -> None:
    redraw()


# ---------------------------------------------------------------- grow, shrink, hollow


class _MorphDialog:
    """Shared invoke and draw of Grow and Shrink: two fields and a scope label."""

    # None of the three remembers its last value: the keymap's dialog=False must
    # not stick to the menu entry, and the dialog's count must not stick to Shift+G.
    count = IntProp(
        name="Voxels",
        description="How many layers of voxels to add or remove",
        min=1,
        max=16,
        default=1,
        options={"SKIP_SAVE"},
    )
    round_corners = BoolProp(
        name="Round corners",
        description="Round the corners instead of leaving them square",
        default=False,
        options={"SKIP_SAVE"},
    )
    dialog = BoolProp(default=True, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        # Shift+G in the scene is Drop (rok.scene_drop_onto).
        return can_edit_voxels()

    def invoke(self, context, event):
        if not self.dialog:
            return self.execute(context)
        return context.prefs.open_dialog(
            self, title=self.title, confirm_text=self.title
        )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "count")
        layout.prop(self, "round_corners")
        _scope_label(layout)


@register
class GrowAction(_MorphDialog, Action):
    command_id = "rok.grow"
    title = "Grow"
    tooltip = "Add layers of voxels around the model or the selection"
    def execute(self, context):
        grid = document.grid
        limit = _limit_mask()
        _cursor(context, "WAIT")
        try:
            new = morph.grow(
                grid.visible, int(self.count), limit=limit, round_corners=self.round_corners
            )
            cells = _apply_cells(new)
        finally:
            _cursor(context, "DEFAULT")
        if limit is not None:
            document.selection.add_cells(cells)  # the new layer joins the selection
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        return {"FINISHED"}


@register
class ShrinkAction(_MorphDialog, Action):
    command_id = "rok.shrink"
    title = "Shrink"
    tooltip = "Remove layers of voxels from the model or the selection"
    def execute(self, context):
        grid = document.grid
        limit = _limit_mask()
        _cursor(context, "WAIT")
        try:
            new = morph.shrink(
                grid.visible, int(self.count), limit=limit, round_corners=self.round_corners
            )
            _apply_cells(new)
        finally:
            _cursor(context, "DEFAULT")
        if document.selection is not None:
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        return {"FINISHED"}


@register
class HollowAction(Action):
    command_id = "rok.hollow"
    title = "Hollow"
    tooltip = "Empty the inside of the model or the selection"
    wall = IntProp(
        name="Wall",
        description="How many voxels thick the wall that stays is",
        min=1,
        max=8,
        default=1,
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def invoke(self, context, event):
        return context.prefs.open_dialog(
            self, title="Hollow", confirm_text="Hollow"
        )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "wall")
        layout.label(text="Keeps a wall this thick and empties the inside")
        _scope_label(layout)

    def execute(self, context):
        grid = document.grid
        _cursor(context, "WAIT")
        try:
            _apply_cells(morph.hollow(grid.visible, int(self.wall), limit=_limit_mask()))
        finally:
            _cursor(context, "DEFAULT")
        if document.selection is not None:
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        return {"FINISHED"}


def _crease_changed(op, context) -> None:
    op.live(context)


@register
class CreaseAction(Action):
    """Bars on the sharp edges. The dialog shows the result in the real voxels
    while its values change (no undo step, as Mirror Half's hover); OK makes it
    one undo step, Cancel puts the voxels back. The split into faces is done
    once, when the dialog opens. Mirror repeats each bar on the other side."""

    command_id = "rok.crease"
    title = "Crease"
    tooltip = "Raise bars in the current colour on the sharp edges of the model or the selection"
    angle = IntProp(
        name="Angle",
        description="How sharp an edge must be: its two faces turn by at least this many degrees",
        min=10, max=90, default=20, update=_crease_changed,
    )
    width = IntProp(
        name="Width",
        description="1 puts a bar on one side of each edge, 2 on both sides",
        min=1, max=2, default=1, update=_crease_changed,
    )
    bars = EnumProp(
        name="Bars",
        description="Raise adds voxels over the edges; Paint recolours the edge voxels; Raise + Paint does both",
        items=(("RAISE", "Raise", "Bars one voxel high over the edges"),
               ("PAINT", "Paint", "The edge voxels take the colour"),
               ("BOTH", "Raise + Paint", "Raised bars over painted edge voxels")),
        default="RAISE", update=_crease_changed,
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def _result(self, voxels, found):
        from rok.core import facets

        mirror = tuple(bool(v) for v in getattr(opkit.CONTEXT.prefs, "rok_mirror", (False, False, False)))
        return facets.crease(voxels, int(document.colour), float(self.angle), int(self.width),
                             self.bars != "PAINT", found=found, mirror=mirror,
                             paint=self.bars != "RAISE")

    def invoke(self, context, event):
        from rok.core import facets

        _cursor(context, "WAIT")
        try:
            before = document.grid.visible.copy()
            object.__setattr__(self, "_before", before)
            object.__setattr__(self, "_faces", facets.Faces.of(before, _limit_mask()))
            object.__setattr__(self, "_key", None)
            self.live(context)
        finally:
            _cursor(context, "DEFAULT")
        return context.prefs.open_dialog(self, title="Crease", confirm_text="Crease")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "angle")
        layout.prop(self, "width")
        layout.prop(self, "bars", expand=True)
        layout.label(text="Bars in the current colour on sharp edges")
        _scope_label(layout)

    def live(self, context) -> None:
        """Show the result for the values now, from the voxels as they were."""
        before = getattr(self, "_before", None)
        grid = document.grid
        if before is None or grid is None or grid.visible.shape != before.shape:
            return
        key = (int(self.angle), int(self.width), self.bars)
        if key == getattr(self, "_key", None):
            return
        object.__setattr__(self, "_key", key)
        self._show(self._result(before, getattr(self, "_faces", None)))

    def _show(self, want) -> None:
        grid = document.grid
        cells, values = morph.changed_cells(grid.visible, want)
        if len(cells):
            grid.set_voxels(cells, values)
        voxel_object.flush_dirty()
        refresh_views(serial=True)

    def cancel(self, context) -> None:
        before = getattr(self, "_before", None)
        if before is not None and document.grid is not None and document.grid.visible.shape == before.shape:
            self._show(before)
        object.__setattr__(self, "_before", None)

    def execute(self, context):
        from rok.core import facets

        before = getattr(self, "_before", None)
        grid = document.grid
        if before is None or grid.visible.shape != before.shape:
            before, found = grid.visible.copy(), None  # run without the dialog (Cmd+K's last values, a script)
        else:
            found = getattr(self, "_faces", None)
            self._show(before)  # back to the start, so the write is one undo step from there
        _cursor(context, "WAIT")
        try:
            want = self._result(before, found if found is not None else facets.Faces.of(before, _limit_mask()))
            cells = _apply_cells(want)
        finally:
            _cursor(context, "DEFAULT")
        object.__setattr__(self, "_before", None)
        refresh_views(serial=True, hover=True)
        if not len(cells):
            self.report({"INFO"}, "Crease found no edge that sharp; try a smaller Angle")
        return {"FINISHED"}


# ---------------------------------------------------------------- bevel, bridge, inset


@register
class BevelAction(Action):
    command_id = "rok.bevel"
    title = "Bevel"
    tooltip = (
        "Take one voxel off every convex edge of the selection, or of the whole model. "
        "Thin parts keep their edge. Run it again for a rounder edge"
    )
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        grid = document.grid
        remove = morph.bevel_mask(grid.visible != 0, limit=_limit_mask())
        count = int(np.count_nonzero(remove))
        if count == 0:
            self.report({"INFO"}, "No edge to bevel")
            return {"CANCELLED"}
        new = grid.visible.copy()
        new[remove] = 0
        _apply_cells(new)
        if document.selection is not None:
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        self.report({"INFO"}, f"Bevelled {count} voxel{'s' if count != 1 else ''} off")
        return {"FINISHED"}


@register
class SmoothAction(Action):
    command_id = "rok.smooth"
    title = "Smooth"
    tooltip = (
        "Smooth the selection, or the whole model: corners and bumps go, dents fill. "
        "Thin walls keep their shape. Run it again for a smoother surface"
    )
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        from .select_ops import mask_of
        from .tools import MASK_NOTE

        grid = document.grid
        limit = _limit_mask()
        _cursor(context, "WAIT")
        try:
            # A masked cell stays, and nothing new is built against its face.
            raw = morph.smooth(grid.visible, limit=limit)
            masked = mask_of()
            new = morph.keep_masked(grid.visible, raw, masked) if masked is not None else raw
            cells = _apply_cells(new)
        finally:
            _cursor(context, "DEFAULT")
        held = masked is not None and not np.array_equal(new, raw)
        if not len(cells):
            self.report({"INFO"}, MASK_NOTE if held else "Already smooth")
            return {"CANCELLED"}
        if held:
            self.report({"INFO"}, MASK_NOTE)
        if limit is not None:
            document.selection.add_cells(cells)  # filled dents join the selection
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        return {"FINISHED"}


@register
class BridgeAction(Action):
    command_id = "rok.bridge"
    title = "Bridge"
    tooltip = (
        "Fill a solid slab between the two separate parts of the selection with the "
        "current colour: flat side walls, height stepping only where the sides differ. "
        "Cmd+K “Bridge (lines)” is the old all-pairs fill"
    )
    kind = EnumProp(
        name="Bridge",
        description="Solid slab between facing sides, or the old all-pairs lines",
        items=(
            (
                shape.BRIDGE_SOLID,
                "Solid",
                "A box loft between the facing sides",
            ),
            (
                shape.BRIDGE_LINES,
                "Lines",
                "Every straight line between the two parts (the old fill)",
            ),
        ),
        default=shape.BRIDGE_SOLID,
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        grid = document.grid
        selection = document.selection
        parts = [] if selection is None or selection.is_empty() else region.components(selection.mask)
        if len(parts) != 2:
            self.report({"WARNING"}, f"Select two separate parts first (the selection has {len(parts)})")
            return {"CANCELLED"}
        _cursor(context, "WAIT")
        try:
            cells = shape.bridge_cells(parts[0], parts[1], grid.size, kind=self.kind)
            new = grid.visible.copy()
            if len(cells):
                empty = new[cells[:, 0], cells[:, 1], cells[:, 2]] == 0
                cells = cells[empty]
                new[cells[:, 0], cells[:, 1], cells[:, 2]] = document.colour
            written = _apply_cells(new)
        finally:
            _cursor(context, "DEFAULT")
        if len(written) == 0:
            self.report({"INFO"}, "Nothing to bridge: the parts touch, or no gap lies straight between them")
            return {"CANCELLED"}
        refresh_views(serial=True, hover=True)
        self.report({"INFO"}, f"Bridged with {len(written)} voxel{'s' if len(written) != 1 else ''}")
        return {"FINISHED"}


@register
class InsetAction(Action):
    command_id = "rok.inset"
    title = "Inset"
    tooltip = (
        "Raise or sink the inside of the face under the mouse, leaving a rim: "
        "a panel with a frame in one step"
    )
    width = IntProp(name="Rim", description="How many voxels wide the rim that stays is", min=1, max=8, default=1)
    depth = IntProp(
        name="Depth",
        description="How many layers the inside rises (positive) or sinks (negative)",
        min=-8, max=8, default=1,
    )
    # The face, taken from the hover when the dialog opens.
    voxel = IntVectorProp(size=3, default=(0, 0, 0), options={"HIDDEN"})
    normal = IntVectorProp(size=3, default=(0, 0, 1), options={"HIDDEN"})
    wall = BoolProp(default=False, options={"HIDDEN"})
    has_face = BoolProp(default=False, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def invoke(self, context, event):
        hover = state.hover
        if hover is None or hover.hit is None or hover.tool == edit.ACTIVATE:
            self.report({"WARNING"}, "Point at a face and press Shift+I")
            return {"CANCELLED"}
        hit = hover.hit
        self.voxel = tuple(int(c) for c in hit.voxel)
        self.normal = tuple(int(c) for c in hit.normal)
        self.wall = bool(hit.wall)
        self.has_face = True
        return context.prefs.open_dialog(self, title="Inset", confirm_text="Inset")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "width")
        layout.prop(self, "depth")
        layout.label(text="The inside of the face rises or sinks, the rim stays")

    def execute(self, context):
        if not self.has_face:
            self.report({"WARNING"}, "Point at a face and press Shift+I")
            return {"CANCELLED"}
        grid = document.grid
        voxel = tuple(int(c) for c in self.voxel)
        normal = tuple(int(c) for c in self.normal)
        depth = int(self.depth)
        if self.wall:
            if depth <= 0:
                self.report({"WARNING"}, "A wall can only rise")
                return {"CANCELLED"}
            target = (voxel[0] + normal[0], voxel[1] + normal[1], voxel[2] + normal[2])
            cells = region.wall_region(grid.tool_voxels, target, normal)
            first = 0  # the wall cells are empty: the first layer is the region itself
        else:
            cells = region.face_region(grid.tool_voxels, voxel, normal)
            first = 1 if depth > 0 else 0  # attach starts above the face; erase at the face
        if len(cells) == 0:
            self.report({"WARNING"}, "No face there")
            return {"CANCELLED"}
        inner = region.inset_region(cells, normal, int(self.width))
        if len(inner) == 0:
            self.report({"WARNING"}, "The rim leaves nothing inside")
            return {"CANCELLED"}
        if depth == 0:
            self.report({"INFO"}, "Depth is 0: nothing to raise or sink")
            return {"CANCELLED"}
        step = np.asarray(normal, dtype=np.int64) * (1 if depth > 0 else -1)
        layers = [inner + (first + k) * step for k in range(abs(depth))]
        target_cells = np.concatenate(layers)
        tool = edit.ATTACH if depth > 0 else edit.ERASE
        mirror = tuple(getattr(context.prefs, "rok_mirror", (False, False, False)))
        recorder = StrokeRecorder(grid)
        changed = edit.apply_cells(grid.voxels, tool, document.colour, target_cells, recorder.set_many, mirror)
        record = recorder.finish()
        if record is not None:
            document.undo.push(ModelRecord(document.active_uid, record))
            document.modified = True
        voxel_object.flush_dirty()
        if document.selection is not None:
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        self.report({"INFO"}, f"Inset: {changed} voxel{'s' if changed != 1 else ''} {'raised' if depth > 0 else 'sunk'}")
        return {"FINISHED"}


# ---------------------------------------------------------------- mirror half


@register
class MirrorHalfAction(Action):
    command_id = "rok.mirror_half"
    title = "Mirror Half"
    tooltip = "Copy one half of the model onto the other, mirrored"
    axis = EnumProp(
        name="Axis", description="The axis to mirror across", items=_AXIS_ITEMS, default="X"
    )
    from_positive = BoolProp(
        name="From Positive", description="Copy the positive half onto the negative one", default=True
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        grid = document.grid
        axis = "XYZ".index(self.axis)
        # The real array on both sides: a cut voxel must still mirror onto its
        # partner. set_voxels keeps the cut cell itself untouched.
        _apply_cells(
            grid_ops.mirror_half(grid.voxels, axis, bool(self.from_positive)),
            before=grid.voxels,
        )
        if document.selection is not None:
            document.selection.prune(grid.visible)
        refresh_views(serial=True, hover=True)
        return {"FINISHED"}


# ---------------------------------------------------------------- whole model


@register
class FillAllAction(Action):
    command_id = "rok.fill_all"
    title = "Fill All"
    tooltip = "Give every filled voxel the current colour"
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        new_voxels = grid_ops.fill_all(document.grid.voxels, document.colour)
        _apply_grid_op(context, new_voxels)
        return {"FINISHED"}


@register
class FillVolumeAction(Action):
    command_id = "rok.fill_volume"
    title = "Fill Volume"
    tooltip = "Make every cell of the grid the current colour"
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        new_voxels = grid_ops.fill_volume(document.grid.size, document.colour)
        _apply_grid_op(context, new_voxels)
        return {"FINISHED"}


@register
class ClearAllAction(Action):
    command_id = "rok.clear_all"
    title = "Clear All"
    tooltip = "Empty every cell of the grid"
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        new_voxels = grid_ops.fill_volume(document.grid.size, 0)
        _apply_grid_op(context, new_voxels)
        return {"FINISHED"}


@register
class InverseAction(Action):
    command_id = "rok.inverse"
    title = "Inverse"
    tooltip = "Empty cells take the current colour; filled cells become empty"
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        new_voxels = grid_ops.inverse(document.grid.voxels, document.colour)
        _apply_grid_op(context, new_voxels)
        return {"FINISHED"}


@register
class FitGridAction(Action):
    command_id = "rok.fit_grid"
    title = "Fit Grid"
    tooltip = "Shrink the grid to the filled voxels, keeping them at their world place"
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        model = document.model
        grid = model.grid
        bounds = grid_ops.fit_bounds(grid.voxels)
        if bounds is None:
            self.report({"INFO"}, "The model is empty; nothing to fit")
            return {"CANCELLED"}
        lo, hi = bounds
        new_voxels = np.ascontiguousarray(
            grid.voxels[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
        )
        position = tuple(int(model.position[i]) + lo[i] for i in range(3))
        _apply_grid_op(context, new_voxels, position,
                       mask_op=lambda v: np.ascontiguousarray(v[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]))
        return {"FINISHED"}


@register
class LoopAction(Action):
    command_id = "rok.loop"
    title = "Loop"
    tooltip = "Shift the voxels along an axis; they wrap around at the edge"
    axis = EnumProp(
        name="Axis", description="The axis to shift along", items=_AXIS_ITEMS, default="X"
    )
    offset = IntProp(
        name="Offset",
        description="How far to shift; positive goes up the axis",
        min=-256,
        max=256,
        default=1,
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def invoke(self, context, event):
        return context.prefs.open_dialog(self, title="Loop", confirm_text="Loop")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "axis")
        layout.prop(self, "offset")

    def execute(self, context):
        axis = "XYZ".index(self.axis)
        offset = int(self.offset)
        new_voxels = grid_ops.loop(document.grid.voxels, axis, offset)
        _apply_grid_op(context, new_voxels, mask_op=lambda v: grid_ops.loop(v, axis, offset))
        return {"FINISHED"}


@register
class ScaleAction(Action):
    command_id = "rok.scale"
    title = "Scale"
    tooltip = "Double or halve the model's resolution"
    mode = EnumProp(
        name="Mode",
        description="Scale direction",
        items=(
            ("UP", "x2", "Repeat every voxel into a 2 x 2 x 2 block"),
            ("DOWN", "x1/2", "Every 2 x 2 x 2 block becomes one voxel; thin parts stay closed"),
        ),
        default="UP",
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        grid = document.grid
        scale = grid_ops.scale_up if self.mode == "UP" else grid_ops.scale_down
        try:
            new_voxels = scale(grid.voxels)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        # Halving can paint a masked cell with a neighbour's colour, or fold
        # several masked cells into one. Either one is refused. Doubling
        # copies each cell, mask included, so it stays.
        if self.mode == "DOWN" and _spoils_mask(new_voxels, scale):
            self.report({"ERROR"}, MASK_LOSS)
            return {"CANCELLED"}
        _apply_grid_op(context, new_voxels, mask_op=scale)
        return {"FINISHED"}


@register
class RepeatAction(Action):
    command_id = "rok.repeat"
    title = "Repeat"
    tooltip = "Tile the model along each axis"
    counts = IntVectorProp(
        name="Counts",
        description="How many times to repeat the model along each axis",
        size=3,
        min=1,
        max=8,
        default=(1, 1, 1),
    )

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def invoke(self, context, event):
        self.counts = (1, 1, 1)
        return context.prefs.open_dialog(
            self, title="Repeat", confirm_text="Repeat"
        )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "counts", text="Counts")
        layout.label(text="Capped so the grid stays within 256 voxels per axis")

    def execute(self, context):
        grid = document.grid
        counts = tuple(int(c) for c in self.counts)
        new_voxels = grid_ops.repeat(grid.voxels, counts)
        applied = tuple(new_voxels.shape[i] // grid.voxels.shape[i] for i in range(3))
        if applied != counts:
            self.report({"WARNING"}, "Repeat capped to keep the grid within 256 per axis")
        _apply_grid_op(context, new_voxels, mask_op=lambda v: grid_ops.repeat(v, counts))
        return {"FINISHED"}


# Drop to Floor's fall, in cells and seconds: 16 cells take about 0.28 s.
FALL_G = 400.0
BOUNCE_SHARE = 0.2  # a bounce rises a fifth of the fall ...
BOUNCE_MAX = 3  # ... at most three cells, and none below three


def fall_offset(t: float, fall: int, max_up: int | None = None) -> int:
    """How many cells below the start the falling voxels are at time t: falling
    faster under gravity, then one small bounce, then resting at `fall`. `max_up`
    caps the bounce (a voxel under a ceiling has little room)."""
    import math

    t_fall = math.sqrt(2.0 * fall / FALL_G)
    if t <= t_fall:
        return min(fall, int(round(0.5 * FALL_G * t * t)))
    up = bounce_height(fall) if max_up is None else min(bounce_height(fall), max_up)
    t_up = math.sqrt(2.0 * up / FALL_G) if up else 0.0
    t -= t_fall
    if t >= 2.0 * t_up:
        return fall
    height = up - 0.5 * FALL_G * (t - t_up) ** 2
    return fall - max(0, int(round(height)))


def bounce_height(fall: int) -> int:
    return min(BOUNCE_MAX, int(round(fall * BOUNCE_SHARE))) if fall >= 3 else 0


def fall_seconds(fall: int, max_up: int | None = None) -> float:
    import math

    up = bounce_height(fall) if max_up is None else min(bounce_height(fall), max_up)
    return math.sqrt(2.0 * fall / FALL_G) + 2.0 * math.sqrt(2.0 * up / FALL_G)


@register
class DropVoxelsAction(Action):
    """Gravity for single voxels: each selected voxel falls straight
    down on its own until it rests on the floor, on a voxel that is not selected, or
    on a selected voxel that has already settled. A column of selected voxels is
    worked from the bottom up and lands as a pile. Masked voxels stay where they are
    and the others land on them. From the rail every voxel falls with gravity and a
    small bounce of its own, moving the real voxels each frame; a click or a key
    skips to the end. One undo step either way."""

    command_id = "rok.drop_voxels"
    title = "Drop Voxels"
    tooltip = ("Each selected voxel falls on its own, with a small bounce, until it rests on the floor or on "
                      "another voxel; a column of selected voxels lands as a pile")
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def _start(self) -> bool:
        """Plan the fall. False (with a report) when nothing can fall."""
        from .select_ops import unmasked

        model = document.model
        self._model = model
        if model.selection.is_empty():
            self.report({"WARNING"}, "Select the voxels to drop first")
            return False
        mask = unmasked(model.selection.mask.copy(), model.uid)  # masked voxels stay up: the rest lands on them
        before = model.grid.voxels
        xs, ys, z0, z1 = grid_ops.settle_moves(before, mask)
        if not (z0 != z1).any():
            self.report({"INFO"}, "Every selected voxel already rests")
            return False
        self._want, self._landed = grid_ops.settle_cells(before, mask)
        self._old_sel = model.selection.mask.copy()
        self._moved = mask & (before != 0)  # the cells that leave
        move = z0 != z1
        self._x, self._y, self._z0 = xs[move], ys[move], z0[move]
        self._fall = (z0 - z1)[move]
        self._colours = before[self._x, self._y, self._z0].copy()
        self._up = self._bounce_room(self._x, self._y, z1[move])
        self._duration = max(fall_seconds(int(f), int(u)) for f, u in set(zip(self._fall.tolist(), self._up.tolist())))
        self._sel_base = self._old_sel & ~self._moved  # what keeps its place in the selection
        self._recorder = StrokeRecorder(model.grid)
        self._shown = None
        return True

    def _bounce_room(self, x, y, z_land) -> np.ndarray:
        """The bounce each moving voxel may make: its share of the fall, but not into the voxel above it
        once everything has landed."""
        occupied = self._want != 0
        size = occupied.shape[2]
        marks = np.where(occupied, np.arange(size, dtype=np.int64)[None, None, :], size)
        nearest = np.minimum.accumulate(marks[:, :, ::-1], axis=2)[:, :, ::-1]  # the lowest filled cell at or above z
        above = np.concatenate([nearest[:, :, 1:], np.full(nearest.shape[:2] + (1,), size, dtype=np.int64)], axis=2)
        room = above[x, y, z_land] - z_land - 1
        return np.minimum(np.array([bounce_height(int(f)) for f in self._fall], dtype=np.int64), room)

    def _offsets(self, t: float) -> np.ndarray:
        """How far each moving voxel is below its start at time t."""
        table = {}
        out = np.empty(len(self._fall), dtype=np.int64)
        key = self._fall * 64 + self._up
        for k in np.unique(key):
            fall, up = int(k) // 64, int(k) % 64
            table[int(k)] = fall_offset(t, fall, up)
        for k, v in table.items():
            out[key == k] = v
        return out

    def _show(self, t: float | None) -> None:
        """The voxels as they are at time t; None: where they come to rest."""
        model = self._model
        self._recorder.revert()
        if t is None:
            self._recorder.set_grid(self._want)
            model.selection.replace((self._old_sel & ~self._moved) | self._landed)
        else:
            drop = self._offsets(t)
            self._recorder.set_many(np.stack([self._x, self._y, self._z0], axis=1), 0)
            now = np.stack([self._x, self._y, self._z0 - drop], axis=1)
            self._recorder.set_many(now, self._colours)
            sel = self._sel_base.copy()
            sel[now[:, 0], now[:, 1], now[:, 2]] = True
            model.selection.replace(sel)
        self._shown = t
        voxel_object.flush_dirty()
        redraw()

    def _finish(self) -> set:
        from rok.core.undo import selection_record

        from .ops_common import commit

        self._show(None)
        model = self._model
        stroke = self._recorder.finish()
        parts = [None if stroke is None else ModelRecord(model.uid, stroke),
                 selection_record(model.uid, self._old_sel, model.selection.mask)]
        commit(stacked_record(*parts), serial=True, hover=True)
        return {"FINISHED"}

    def execute(self, context):
        if not self._start():
            return {"CANCELLED"}
        return self._finish()

    def invoke(self, context, event):
        if not self._start():
            return {"CANCELLED"}
        import time

        self._t0 = time.monotonic()
        self._timer = context.prefs.add_timer(1.0 / 60.0, window=context.window)
        state.stroke_active = True
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        import time

        t = time.monotonic() - self._t0
        skip = event.value == "PRESS" and event.type not in {"TIMER", "MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}
        if event.type == "TIMER" or skip:
            if skip or t >= self._duration:
                return self._end(context)
            self._show(t)
        return {"RUNNING"}

    def _end(self, context) -> set:
        context.prefs.remove_timer(self._timer)
        state.stroke_active = False
        return self._finish()

    def cancel(self, context):
        self._end(context)


@register
class DropToFloorAction(Action):
    """Gravity inside a model: the selected voxels fall straight
    down as one piece until one of them lands on a voxel or on the grid's floor.
    With nothing selected the whole model falls to the floor. From the rail it
    falls with gravity and a small bounce, moving the real voxels each frame;
    a click or a key skips to the end. One undo step either way."""

    command_id = "rok.drop_to_floor"
    title = "Drop to Floor"
    tooltip = ("The selected voxels fall straight down, all together, until one of them lands on a voxel; "
                      "with nothing selected the whole model falls to the floor")
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def _start(self) -> bool:
        """Plan the fall. False (with a report) when nothing can fall."""
        model = document.model
        self._model = model
        self._picked = not model.selection.is_empty()
        mask = model.selection.mask.copy() if self._picked else (model.grid.voxels != 0)
        from .select_ops import unmasked

        mask = unmasked(mask, model.uid)  # masked voxels stay up: the rest lands on them
        self._mask = mask
        self._fall = grid_ops.fall_distance(model.grid.voxels, mask)
        if self._fall == 0:
            if not mask.any():
                self.report({"INFO"}, "The model is empty")
            else:
                self.report({"INFO"}, "Already resting on a voxel" if self._picked else "Already on the floor")
            return False
        self._recorder = StrokeRecorder(model.grid)
        self._shown = 0
        return True

    def _show(self, offset: int) -> None:
        """The voxels `offset` cells below where they started, from the start state."""
        from rok.core.select import move_cells

        if offset == self._shown:
            return
        model = self._model
        self._recorder.revert()
        moving = Selection(model.grid.size)
        moving.replace(self._mask)
        new_mask = move_cells(model.grid.voxels, moving, (0, 0, -offset), False, self._recorder.set_many)
        if self._picked:
            model.selection.replace(new_mask)
        self._shown = offset
        voxel_object.flush_dirty()
        redraw()

    def _finish(self) -> set:
        from rok.core.undo import selection_record

        from .ops_common import commit

        self._show(self._fall)
        model = self._model
        record = self._recorder.finish()
        parts = [None if record is None else ModelRecord(model.uid, record)]
        if self._picked:
            parts.append(selection_record(model.uid, self._mask, model.selection.mask))
        commit(stacked_record(*parts), serial=True, hover=True)
        return {"FINISHED"}

    def execute(self, context):
        if not self._start():
            return {"CANCELLED"}
        return self._finish()

    def invoke(self, context, event):
        if not self._start():
            return {"CANCELLED"}
        import time

        self._t0 = time.monotonic()
        self._timer = context.prefs.add_timer(1.0 / 60.0, window=context.window)
        state.stroke_active = True
        context.prefs.add_modal(self)
        return {"RUNNING"}

    def modal(self, context, event):
        import time

        t = time.monotonic() - self._t0
        skip = event.value == "PRESS" and event.type not in {"TIMER", "MOUSE_MOVE", "MOUSE_MOVE_BETWEEN"}
        if event.type == "TIMER" or skip:
            if skip or t >= fall_seconds(self._fall):
                return self._end(context)
            self._show(fall_offset(t, self._fall))
            return {"RUNNING"}
        return {"RUNNING"}

    def _end(self, context) -> set:
        context.prefs.remove_timer(self._timer)
        state.stroke_active = False
        return self._finish()

    def cancel(self, context):
        self._end(context)


@register
class DiagonalAction(Action):
    command_id = "rok.diagonal"
    title = "Diagonal"
    tooltip = "Mirror across the x = y diagonal, swapping the x and y axes"
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        model = document.model
        new_voxels = grid_ops.diagonal(model.grid.voxels)
        position = (model.position[1], model.position[0], model.position[2])
        _apply_grid_op(context, new_voxels, position, mask_op=grid_ops.diagonal)
        return {"FINISHED"}


@register
class LatheAction(Action):
    command_id = "rok.lathe"
    title = "Lathe"
    tooltip = (
        "Spin what you drew round the grid's centre line into a round solid, like a lathe: "
        "draw a profile on one side of the centre line, then turn it"
    )
    axis = EnumProp(name="Axis", description="The centre line to spin round", items=_AXIS_ITEMS, default="Z")

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        grid = document.grid
        if not grid.voxels.any():
            self.report({"INFO"}, "Draw a profile first: Lathe spins it round the centre line")
            return {"CANCELLED"}
        from .select_ops import mask_of

        axis = "XYZ".index(self.axis)
        new_voxels = reshape.lathe(grid.voxels, axis)
        # A masked cell stays. The spin still reads it, so its colour can
        # land on the rest of the ring. "Already round" looks at the grid
        # after that cell is put back.
        shown = new_voxels
        mask = mask_of()
        if mask is not None and new_voxels.shape == grid.voxels.shape:
            shown = np.where(mask, grid.voxels, new_voxels).astype(np.uint8)
        if np.array_equal(shown, grid.voxels):
            self.report({"INFO"}, "Already round")
            return {"CANCELLED"}
        _apply_grid_op(context, new_voxels)
        return {"FINISHED"}


@register
class TwistAction(Action):
    command_id = "rok.twist"
    title = "Twist"
    tooltip = "Turn each layer a little more than the one below, round the grid's centre line"
    angle = IntProp(name="Angle", description="How far the last layer turns, in degrees",
                       min=-720, max=720, default=90)
    axis = EnumProp(name="Axis", description="The centre line to turn round", items=_AXIS_ITEMS, default="Z")

    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def invoke(self, context, event):
        return context.prefs.open_dialog(self, title="Twist", confirm_text="Twist")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "angle")
        layout.prop(self, "axis")
        layout.label(text="Voxels that turn out of the grid are lost")

    def execute(self, context):
        angle, axis = float(self.angle), "XYZ".index(self.axis)
        new_voxels = reshape.twist(document.grid.voxels, angle, axis)

        def turn(v):
            return reshape.twist(v, angle, axis)

        # A twist samples each cell from one source. Two masked cells can
        # land on one, and the mask count can stay while a colour is gone.
        if _spoils_mask(new_voxels, turn):
            self.report({"ERROR"}, MASK_LOSS)
            return {"CANCELLED"}
        _apply_grid_op(context, new_voxels, mask_op=turn)
        return {"FINISHED"}


@register
class RemoveSliceAction(Action):
    command_id = "rok.remove_slice"
    title = "Remove Slice"
    tooltip = (
        "Take the layers the selection covers out of the model and close the gap: "
        "across the selection's thinnest side, so select a slice and remove it"
    )
    @classmethod
    def poll(cls, context):
        return can_edit_model()

    def execute(self, context):
        mask = _limit_mask()
        found = None if mask is None else reshape.slice_axis(mask)
        if found is None:
            self.report({"INFO"}, "Select the layers to remove first")
            return {"CANCELLED"}
        axis, first, last = found
        try:
            new_voxels = reshape.remove_slices(document.grid.voxels, axis, first, last)
        except ValueError as exc:
            self.report({"ERROR"}, f"Remove Slice: {exc}")
            return {"CANCELLED"}
        if _loses_masked(lambda v: reshape.remove_slices(v, axis, first, last)):
            self.report({"ERROR"}, MASK_LOSS)
            return {"CANCELLED"}
        _apply_grid_op(context, new_voxels, mask_op=lambda v: reshape.remove_slices(v, axis, first, last))
        count = last - first + 1
        self.report({"INFO"}, f"Removed {count} layer{'s' if count != 1 else ''} along {'XYZ'[axis]}")
        return {"FINISHED"}
