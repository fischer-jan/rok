"""Selection operators: all, deselect, delete, fill, by colour, connected, invert,
grow and shrink, and the section plane.

Every one of them reads `grid.visible`: a voxel beyond a section plane is not there for the tools.
Writes go through a StrokeRecorder on `grid.voxels`. Selection changes push a
SelectionRecord (or a CompoundRecord with the voxel write).
"""

from __future__ import annotations

import time

import numpy as np

from rok.core import edit, morph, region, select
from rok.core.undo import ModelRecord, StrokeRecorder, selection_record, stacked_record

from . import object as voxel_object
from .handlers import registry
from .object import document
from . import opkit
from .opkit import EnumProp, IntProp, Action, register
from .ops_common import can_edit_voxels, refresh_views, rehover
from .state import state


def _has_selection() -> bool:
    return can_edit_voxels() and document.selection is not None and not document.selection.is_empty()


def _voxel_hover():
    """The hover when it sits on a voxel of the active model, else None."""
    hover = state.hover
    if hover is None or hover.hit is None or hover.hit.wall:
        return None
    if hover.tool == edit.ACTIVATE:
        return None  # the hit is in another model's space
    return hover


def _push_selection(before: np.ndarray) -> None:
    """Push a SelectionRecord when the active mask differs from `before`."""
    sel = document.selection
    if sel is None:
        return
    record = selection_record(document.active_uid, before, sel.mask)
    if record is not None:
        document.undo.push(record)


def _after_voxel_change() -> None:
    voxel_object.flush_dirty()
    state.hover = None
    state.preview = None
    refresh_views()


@register
class SelectAllAction(Action):
    command_id = "rok.select_all"
    title = "Select All"
    tooltip = "Select every voxel"
    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        grid = document.grid
        before = document.selection.mask.copy()
        document.selection.replace(grid.visible != 0)
        _push_selection(before)
        refresh_views()
        return {"FINISHED"}


# --- saved selections: Ctrl+1..4 saves, 1..4 brings one back ---------


def _fit_mask(mask: np.ndarray, size) -> np.ndarray:
    """A saved mask cut or padded to the grid's size now (the grid may have been resized)."""
    out = np.zeros(tuple(size), dtype=bool)
    sx, sy, sz = (min(a, b) for a, b in zip(mask.shape, size))
    out[:sx, :sy, :sz] = mask[:sx, :sy, :sz]
    return out


@register
class SelectionSaveAction(Action):
    command_id = "rok.selection_save"
    title = "Save Selection"
    tooltip = "Keep the selection in a slot; the slot's number key brings it back"

    slot = IntProp(name="Slot", description="Which of the four saved selections", default=1, min=1, max=4)

    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        selection = document.selection
        if selection is None or selection.is_empty():
            self.report({"INFO"}, "Select something first")
            return {"CANCELLED"}
        document.saved_selections[(document.active_uid, self.slot)] = selection.mask.copy()
        n = selection.count()
        self.report({"INFO"}, f"Selection {self.slot} saved: {n} voxel{'s' if n != 1 else ''} · {self.slot} brings it back")
        return {"FINISHED"}


@register
class SelectionRecallAction(Action):
    command_id = "rok.selection_recall"
    title = "Recall Selection"
    tooltip = "Select what a slot holds again (Ctrl and the number saves one)"

    slot = IntProp(name="Slot", description="Which of the four saved selections", default=1, min=1, max=4)

    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        saved = document.saved_selections.get((document.active_uid, self.slot))
        if saved is None:
            self.report({"INFO"}, f"Selection {self.slot} is empty · Ctrl+{self.slot} saves one")
            return {"CANCELLED"}
        grid = document.grid
        before = document.selection.mask.copy()
        document.selection.replace(_fit_mask(saved, grid.size) & (grid.visible != 0))
        _push_selection(before)
        refresh_views()
        return {"FINISHED"}


# --- the mask: voxels nothing changes until it is cleared ---------------
#
# The mask is the grid's lock (VoxelGrid.lock): the grid itself refuses every
# write to a masked cell, so no command can change one (the
# commands that wrote past the tools deleted masked voxels). Masking and
# clearing are undo steps.


def mask_of(uid: int | None = None) -> np.ndarray | None:
    """The active model's mask (or `uid`'s), fitted to its grid; None when nothing is masked."""
    scene = document.scene
    uid = document.active_uid if uid is None else uid
    if scene is None:
        return None
    try:
        grid = scene.by_uid(uid).grid
    except ValueError:
        return None
    mask = grid.lock
    if mask is None:
        return None
    if mask.shape != tuple(grid.size):
        # A change of size that did not take the mask along: keep it at the origin.
        mask = _fit_mask(mask, grid.size)
        grid.lock = mask
    return mask if mask.any() else None


def carried_mask(model, change):
    """Take `model`'s mask through a whole-grid change `change` (voxels ->
    voxels, the same one its voxels went through). Returns the MaskRecord,
    not pushed, or None when nothing is masked."""
    mask = mask_of(model.uid)
    if mask is None:
        return None
    return set_mask(np.asarray(change(mask.astype(np.uint8))) != 0, model.uid)


def unmasked(cells: np.ndarray, uid: int | None = None) -> np.ndarray:
    """`cells` (a bool array of the grid's shape) without the masked ones:
    the voxels a move may take along; masked ones stay where they are."""
    mask = mask_of(uid)
    return cells if mask is None else cells & ~mask


def set_mask(mask: np.ndarray | None, uid: int | None = None):
    """Make `mask` the model's mask (None: no mask). Returns the MaskRecord, not pushed."""
    from rok.core.undo import mask_record

    uid = document.active_uid if uid is None else uid
    grid = document.scene.by_uid(uid).grid
    before = mask_of(uid)
    after = None if mask is None or not np.any(mask) else np.asarray(mask, dtype=bool).copy()
    grid.lock = after
    document.mask_serial += 1
    return mask_record(uid, before, after, grid.size)


@register
class MaskSelectionAction(Action):
    command_id = "rok.mask_selection"
    title = "Mask Selection"
    tooltip = "Protect the selected voxels: nothing changes them until the mask is cleared"

    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        selection = document.selection
        if selection is None or selection.is_empty():
            self.report({"INFO"}, "Select something first")
            return {"CANCELLED"}
        before = selection.mask.copy()
        record = set_mask(before & (document.grid.voxels != 0))
        selection.clear()  # the mask is not a selection: Paint would stay inside it
        document.undo.push(stacked_record(record, selection_record(document.active_uid, before, selection.mask)))
        n = int(before.sum())
        self.report({"INFO"}, f"Masked {n} voxel{'s' if n != 1 else ''} · Option+M clears the mask")
        refresh_views()
        return {"FINISHED"}


@register
class MaskClearAction(Action):
    command_id = "rok.mask_clear"
    title = "Clear Mask"
    tooltip = "Let every tool and command reach the masked voxels again"

    @classmethod
    def poll(cls, context):
        return document.grid is not None

    def execute(self, context):
        if mask_of() is None:
            return {"CANCELLED"}
        document.undo.push(set_mask(None))
        self.report({"INFO"}, "Mask cleared")
        refresh_views()
        return {"FINISHED"}


last_deselect = 0.0  # time.monotonic() of the last Deselect


@register
class DeselectAction(Action):
    command_id = "rok.deselect"
    title = "Deselect"
    tooltip = "Clear the selection"
    @classmethod
    def poll(cls, context):
        # Needs a selection, so with nothing selected Esc goes on to the next key item.
        return _has_selection()

    def execute(self, context):
        global last_deselect
        before = document.selection.mask.copy()
        document.selection.clear()
        _push_selection(before)
        refresh_views()
        last_deselect = time.monotonic()  # Esc's step out to the scene waits a moment after this
        return {"FINISHED"}


@register
class SelectionDeleteAction(Action):
    command_id = "rok.selection_delete"
    title = "Delete Selection"
    tooltip = "Remove the selected voxels"
    @classmethod
    def poll(cls, context):
        # Polls True with an empty selection too, so Delete is always taken here.
        return can_edit_voxels()

    def execute(self, context):
        grid = document.grid
        selection = document.selection
        if selection is None or selection.is_empty():
            return {"CANCELLED"}
        before = selection.mask.copy()
        recorder = StrokeRecorder(grid)
        select.delete_cells(grid.voxels, selection, recorder.set_many)
        record = recorder.finish()
        selection.clear()
        parts = []
        if record is not None:
            parts.append(ModelRecord(document.active_uid, record))
            document.modified = True
        sel_record = selection_record(document.active_uid, before, selection.mask)
        if sel_record is not None:
            parts.append(sel_record)
        stacked = stacked_record(*parts)
        if stacked is not None:
            document.undo.push(stacked)
        document.stroke_serial += 1
        voxel_object.flush_dirty()
        state.hover = None
        refresh_views()
        return {"FINISHED"}


@register
class PaintSelectionAction(Action):
    command_id = "rok.paint_selection"
    title = "Paint Selection"
    tooltip = "Give every selected voxel the current colour"

    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        grid = document.grid
        selection = document.selection
        if selection is None or selection.is_empty():
            self.report({"INFO"}, "Select something first")
            return {"CANCELLED"}
        # The selected cells that hold a voxel and are in view: no mirror, the
        # selection says exactly which voxels.
        take = selection.mask & (grid.visible != 0)
        mask = mask_of()
        if mask is not None:
            take &= ~mask  # masked voxels keep their colour
        cells = np.argwhere(take)
        recorder = StrokeRecorder(grid)
        recorder.set_many(cells, np.full(len(cells), document.colour, dtype=np.uint8))
        record = recorder.finish()
        if record is None:
            return {"CANCELLED"}
        from . import palette_ui

        palette_ui.note_used(document.colour)  # the star's recent colours
        document.undo.push(ModelRecord(document.active_uid, record))
        document.modified = True
        document.stroke_serial += 1
        voxel_object.flush_dirty()
        refresh_views(hover=True)
        return {"FINISHED"}


@register
class FillAction(Action):
    command_id = "rok.fill"
    title = "Fill Connected Colour"
    tooltip = "Paint every joined voxel of the colour under the mouse"
    @classmethod
    def poll(cls, context):
        return can_edit_voxels() and _voxel_hover() is not None

    def execute(self, context):
        grid = document.grid
        hover = _voxel_hover()
        if hover is None:
            return {"CANCELLED"}
        mask = region.connected_colour(grid.visible, hover.hit.voxel)
        cells = np.argwhere(mask)
        mirror = tuple(getattr(context.prefs, "rok_mirror", (False, False, False)))
        recorder = StrokeRecorder(grid)
        edit.apply_cells(grid.voxels, edit.PAINT, document.colour, cells, recorder.set_many, mirror)
        record = recorder.finish()
        if record is not None:
            document.undo.push(ModelRecord(document.active_uid, record))
            document.modified = True
        document.stroke_serial += 1
        voxel_object.flush_dirty()
        if document.selection is not None:
            document.selection.prune(grid.visible)
        refresh_views(hover=True)
        return {"FINISHED"}


# ---------------------------------------------------------------- by colour, connected, invert


@register
class SelectColourAction(Action):
    command_id = "rok.select_colour"
    title = "Select by Colour"
    tooltip = "Add every voxel of the same colour to the selection"
    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        grid = document.grid
        hover = _voxel_hover()
        if hover is not None:
            x, y, z = hover.hit.voxel
            index = int(grid.visible[x, y, z])
        else:
            index = int(document.colour)
        if index == 0:
            return {"CANCELLED"}
        before = document.selection.mask.copy()
        document.selection.add_mask(grid.visible == index)
        _push_selection(before)
        refresh_views()
        return {"FINISHED"}


@register
class SelectConnectedAction(Action):
    command_id = "rok.select_connected"
    title = "Select Connected"
    tooltip = "Add every voxel joined to the one under the mouse, or to the selection"
    @classmethod
    def poll(cls, context):
        # Polls True without a hover too: the menu entry grows from the selection.
        return can_edit_voxels()

    def execute(self, context):
        grid = document.grid
        selection = document.selection
        hover = _voxel_hover()
        if hover is not None:
            seeds = np.asarray([hover.hit.voxel], dtype=np.int64)
        else:
            seeds = selection.cells()  # the menu entry: the mouse is off the model
        if len(seeds) == 0:
            return {"CANCELLED"}
        before = selection.mask.copy()
        selection.add_mask(region.connected_blobs(grid.visible, seeds))
        _push_selection(before)
        refresh_views()
        return {"FINISHED"}


@register
class SelectInvertAction(Action):
    command_id = "rok.select_invert"
    title = "Invert Selection"
    tooltip = "Select the unselected voxels and drop the selected ones"
    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        grid = document.grid
        selection = document.selection
        before = selection.mask.copy()
        selection.replace(select.invert_mask(grid.visible, selection.mask))
        _push_selection(before)
        refresh_views()
        return {"FINISHED"}


@register
class SelectGrowAction(Action):
    command_id = "rok.select_grow"
    title = "Grow Selection"
    tooltip = "Add the filled voxels next to the selection"
    @classmethod
    def poll(cls, context):
        return _has_selection()

    def execute(self, context):
        grid = document.grid
        selection = document.selection
        before = selection.mask.copy()
        selection.add_mask(morph.dilate(selection.mask, 1) & (grid.visible != 0))
        _push_selection(before)
        refresh_views()
        return {"FINISHED"}


@register
class SelectShrinkAction(Action):
    command_id = "rok.select_shrink"
    title = "Shrink Selection"
    tooltip = "Drop the selected voxels that touch an unselected voxel"
    @classmethod
    def poll(cls, context):
        return _has_selection()

    def execute(self, context):
        grid = document.grid
        selection = document.selection
        # An empty neighbour does not shrink the selection; only an unselected filled one
        # does. The grid wall is padded True so it does not count as one either.
        before = selection.mask.copy()
        soft = np.pad(selection.mask | (grid.visible == 0), 1, constant_values=True)
        keep = morph.erode(soft, 1)[1:-1, 1:-1, 1:-1]
        selection.replace(selection.mask & keep)
        _push_selection(before)
        refresh_views()
        return {"FINISHED"}


def _view_direction(context):
    """The direction the view looks along, or None without one."""
    view_state = getattr(context, "view_state", None)
    if view_state is None:
        return None
    return tuple(float(v) for v in view_state.view_direction())


@register
class SectionAction(Action):
    command_id = "rok.section"
    title = "Section Plane"
    tooltip = (
        "Cut the view with a plane across an axis: voxels beyond it are out of view "
        "and take no write. The side facing the view is cut. Cmd+Up and Cmd+Down move "
        "the plane last touched"
    )
    axis = EnumProp(
        name="Axis",
        description="Which plane",
        items=(
            ("X", "X", "The plane across x"),
            ("Y", "Y", "The plane across y"),
            ("Z", "Z", "The plane across z"),
            ("CURRENT", "Current", "The plane the keys move: the last one touched"),
        ),
        default="CURRENT",
    )
    action = EnumProp(
        name="Action",
        description="What to do with the plane",
        items=(
            ("TOGGLE", "Toggle", "Put the plane at the middle, or take it away"),
            ("DOWN", "Down", "Cut one layer more; from off, start one layer in"),
            ("UP", "Up", "Cut one layer less; at the edge the plane goes away"),
            ("SET", "Set", "Put the plane at `coord`, kept inside the grid (the nav cube's slide)"),
        ),
        default="TOGGLE",
    )
    coord = IntProp(name="Layer", description="Where SET puts the plane", default=1)

    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        grid = document.grid
        axis = grid.section_axis if self.axis == "CURRENT" else "XYZ".index(self.axis)
        size = grid.size[axis]
        current = grid.sections.get(axis)
        if current is None:
            # A new plane cuts the side that faces the view, so the inside shows.
            view = _view_direction(context)
            from_positive = view is None or view[axis] <= 0.0
        else:
            from_positive = current[1]
        if self.action == "SET":
            coord = min(max(int(self.coord), 1), size - 1)
        elif self.action == "TOGGLE":
            coord = None if current is not None else max(1, size // 2)
        elif current is None:
            if self.action == "UP":
                return {"CANCELLED"}
            coord = size - 1 if from_positive else 1  # one layer in
        else:
            more = self.action == "DOWN"
            step = -1 if more == from_positive else 1
            coord = current[0] + step
            if coord < 1 or coord >= size:
                coord = None  # the plane left the grid: off
        if current is not None and coord == current[0]:
            return {"CANCELLED"}
        grid.set_section(coord, axis, from_positive)
        if document.selection is not None:
            document.selection.prune(grid.visible)
        _after_voxel_change()
        rehover()
        _update_menu(context)  # the View menu ticks the planes that are on
        return {"FINISHED"}


def _update_menu(context) -> None:
    app = getattr(context, "app", None)
    if app is not None:
        app.set_menu()


registry.load_post.append(_update_menu)  # a new file has its own section planes


def _note_locked(_idname, _props) -> None:
    """After a command: when the mask stopped some of its writes, say so."""
    scene = document.scene
    if scene is None:
        return
    stopped = 0
    for model in scene.models:
        stopped += model.grid.locked_writes
        model.grid.locked_writes = 0
    if stopped:
        from . import tools

        tools._note(tools.MASK_NOTE)


opkit.finished.append(_note_locked)


def _forget_saved_selections(_context) -> None:
    """New and Open: the saved selections and the masks were the last file's (its model ids come again)."""
    document.saved_selections = {}
    document.mask_serial += 1  # the new file's grids have no lock


registry.load_post.append(_forget_saved_selections)
