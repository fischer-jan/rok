"""Undo and redo on Rok's own stack."""

from __future__ import annotations

from .opkit import Action, register
from rok.core.select import Selection
from rok.core.undo import (
    CompoundRecord,
    ModelRecord,
    PaletteRecord,
    SceneRecord,
    SelectionRecord,
    changes_saved_data,
)
from . import object as voxel_object
from .object import document
from .ops_common import refresh_views
from .state import state


def _has_document() -> bool:
    return document.scene is not None


# The two actions poll True whenever a document is open, and refuse inside
# execute while a stroke runs. The poll must not fail then: a keymap item whose
# poll fails hands Ctrl/Cmd+Z on to the next item. A CANCELLED execute ends
# the key's journey instead, and nothing happens, which is right mid-stroke.


@register
class UndoAction(Action):
    command_id = "rok.undo"
    title = "Undo"
    tooltip = "Undo the last change"
    @classmethod
    def poll(cls, context):
        return _has_document()

    def execute(self, context):
        if state.stroke_active or not document.undo.undo(document.scene):
            return {"CANCELLED"}
        _after_step(undone=True)
        return {"FINISHED"}


@register
class RedoAction(Action):
    command_id = "rok.redo"
    title = "Redo"
    tooltip = "Do the change that was undone again"
    @classmethod
    def poll(cls, context):
        return _has_document()

    def execute(self, context):
        if state.stroke_active or not document.undo.redo(document.scene):
            return {"CANCELLED"}
        _after_step(undone=False)
        return {"FINISHED"}


def _after_step(undone: bool) -> None:
    """Bring the document and the objects in line with the record just applied.

    The active model follows the change so it is on screen. A SelectionRecord
    already restored the mask; a model whose grid changed size and has no
    selection in the record gets a fresh empty one, and every other model loses
    selected cells that are empty now.
    """
    record = document.undo.last_applied
    if isinstance(record, SceneRecord):
        document.active_uid = record.active_before if undone else record.active_after
        inner = record.inner
        _mark_returned_models(record)
    elif isinstance(record, ModelRecord):
        document.active_uid = record.uid
        inner = record.inner
    elif isinstance(record, SelectionRecord):
        document.active_uid = record.uid
        inner = None
    elif isinstance(record, CompoundRecord):
        model_part = record.first(ModelRecord)
        if model_part is not None:
            document.active_uid = model_part.uid
        else:
            sel_part = record.first(SelectionRecord)
            if sel_part is not None:
                document.active_uid = sel_part.uid
        inner = record.first(PaletteRecord)
    else:
        inner = record
    if isinstance(record, PaletteRecord) or isinstance(inner, PaletteRecord):
        voxel_object.update_palette()
    document.pivot_override = None
    voxel_object.sync_scene()  # Empties, deleted objects, dirty chunks
    scene = document.scene
    if scene is not None:
        restored = _selection_uids(record)
        for model in scene.models:
            if model.selection.size != model.grid.size:
                model.selection = Selection(model.grid.size)
            elif model.uid not in restored:
                model.selection.prune(model.grid.visible)
    document.mask_serial += 1  # the step may have changed a mask (MaskRecord) or the grid under it
    if changes_saved_data(record):
        document.modified = True  # a selection-only step leaves a clean file clean
    state.hover = None
    state.preview = None
    refresh_views(serial=True)


def _selection_uids(record) -> set[int]:
    """Model uids whose selection this step already restored."""
    if isinstance(record, SelectionRecord):
        return {record.uid}
    if isinstance(record, CompoundRecord):
        return {part.uid for part in record.parts if isinstance(part, SelectionRecord)}
    return set()


def _mark_returned_models(record: SceneRecord) -> None:
    """A model that a SceneRecord step put back has no chunk objects: mark it all dirty."""
    before = {m.uid for m in record.before}
    after = {m.uid for m in record.after}
    scene = document.scene
    if scene is None:
        return
    for model in scene.models:
        if model.uid in before ^ after:
            model.grid.mark_all_dirty()
