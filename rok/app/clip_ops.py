"""Copy, cut and paste of the selected voxels.

The clip lives in `state.clipboard` and survives New and Open. A paste lands at
the clip's own origin in the active model (moved inside the grid when it does
not fit), selects the pasted voxels and switches to Select, so the gizmo can
move them. A clip from another file brings its own palette and materials; the
colours are fitted in as Add From File does. One undo step takes it all back.
"""

from __future__ import annotations

import numpy as np

from rok.core import clipboard, edit
from rok.core import palette as palette_core
from rok.core.undo import ModelRecord, StrokeRecorder, selection_record, stacked_record

from . import object as voxel_object
from . import opkit
from .object import document
from .opkit import CONTEXT, Action, BoolProp, register
from .ops_common import can_edit_voxels, refresh_views
from .state import state


def _copy(report) -> bool:
    """Put the selection on the clipboard. False (with a report) when nothing is selected."""
    scene = document.scene
    grid = document.grid
    clip = clipboard.copy_selection(grid.visible, document.selection, scene.palette, scene.materials)
    if clip is None:
        report({"WARNING"}, "Select something first")
        return False
    state.clipboard = clip
    report({"INFO"}, f"Copied {clip.count()} voxel{'s' if clip.count() != 1 else ''}")
    return True


@register
class CopyAction(Action):
    command_id = "rok.copy"
    title = "Copy"
    tooltip = "Copy the selected voxels"
    @classmethod
    def poll(cls, context):
        # Polls True with an empty selection too, so Cmd+C is always taken here.
        return can_edit_voxels()

    def execute(self, context):
        return {"FINISHED"} if _copy(self.report) else {"CANCELLED"}


@register
class CutAction(Action):
    command_id = "rok.cut"
    title = "Cut"
    tooltip = "Copy the selected voxels and remove them"
    @classmethod
    def poll(cls, context):
        return can_edit_voxels()

    def execute(self, context):
        if not _copy(self.report):
            return {"CANCELLED"}
        return opkit.ops.rok.selection_delete()


def clip_short() -> int:
    """How many colours of the clip would find no free place and take the
    nearest colour you have. Changes nothing."""
    scene = document.scene
    clip = state.clipboard
    merged, remap, _claimed_by = palette_core.merge_claimed(
        scene.palette, document.used_counts(), clip.palette, clip.used_counts(),
        protect=document.colour, keep=document.made_unused(), materials=scene.materials,
        incoming_materials=clip.materials,
    )
    counts = clip.used_counts()
    return sum(1 for i in range(1, 256)
               if counts[i] and not np.array_equal(merged[remap[i]], clip.palette[i]))


def fit_clip_palette():
    """Fit the clip's colours into the scene's colours. Returns (remap, changed).

    Shared with the pattern brush. Nothing changes when every used colour
    already sits in the palette (a clip from this file). The change travels
    with the next undo record (UndoStack.track)."""
    scene = document.scene
    clip = state.clipboard
    merged, remap, claimed_by = palette_core.merge_claimed(
        scene.palette, document.used_counts(), clip.palette, clip.used_counts(),
        protect=document.colour, keep=document.made_unused(), materials=scene.materials,
        incoming_materials=clip.materials,
    )
    if not claimed_by.any():
        return remap, False
    voxel_object.merge_colours(merged, claimed_by, clip.materials)
    return remap, True


@register
class PasteAction(Action):
    command_id = "rok.paste"
    title = "Paste"
    tooltip = "Paste the copied voxels into the active model and select them"
    merge = BoolProp(
        name="Use Nearest Colours",
        description="Colours of the copy without a free place in the palette take the nearest colour you have",
        default=False, options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        # Polls True with an empty clipboard too, so Cmd+V is always taken here.
        return can_edit_voxels()

    def execute(self, context):
        clip = state.clipboard
        if clip is None:
            self.report({"WARNING"}, "Nothing to paste")
            return {"CANCELLED"}
        short = clip_short()
        if short and not self.merge and CONTEXT.app is not None:
            self.merge = True  # the answer runs this action again, merging
            return context.prefs.ask_confirm(
                self, None, title="The palette is full",
                message=f"{short} colour{'s' if short != 1 else ''} of the copy "
                        f"{'have' if short != 1 else 'has'} no free place in the palette. "
                        "Use the nearest colours you have instead?", confirm_text="Use Nearest")
        grid = document.grid
        sel_before = document.selection.mask.copy() if document.selection is not None else None
        remap, colours_changed = fit_clip_palette()
        origin = clipboard.paste_origin(clip.size, clip.origin, grid.size)
        recorder = StrokeRecorder(grid)
        mask = clipboard.paste(grid.voxels, clip, origin, recorder.set_many, remap)
        stroke = recorder.finish()

        document.selection.replace(mask)
        parts = []
        if stroke is not None:
            parts.append(ModelRecord(document.active_uid, stroke))
        if sel_before is not None:
            sel_record = selection_record(document.active_uid, sel_before, document.selection.mask)
            if sel_record is not None:
                parts.append(sel_record)
        stacked = stacked_record(*parts)
        if stacked is not None:
            document.undo.push(stacked)  # carries the colours the clip brought in
        else:
            document.undo.push_colours()
        if colours_changed or stroke is not None:
            document.modified = True
        try:
            context.prefs.rok_tool = edit.SELECT
        except AttributeError:
            pass  # the action property is not registered (headless check)
        document.stroke_serial += 1
        voxel_object.flush_dirty()
        state.hover = None
        refresh_views()

        pasted = int(np.count_nonzero(mask))
        message = f"Pasted {pasted} voxel{'s' if pasted != 1 else ''}"
        if pasted < clip.count():
            message += f", {clip.count() - pasted} outside the grid"
        self.report({"INFO"}, message)
        return {"FINISHED"}
