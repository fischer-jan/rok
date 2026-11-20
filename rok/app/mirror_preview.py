"""Mirror Half shows its result while the mouse rests on a choice.

The preview writes the mirrored voxels into the model, with no undo record and
no unsaved mark, so it shows the real voxels. It is taken back when the mouse
leaves the choice and before any operator runs: a click on the choice then
runs Mirror Half on the model as it was, and that is one undo step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rok.core import chrome as kit
from rok.core import grid_ops, morph

from . import object as voxel_object
from . import opkit
from .handlers import registry
from .object import document
from .ops_common import can_edit_model, refresh_views


@dataclass
class _Shown:
    widget_id: str
    uid: int
    cells: np.ndarray  # the cells the preview changed
    before: np.ndarray  # their values before it


_shown: _Shown | None = None


def _mirror_choice(widget_id: str | None):
    """(axis 0..2, from_positive) of a Mirror Half choice, else None."""
    choice = kit.CHOICE_OF_ID.get(widget_id) if widget_id is not None else None
    if choice is None or choice.op != "mirror_half":
        return None
    props = dict(choice.props)
    return "XYZ".index(props["axis"]), bool(props["from_positive"])


def shown_axis() -> int | None:
    """The axis of the preview on the model, for the overlay's mirror plane."""
    if _shown is None:
        return None
    return _mirror_choice(_shown.widget_id)[0]


def hover(widget_id: str | None) -> None:
    """The rail's hovered widget changed: show its result, or take the last one back."""
    if _shown is not None and _shown.widget_id == widget_id:
        return
    restore()
    found = _mirror_choice(widget_id)
    if found is None or not can_edit_model():
        return
    show(widget_id, *found)


def show(widget_id: str, axis: int, from_positive: bool) -> None:
    """Write the mirrored model into the grid, as the operator would."""
    global _shown
    grid = document.grid
    new = grid_ops.mirror_half(grid.voxels, axis, from_positive)
    cells, values = morph.changed_cells(grid.voxels, new)
    # With no change it still counts as shown: the plane says which axis it is.
    before = grid.voxels[tuple(cells.T)].copy()
    if len(cells):
        grid.set_voxels(cells, values)  # a cut or masked cell stays, as in the operator
    _shown = _Shown(widget_id, document.active_uid, cells, before)
    voxel_object.flush_dirty()
    refresh_views(serial=True)


def restore() -> None:
    """Put the model back as it was before the preview."""
    global _shown
    shown, _shown = _shown, None
    if shown is None:
        return
    grid = document.grid
    if grid is None or document.active_uid != shown.uid:
        return
    if len(shown.cells):
        grid.set_voxels(shown.cells, shown.before)
    voxel_object.flush_dirty()
    refresh_views(serial=True)


def _before_operator(_idname: str) -> None:
    restore()


def _on_load(_context) -> None:
    global _shown
    _shown = None  # a new file: the cells belong to a model that is gone


opkit.before.append(_before_operator)
registry.load_post.append(_on_load)
