"""Helpers the operator modules share: the tests that gate an edit and the tail
that follows one. Nothing here registers; the operator modules import it."""

from __future__ import annotations

from . import object as voxel_object
from .object import document
from .state import state


def can_touch_model() -> bool:
    """A model is open and no stroke is running, not in Animate mode: the
    commands that change no cell (a layer, the active model) may run."""
    return document.grid is not None and not state.stroke_active and not state.animating


def is_brick_model(model=None) -> bool:
    """True when `model` (default: the active one) is built from bricks: its
    cells are locked to everything but Brick mode and the Convert commands."""
    grid = document.grid if model is None else model.grid
    return grid is not None and grid.bricks is not None


def can_edit_model() -> bool:
    """A model is open and no stroke is running: whole-model operations may run.
    Not in Animate mode: it poses, it does not build. Not on a brick model:
    its cells are locked."""
    return can_touch_model() and not is_brick_model()


def can_edit_voxels() -> bool:
    """`can_edit_model` inside a model, in Model level. The scene works on
    whole models: the voxel selection waits inside its model; Brick mode has
    tools of its own."""
    from rok.core.chrome import LEVEL_MODEL

    return can_edit_model() and state.level == LEVEL_MODEL


def can_edit_scene() -> bool:
    """A scene is open and no stroke is running: the scene operators may run.
    Not in Animate mode: it poses, it does not build."""
    return document.scene is not None and not state.stroke_active and not state.animating


def rehover() -> None:
    _late("mode", "rehover")()




def redraw() -> None:
    """Tag every 3D viewport and the status bar."""
    voxel_object.redraw_3d_views()
    voxel_object.redraw_statusbar()


def refresh_views(*, serial: bool = False, sync: bool = False, hover: bool = False) -> None:
    """The tail of a change: refresh the gizmo and redraw.

    `serial` bumps the stroke serial, `sync` brings the objects in line with the
    scene first (a change of the model list, a size or a position), `hover`
    finds the voxel under the mouse again.

    `modified` is not set here, but where the record is pushed: an operation that
    changed nothing pushes no record and must not mark the file unsaved.
    """


    if serial:
        document.stroke_serial += 1
    if sync:
        voxel_object.sync_scene()
    _late("gizmo_ops", "refresh")()
    redraw()
    if hover:
        rehover()


def commit(record, *, modified: bool = True, **views) -> None:
    """Push `record` (nothing when it is None), mark the file changed and refresh
    the views; `views` are the arguments of `refresh_views`.

    `modified=False` leaves the flag alone, for a record that changes no saved data.
    """
    if record is not None:
        document.undo.push(record)
        if modified:
            document.modified = True
    refresh_views(**views)


def _late(module: str, name: str):
    """A function of a module that may import this one (or is not imported yet):
    looked up when called; a missing one does nothing."""
    import importlib

    try:
        mod = importlib.import_module(f"{__package__}.{module}")
    except ImportError:
        return lambda *a, **k: None
    return getattr(mod, name, lambda *a, **k: None)
