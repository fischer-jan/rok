"""The cage: a lattice of points around the selection that bends the real voxels.

A Model menu command (`rok.cage`), not a rail tool. Running it puts the cage on
the selection, or on every voxel when nothing is selected. It then works beside
whatever tool is active, as the corner gizmo does: a press on a point is the
cage's (`tools.py` asks `overlay.cage_hit` first), a press anywhere else is the
tool's. The maths is rok/core/lattice.py, the drag rok/core/cage_drag.py.

`CageState` keeps the points, the voxels and the selection as they were when the
cage was placed (the rest) and what the overlay drew last, for the hit test.
`refresh()` runs where `gizmo_ops.refresh()` runs: the rest goes stale when the
selection changes, when another edit changes the grid (`document.stroke_serial`
moved: a stroke, undo, redo), or when the model changes; then the cage resets
(rest := the voxels now, points := the box). A bent cage also starts again when
the Mirror toggles change which mirrored copies it bends, so the other side keeps
its bend. It hides on Done, Return, Esc and
when the mode leaves Model. The Reset chip is an undo of the whole cage session:
the voxels and the selection go back to what they were when the cage first
appeared (`origin_voxels`, `origin_sel`), as one undo step, and the cage starts
over with the default points (3 on every axis, Soft off). Imports object, state and the core modules only,
never tools, mode or overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rok.core import cage_chips, lattice
from rok.core.cage_drag import CageDrag, CageRest, ScreenBasis, mirror_copies

from . import object as voxel_object
from . import opkit
from .object import document
from .opkit import IntProp, Action, register
from .ops_common import can_edit_voxels, commit
from .state import state

DEFAULT_COUNT = 3


@dataclass
class CageState(CageRest):
    """The cage on screen: the rest, and what the overlay drew last."""

    counts: tuple[int, int, int] = (DEFAULT_COUNT,) * 3
    whole: bool = False  # nothing was selected: the cage acts on every voxel
    uid: int = 0  # the model it belongs to
    serial: int = 0  # document.stroke_serial the rest was taken at
    origin_voxels: np.ndarray | None = None  # the grid when the cage first appeared; Reset goes back to it
    origin_sel: np.ndarray | None = None  # the selection mask then, or None when there was no selection
    locks: tuple[bool, bool, bool] = (False, False, False)  # Lock X, Y, Z: a point does not move along a locked axis
    sel_version: int = 0  # the selection's version the rest was taken at
    copies: tuple = ()  # Mirror's mirrored copies (cage_drag.mirror_copies) the rest was taken with
    # (region pointer, ids, projected points (N, 2), depth (N,)) of the last drawn frame, for the hit test
    layout: tuple | None = None
    # (region pointer, ChipLayout) of the last drawn chip row
    chips: tuple | None = None
    lines: tuple | None = None  # (lattice object, (2N, 3) segment points in grid space): the overlay's cache


def _scope():
    """(mask, bounds, whole) the cage would act on, or None when there is nothing to bend."""
    grid = document.grid
    sel = document.selection
    if grid is None:
        return None
    if sel is not None and not sel.is_empty():
        return sel.mask.copy(), sel.bounds(), False
    mask = grid.visible != 0
    cells = np.argwhere(mask)
    if not len(cells):
        return None
    return mask, (tuple(int(v) for v in cells.min(axis=0)), tuple(int(v) for v in cells.max(axis=0))), True


def _mirror() -> tuple[bool, bool, bool]:
    return tuple(bool(v) for v in getattr(opkit.CONTEXT.prefs, "rok_mirror", (False, False, False)))


def _bent(cage: CageState) -> bool:
    """The points are off the box: a drag has bent the voxels."""
    lat = cage.lattice
    box = (lat.lo, tuple(l + s - 1 for l, s in zip(lat.lo, lat.size)))
    return not np.allclose(lat.points, lattice.rest_lattice(box, lat.counts, lat.bernstein).points)


def place(count: int | tuple = DEFAULT_COUNT, bernstein: bool | None = None, keep_origin: bool = False,
          locks: tuple | None = None) -> bool:
    """Show the cage on the selection (or the whole model) with `count` points per axis.
    `bernstein` picks the full-degree basis; None keeps what the cage shown now has.
    `keep_origin` keeps what Reset goes back to (a new point count is still the same
    session); otherwise the voxels and the selection now are the new origin.
    `locks` are the axis locks; None keeps the ones the cage shown now has."""
    found = _scope()
    model = document.model
    if found is None or model is None:
        state.cage = None
        return False
    mask, bounds, whole = found
    counts = lattice._counts(count)
    sel = document.selection
    old = state.cage
    if bernstein is None:
        bernstein = old is not None and old.lattice.bernstein
    keep = (keep_origin and old is not None and old.origin_voxels is not None and old.uid == model.uid
            and old.origin_voxels.shape == document.grid.visible.shape)
    rest = lattice.rest_lattice(bounds, counts, bool(bernstein))
    state.cage = CageState(
        origin_voxels=old.origin_voxels if keep else document.grid.visible.copy(),
        origin_sel=old.origin_sel if keep else (sel.mask.copy() if sel is not None else None),
        locks=tuple(locks) if locks is not None else (old.locks if old is not None else (False, False, False)),
        lattice=rest,
        copies=mirror_copies(rest, _mirror(), document.grid.visible.shape),
        rest_voxels=document.grid.visible.copy(),
        rest_mask=mask,
        counts=counts,
        whole=whole,
        uid=model.uid,
        serial=document.stroke_serial,
        sel_version=sel.version if sel is not None else 0,
    )
    voxel_object.redraw_3d_views()
    return True


def restart() -> bool:
    """Points back on the box, the voxels as they are now the rest and the new origin."""
    cage = state.cage
    return place(cage.counts if cage is not None else DEFAULT_COUNT)


def reset() -> bool:
    """Undo the whole cage session: the voxels and the selection go back to what
    they were when the cage appeared (one undo step), and the cage starts over with
    the default points."""
    cage = state.cage
    if cage is not None:
        _restore_origin(cage)
    return place(DEFAULT_COUNT, False, locks=(False, False, False))


def _restore_origin(cage: CageState) -> None:
    from rok.core.undo import ModelRecord, StrokeRecorder, selection_record, stacked_record

    grid = document.grid
    model = document.model
    if (cage.origin_voxels is None or grid is None or model is None or cage.uid != model.uid
            or cage.origin_voxels.shape != grid.visible.shape):
        return
    recorder = StrokeRecorder(grid)
    recorder.set_grid(cage.origin_voxels)
    stroke = recorder.finish()
    parts = [ModelRecord(document.active_uid, stroke)] if stroke is not None else []
    sel = document.selection
    if sel is not None and cage.origin_sel is not None:
        before = sel.mask.copy()
        sel.replace(cage.origin_sel)
        parts.append(selection_record(document.active_uid, before, sel.mask))
    voxel_object.flush_dirty()
    record = stacked_record(*parts)
    document.stroke_serial += 1
    commit(record, serial=False)


def hide() -> None:
    if state.cage is not None:
        state.cage = None
        voxel_object.redraw_3d_views()


def shown() -> bool:
    return state.cage is not None


def adopt() -> None:
    """The cage's own drag just changed the grid and the selection: they are not
    stale, so the rest keeps its meaning."""
    cage = state.cage
    if cage is None:
        return
    sel = document.selection
    cage.serial = document.stroke_serial
    cage.sel_version = sel.version if sel is not None else 0


def refresh() -> None:
    """Drop or reset the cage when what it was placed on has changed."""
    cage = state.cage
    if cage is None or cage.live is not None:
        return  # nothing shown, or the cage's own drag is running
    model = document.model
    if (model is None or document.grid is None or state.level != "MODEL" or state.animating
            or model.uid != cage.uid or document.grid.visible.shape != cage.rest_voxels.shape):
        hide()
        return
    sel = document.selection
    if document.stroke_serial != cage.serial or (sel.version if sel is not None else 0) != cage.sel_version:
        if not restart():
            hide()
        return
    copies = mirror_copies(cage.lattice, _mirror(), cage.rest_voxels.shape)
    if copies != cage.copies:
        # Every drag bends from the rest, so a mirrored copy that Mirror no longer asks for
        # would go back to its rest shape: a bent cage starts again from the voxels as they
        # are now, in the same session for Reset.
        if _bent(cage):
            place(cage.counts, keep_origin=True)
        else:
            cage.copies = copies


def points_world() -> np.ndarray | None:
    """(N, 3) the points in world space, in `point_ids` order, as drawn now."""
    cage = state.cage
    model = document.model
    if cage is None or model is None:
        return None
    lat = cage.live or cage.lattice
    return lat.points.reshape(-1, 3) + np.asarray(voxel_object.model_origin(model), dtype=np.float64)


def screen_basis(region, view_state, world, index) -> tuple[ScreenBasis, dict] | None:
    """The screen's plane at a point for a planar drag, and one cell along each
    world axis at the point in region pixels. None when the point is not on screen."""
    from .view3d import world_to_region

    p0 = world_to_region(region, view_state, world)
    if p0 is None:
        return None
    rot = view_state.axes_matrix()
    right, up = rot[:, 0].copy(), rot[:, 1].copy()

    def px(step) -> np.ndarray | None:
        p1 = world_to_region(region, view_state, world + step)
        return None if p1 is None else np.asarray(p1 - p0, dtype=np.float64)

    r_px, u_px = px(right), px(up)
    cell_px = {}
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0
        v = px(step)
        if v is not None:
            cell_px[axis] = (float(v[0]), float(v[1]))
    if r_px is None or u_px is None:
        return None
    return ScreenBasis(right, up, r_px, u_px), cell_px


def build_drag(point_id: str, region, view_state, recorder, mirror=(False, False, False)) -> CageDrag | None:
    """The drag of one point, with the screen as it was drawn. With Mirror on, the
    mirrored points move too."""
    cage = state.cage
    index = lattice.parse_point_id(point_id)
    model = document.model
    if cage is None or index is None or model is None:
        return None
    world = cage.lattice.points[index] + np.asarray(voxel_object.model_origin(model), dtype=np.float64)
    found = screen_basis(region, view_state, world, index)
    if found is None:
        return None
    basis, cell_px = found
    sel = None if cage.whole else document.selection
    return CageDrag(point_id, cage, document.grid.visible, sel, recorder.set_many, basis, cell_px,
                    set_grid=recorder.set_grid, mirror=mirror, grid_shape=document.grid.visible.shape,
                    locks=cage.locks)


def press_chip(name: str) -> None:
    """A click on the chip row."""
    lock = cage_chips.parse_lock(name)
    if name == cage_chips.POINTS and state.cage is not None:
        if "rok.cage_points" in opkit.REGISTRY and opkit.ops.rok.cage_points.poll():
            opkit.ops.rok.cage_points("INVOKE")
    elif lock is not None and state.cage is not None:
        locks = list(state.cage.locks)
        locks[lock] = not locks[lock]
        state.cage.locks = tuple(locks)
        voxel_object.redraw_3d_views()
    elif name == cage_chips.SOFT and state.cage is not None:
        place(state.cage.counts, not state.cage.lattice.bernstein, keep_origin=True)
    elif name == cage_chips.RESET:
        reset()
    elif name == cage_chips.DONE:
        hide()


@register
class CageAction(Action):
    command_id = "rok.cage"
    title = "Cage"
    tooltip = "A cage of points around the selection; drag a point to bend the voxels"
    @classmethod
    def poll(cls, context):
        return can_edit_voxels() and _scope() is not None

    def execute(self, context):
        # Run again while it shows: the cage starts over.
        (reset if shown() else place)()
        return {"FINISHED"}


@register
class CageDoneAction(Action):
    command_id = "rok.cage_done"
    title = "Cage Done"
    tooltip = "Hide the cage; the voxels stay as they are"
    @classmethod
    def poll(cls, context):
        return shown()

    def execute(self, context):
        hide()
        return {"FINISHED"}


@register
class CagePointsAction(Action):
    command_id = "rok.cage_points"
    title = "Cage Points"
    tooltip = "Place the cage again with this many points along X, Y and Z"
    points_x = IntProp(name="Points along X", min=lattice.MIN_COUNT, max=lattice.MAX_COUNT, default=DEFAULT_COUNT)
    points_y = IntProp(name="Points along Y", min=lattice.MIN_COUNT, max=lattice.MAX_COUNT, default=DEFAULT_COUNT)
    points_z = IntProp(name="Points along Z", min=lattice.MIN_COUNT, max=lattice.MAX_COUNT, default=DEFAULT_COUNT)

    @classmethod
    def poll(cls, context):
        return can_edit_voxels() and _scope() is not None

    def invoke(self, context, event):
        counts = state.cage.counts if state.cage is not None else (DEFAULT_COUNT,) * 3
        self.points_x, self.points_y, self.points_z = counts
        return context.prefs.open_dialog(self, title="Cage Points", confirm_text="Apply")

    def draw(self, context):
        for name in ("points_x", "points_y", "points_z"):
            self.layout.prop(self, name)

    def execute(self, context):
        place((int(self.points_x), int(self.points_y), int(self.points_z)), keep_origin=True)
        return {"FINISHED"}
