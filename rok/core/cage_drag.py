"""The cage's drag: pull one control point and the real voxels bend.

`CageDrag` is a `Gesture` in the shape of `corner_gizmo.CornerDrag`: it runs
through `rok.stroke`'s `extend_screen` path (`screen_drag`), writes through
`set_grid`, and knows the screen from the cell lengths it was drawn with. The
point moves in the plane through it parallel to the screen, or along one world
axis after X, Y or Z. Every move deforms the voxels *as they were when the cage
was placed* (`CageRest`) through the full set of points, so chained drags never
wear the model down; `LiveDeform` is `gizmo.LiveTurn`'s shape and writes only
the cells that differ. Pure numpy: nothing from the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edit import SELECT, SetMany
from .gesture_base import Gesture
from .lattice import Lattice, greville, live_deform_grid, parse_point_id
from .select import MovePreview, Selection

AXES = "XYZ"
FINE_STEP = 0.1  # Shift: a point moves in tenths of a cell
DRAG_TIP = "Drag to bend · X, Y, Z lock an axis · Shift fine"
MIN_BASIS_DET = 1e-9  # a screen basis flatter than this moves nothing


@dataclass
class CageRest:
    """What a cage deforms: the points as they stand, and the voxels and the
    selection as they were when the cage was placed (copies)."""

    lattice: Lattice
    rest_voxels: np.ndarray
    rest_mask: np.ndarray
    live: Lattice | None = None  # the points while a drag runs, for the overlay; None at rest


@dataclass(frozen=True)
class ScreenBasis:
    """The screen's plane at a point, for a planar drag: the grid-space unit
    vectors of the camera's right and up, and what one cell along each looks
    like in region pixels at the point."""

    right: np.ndarray  # (3,)
    up: np.ndarray  # (3,)
    right_px: np.ndarray  # (2,)
    up_px: np.ndarray  # (2,)


def mirror_partners(lat: Lattice, index, mirror, shape):
    """The points that move with `index` when Mirror is on, and the axes it may not move along.

    Mirror reflects a cell `c` to `shape - 1 - c`, so a rest place `x` goes to
    `shape - x`. An axis counts only when the cage's rest places are symmetric
    about that plane (a partner exists for every point); on any other axis Mirror
    does nothing. Returns ([(partner index, signs)], frozen axes): a partner moves
    by the dragged offset times `signs` (-1 on each reflected axis), and a point
    that is its own image on an active axis (it sits on the plane) is frozen on
    that axis, so the bend stays symmetric. The partners are the images for every
    non-empty set of reflecting axes, without repeats."""
    partner: dict[int, int] = {}
    for a in range(3):
        if not mirror[a]:
            continue
        n = lat.counts[a]
        rest = lat.lo[a] + lat.size[a] * greville(n, lat.bernstein)
        if np.allclose(rest, float(shape[a]) - rest[::-1], atol=1e-6):  # greville is symmetric: the image of k is n - 1 - k
            partner[a] = n - 1 - index[a]
    reflecting = [a for a in partner if partner[a] != index[a]]
    frozen = tuple(a for a in partner if partner[a] == index[a])
    out = []
    for mask in range(1, 1 << len(reflecting)):
        picked = [reflecting[b] for b in range(len(reflecting)) if mask >> b & 1]
        idx = tuple(partner[a] if a in picked else index[a] for a in range(3))
        signs = tuple(-1 if a in picked else 1 for a in range(3))
        out.append((idx, signs))
    return out, frozen


def mirror_copies(lat: Lattice, mirror, shape) -> tuple[int, ...]:
    """The axes on which Mirror bends a mirrored copy: Mirror is on and the cage's
    box lies wholly on one side of the plane (a wing, not the body). The voxels
    in the mirror image of the selection bend the mirrored way, as a brush stroke
    repeats on the other side."""
    out = []
    for a in range(3):
        if not mirror[a]:
            continue
        lo = float(lat.lo[a])
        hi = lo + float(lat.size[a])
        s = float(shape[a])
        if hi <= s - hi + 1e-6 or lo >= s - lo - 1e-6:
            out.append(a)
    return tuple(out)


def image_axes(copies) -> list[tuple[int, ...]]:
    """Every non-empty set of the axes in `copies`: one mirrored copy each."""
    copies = tuple(copies)
    return [tuple(copies[b] for b in range(len(copies)) if mask >> b & 1) for mask in range(1, 1 << len(copies))]


def mirrored_deform_grid(rest_voxels: np.ndarray, rest_mask: np.ndarray, lattice: Lattice, copies=()):
    """(want, new mask) as `live_deform_grid`, and on every axis in `copies` the
    mirror image of the selection bent by the mirrored cage too. A copy deforms
    the flipped grid with the same cage and is flipped back. Every rest region is
    cleared first; where two sides land on the same cell, the dragged side wins.
    `new mask` is the dragged side's cells only: the selection stays where the cage is."""
    if not copies:
        return live_deform_grid(rest_voxels, rest_mask, lattice)
    images = image_axes(copies)
    want = rest_voxels.copy()
    want[rest_mask] = 0
    for axes in images:
        want[np.flip(rest_mask, axes)] = 0
    for axes in images:
        w, n = live_deform_grid(np.ascontiguousarray(np.flip(rest_voxels, axes)), rest_mask, lattice)
        w, n = np.flip(w, axes), np.flip(n, axes)
        want[n] = w[n]
    w, new_mask = live_deform_grid(rest_voxels, rest_mask, lattice)
    want[new_mask] = w[new_mask]
    return want, new_mask


class LiveDeform:
    """A cage's deformation shown in the real voxels during a point drag, always
    from the rest voxels. Each step is one write of the cells that differ: it
    deforms the rest copy and `set_grid` (StrokeRecorder.set_grid), or `set_many`,
    writes the difference from the grid, so the mask and the section planes hold
    as for any write. The selection becomes the cells written, unless the cage
    acts on the whole model (`selection` is None)."""

    def __init__(self, rest: CageRest, voxels: np.ndarray, selection: Selection | None, set_many: SetMany,
                 set_grid=None, copies=()) -> None:
        self.rest = rest
        self.copies = tuple(copies)  # the axes of Mirror's mirrored copies (`mirror_copies`)
        self.voxels = voxels  # the live grid, to take the difference from
        self.selection = selection
        self.set_many = set_many
        self.set_grid = set_grid
        self.key = None  # the deformation shown
        self.total = 0  # voxels the shown deformation changed, for the stroke's report

    def _write(self, want: np.ndarray) -> int:
        """The grid becomes `want`: one write of the cells that differ."""
        if self.set_grid is not None:
            return int(self.set_grid(want))
        diff = want != self.voxels
        cells = np.argwhere(diff)
        return int(self.set_many(cells, want[diff])) if len(cells) else 0

    def show(self, key, lattice: Lattice) -> None:
        if self.set_many is None or key == self.key:
            return
        want, new_mask = mirrored_deform_grid(self.rest.rest_voxels, self.rest.rest_mask, lattice, self.copies)
        self.total = self._write(want)
        if self.selection is not None:
            self.selection.replace(new_mask)
        self.key = key


class CageDrag(Gesture):
    """One press-drag-release on a cage point (the stroke's extend_screen path)."""

    tool = SELECT
    screen_drag = True
    locks_axis = True

    def __init__(
        self,
        point_id: str,
        cage: CageRest,
        voxels: np.ndarray,
        selection: Selection | None,
        set_many: SetMany,
        basis: ScreenBasis,
        cell_px: dict[int, tuple[float, float]],
        *,
        set_grid=None,
        mirror=(False, False, False),
        grid_shape=None,
        locks=(False, False, False),
    ) -> None:
        index = parse_point_id(point_id)
        if index is None or any(not 0 <= i < n for i, n in zip(index, cage.lattice.counts)):
            raise ValueError(f"not a point of this cage: {point_id!r}")
        self.point_id = point_id
        self.index = index
        self.cage = cage
        self.basis = basis
        self.cell_px = {a: np.asarray(v, dtype=np.float64) for a, v in cell_px.items()}
        self.start: tuple[float, float] | None = None
        self.mouse: tuple[float, float] | None = None
        self.axis: int | None = None  # X, Y or Z during the drag locks the move to it
        self.locks = tuple(bool(v) for v in locks)  # the chips Lock X, Y, Z: no move along a locked axis
        self.fine = False  # Shift: tenths of a cell
        self.offset = (0.0, 0.0, 0.0)
        self.dragged = False
        self._base = cage.lattice.copy()  # the points at the press
        # Mirror: the points that move with this one, and the axes it is frozen on, when the
        # cage is symmetric about a plane; a mirrored copy when it lies on one side of it
        on = grid_shape is not None and any(mirror)
        self.partners, self.frozen = mirror_partners(cage.lattice, index, mirror, grid_shape) if on else ([], ())
        self.copies = mirror_copies(cage.lattice, mirror, grid_shape) if on else ()
        self._shown: Lattice | None = None
        self._live = LiveDeform(cage, voxels, selection, set_many, set_grid, self.copies)

    # --- the mouse -----------------------------------------------------------------------

    def set_shift(self, shift: bool) -> None:
        self.fine = bool(shift)

    def begin_screen(self, mx: float, my: float) -> bool:
        self.start = self.mouse = (float(mx), float(my))
        return True

    def toggle_axis_lock(self, axis: int) -> bool:
        self.axis = None if self.axis == axis else int(axis)
        return self._update()

    def _raw_offset(self) -> np.ndarray:
        """The mouse's move since the press in grid units, before rounding."""
        out = np.zeros(3, dtype=np.float64)
        if self.start is None or self.mouse is None:
            return out
        delta = np.array([self.mouse[0] - self.start[0], self.mouse[1] - self.start[1]], dtype=np.float64)
        free = [a for a in range(3) if not self.locks[a]]
        if not free:
            return out  # all three locked: nothing moves
        axis = self.axis
        if axis is None and len(free) == 1:
            axis = free[0]  # two locked: the mouse moves the point along the third
        if axis is not None:
            v = self.cell_px.get(axis)
            if v is not None and float(v @ v) > 0.0:
                out[axis] = float(delta @ v) / float(v @ v)
            return out
        b = self.basis
        m = np.stack([b.right_px, b.up_px], axis=1)  # pixels per unit along right and up
        det = float(np.linalg.det(m))
        if abs(det) < MIN_BASIS_DET:
            return out
        a, c = np.linalg.solve(m, delta)
        return a * np.asarray(b.right, dtype=np.float64) + c * np.asarray(b.up, dtype=np.float64)

    def _quantised(self) -> tuple[float, float, float]:
        step = FINE_STEP if self.fine else 1.0
        raw = self._raw_offset()
        for a in self.frozen:
            raw[a] = 0.0  # a point on the mirror plane stays on it
        for a in range(3):
            if self.locks[a]:
                raw[a] = 0.0  # a locked axis does not move
        # Whole cells (tenths with Shift); round() on a float keeps the key free of 0.30000000000000004
        return tuple(round(float(np.round(v / step)) * step, 6) for v in raw)

    def _update(self) -> bool:
        offset = self._quantised()
        if offset == self.offset:
            return False
        self.offset = offset
        self._show()
        return True

    def extend_screen(self, mx: float, my: float) -> bool:
        self.mouse = (float(mx), float(my))
        if self.start is not None and not self.dragged:
            self.dragged = float(np.hypot(mx - self.start[0], my - self.start[1])) >= 2.0
        return self._update()

    # --- the deformation -------------------------------------------------------------------

    def lattice_now(self) -> Lattice:
        """The cage with the dragged point at its place now."""
        out = self._base.copy()
        i, j, k = self.index
        offset = np.asarray(self.offset, dtype=np.float64)
        out.points[i, j, k] = self._base.points[i, j, k] + offset
        for idx, signs in self.partners:
            out.points[idx] = self._base.points[idx] + offset * np.asarray(signs, dtype=np.float64)
        return out

    def _show(self) -> None:
        lattice = self.lattice_now()
        self._live.show(self.offset, lattice)
        self._shown = lattice
        self.cage.live = lattice

    # --- the stroke --------------------------------------------------------------------------

    def label(self) -> str:
        parts = [f"{v:+g} {AXES[a]}" for a, v in enumerate(self.offset) if v != 0.0]
        return " ".join(parts)

    def preview(self):
        # The voxels themselves show the bend (see LiveDeform): only the move.
        return MovePreview(kind="LABEL", label=self.label())

    def finish(self) -> int:
        """Keep the points where the drag left them, for the next drag."""
        self.cage.live = None
        if self._shown is None:
            return 0
        self.cage.lattice = self._shown
        return self._live.total
