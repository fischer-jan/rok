"""Selection mask, column masks, move, copy, delete and the Select gestures."""

import numpy as np
import pytest

from rok.core.edit import SELECT
from rok.core.gizmo import snap_pivot
from rok.core.grid import VoxelGrid
from rok.core.raycast import Hit
from rok.core.select import (
    MoveGesture,
    MovePreview,
    PickGesture,
    Selection,
    brush_cells,
    column_mask,
    delete_cells,
    flip_cells,
    flipped_cells,
    footprint_mask,
    invert_mask,
    move_cells,
    point_in_polygon,
    rotate_cells,
    screen_cells,
    turned_cells,
)
from rok.core.undo import StrokeRecorder
from rok.core.vox_read import read_vox

from helpers import FIXTURES, ArraySetMany, filled, bird_slab, top_down, top_ray

SIZE = (8, 8, 8)


def selected(sel: Selection) -> set:
    return {tuple(c) for c in sel.cells().tolist()}


def column_tops(voxels: np.ndarray) -> np.ndarray:
    """Top filled z per (x, y) column, -1 for an empty column."""
    solid = voxels != 0
    sz = solid.shape[2]
    tops = sz - 1 - np.argmax(solid[:, :, ::-1], axis=2)
    return np.where(solid.any(axis=2), tops, -1)


# ---------------------------------------------------------------- Selection


def test_fresh_selection_is_empty_with_new_version():
    a = Selection(SIZE)
    b = Selection(SIZE)
    assert a.is_empty() and b.is_empty()
    assert a.count() == 0
    assert a.bounds() is None
    assert a.cells().shape == (0, 3)
    assert a.mask.shape == SIZE and a.mask.dtype == bool
    assert a.size == SIZE
    assert b.version > a.version


def test_add_cells_count_bounds_contains():
    sel = Selection(SIZE)
    sel.add_cells([(1, 2, 3), (4, 5, 6), (1, 2, 3)])
    assert sel.count() == 2
    assert not sel.is_empty()
    assert sel.bounds() == ((1, 2, 3), (4, 5, 6))
    assert selected(sel) == {(1, 2, 3), (4, 5, 6)}
    assert sel.contains((1, 2, 3))
    assert not sel.contains((0, 0, 0))
    assert not sel.contains((-1, 2, 3))
    assert not sel.contains((8, 0, 0))


def test_add_cells_ignores_out_of_bounds_rows():
    sel = Selection(SIZE)
    v = sel.version
    sel.add_cells([(-1, 0, 0), (8, 0, 0), (0, 0, -1)])
    assert sel.is_empty()
    assert sel.version == v
    sel.add_cells(np.array([(0, 0, 0), (9, 9, 9)]))
    assert selected(sel) == {(0, 0, 0)}


def test_version_bumps_only_on_change():
    sel = Selection(SIZE)
    v0 = sel.version
    sel.add_cells([(1, 1, 1)])
    v1 = sel.version
    assert v1 > v0
    sel.add_cells([(1, 1, 1)])
    assert sel.version == v1
    sel.add_cells(np.zeros((0, 3), dtype=np.int64))
    assert sel.version == v1
    mask = np.zeros(SIZE, dtype=bool)
    mask[1, 1, 1] = True
    sel.add_mask(mask)
    assert sel.version == v1
    mask[2, 2, 2] = True
    sel.add_mask(mask)
    v2 = sel.version
    assert v2 > v1
    assert sel.count() == 2
    sel.replace(mask)
    assert sel.version == v2
    sel.clear()
    v3 = sel.version
    assert v3 > v2
    assert sel.is_empty()
    sel.clear()
    assert sel.version == v3


def test_replace_copies_the_mask():
    sel = Selection(SIZE)
    mask = np.zeros(SIZE, dtype=bool)
    mask[3, 3, 3] = True
    sel.replace(mask)
    mask[4, 4, 4] = True
    assert sel.count() == 1
    assert sel.bounds() == ((3, 3, 3), (3, 3, 3))


def test_replace_rejects_a_larger_mask_and_pads_a_smaller_one():
    sel = Selection(SIZE)
    with pytest.raises(ValueError):
        sel.replace(np.zeros((9, 8, 8), dtype=bool))
    with pytest.raises(ValueError):
        sel.add_mask(np.zeros((8, 8), dtype=bool))
    # A grid cut at the section plane gives masks with fewer layers: padded with False.
    cut = np.zeros((8, 8, 3), dtype=bool)
    cut[1, 1, 2] = True
    sel.replace(cut)
    assert selected(sel) == {(1, 1, 2)} and sel.mask.shape == SIZE
    more = np.zeros((8, 8, 3), dtype=bool)
    more[2, 2, 0] = True
    sel.add_mask(more)
    assert selected(sel) == {(1, 1, 2), (2, 2, 0)}


def test_remove_cells_drops_only_what_is_selected():
    sel = Selection(SIZE)
    sel.add_cells([(1, 2, 3), (4, 5, 6)])
    v = sel.version
    sel.remove_cells([(0, 0, 0), (8, 0, 0)])  # not selected, out of bounds: no change
    assert sel.version == v
    sel.remove_cells([(1, 2, 3)])
    assert selected(sel) == {(4, 5, 6)}
    assert sel.version != v


def test_remove_mask_pads_a_smaller_mask_like_add_mask():
    sel = Selection(SIZE)
    sel.add_cells([(1, 1, 2), (2, 2, 0)])
    v = sel.version
    empty_cut = np.zeros((8, 8, 3), dtype=bool)
    sel.remove_mask(empty_cut)
    assert sel.version == v  # nothing in the padded area is selected
    cut = np.zeros((8, 8, 3), dtype=bool)
    cut[1, 1, 2] = True
    sel.remove_mask(cut)
    assert selected(sel) == {(2, 2, 0)}
    assert sel.version != v


def test_count_and_bounds_are_cached_per_version():
    sel = Selection(SIZE)
    sel.add_cells([(2, 3, 4)])
    assert sel.count() == 1
    assert sel.bounds() == ((2, 3, 4), (2, 3, 4))
    # A write behind the API is not seen until the version changes.
    sel.mask[0, 0, 0] = True
    assert sel.count() == 1
    assert sel.bounds() == ((2, 3, 4), (2, 3, 4))
    sel.add_cells([(7, 7, 7)])
    assert sel.count() == 3
    assert sel.bounds() == ((0, 0, 0), (7, 7, 7))


def test_prune_bumps_only_on_removal():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 1] = 5
    voxels[2, 2, 2] = 5
    sel = Selection(SIZE)
    sel.add_cells([(1, 1, 1), (2, 2, 2)])
    v = sel.version
    sel.prune(voxels)
    assert sel.version == v
    assert sel.count() == 2
    voxels[2, 2, 2] = 0
    # A filled cell outside the selection bounds must not matter.
    voxels[7, 7, 7] = 5
    sel.prune(voxels)
    assert sel.version > v
    assert selected(sel) == {(1, 1, 1)}
    assert sel.bounds() == ((1, 1, 1), (1, 1, 1))


def test_prune_on_empty_selection_is_a_no_op():
    sel = Selection(SIZE)
    v = sel.version
    sel.prune(np.zeros(SIZE, dtype=np.uint8))
    assert sel.version == v


# ---------------------------------------------------------------- column masks


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_column_mask_is_the_whole_line(axis):
    mask = column_mask(SIZE, axis, [(1, 2, 3)])
    assert mask.shape == SIZE and mask.dtype == bool
    assert mask.sum() == SIZE[axis]
    idx = [1, 2, 3]
    for k in range(SIZE[axis]):
        idx[axis] = k
        assert mask[idx[0], idx[1], idx[2]]


def test_column_mask_ignores_free_coordinates_outside_and_empty_input():
    mask = column_mask(SIZE, 2, [(1, 1, -5), (9, 1, 0), (1, -1, 0)])
    assert mask.sum() == SIZE[2]
    assert mask[1, 1, :].all()
    assert column_mask(SIZE, 2, np.zeros((0, 3), dtype=np.int64)).sum() == 0


def test_column_mask_is_writable():
    mask = column_mask(SIZE, 2, [(0, 0, 0)])
    mask[0, 0, 0] = False
    assert mask.sum() == SIZE[2] - 1


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_footprint_mask_golden_robot(axis):
    voxels = read_vox(FIXTURES / "robot.vox").voxels
    solid = voxels != 0
    u, v = (i for i in range(3) if i != axis)
    # Two cells: a filled voxel in the front of the body and one on the empty ground.
    cells = [(12, 9, 9), (2, 2, 0)]
    expected = 0
    for c in cells:
        idx = [slice(None)] * 3
        idx[u] = c[u]
        idx[v] = c[v]
        expected += int(solid[tuple(idx)].sum())
    assert expected > 0
    mask = footprint_mask(voxels, axis, cells)
    assert int(mask.sum()) == expected
    assert not (mask & ~solid).any()
    assert not (mask & ~column_mask(voxels.shape, axis, cells)).any()


# ---------------------------------------------------------------- move, delete


def test_move_cells_in_place_with_overlap():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1:4, 2, 2] = [1, 2, 3]
    sel = Selection(SIZE)
    sel.add_mask(voxels != 0)
    sm = ArraySetMany(voxels)
    new = move_cells(voxels, sel, (1, 0, 0), False, sm)
    assert voxels[1, 2, 2] == 0
    assert voxels[2:5, 2, 2].tolist() == [1, 2, 3]
    assert filled(voxels) == {(2, 2, 2), (3, 2, 2), (4, 2, 2)}
    assert new.dtype == bool and new.shape == SIZE
    assert {tuple(c) for c in np.argwhere(new).tolist()} == filled(voxels)
    # Source cleared first, then the destination written.
    assert len(sm.calls) == 2
    assert sm.calls[0][1].tolist() == [0, 0, 0]
    assert sm.calls[1][1].tolist() == [1, 2, 3]


def test_move_cells_clips_at_the_border():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[6, 0, 0] = 4
    voxels[7, 0, 0] = 5
    sel = Selection(SIZE)
    sel.add_mask(voxels != 0)
    new = move_cells(voxels, sel, (1, 0, 0), False, ArraySetMany(voxels))
    assert filled(voxels) == {(7, 0, 0)}
    assert voxels[7, 0, 0] == 4
    assert np.argwhere(new).tolist() == [[7, 0, 0]]


def test_move_cells_copy_keeps_the_source():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 1] = 9
    sel = Selection(SIZE)
    sel.add_cells([(1, 1, 1)])
    sm = ArraySetMany(voxels)
    new = move_cells(voxels, sel, (0, 0, 3), True, sm)
    assert voxels[1, 1, 1] == 9
    assert voxels[1, 1, 4] == 9
    assert len(sm.calls) == 1
    assert np.argwhere(new).tolist() == [[1, 1, 4]]


def test_move_cells_destination_overwrites():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[0, 0, 0] = 1
    voxels[0, 0, 1] = 7
    sel = Selection(SIZE)
    sel.add_cells([(0, 0, 0)])
    move_cells(voxels, sel, (0, 0, 1), False, ArraySetMany(voxels))
    assert voxels[0, 0, 0] == 0
    assert voxels[0, 0, 1] == 1


def test_move_cells_empty_selection_writes_nothing():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    sm = ArraySetMany(voxels)
    new = move_cells(voxels, Selection(SIZE), (1, 0, 0), False, sm)
    assert sm.calls == []
    assert not new.any()


def test_delete_cells():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[0:3, 0, 0] = 2
    sel = Selection(SIZE)
    sel.add_cells([(0, 0, 0), (1, 0, 0)])
    sm = ArraySetMany(voxels)
    assert delete_cells(voxels, sel, sm) == 2
    assert filled(voxels) == {(2, 0, 0)}
    assert delete_cells(voxels, Selection(SIZE), sm) == 0
    assert len(sm.calls) == 1


# ---------------------------------------------------------------- PickGesture


def test_pick_gesture_adds_hovered_voxels():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 1
    voxels[2, 1, 0] = 1
    voxels[2, 1, 1] = 1
    sel = Selection(SIZE)
    g = PickGesture(voxels, sel)
    assert g.tool == SELECT
    hit = Hit((1, 1, 0), (0, 0, 1), 1.0)
    assert g.begin(hit) is True
    assert g.hit is hit
    assert g.last == (1, 1, 0)
    assert selected(sel) == {(1, 1, 0)}
    assert g.extend(*top_ray(2, 1)) is True
    assert g.last == (2, 1, 1)
    assert selected(sel) == {(1, 1, 0), (2, 1, 1)}
    # The same voxel again is not new.
    assert g.extend(*top_ray(2, 1)) is False
    # A miss on the empty ground adds nothing: walls are off.
    v = sel.version
    assert g.extend(*top_ray(5, 5)) is False
    assert sel.version == v
    assert g.finish() == 0
    assert g.preview() is None


def test_pick_gesture_ignores_walls():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    sel = Selection(SIZE)
    g = PickGesture(voxels, sel)
    assert g.begin(Hit((3, 3, -1), (0, 0, 1), 1.0, wall=True)) is False
    assert sel.is_empty()
    assert g.last is None


def test_pick_gesture_subtract_removes_the_hovered_voxels():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 0] = 1
    voxels[2, 1, 0] = 1
    sel = Selection(SIZE)
    sel.add_cells([(1, 1, 0), (2, 1, 0), (3, 3, 3)])
    g = PickGesture(voxels, sel, subtract=True)
    assert g.begin(Hit((1, 1, 0), (0, 0, 1), 1.0)) is True
    assert selected(sel) == {(2, 1, 0), (3, 3, 3)}
    # A voxel already outside the selection: nothing changes.
    assert g.extend(*top_ray(5, 5)) is False
    assert g.extend(*top_ray(2, 1)) is True
    assert selected(sel) == {(3, 3, 3)}


# ---------------------------------------------------------------- MoveGesture


def make_move(copy=False):
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 0] = 3
    voxels[3, 2, 0] = 4
    sel = Selection(SIZE)
    sel.add_cells([(2, 2, 0), (3, 2, 0)])
    sm = ArraySetMany(voxels)
    return MoveGesture(voxels, sel, copy, sm), voxels, sel, sm


def test_move_gesture_begin_needs_a_selected_voxel():
    g, voxels, sel, _ = make_move()
    assert g.tool == SELECT
    assert g.begin(Hit((2, 2, -1), (0, 0, 1), 1.0, wall=True)) is False
    assert g.begin(Hit((5, 5, 0), (0, 0, 1), 1.0)) is False
    hit = Hit((2, 2, 0), (0, 0, 1), 1.0)
    assert g.begin(hit) is True
    assert g.hit is hit
    assert g.last == (2, 2, 0)
    assert g.offset == (0, 0, 0)
    assert g.lock.axis == 2 and g.lock.plane == 1.0 and g.lock.layer == 0


def test_move_gesture_plane_lock_zeroes_the_locked_axis():
    g, voxels, sel, _ = make_move()
    # Press on the +X face of (3, 2, 0): moves are locked to the x == 4 plane.
    assert g.begin(Hit((3, 2, 0), (1, 0, 0), 1.0)) is True
    assert g.lock.axis == 0 and g.lock.plane == 4.0 and g.lock.layer == 3
    # A ray from +X toward the plane, aimed at cell (3, 4, 2).
    assert g.extend((10.0, 4.5, 2.5), (-1.0, 0.0, 0.0)) is True
    assert g.offset == (0, 2, 2)
    assert g.last == (3, 4, 2)
    # The same target again changes nothing.
    assert g.extend((10.0, 4.5, 2.5), (-1.0, 0.0, 0.0)) is False
    # A ray parallel to the plane misses.
    assert g.extend((10.0, 4.5, 2.5), (0.0, 1.0, 0.0)) is False
    assert g.offset == (0, 2, 2)
    p = g.preview()
    assert isinstance(p, MovePreview)
    assert p.kind == "MOVE" and p.tool == SELECT and p.offset == (0, 2, 2)
    assert p.cells.shape == (0, 3)
    assert p.box is None and p.label == "" and p.normal == (0, 0, 1)


def test_move_gesture_finish_moves_and_replaces_the_selection():
    g, voxels, sel, sm = make_move()
    assert g.begin(Hit((2, 2, 0), (0, 0, 1), 1.0)) is True
    assert g.extend(*top_ray(3, 5)) is True
    assert g.offset == (1, 3, 0)
    assert sm.calls == []  # nothing written during the drag
    assert g.finish() == 4
    assert filled(voxels) == {(3, 5, 0), (4, 5, 0)}
    assert voxels[3, 5, 0] == 3 and voxels[4, 5, 0] == 4
    assert selected(sel) == {(3, 5, 0), (4, 5, 0)}


def test_move_gesture_copy_keeps_the_source_and_selects_the_copy():
    g, voxels, sel, sm = make_move(copy=True)
    assert g.begin(Hit((2, 2, 0), (0, 0, 1), 1.0)) is True
    assert g.extend(*top_ray(2, 4)) is True
    assert g.finish() == 2
    assert filled(voxels) == {(2, 2, 0), (3, 2, 0), (2, 4, 0), (3, 4, 0)}
    assert selected(sel) == {(2, 4, 0), (3, 4, 0)}


def test_move_gesture_zero_offset_writes_nothing():
    g, voxels, sel, sm = make_move()
    assert g.begin(Hit((2, 2, 0), (0, 0, 1), 1.0)) is True
    v = sel.version
    assert g.finish() == 0
    assert sm.calls == []
    assert sel.version == v
    # Away and back again is also zero.
    assert g.extend(*top_ray(3, 2)) is True
    assert g.extend(*top_ray(2, 2)) is True
    assert g.offset == (0, 0, 0)
    assert g.finish() == 0
    assert sm.calls == []


# ---------------------------------------------------------------- golden


def test_golden_bird_slab_moves_up_by_one():
    voxels, sel = bird_slab()
    before = voxels.copy()
    tops_before = column_tops(before)
    count = int((before != 0).sum())
    assert tops_before[30:40, 30:40].max() < before.shape[2] - 1
    sm = ArraySetMany(voxels)
    new = move_cells(voxels, sel, (0, 0, 1), False, sm)
    assert int((voxels != 0).sum()) == count
    tops_after = column_tops(voxels)
    moved = tops_before[30:40, 30:40] >= 0
    assert (tops_after[30:40, 30:40][moved] == tops_before[30:40, 30:40][moved] + 1).all()
    # Columns outside the rectangle are untouched.
    outside = np.ones(before.shape[:2], dtype=bool)
    outside[30:40, 30:40] = False
    assert np.array_equal(voxels[outside], before[outside])
    assert int(new.sum()) == sel.count()


def test_golden_bird_move_through_recorder_and_revert():
    model = read_vox(FIXTURES / "bird.vox")
    grid = VoxelGrid.from_model(model)
    _, sel = bird_slab()
    rec = StrokeRecorder(grid)
    g = MoveGesture(grid.voxels, sel, False, rec.set_many)
    cell = tuple(int(c) for c in sel.cells()[0])
    assert g.begin(Hit(cell, (0, 0, 1), 1.0)) is True
    assert g.extend(*top_ray(cell[0] + 1, cell[1])) is True
    assert g.offset == (1, 0, 0)
    assert g.finish() > 0
    assert not np.array_equal(grid.voxels, model.voxels)
    record = rec.finish()
    assert record is not None
    rec.revert()
    assert np.array_equal(grid.voxels, model.voxels)
    # The record replays the move, and undoes it again.
    record.redo(grid)
    assert not np.array_equal(grid.voxels, model.voxels)
    record.undo(grid)
    assert grid.voxels.tobytes() == model.voxels.tobytes()


# ---------------------------------------------------------------- brush


def test_brush_cells_radius_one_is_the_single_cell():
    voxels = read_vox(FIXTURES / "bird.vox").voxels
    cell = tuple(int(c) for c in np.argwhere(voxels != 0)[100])
    cells = brush_cells(voxels, cell, 1)
    assert cells.shape == (1, 3) and tuple(cells[0]) == cell
    # An empty centre gives nothing.
    empty = tuple(int(c) for c in np.argwhere(voxels == 0)[0])
    assert brush_cells(voxels, empty, 1).shape == (0, 3)


def test_brush_cells_size_two_is_the_filled_cells_of_the_small_sphere():
    voxels = read_vox(FIXTURES / "bird.vox").voxels
    inner = np.argwhere(voxels[1:-1, 1:-1, 1:-1] != 0) + 1
    c = tuple(int(v) for v in inner[len(inner) // 2])
    cells = brush_cells(voxels, c, 2)
    d = cells - np.array(c)
    assert ((d * d).sum(axis=1) <= 2).all()  # no corner of the 3-cube
    assert all(voxels[tuple(p)] != 0 for p in cells)


def test_brush_cells_is_clipped_to_the_grid():
    voxels = np.ones(SIZE, dtype=np.uint8)
    cells = brush_cells(voxels, (0, 0, 0), 3)
    assert cells.min() == 0 and cells.max() == 2  # clipped: only the octant inside the grid
    assert all(((p * p).sum() <= 6) for p in cells)
    assert brush_cells(voxels, (7, 7, 7), 2).shape == (7, 3)  # the corner cell, 3 faces, 3 edges
    assert brush_cells(voxels, (20, 20, 20), 2).shape == (0, 3)


def test_pick_gesture_with_radius_adds_the_brush():
    voxels = np.ones(SIZE, dtype=np.uint8)
    sel = Selection(SIZE)
    g = PickGesture(voxels, sel, radius=2)
    assert g.begin(Hit((3, 3, 7), (0, 0, 1), 1.0)) is True
    first = {tuple(p) for p in brush_cells(voxels, (3, 3, 7), 2).tolist()}
    assert sel.count() == len(first)  # the size-2 sphere, clipped at the top (z 6..7)
    # The same brush again is nothing new; one cell further is.
    assert g.extend(*top_ray(3, 3)) is False
    assert g.extend(*top_ray(4, 3)) is True
    second = {tuple(p) for p in brush_cells(voxels, (4, 3, 7), 2).tolist()}
    assert sel.count() == len(first | second)
    assert PickGesture(voxels, sel).radius == 1


# ---------------------------------------------------------------- turn, flip


def cells_of(mask) -> set:
    return {tuple(c) for c in np.argwhere(mask).tolist()}


def test_turned_cells_quarter_turns_about_each_axis():
    # One cell off the pivot (the pivot is the centre of cell (1, 1, 1)).
    pivot2 = (3, 3, 3)
    cell = np.array([[2, 1, 1]])
    assert tuple(turned_cells(cell, 2, 1, pivot2)[0]) == (1, 2, 1)
    assert tuple(turned_cells(cell, 2, 2, pivot2)[0]) == (0, 1, 1)
    assert tuple(turned_cells(cell, 2, 3, pivot2)[0]) == (1, 0, 1)
    assert tuple(turned_cells(cell, 2, 4, pivot2)[0]) == (2, 1, 1)
    # About y: (z, x) -> (-x, z); x = +1 lands at z = -1.
    assert tuple(turned_cells(cell, 1, 1, pivot2)[0]) == (1, 1, 0)
    # About x: (y, z) -> (-z, y).
    assert tuple(turned_cells([[1, 2, 1]], 0, 1, pivot2)[0]) == (1, 1, 2)
    assert turned_cells(np.zeros((0, 3), np.int64), 0, 1, pivot2).shape == (0, 3)


def test_flipped_cells_across_the_pivot_plane():
    # pivot2 4 is the plane x = 2, a cell centre: cell 0 mirrors to cell 3.
    assert tuple(flipped_cells([[0, 5, 5]], 0, (4, 0, 0))[0]) == (3, 5, 5)
    # pivot2 2 is x = 1, a cell corner: cell 0 mirrors to cell 1.
    assert tuple(flipped_cells([[0, 5, 5]], 0, (2, 0, 0))[0]) == (1, 5, 5)
    assert tuple(flipped_cells([[0, 5, 5]], 0, (1, 0, 0))[0]) == (0, 5, 5)


def test_rotate_cells_four_times_on_a_bird_slab_is_identity():
    model = read_vox(FIXTURES / "bird.vox")
    grid = VoxelGrid.from_model(model)
    _, sel = bird_slab()
    lo, hi = sel.bounds()
    pivot2 = snap_pivot((lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1), 2)
    rec = StrokeRecorder(grid)
    first = None
    for _ in range(4):
        new = rotate_cells(grid.voxels, sel, 2, 1, pivot2, rec.set_many)
        assert int(new.sum()) == sel.count()
        sel.replace(new)
        if first is None:
            first = grid.voxels.copy()
    assert not np.array_equal(first, model.voxels)
    assert grid.voxels.tobytes() == model.voxels.tobytes()
    assert cells_of(sel.mask) == cells_of(bird_slab()[1].mask)
    # revert restores the fixture too after a single turn.
    rotate_cells(grid.voxels, sel, 2, 1, pivot2, rec.set_many)
    assert not np.array_equal(grid.voxels, model.voxels)
    rec.revert()
    assert grid.voxels.tobytes() == model.voxels.tobytes()


def test_rotate_cells_zero_turns_writes_nothing():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 1, 1] = 3
    sel = Selection(SIZE)
    sel.add_cells([(1, 1, 1)])
    sm = ArraySetMany(voxels)
    new = rotate_cells(voxels, sel, 2, 4, (3, 3, 3), sm)
    assert sm.calls == [] and cells_of(new) == {(1, 1, 1)}


def test_rotate_cells_mixed_parity_after_snap_maps_cells_to_cells():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[0:3, 0:4, 0] = np.arange(1, 13, dtype=np.uint8).reshape(3, 4)
    sel = Selection(SIZE)
    sel.replace(voxels != 0)
    lo, hi = sel.bounds()
    raw = (lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1)
    assert raw == (3, 4, 1)
    pivot2 = snap_pivot(raw, 2)
    assert pivot2 == (3, 5, 1)
    before = voxels.copy()
    sm = ArraySetMany(voxels)
    new = rotate_cells(voxels, sel, 2, 1, pivot2, sm)
    # 3 x 4 turns into 4 x 3, every colour kept, all inside the grid.
    assert cells_of(new) == {(x, y, 0) for x in range(4) for y in range(1, 4)}
    assert sorted(voxels[new].tolist()) == sorted(before[before != 0].tolist())
    assert int((voxels != 0).sum()) == 12
    # Sources cleared first, then the destinations.
    assert len(sm.calls) == 2 and (sm.calls[0][1] == 0).all()


def test_rotate_cells_drops_cells_outside_the_grid():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[7, 0, 0] = 1
    voxels[7, 1, 0] = 2
    sel = Selection(SIZE)
    sel.replace(voxels != 0)
    sm = ArraySetMany(voxels)
    # Turn about the centre of cell (7, 0, 0): (7, 1) -> (6, 0); (7, 0) stays.
    new = rotate_cells(voxels, sel, 2, 1, (15, 1, 1), sm)
    assert cells_of(new) == {(7, 0, 0), (6, 0, 0)}
    assert int(voxels[6, 0, 0]) == 2
    # Turn the other way: (7, 1) -> (8, 0), off the grid and dropped.
    sel.replace(voxels != 0)
    voxels[7, 1, 0] = 2
    voxels[6, 0, 0] = 0
    sel.replace(voxels != 0)
    new = rotate_cells(voxels, sel, 2, 3, (15, 1, 1), sm)
    assert cells_of(new) == {(7, 0, 0)}
    assert int((voxels != 0).sum()) == 1


def test_flip_cells_twice_is_identity():
    model = read_vox(FIXTURES / "bird.vox")
    grid = VoxelGrid.from_model(model)
    _, sel = bird_slab()
    lo, hi = sel.bounds()
    pivot2 = (lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1)
    rec = StrokeRecorder(grid)
    for axis in range(3):
        new = flip_cells(grid.voxels, sel, axis, pivot2, rec.set_many)
        assert int(new.sum()) == sel.count()
        sel.replace(new)
        new = flip_cells(grid.voxels, sel, axis, pivot2, rec.set_many)
        sel.replace(new)
        assert grid.voxels.tobytes() == model.voxels.tobytes()
    assert cells_of(sel.mask) == cells_of(bird_slab()[1].mask)


def test_flip_cells_small():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1, 2, 0] = 5
    voxels[2, 2, 0] = 6
    sel = Selection(SIZE)
    sel.replace(voxels != 0)
    sm = ArraySetMany(voxels)
    # Flip across x = 2 (the corner between cells 1 and 2): 1 <-> 2.
    new = flip_cells(voxels, sel, 0, (4, 5, 1), sm)
    assert cells_of(new) == {(1, 2, 0), (2, 2, 0)}
    assert int(voxels[1, 2, 0]) == 6 and int(voxels[2, 2, 0]) == 5
    # Flip across x = 1: cell 1 lands on cell 0, cell 2 at -1 and is dropped.
    new = flip_cells(voxels, sel, 0, (2, 5, 1), sm)
    assert cells_of(new) == {(0, 2, 0)}
    assert int(voxels[0, 2, 0]) == 6 and int((voxels != 0).sum()) == 1


# ---------------------------------------------------------------- screen space


def in_polygon_ref(px, py, polygon) -> bool:
    """Plain even-odd loop over the edges."""
    inside = False
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if (ay > py) != (by > py):
            x_at = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < x_at:
                inside = not inside
    return inside


def test_point_in_polygon_concave_against_the_plain_loop():
    # A "U": the notch lies inside the bounding box but outside the polygon.
    polygon = [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
    rng = np.random.default_rng(3)
    pts = rng.uniform(-2, 12, size=(500, 2))
    got = point_in_polygon(pts, polygon)
    expected = np.array([in_polygon_ref(x, y, polygon) for x, y in pts])
    assert np.array_equal(got, expected)
    assert got.any() and not got.all()
    # Hand-picked points: in the notch, in a leg, outside.
    assert point_in_polygon([(5, 6)], polygon).tolist() == [False]
    assert point_in_polygon([(1.5, 6)], polygon).tolist() == [True]
    assert point_in_polygon([(11, 5), (5, -1)], polygon).tolist() == [False, False]


def test_point_in_polygon_many_edges_run_in_chunks():
    # A 100-gon: more than three chunks of 32 edges.
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    polygon = np.stack([np.cos(t) * 5, np.sin(t) * 5], axis=1)
    rng = np.random.default_rng(4)
    pts = rng.uniform(-6, 6, size=(300, 2))
    got = point_in_polygon(pts, polygon)
    expected = np.array([in_polygon_ref(x, y, polygon.tolist()) for x, y in pts])
    assert np.array_equal(got, expected)


def test_point_in_polygon_box_case_includes_the_edge():
    box = [(0, 0), (4, 0), (4, 2), (0, 2)]
    pts = [(0, 0), (4, 2), (2, 1), (4.1, 1), (2, -0.1)]
    assert point_in_polygon(pts, box).tolist() == [True, True, True, False, False]
    # Any corner order works.
    assert point_in_polygon([(2, 1)], [(4, 2), (0, 2), (0, 0), (4, 0)]).tolist() == [True]
    assert point_in_polygon(np.zeros((0, 2)), box).shape == (0,)
    assert point_in_polygon([(1, 1)], [(0, 0), (1, 1)]).tolist() == [False]


def test_screen_cells_top_down_rectangle_equals_footprint_mask():
    voxels = read_vox(FIXTURES / "bird.vox").voxels
    x0, x1, y0, y1 = 30, 39, 30, 39
    polygon = [
        (x0 * 10, y0 * 10),
        ((x1 + 1) * 10, y0 * 10),
        ((x1 + 1) * 10, (y1 + 1) * 10),
        (x0 * 10, (y1 + 1) * 10),
    ]
    cells = screen_cells(voxels, top_down, polygon)
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1), indexing="ij")
    rect = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size, np.int64)], axis=1)
    expected = footprint_mask(voxels, 2, rect)
    got = np.zeros(voxels.shape, dtype=bool)
    got[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    assert np.array_equal(got, expected)
    assert int(expected.sum()) > 10 * 10  # interior cells are included


def test_screen_cells_drops_nan_rows_and_empty_input():
    voxels = np.ones(SIZE, dtype=np.uint8)

    def behind(points):
        px = top_down(points)
        px[np.asarray(points)[:, 2] > 4] = np.nan
        return px

    cells = screen_cells(voxels, behind, [(0, 0), (80, 0), (80, 80), (0, 80)])
    assert cells.shape == (8 * 8 * 4, 3) and cells[:, 2].max() == 3
    empty = np.zeros(SIZE, np.uint8)
    assert screen_cells(empty, top_down, [(0, 0), (9, 0), (9, 9)]).shape == (0, 3)
    assert screen_cells(voxels, top_down, [(0, 0), (9, 9)]).shape == (0, 3)
    far = [(200, 200), (300, 200), (300, 300)]
    assert screen_cells(voxels, top_down, far).shape == (0, 3)


def test_invert_mask():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[0:2, 0, 0] = 1
    voxels[5, 5, 5] = 2
    mask = np.zeros(SIZE, dtype=bool)
    mask[0, 0, 0] = True
    mask[3, 3, 3] = True  # an empty cell in the mask stays out
    assert cells_of(invert_mask(voxels, mask)) == {(1, 0, 0), (5, 5, 5)}
    assert invert_mask(voxels, invert_mask(voxels, mask)).sum() == 1


def test_move_gesture_axis_lock_keeps_one_component_and_relabels():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 1
    sel = Selection(SIZE)
    sel.add_cells(np.array([[2, 2, 0]]))
    g = MoveGesture(voxels, sel, copy=False, set_many=ArraySetMany(voxels))
    assert g.begin(Hit((2, 2, 0), (0, 0, 1), 1.0))
    ray = ((5.5, 4.5, 50.0), (0.0, 0.0, -1.0))  # three over in x, two in y
    assert g.extend(*ray) and g.offset == (3, 2, 0)
    assert g.toggle_axis_lock(0) and g.offset == (3, 0, 0)
    assert g.preview().label == "lock X"
    assert g.toggle_axis_lock(1) and g.offset == (0, 2, 0)
    assert g.preview().label == "lock Y"
    assert g.toggle_axis_lock(1) and g.offset == (3, 2, 0)  # the same key again unlocks
    assert g.preview().label == ""
    # Locked, a move along the other axis changes nothing.
    g.toggle_axis_lock(0)
    assert not g.extend((5.5, 7.5, 50.0), (0.0, 0.0, -1.0)) and g.offset == (3, 0, 0)



# ------------------------------------------------------------ what the camera sees

from rok.core.select import seen_cells  # noqa: E402


def cube(n=6, pad=2):
    voxels = np.zeros((n + 2 * pad,) * 3, dtype=np.uint8)
    voxels[pad : pad + n, pad : pad + n, pad : pad + n] = 1
    return voxels


def test_from_above_only_the_top_layer_is_seen():
    voxels = cube()
    cells = np.argwhere(voxels)
    seen = cells[seen_cells(voxels, cells, toward=(0, 0, 1))]
    assert len(seen) == 36 and set(seen[:, 2].tolist()) == {7}


@pytest.mark.parametrize("camera", [{"eye": (20.0, 20.0, 20.0)}, {"toward": (1.0, 1.0, 1.0)}])
def test_from_a_corner_three_faces_are_seen(camera):
    voxels = cube()
    cells = np.argwhere(voxels)
    seen = seen_cells(voxels, cells, **camera)
    assert seen.sum() == 6**3 - 5**3  # everything but the far core
    at = {tuple(c): bool(s) for c, s in zip(cells.tolist(), seen)}
    assert at[(7, 7, 7)] and not at[(2, 2, 2)]  # the near corner shows, the far one not


def test_a_roof_hides_what_is_under_it():
    voxels = np.zeros((20, 20, 20), dtype=np.uint8)
    voxels[5:15, 5:15, 10] = 1
    voxels[8:12, 8:12, 2:4] = 2
    under = np.argwhere(voxels == 2)
    assert not seen_cells(voxels, under, toward=(0, 0, 1)).any()
    assert seen_cells(voxels, under, toward=(0, 0, -1)).any()  # from below it shows


def test_the_seen_model_is_its_near_side():
    voxels = np.asarray(read_vox(FIXTURES / "bird.vox").voxels)
    cells = np.argwhere(voxels)
    seen = seen_cells(voxels, cells, toward=(0.0, 0.0, 1.0))
    assert 0 < seen.sum() < len(cells)
    # Seen from above, no seen cell has a filled cell anywhere above it.
    for x, y, z in cells[seen][::50]:
        assert not voxels[x, y, z + 1 :].any()


def test_no_cells_no_work():
    assert seen_cells(cube(), np.zeros((0, 3), dtype=np.int64), toward=(0, 0, 1)).shape == (0,)


def test_a_connected_click_takes_the_region_by_faces_or_by_corners():
    from rok.core.edit import CONNECTED, CONNECTED_CORNERS, SELECT
    from rok.core.gesture import make_gesture
    from rok.core.raycast import Hit
    from rok.core.select import ConnectedGesture, Selection

    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    voxels[0, 0, 0] = voxels[1, 0, 0] = 1  # two blocks face to face
    voxels[2, 1, 1] = 2  # touches (1, 0, 0) at one corner only
    voxels[3, 3, 3] = 3  # alone
    hit = Hit((0, 0, 0), (0, 0, 1), 1.0)

    def take(mode, subtract=False, selection=None):
        selection = selection if selection is not None else Selection(voxels.shape)
        gesture = make_gesture(mode, SELECT, voxels, 1, (False, False, False), None, None, selection,
                               subtract=subtract)
        assert isinstance(gesture, ConnectedGesture)
        assert gesture.begin(hit)
        return selection

    faces = take(CONNECTED)
    assert sorted(map(tuple, faces.cells())) == [(0, 0, 0), (1, 0, 0)]
    corners = take(CONNECTED_CORNERS)
    assert sorted(map(tuple, corners.cells())) == [(0, 0, 0), (1, 0, 0), (2, 1, 1)]
    # Option+Shift takes the region back out.
    take(CONNECTED_CORNERS, subtract=True, selection=corners)
    assert len(corners.cells()) == 0


def test_select_in_face_mode_takes_the_flat_face_under_the_mouse():
    """Select face. Select's Face mode takes the connected,
    coplanar, same-colour face region, as Attach's Face mode would extrude it."""
    from rok.core.edit import FACE, SELECT
    from rok.core.gesture import FaceGesture, make_gesture
    from rok.core.raycast import Hit
    from rok.core.select import Selection

    voxels = np.zeros((4, 4, 3), dtype=np.uint8)
    voxels[:, :, 0] = 1  # a floor slab
    voxels[0:2, 0:2, 0] = 2  # one corner in another colour
    voxels[3, 3, 1] = 1  # a block on the slab: its top is another plane
    selection = Selection(voxels.shape)
    gesture = make_gesture(FACE, SELECT, voxels, 1, (False, False, False), None, None, selection)
    assert isinstance(gesture, FaceGesture)
    assert gesture.begin(Hit((2, 2, 0), (0, 0, 1), 1.0))
    cells = {tuple(c) for c in selection.cells()}
    assert (2, 2, 0) in cells and (3, 0, 0) in cells
    assert (0, 0, 0) not in cells  # the other colour
    assert (3, 3, 0) not in cells and (3, 3, 1) not in cells  # under the block, and the block's top


def test_other_tools_in_a_connected_mode_work_one_voxel_at_a_time():
    from rok.core.edit import CONNECTED, PAINT, Stroke
    from rok.core.gesture import make_gesture

    voxels = np.zeros((2, 2, 2), dtype=np.uint8)
    gesture = make_gesture(CONNECTED, PAINT, voxels, 1, (False, False, False), lambda *a: True, None, None)
    assert isinstance(gesture, Stroke)


def test_a_by_colour_click_takes_every_voxel_of_that_colour():
    from rok.core.edit import BY_COLOUR, SELECT
    from rok.core.gesture import make_gesture
    from rok.core.raycast import Hit
    from rok.core.select import ColourGesture, Selection

    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    voxels[0, 0, 0] = voxels[3, 3, 3] = 5  # one colour, far apart
    voxels[1, 0, 0] = 6
    selection = Selection(voxels.shape)
    gesture = make_gesture(BY_COLOUR, SELECT, voxels, 1, (False, False, False), None, None, selection)
    assert isinstance(gesture, ColourGesture)
    assert gesture.begin(Hit((0, 0, 0), (0, 0, 1), 1.0))
    assert sorted(map(tuple, selection.cells())) == [(0, 0, 0), (3, 3, 3)]
    # A drag over the other colour takes it too.
    assert gesture._take((1, 0, 0))
    assert len(selection.cells()) == 3


def test_by_colour_connected_takes_the_colour_only_where_it_joins():
    from rok.core.edit import COLOUR_CONNECTED, COLOUR_CONNECTED_CORNERS, SELECT
    from rok.core.gesture import make_gesture
    from rok.core.raycast import Hit
    from rok.core.select import ColourGesture, Selection

    voxels = np.zeros((4, 4, 4), dtype=np.uint8)
    voxels[0, 0, 0] = voxels[1, 0, 0] = 5  # one colour, face to face
    voxels[2, 1, 1] = 5  # the same colour, at a corner of (1, 0, 0)
    voxels[3, 3, 3] = 5  # the same colour, far off
    voxels[0, 1, 0] = 6  # another colour, face to face with (0, 0, 0)

    def take(mode):
        selection = Selection(voxels.shape)
        gesture = make_gesture(mode, SELECT, voxels, 1, (False, False, False), None, None, selection)
        assert isinstance(gesture, ColourGesture) and gesture.begin(Hit((0, 0, 0), (0, 0, 1), 1.0))
        return sorted(map(tuple, selection.cells()))

    assert take(COLOUR_CONNECTED) == [(0, 0, 0), (1, 0, 0)]
    assert take(COLOUR_CONNECTED_CORNERS) == [(0, 0, 0), (1, 0, 0), (2, 1, 1)]


def test_only_exposed_cells_can_be_seen():
    """A pick of what the camera sees tests the exposed cells
    only; the others have no open face, so seen_cells never takes them."""
    from rok.core.select import exposed_cells

    voxels = np.zeros((10, 10, 10), dtype=np.uint8)
    voxels[2:8, 2:8, 2:8] = 1  # a 6-cube: 216 cells, 64 inside
    voxels[0, 0, 0] = 2  # a lone cell at the grid's corner: exposed by the edge
    exposed = exposed_cells(voxels)
    assert len(exposed) == 216 - 64 + 1
    inside = np.argwhere(voxels != 0)
    keys = {tuple(c) for c in exposed.tolist()}
    hidden = np.array([c for c in inside.tolist() if tuple(c) not in keys])
    assert len(hidden) == 64
    for camera in ({"toward": (0.0, 0.0, 1.0)}, {"eye": (30.0, -20.0, 25.0)}):
        assert not seen_cells(voxels, hidden, **camera).any()


def test_colour_tolerance_takes_close_colours_and_zero_takes_the_same_only():
    from rok.core.select import ColourGesture, colour_mask, similar_colours

    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[1, :3] = (200, 40, 40)   # red
    palette[2, :3] = (190, 50, 45)   # a red very close to it
    palette[3, :3] = (40, 60, 200)   # blue
    assert similar_colours(palette, 1, 0).nonzero()[0].tolist() == [1]
    assert similar_colours(palette, 1, 10).nonzero()[0].tolist() == [1, 2]
    assert 3 in similar_colours(palette, 1, 100).nonzero()[0].tolist()  # far colours at the top end
    voxels = np.zeros((6, 1, 1), dtype=np.uint8)
    voxels[:, 0, 0] = [1, 2, 3, 2, 0, 1]
    assert colour_mask(voxels, palette, 1, 0)[:, 0, 0].tolist() == [1, 0, 0, 0, 0, 1]
    assert colour_mask(voxels, palette, 1, 10)[:, 0, 0].tolist() == [1, 1, 0, 1, 0, 1]
    joined = ColourGesture(voxels, None, joined=True, palette=palette, tolerance=10)
    assert joined.region((0, 0, 0))[:, 0, 0].tolist() == [1, 1, 0, 0, 0, 0]  # the blue cell ends the run
