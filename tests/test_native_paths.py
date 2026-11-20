"""The native module's helpers build the same thing as the numpy paths they
replace (native code where the Python side was slow): the
exposed faces of a mask (hud.mask_surface) and the free turn of a voxel
block (free_rotate.rotate_voxels). Without the native module these tests
skip; the numpy paths are what runs then."""

from __future__ import annotations

import numpy as np
import pytest

from rok.core import free_rotate, hud

rok_native = pytest.importorskip("rok_native")


def _mask(rng, shape, fill=0.4):
    m = rng.random(shape) < fill
    m[0, 0, 0] = True  # a corner cell: faces on the grid's edge
    return m


@pytest.mark.skipif(not hasattr(rok_native, "mask_surface"), reason="old native module")
@pytest.mark.parametrize("neighbours", [False, True])
@pytest.mark.parametrize("shape, lo, hi", [
    ((9, 7, 5), (0, 0, 0), (8, 6, 4)),  # the whole grid
    ((9, 7, 5), (2, 1, 0), (6, 6, 3)),  # a window inside
    ((40, 36, 33), (32, 0, 0), (63, 31, 31)),  # a skin block that reaches past the grid
    ((9, 7, 5), (-3, -1, 2), (20, 2, 9)),  # a box larger than the grid
])
def test_the_native_surface_is_the_numpy_surface(shape, lo, hi, neighbours):
    rng = np.random.default_rng(7)
    mask = _mask(rng, shape)
    for offset in (0.0, 0.003):
        want_t, want_l = hud._mask_surface_numpy(mask, lo, hi, offset, neighbours)
        got_t, got_l = rok_native.mask_surface(mask, tuple(lo), tuple(hi), offset, neighbours)
        assert got_t.shape == want_t.shape and got_l.shape == want_l.shape
        assert np.array_equal(got_t, want_t) and np.array_equal(got_l, want_l)  # same quads, same order, bit for bit
        assert got_t.dtype == np.float32 and got_l.dtype == np.float32


@pytest.mark.skipif(not hasattr(rok_native, "mask_surface"), reason="old native module")
def test_an_empty_window_gives_empty_arrays_in_both():
    mask = np.ones((4, 4, 4), dtype=bool)
    got_t, got_l = rok_native.mask_surface(mask, (3, 0, 0), (1, 3, 3), 0.003, False)
    want_t, want_l = hud._mask_surface_numpy(mask, (3, 0, 0), (1, 3, 3))
    assert got_t.shape == want_t.shape == (0, 3) and got_l.shape == want_l.shape == (0, 3)
    assert hud.mask_surface(np.zeros((3, 3, 3), dtype=bool), (0, 0, 0), (2, 2, 2))[0].shape == (0, 3)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("degrees", [37.4, -100.0, 12.3, 179.9, 45.0])
def test_the_native_free_turn_is_the_numpy_free_turn(axis, degrees):
    rng = np.random.default_rng(11)
    voxels = (rng.random((13, 9, 6)) < 0.5).astype(np.uint8) * rng.integers(1, 255, (13, 9, 6), dtype=np.uint8)
    new = free_rotate.turned_size(voxels.shape, axis, degrees)
    want = free_rotate._rotate_voxels_numpy(voxels, axis, degrees, new)
    got = np.ascontiguousarray(rok_native.bake_rotate(voxels, axis, degrees))
    assert got.shape == want.shape == tuple(new)
    assert np.array_equal(got, want)
    assert np.array_equal(free_rotate.rotate_voxels(voxels, axis, degrees), want)  # whichever path runs


@pytest.mark.skipif(not hasattr(rok_native, "mask_surface_blocks"), reason="old native module")
def test_the_batched_blocks_are_the_single_calls():
    rng = np.random.default_rng(3)
    mask = _mask(rng, (70, 40, 33))
    boxes = []
    for i in range(3):
        for j in range(2):
            for k in range(2):
                lo = np.array([i, j, k]) * 32
                boxes.append((lo, np.minimum(lo + 31, np.array(mask.shape) - 1)))
    built = hud.mask_surface_blocks(mask, boxes, neighbours=True)
    assert len(built) == len(boxes)
    for (lo, hi), (tris, lines) in zip(boxes, built):
        want_t, want_l = hud._mask_surface_numpy(mask, lo, hi, neighbours=True)
        assert np.array_equal(tris, want_t) and np.array_equal(lines, want_l)


@pytest.mark.skipif(not hasattr(rok_native, "live_turn"), reason="old native module")
@pytest.mark.parametrize("degrees, axis", [(37.4, 2), (-100.0, 0), (12.3, 1), (45.0, 2)])
def test_the_native_live_turn_is_the_numpy_live_turn(degrees, axis):
    from rok.core import gizmo

    rng = np.random.default_rng(5)
    shape = (23, 19, 11)
    press = (rng.random(shape) < 0.6).astype(np.uint8) * rng.integers(1, 255, shape, dtype=np.uint8)
    mask = np.zeros(shape, dtype=bool)
    mask[4:16, 3:14, 2:9] = rng.random((12, 11, 7)) < 0.7
    cells = np.argwhere(mask)
    bounds = (tuple(cells.min(axis=0)), tuple(cells.max(axis=0)))
    pivot2 = (23, 19, 11)
    want_n, mask_n = gizmo._live_turn_grid_numpy(press, mask, bounds, axis, degrees, pivot2)
    want, new_mask = gizmo.live_turn_grid(press, mask, bounds, axis, degrees, pivot2)
    assert np.array_equal(want, want_n) and np.array_equal(new_mask, mask_n)
    assert want.dtype == np.uint8 and new_mask.dtype == bool


@pytest.mark.skipif(not hasattr(rok_native, "apply_grid"), reason="old native module")
def test_the_native_grid_write_is_the_numpy_grid_write():
    from rok.core.grid import VoxelGrid

    rng = np.random.default_rng(6)
    size = (70, 40, 33)
    start = rng.integers(0, 4, size, dtype=np.uint8)
    want = rng.integers(0, 4, size, dtype=np.uint8)
    lock = rng.random(size) < 0.1
    a, b = VoxelGrid(size, start.copy()), VoxelGrid(size, start.copy())
    for g in (a, b):
        g.lock = lock
        g.set_section(20, axis=0)
        g.clear_dirty()
    assert a._changed_chunks_numpy(want) == a.changed_chunks(want)
    n_a = a._set_grid_numpy(want)
    n_b = b.set_grid(want)
    assert n_a == n_b > 0
    assert np.array_equal(a.voxels, b.voxels) and np.array_equal(a.visible, b.visible)
    assert a.dirty_chunks() == b.dirty_chunks()
    assert a.locked_writes == b.locked_writes > 0
    # ignore_lock and ignore_cut, as a palette swap writes.
    c, d = VoxelGrid(size, start.copy()), VoxelGrid(size, start.copy())
    for g in (c, d):
        g.lock = lock
        g.set_section(20, axis=0)
    c._set_grid_numpy(want, ignore_cut=True, ignore_lock=True)
    d.set_grid(want, ignore_cut=True, ignore_lock=True)
    assert np.array_equal(c.voxels, d.voxels) and np.array_equal(c.visible, d.visible)


@pytest.mark.skipif(not hasattr(rok_native, "seen_cells"), reason="old native module")
@pytest.mark.parametrize("camera", [
    {"eye": (40.0, -30.0, 60.0)}, {"eye": (-5.0, 12.0, 3.0)}, {"toward": (0.0, 0.0, 1.0)}, {"toward": (0.3, -0.8, 0.5)},
])
def test_the_native_seen_cells_are_the_numpy_seen_cells(camera):
    from rok.core import select

    rng = np.random.default_rng(13)
    shape = (24, 20, 16)
    voxels = (rng.random(shape) < 0.35).astype(np.uint8) * 3
    voxels[5:12, 5:12, 2:9] = 4  # a solid block: an inside no camera sees
    cells = np.argwhere(voxels != 0)
    want = select._seen_cells_numpy(voxels, cells, **camera)
    got = select.seen_cells(voxels, cells, **camera)
    assert got.dtype == bool and got.shape == want.shape
    assert np.array_equal(got, want)
    assert 0 < got.sum() < len(cells)


@pytest.mark.skipif(not hasattr(rok_native, "brush_write"), reason="old native module")
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_the_native_brush_write_is_the_numpy_brush_write(mode):
    """The brush as one box write (grid.write_brush): native and numpy agree on
    the voxels, the visible copy, the mark, the count and the dirty chunks,
    with a lock and a section plane in the way."""
    from rok.core import edit
    from rok.core.grid import VoxelGrid

    rng = np.random.default_rng(21)
    size = (70, 40, 33)
    start = rng.integers(0, 4, size, dtype=np.uint8)
    lock = rng.random(size) < 0.1
    mask = edit.sphere_mask(9)  # 17 on a side
    lo = (60, 25, 20)  # reaches past the grid on every axis: the caller clips
    clip = tuple(slice(0, min(17, size[k] - lo[k])) for k in range(3))
    mask = np.ascontiguousarray(mask[clip])
    out = []
    for native in (False, True):
        g = VoxelGrid(size, start.copy())
        g.lock = lock
        g.set_section(65, axis=0)  # cuts the box's far end
        g.clear_dirty()
        mark = np.zeros(size, dtype=bool)
        if native:
            n = g.write_brush(lo, mask, mode, 7, mark)
        else:
            cut = np.zeros(mask.shape, dtype=bool)
            cut[65 - lo[0]:, :, :] = True
            n = g._write_brush_numpy(lo, mask, mode, 7, cut, mark)
        out.append((n, g.voxels.copy(), g.visible.copy(), mark, g.dirty_chunks(), g.locked_writes))
    (n_a, v_a, vis_a, m_a, d_a, l_a), (n_b, v_b, vis_b, m_b, d_b, l_b) = out
    assert n_a == n_b > 0 and l_a == l_b > 0
    assert np.array_equal(v_a, v_b) and np.array_equal(vis_a, vis_b) and np.array_equal(m_a, m_b)
    assert d_a == d_b and len(d_a) > 1
    assert not np.array_equal(v_a, start)  # the brush wrote something
    assert (v_a[65:] == start[65:]).all()  # nothing beyond the section plane


@pytest.mark.skipif(not hasattr(rok_native, "count_colours"), reason="old native module")
def test_the_native_colour_count_is_numpys_bincount():
    from rok.core import scene

    rng = np.random.default_rng(4)
    voxels = rng.integers(0, 256, (37, 23, 19), dtype=np.uint8)
    want = np.bincount(voxels.ravel(), minlength=256)
    assert np.array_equal(scene.count_colours(voxels), want)
    assert np.array_equal(rok_native.count_colours(voxels), want)
    assert scene.count_colours(np.zeros((2, 2, 2), np.uint8))[0] == 8


# --- the cage -------------------------------------------------

from rok.core import lattice as L  # noqa: E402
from rok.core.vox_read import read_vox  # noqa: E402
from tests.helpers import FIXTURES  # noqa: E402

needs_deform = pytest.mark.skipif(not hasattr(rok_native, "live_deform"), reason="old native module")


def _lump(shape=(44, 38, 30), lo=(6, 5, 4), hi=(33, 29, 21), seed=5):
    v = np.zeros(shape, dtype=np.uint8)
    box = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
    v[box] = np.random.default_rng(seed).integers(0, 40, v[box].shape, dtype=np.uint8)  # holes and colours
    return v, v != 0, (lo, hi)


def _both(voxels, mask, lat):
    want, new = L.deform_grid(voxels, mask, lat)
    got, got_new = L.live_deform_grid(voxels, mask, lat, exact=True)
    return want, new, got, got_new


def _waived(lat, voxels, mask, want, got):
    """The cells where the two grids differ must be on a boundary the plan waives: the
    rest position within 1e-5 of a cell edge, or the residual within 1e-5 of half a cell."""
    diff = np.argwhere(want != got)
    for cell in diff:
        x = cell.astype(np.float64) + 0.5
        u, _ok = L.inverse(lat, x[None])
        q = np.asarray(lat.lo) + u[0] * np.asarray(lat.size)
        res = np.abs(L._eval(lat, u, jac=False)[0] - x).max()
        near_edge = np.abs(q - np.round(q)).min() < 1e-5
        near_half = abs(res - L.ACCEPT_RESIDUAL) < 1e-5
        if not (near_edge or near_half):
            return False, cell
    return True, None


@needs_deform
def test_the_native_cage_is_the_numpy_cage_on_the_rest_a_move_and_a_scale():
    v, m, bounds = _lump()
    base = L.rest_lattice(bounds, 3)
    moved = base.copy()
    moved.points = moved.points + np.array([3.0, -2.0, 1.0])
    scaled = base.copy()
    lo = np.asarray(base.lo, dtype=np.float64)
    scaled.points = lo + (base.points - lo) * np.array([2.0, 1.0, 1.5])  # an axis-aligned scale
    for lat in (base, moved, scaled):
        want, new, got, got_new = _both(v, m, lat)
        assert got.shape == want.shape and got.dtype == np.uint8 and got_new.dtype == bool
        assert np.array_equal(got, want) and np.array_equal(got_new, new)  # exact, no waiver


@needs_deform
@pytest.mark.parametrize("counts", [(3, 3, 3), (2, 2, 2), (4, 3, 2), (5, 5, 5), (6, 4, 5)])
def test_the_native_cage_is_the_numpy_cage_on_random_cages(counts):
    seed = 1
    v, m, bounds = _lump(seed=seed)
    rng = np.random.default_rng(100 * seed + sum(counts))
    lat = L.rest_lattice(bounds, counts)
    gap = np.asarray(lat.size) / (np.asarray(counts) - 1.0)
    lat.points = lat.points + rng.uniform(-0.45, 0.45, lat.points.shape) * gap[None, None, None, :] * 1.5
    want, new, got, got_new = _both(v, m, lat)
    ok, cell = _waived(lat, v, m, want, got)
    assert ok, f"the paths differ at {cell}, not on a waived boundary"
    assert np.array_equal(got != 0, got_new)
    assert new.sum() > 100  # the cage did something


@needs_deform
def test_the_native_cage_is_the_numpy_cage_on_a_fold_and_a_near_singular_cage():
    v, m, bounds = _lump()
    base = L.rest_lattice(bounds, 3)
    fold = L.offset_lattice(base, {"2,1,1": [-30, 0, 0], "2,2,2": [-25, -20, 5]})
    squash = base.copy()
    squash.points[1, 1, 1] = squash.points[0, 1, 1]
    flat = base.copy()
    flat.points[:, :, 1:, 2] = flat.points[0, 0, 0, 2]
    for lat in (fold, squash, flat):
        want, _new, got, _got_new = _both(v, m, lat)
        ok, cell = _waived(lat, v, m, want, got)
        assert ok, f"the paths differ at {cell}"


@needs_deform
def test_the_native_cage_clips_a_box_that_reaches_past_the_grid():
    v, m, bounds = _lump()
    lat = L.offset_lattice(L.rest_lattice(bounds, 3), {"2,0,0": [40, 0, 0], "2,2,2": [30, 9, -20], "0,0,0": [-12, -3, 0]})
    lo, hi = L.deformed_bounds(lat, v.shape)
    assert hi[0] == v.shape[0] - 1 and lo[0] == 0  # the box was clipped on both sides
    want, new, got, got_new = _both(v, m, lat)
    ok, cell = _waived(lat, v, m, want, got)
    assert ok, f"the paths differ at {cell}"
    assert np.array_equal(got_new, got_new & (got != 0))


@needs_deform
def test_the_native_wing_is_the_golden_wing():
    import json

    cage = FIXTURES / "cage"
    spec = json.loads((cage / "wing_sweep.json").read_text())
    wing = read_vox(cage / "wing.vox").voxels
    expected = read_vox(cage / "wing_sweep_expected.vox").voxels
    lat = L.offset_lattice(L.rest_lattice(spec["bounds"], spec["counts"]), spec["offsets"])
    for exact in (True, False):  # the fast pass gives the golden wing too
        got, got_new = L.live_deform_grid(wing, wing != 0, lat, exact=exact)
        assert np.array_equal(got, expected) and np.array_equal(got_new, expected != 0), f"exact={exact}"


# --- the fast pass: the inverse by checked interpolation -----

FAST_WRONG = 0.005  # share of the target cells the fast pass may colour differently from the exact one


def _target_count(lat, shape):
    return len(L.target_cells(lat, shape))


def _share_wrong(lat, v, m):
    want, _new = L.deform_grid(v, m, lat)
    got, got_new = L.live_deform_grid(v, m, lat)  # fast
    assert np.array_equal(got != 0, got_new)
    return float((want != got).sum()) / max(1, _target_count(lat, v.shape))


@needs_deform
def test_the_fast_pass_is_exact_on_the_rest_a_move_and_a_scale():
    v, m, bounds = _lump()
    base = L.rest_lattice(bounds, 3)
    moved = base.copy()
    moved.points = moved.points + np.array([3.0, -2.0, 1.0])
    scaled = base.copy()
    lo = np.asarray(base.lo, dtype=np.float64)
    scaled.points = lo + (base.points - lo) * np.array([2.0, 1.0, 1.5])
    for lat in (base, moved):
        want, new = L.deform_grid(v, m, lat)
        got, got_new = L.live_deform_grid(v, m, lat)
        assert np.array_equal(got, want) and np.array_equal(got_new, new)  # the blend of an affine map is exact
    # A scale by 1.5 puts every third cell plane exactly on a rest cell's edge: either side is right, so the
    # paths may differ there (the waiver of section 4) and nowhere else.
    want, _new = L.deform_grid(v, m, scaled)
    got, _got_new = L.live_deform_grid(v, m, scaled)
    ok, cell = _waived(scaled, v, m, want, got)
    assert ok, f"the fast pass differs at {cell}, not on a rest cell's edge"


@needs_deform
@pytest.mark.parametrize("bernstein", [False, True])
@pytest.mark.parametrize("counts", [(3, 3, 3), (2, 2, 2), (4, 3, 2), (5, 5, 5), (6, 4, 5)])
def test_the_fast_pass_agrees_with_the_exact_one_on_random_cages(counts, bernstein):
    seed = 1
    v, m, bounds = _lump(seed=seed)
    rng = np.random.default_rng(100 * seed + sum(counts))
    lat = L.rest_lattice(bounds, counts, bernstein)
    gap = np.asarray(lat.size) / (np.asarray(counts) - 1.0)
    lat.points = lat.points + rng.uniform(-0.45, 0.45, lat.points.shape) * gap[None, None, None, :] * 1.5
    assert _share_wrong(lat, v, m) <= FAST_WRONG


@needs_deform
def test_the_fast_pass_agrees_on_a_moderate_drag_to_the_cell():
    v, m, bounds = _lump()
    lat = L.offset_lattice(L.rest_lattice(bounds, 3), {"2,1,1": [8, 0, 0], "2,2,2": [4, -4, 3]})
    assert _share_wrong(lat, v, m) <= 0.002


@needs_deform
def test_the_fast_pass_survives_a_fold_a_near_singular_cage_and_a_box_past_the_grid():
    v, m, bounds = _lump()
    base = L.rest_lattice(bounds, 3)
    cages = [L.offset_lattice(base, {"2,1,1": [-30, 0, 0], "2,2,2": [-25, -20, 5]}),
             L.offset_lattice(base, {"2,0,0": [40, 0, 0], "2,2,2": [30, 9, -20], "0,0,0": [-12, -3, 0]})]
    squash = base.copy()
    squash.points[1, 1, 1] = squash.points[0, 1, 1]
    cages.append(squash)
    for lat in cages:
        got, got_new = L.live_deform_grid(v, m, lat)
        assert got.shape == v.shape and np.array_equal(got != 0, got_new | ((v != 0) & ~m))
        assert _share_wrong(lat, v, m) <= 0.05  # folds fall back to the exact solve block by block


@needs_deform
def test_a_bernstein_cage_matches_numpy_exactly_on_the_exact_kernel():
    v, m, bounds = _lump()
    for counts in ((5, 5, 5), (6, 6, 6), (4, 3, 6)):
        lat = L.rest_lattice(bounds, counts, bernstein=True)
        lat.points[-1, -1, -1] += (7.0, 3.0, -4.0)
        lat.points[0, 2, 1] += (-3.0, 5.0, 2.0)
        want, new, got, got_new = _both(v, m, lat)
        ok, cell = _waived(lat, v, m, want, got)
        assert ok, f"{counts}: the paths differ at {cell}"
        assert new.sum() > 100


@needs_deform
def test_the_pure_switch_runs_the_numpy_path(monkeypatch):
    v, m, bounds = _lump()
    lat = L.offset_lattice(L.rest_lattice(bounds, 3), {"2,1,1": [4, 0, 0]})
    monkeypatch.setattr(L, "_NATIVE", [None])  # what ROK_PURE_DEFORM gives
    want, new = L.live_deform_grid(v, m, lat)
    ref, ref_new = L.deform_grid(v, m, lat)
    assert np.array_equal(want, ref) and np.array_equal(new, ref_new)


@needs_deform
def test_the_native_cage_refuses_bad_input():
    v, m, bounds = _lump()
    lat = L.rest_lattice(bounds, 3)
    boxes = np.array([[0, 0, 0, 3, 3, 3]], dtype=np.int64)
    rok_native.live_deform(v, m, tuple(lat.lo), tuple(lat.size), (3, 3, 3), lat.points, boxes)
    with pytest.raises(ValueError):
        rok_native.live_deform(v, m, tuple(lat.lo), tuple(lat.size), (3, 3, 4), lat.points, boxes)  # points disagree
    with pytest.raises(ValueError):
        rok_native.live_deform(v, m, tuple(lat.lo), tuple(lat.size), (1, 3, 3), lat.points[:1], boxes)  # one point wide
    with pytest.raises(ValueError):
        rok_native.live_deform(v, m, tuple(lat.lo), tuple(lat.size), (3, 3, 3), lat.points,
                                np.array([[0, 0, 0, 99, 3, 3]], dtype=np.int64))  # past the grid
