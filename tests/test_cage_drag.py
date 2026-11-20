"""The cage's drag: one point pulled, the real voxels bent (pattern: test_corner_gizmo)."""

import numpy as np
import pytest

from rok.core import lattice as L
from rok.core.cage_drag import CageDrag, CageRest, ScreenBasis
from rok.core.select import Selection

# The camera looks down z: right is x, up is y, one cell is 10 pixels.
BASIS = ScreenBasis(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([10.0, 0]), np.array([0, 10.0]))
CELL_PX = {0: (10.0, 0.0), 1: (0.0, 10.0), 2: (4.0, 4.0)}
PX = 10.0
POINT_X_FACE = "CAGE_2_1_1"  # the +X face centre of a 3 x 3 x 3 cage


def block():
    voxels = np.zeros((30, 30, 30), dtype=np.uint8)
    rng = np.random.default_rng(3)
    voxels[8:20, 8:20, 8:20] = rng.integers(1, 30, (12, 12, 12), dtype=np.uint8)
    sel = Selection(voxels.shape)
    sel.replace(voxels != 0)
    return voxels, sel


def writer(voxels):
    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return len(cells)

    def set_grid(want):
        changed = int((want != voxels).sum())
        voxels[:] = want
        return changed

    return set_many, set_grid


def setup(counts=3, use_grid=True, whole=False):
    voxels, sel = block()
    rest = CageRest(L.rest_lattice(sel.bounds(), counts), voxels.copy(), sel.mask.copy())
    set_many, set_grid = writer(voxels)
    drag = lambda pid: CageDrag(pid, rest, voxels, None if whole else sel, set_many, BASIS, CELL_PX,
                                set_grid=set_grid if use_grid else None)
    return voxels, sel, rest, drag


def pull(g, dx_cells, dy_cells=0.0, start=(100.0, 100.0)):
    g.begin_screen(*start)
    return g.extend_screen(start[0] + dx_cells * PX, start[1] + dy_cells * PX)


@pytest.mark.parametrize("use_grid", [True, False])
def test_pulling_the_plus_x_face_point_three_cells_bulges_the_block(use_grid):
    voxels, sel, rest, drag = setup(use_grid=use_grid)
    before = rest.lattice.points[2, 1, 1].copy()
    rest_count = int((voxels != 0).sum())
    g = drag(POINT_X_FACE)
    assert pull(g, 3) is True
    assert g.offset == (3.0, 0.0, 0.0) and g.label() == "+3 X"
    assert int((voxels != 0).sum()) > rest_count  # a bulge: more cells
    assert set(np.unique(voxels)) - {0} <= set(np.unique(rest.rest_voxels))  # the colours are all from the rest
    assert sel.count() == int((voxels != 0).sum())  # the selection is the cells written
    assert g.finish() > 0
    assert np.allclose(rest.lattice.points[2, 1, 1], before + (3, 0, 0))  # written back for the next drag
    untouched = L.rest_lattice(((8, 8, 8), (19, 19, 19)), 3).points
    untouched[2, 1, 1] += (3, 0, 0)
    assert np.allclose(rest.lattice.points, untouched)  # only that point moved


def test_a_second_drag_starts_from_the_moved_lattice_and_from_the_rest_voxels():
    voxels, sel, rest, drag = setup()
    g = drag(POINT_X_FACE)
    pull(g, 3)
    g.finish()
    g2 = drag("CAGE_0_1_1")
    assert g2._base.points[2, 1, 1, 0] == pytest.approx(20.0 + 3)  # the press starts from the moved point
    pull(g2, -2, 0.0, start=(50.0, 50.0))
    g2.finish()
    assert rest.lattice.points[2, 1, 1, 0] == pytest.approx(20.0 + 3)  # the first drag held
    assert rest.lattice.points[0, 1, 1, 0] == pytest.approx(8.0 - 2)
    # the voxels are the rest deformed by both points, not a bend of a bend
    want, _ = L.deform_grid(rest.rest_voxels, rest.rest_mask, rest.lattice)
    assert np.array_equal(voxels, want)


def test_pulling_back_to_the_start_restores_the_rest_exactly():
    voxels, sel, rest, drag = setup()
    g = drag(POINT_X_FACE)
    pull(g, 4)
    assert not np.array_equal(voxels, rest.rest_voxels)
    g.extend_screen(100.0, 100.0)
    assert g.offset == (0.0, 0.0, 0.0)
    assert np.array_equal(voxels, rest.rest_voxels)
    assert np.array_equal(sel.mask, rest.rest_mask)


def test_a_click_writes_nothing():
    voxels, sel, rest, drag = setup()
    g = drag(POINT_X_FACE)
    g.begin_screen(100, 100)
    assert g.extend_screen(100.2, 100.1) is False
    assert g.finish() == 0
    assert np.array_equal(voxels, rest.rest_voxels)


def test_shift_moves_by_tenths():
    voxels, sel, rest, drag = setup()
    g = drag(POINT_X_FACE)
    g.set_shift(True)
    pull(g, 1.55)
    assert g.offset[0] == pytest.approx(1.6) and g.offset[1:] == (0.0, 0.0)
    g.set_shift(False)
    g.extend_screen(100.0 + 1.55 * PX, 100.0)
    assert g.offset[0] == 2.0  # whole cells without it


def test_x_y_and_z_lock_the_move_to_one_axis_and_the_same_key_unlocks():
    voxels, sel, rest, drag = setup()
    g = drag(POINT_X_FACE)
    pull(g, 3, 2)  # a diagonal pull on screen
    assert g.offset == (3.0, 2.0, 0.0)
    assert g.toggle_axis_lock(0) is True  # lock X: the pull's share along x only
    assert g.offset == (3.0, 0.0, 0.0)
    assert g.toggle_axis_lock(1) is True
    assert g.offset == (0.0, 2.0, 0.0)
    assert g.toggle_axis_lock(1) is True  # the same key unlocks
    assert g.offset == (3.0, 2.0, 0.0)
    g.toggle_axis_lock(2)  # Z has its own cell length on screen (4, 4): (30 + 20) * 4 / 32 steps
    assert g.offset[0] == 0.0 and g.offset[1] == 0.0 and g.offset[2] == round((30 * 4 + 20 * 4) / 32)


def test_a_tilted_camera_moves_the_point_in_the_screen_plane():
    voxels, sel, rest, drag = setup()
    s = 1 / np.sqrt(2)
    basis = ScreenBasis(np.array([s, 0, s]), np.array([0, 1.0, 0]), np.array([8.0, 0]), np.array([0, 10.0]))
    set_many, set_grid = writer(voxels)
    g = CageDrag(POINT_X_FACE, rest, voxels, sel, set_many, basis, CELL_PX, set_grid=set_grid)
    g.begin_screen(0, 0)
    g.extend_screen(80, 0)  # 10 units along the camera's right: x and z each by 7.07
    assert g.offset == (7.0, 0.0, 7.0)


def test_the_whole_model_cage_leaves_the_selection_alone():
    voxels, sel, rest, drag = setup(whole=True)
    sel.clear()
    g = drag(POINT_X_FACE)
    pull(g, 5)  # a face centre moves a quarter of the pull: five cells is needed to show
    assert sel.is_empty()
    assert not np.array_equal(voxels, rest.rest_voxels)


def test_a_point_that_is_not_in_the_cage_is_refused():
    voxels, sel, rest, drag = setup()
    for bad in ("CAGE_3_0_0", "CG_MOVE_X", "CAGE_0_0"):
        with pytest.raises(ValueError):
            drag(bad)


# --- Mirror -----------------------------------------------------


def symmetric_setup(mirror, counts=3, shape=30):
    """A block centred in the grid, its colours mirrored in x, y and z, so a symmetric bend stays so."""
    voxels = np.zeros((shape,) * 3, dtype=np.uint8)
    rng = np.random.default_rng(5)
    lo, hi = 9, shape - 9
    block_ = rng.integers(1, 30, (hi - lo,) * 3, dtype=np.uint8)
    for a in range(3):
        block_ = np.where(np.arange(hi - lo).reshape([-1 if i == a else 1 for i in range(3)]) < (hi - lo) // 2,
                          block_, np.flip(block_, axis=a))
    voxels[lo:hi, lo:hi, lo:hi] = block_
    sel = Selection(voxels.shape)
    sel.replace(voxels != 0)
    rest = CageRest(L.rest_lattice(sel.bounds(), counts), voxels.copy(), sel.mask.copy())
    set_many, set_grid = writer(voxels)
    drag = lambda pid: CageDrag(pid, rest, voxels, sel, set_many, BASIS, CELL_PX, set_grid=set_grid,
                                mirror=mirror, grid_shape=voxels.shape)
    return voxels, rest, drag


def test_with_mirror_x_the_mirrored_point_moves_the_other_way_and_the_bend_is_symmetric():
    voxels, rest, drag = symmetric_setup((True, False, False))
    base = rest.lattice.points.copy()
    g = drag("CAGE_2_2_1")  # a +X, +Y edge middle
    assert pull(g, 3, 2) and g.offset == (3.0, 2.0, 0.0)
    assert np.allclose(g.cage.live.points[2, 2, 1], base[2, 2, 1] + (3, 2, 0))
    assert np.allclose(g.cage.live.points[0, 2, 1], base[0, 2, 1] + (-3, 2, 0))  # x reversed, y the same
    assert np.array_equal(voxels, voxels[::-1])  # symmetric in x
    assert not np.array_equal(voxels, voxels[:, ::-1])  # y was not mirrored
    g.finish()
    moved = np.argwhere(~np.isclose(rest.lattice.points, base).all(axis=-1))
    assert {tuple(m) for m in moved} == {(2, 2, 1), (0, 2, 1)}


def test_with_mirror_on_two_axes_four_points_move_and_the_bend_is_symmetric_in_both():
    voxels, rest, drag = symmetric_setup((True, True, False))
    base = rest.lattice.points.copy()
    g = drag("CAGE_2_2_2")  # a corner
    pull(g, 3, 2)
    live = g.cage.live.points
    assert np.allclose(live[2, 2, 2], base[2, 2, 2] + (3, 2, 0))
    assert np.allclose(live[0, 2, 2], base[0, 2, 2] + (-3, 2, 0))
    assert np.allclose(live[2, 0, 2], base[2, 0, 2] + (3, -2, 0))
    assert np.allclose(live[0, 0, 2], base[0, 0, 2] + (-3, -2, 0))
    assert np.array_equal(voxels, voxels[::-1]) and np.array_equal(voxels, voxels[:, ::-1])
    assert len(g.partners) == 3


def test_a_point_on_the_mirror_plane_stays_on_it():
    voxels, rest, drag = symmetric_setup((True, False, False))
    g = drag("CAGE_1_2_1")  # a +Y face centre: its x is the plane
    pull(g, 3, 2)
    assert g.offset == (0.0, 2.0, 0.0) and g.partners == []  # the x move is dropped, no partner
    assert np.array_equal(voxels, voxels[::-1])


def test_mirror_does_nothing_on_an_axis_where_the_cage_is_not_symmetric_about_the_plane():
    voxels, sel = block()  # 8..19 in a 30 grid: the plane is at 15, the block's centre at 14
    rest = CageRest(L.rest_lattice(sel.bounds(), 3), voxels.copy(), sel.mask.copy())
    set_many, set_grid = writer(voxels)
    g = CageDrag(POINT_X_FACE, rest, voxels, sel, set_many, BASIS, CELL_PX, set_grid=set_grid,
                 mirror=(True, False, False), grid_shape=voxels.shape)
    assert g.partners == [] and g.frozen == () and g.copies == ()  # it straddles the plane: no copy either
    pull(g, 3)
    assert g.offset == (3.0, 0.0, 0.0)


def wing_setup(mirror, right_colour=None, quadrant=False):
    """Two random blocks, mirror images in x (and in y with `quadrant`), in a 30 grid;
    only the low one is selected, as a wing is. `right_colour` paints the others one colour."""
    voxels = np.zeros((30, 30, 30), dtype=np.uint8)
    rng = np.random.default_rng(7)
    ys = slice(3, 10) if quadrant else slice(10, 20)
    voxels[3:10, ys, 10:20] = rng.integers(1, 30, (7, voxels[0, ys, 0].size, 10), dtype=np.uint8)
    wing = voxels != 0
    images = [(0,), (1,), (0, 1)] if quadrant else [(0,)]
    for a in images:
        other = np.flip(voxels * wing, a)
        voxels[other != 0] = right_colour if right_colour is not None else other[other != 0]
    sel = Selection(voxels.shape)
    sel.replace(wing)
    rest = CageRest(L.rest_lattice(sel.bounds(), 3), voxels.copy(), sel.mask.copy())
    set_many, set_grid = writer(voxels)
    g = CageDrag(POINT_X_FACE, rest, voxels, sel, set_many, BASIS, CELL_PX, set_grid=set_grid,
                 mirror=mirror, grid_shape=voxels.shape)
    return voxels, sel, g


def test_with_mirror_x_a_cage_on_one_wing_bends_the_other_wing_the_mirrored_way():
    voxels, sel, g = wing_setup((True, False, False))
    assert g.copies == (0,) and g.partners == []
    wing = sel.mask.copy()
    pull(g, 3)
    assert np.array_equal(voxels, voxels[::-1])  # both wings bent, symmetric in x
    assert int((voxels[:15] != 0).sum()) > int(wing.sum())  # the bulge is real
    assert sel.count() and not sel.mask[15:].any()  # the selection stays on the dragged wing


def test_without_mirror_the_other_wing_stays_as_it_was():
    voxels, _sel, g = wing_setup((False, False, False))
    before = voxels[15:].copy()
    pull(g, 3)
    assert g.copies == () and np.array_equal(voxels[15:], before)
    assert not np.array_equal(voxels, voxels[::-1])


def test_the_mirrored_copy_bends_the_voxels_that_are_there_with_their_own_colours():
    voxels, _sel, g = wing_setup((True, False, False), right_colour=200)
    pull(g, 3)
    assert np.array_equal(voxels != 0, (voxels != 0)[::-1])  # the same shape on both sides
    assert set(np.unique(voxels[15:])) == {0, 200}  # the right wing keeps its colour


def test_with_mirror_on_two_axes_a_cage_in_one_quadrant_bends_all_four():
    voxels, _sel, g = wing_setup((True, True, False), quadrant=True)
    assert g.copies == (0, 1)
    pull(g, 3)
    assert np.array_equal(voxels, voxels[::-1]) and np.array_equal(voxels, voxels[:, ::-1])


@pytest.mark.parametrize("counts", [2, 4, 5, 6])
def test_mirror_partners_are_the_images_at_every_count(counts):
    voxels, rest, drag = symmetric_setup((True, True, True), counts=counts)
    n = counts
    g = drag(f"CAGE_0_0_{n - 1}")
    assert sorted(i for i, _s in g.partners) == sorted(
        [(n - 1, 0, n - 1), (0, n - 1, n - 1), (n - 1, n - 1, n - 1), (0, 0, 0), (n - 1, 0, 0), (0, n - 1, 0),
         (n - 1, n - 1, 0)])
    assert len(g.partners) == 7 and g.frozen == ()


# --- the axis locks ------------------------------------------------------------


def locked_drag(locks):
    voxels, sel, rest, _drag = setup()
    set_many, set_grid = writer(voxels)
    return CageDrag(POINT_X_FACE, rest, voxels, sel, set_many, BASIS, CELL_PX, set_grid=set_grid, locks=locks)


def test_a_locked_axis_stays_put_and_the_others_move():
    g = locked_drag((True, False, False))
    pull(g, 3, 2)
    assert g.offset == (0.0, 2.0, 0.0)  # the screen plane's x went nowhere
    g = locked_drag((False, True, False))
    pull(g, 3, 2)
    assert g.offset == (3.0, 0.0, 0.0)


def test_two_locks_send_the_mouse_along_the_third_axis():
    g = locked_drag((False, True, True))
    pull(g, 3, 2)
    assert g.offset == (3.0, 0.0, 0.0)  # along X by the mouse's way across the screen
    g = locked_drag((True, True, False))
    pull(g, 2, 2)  # Z is drawn at (4, 4) pixels per cell: 20 + 20 px = 5 cells
    assert g.offset[:2] == (0.0, 0.0) and g.offset[2] == 5.0


def test_three_locks_move_nothing_and_the_axis_key_cannot_get_past_a_lock():
    g = locked_drag((True, True, True))
    assert pull(g, 3, 2) is False and g.offset == (0.0, 0.0, 0.0)
    g = locked_drag((True, False, False))
    g.begin_screen(100.0, 100.0)
    g.toggle_axis_lock(0)  # the X key on a locked X
    g.extend_screen(130.0, 100.0)
    assert g.offset == (0.0, 0.0, 0.0)
