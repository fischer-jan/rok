"""The transform gizmo: geometry, projection, hit test and the drag."""

import math

import numpy as np
import pytest

from rok.core.edit import SELECT
from rok.core.gizmo import (
    degrees_label,
    snap_degrees,
    wrap_degrees,
    AXIS_COLOURS,
    FLIP_AT,
    HANDLES,
    LINE_END,
    MODEL,
    MOVE_ANCHOR,
    PLANE_AT,
    RING_RADIUS,
    RING_SEGMENTS,
    SELECTION,
    GizmoDrag,
    free_rotate_cells,
    GizmoGeometry,
    GizmoResult,
    GizmoScreen,
    geometry,
    hit_test,
    plane_point,
    project,
    screen_geometry,
    snap_pivot,
)
from rok.core.grid import VoxelGrid
from rok.core.select import Selection
from rok.core.undo import StrokeRecorder
from rok.core.vox_read import read_vox

from helpers import DOWN, FIXTURES, ORTHO, ArraySetMany, down_ray, px_of, bird_slab

SIZE = (8, 8, 8)
WIDTH = HEIGHT = 200
LENGTH = 9.0  # 90 px at 10 px per unit
PIVOT_PX = (100.0, 100.0)



def make_screen(pivot=(0.0, 0.0, 0.0)) -> GizmoScreen:
    return screen_geometry(geometry(pivot, LENGTH, DOWN), ORTHO, WIDTH, HEIGHT)


# ---------------------------------------------------------------- geometry


def test_handles_and_colours():
    assert len(set(HANDLES)) == len(HANDLES)
    assert "CENTER" in HANDLES and "SCALE_ALL" in HANDLES
    assert len(AXIS_COLOURS) == 3


def test_geometry_layout():
    g = geometry((1.0, 2.0, 3.0), LENGTH, DOWN)
    assert isinstance(g, GizmoGeometry)
    assert g.pivot.tolist() == [1.0, 2.0, 3.0] and g.length == LENGTH
    assert set(g.tris) == set(HANDLES) - {"ROT_X", "ROT_Y", "ROT_Z"}
    assert set(g.anchors) == set(HANDLES) - {"ROT_X", "ROT_Y", "ROT_Z"}
    assert set(g.rings) == {"ROT_X", "ROT_Y", "ROT_Z"}
    assert set(g.lines) == {"MOVE_X", "MOVE_Y", "MOVE_Z"}
    for axis, name in enumerate("XYZ"):
        line = g.lines[f"MOVE_{name}"]
        assert line.shape == (2, 3)
        expected = g.pivot.copy()
        expected[axis] += LINE_END * LENGTH
        assert np.allclose(line[1], expected)
        cone = g.tris[f"MOVE_{name}"]
        assert cone.shape == (12 * 2 * 3, 3)
        assert np.isclose(cone[:, axis].max(), g.pivot[axis] + LENGTH)
        assert np.isclose(cone[:, axis].min(), g.pivot[axis] + LINE_END * LENGTH)
        cube = g.tris[f"FLIP_{name}"]
        assert cube.shape == (36, 3)
        assert np.isclose(cube[:, axis].mean(), g.pivot[axis] + FLIP_AT * LENGTH)
        assert np.isclose(cube.max(axis=0)[axis] - cube.min(axis=0)[axis], 0.09 * LENGTH)
        ring = g.rings[f"ROT_{name}"]
        assert ring.shape == (RING_SEGMENTS, 3)
        assert np.allclose(ring[:, axis], g.pivot[axis])
        radii = np.linalg.norm(ring - g.pivot, axis=1)
        assert np.allclose(radii, RING_RADIUS * LENGTH, atol=1e-4)
    # PLANE_YZ sits in the yz plane at x of the pivot.
    sq = g.tris["PLANE_YZ"]
    assert sq.shape == (6, 3) and np.allclose(sq[:, 0], g.pivot[0])
    assert np.isclose(sq[:, 1].mean(), g.pivot[1] + PLANE_AT * LENGTH)
    assert np.isclose(sq[:, 2].mean(), g.pivot[2] + PLANE_AT * LENGTH)
    assert np.isclose(sq[:, 1].max() - sq[:, 1].min(), 0.13 * LENGTH)
    # The centre square faces the viewer: flat in z for a top-down view.
    centre = g.tris["CENTER"]
    assert centre.shape == (6, 3) and np.allclose(centre[:, 2], g.pivot[2])
    assert np.isclose(centre[:, 0].max() - centre[:, 0].min(), 0.1 * LENGTH)


def test_project_orthographic_and_behind_the_eye():
    pts = [(0, 0, 0), (1, -2, 3), (-10, 10, 0)]
    px = project(ORTHO, pts, WIDTH, HEIGHT)
    assert px.shape == (3, 2)
    assert px[0].tolist() == [100.0, 100.0]
    assert np.allclose(px[1], [110.0, 80.0])
    assert np.allclose(px[2], [0.0, 200.0])
    # A perspective row: w = -z; a point at z >= 0 is behind the eye.
    persp = ORTHO.copy()
    persp[3] = [0.0, 0.0, -1.0, 0.0]
    px = project(persp, [(1, 0, -2), (1, 0, 0), (1, 0, 5)], WIDTH, HEIGHT)
    assert np.allclose(px[0], [105.0, 100.0])
    assert np.isnan(px[1]).all() and np.isnan(px[2]).all()
    assert project(ORTHO, np.zeros((0, 3)), WIDTH, HEIGHT).shape == (0, 2)


def test_screen_geometry_anchors_project_where_expected():
    s = make_screen()
    assert s.pivot.tolist() == [100.0, 100.0]
    assert np.allclose(s.anchors["CENTER"][0], PIVOT_PX)
    assert np.allclose(s.anchors["MOVE_X"][0], px_of((MOVE_ANCHOR * LENGTH, 0)))
    assert np.allclose(s.anchors["MOVE_Y"][0], px_of((0, MOVE_ANCHOR * LENGTH)))
    assert np.allclose(s.anchors["MOVE_Z"][0], PIVOT_PX)  # z is the view axis
    assert np.allclose(s.anchors["FLIP_X"][0], px_of((FLIP_AT * LENGTH, 0)))
    plane_at = PLANE_AT * LENGTH
    assert np.allclose(s.anchors["PLANE_XY"][0], px_of((plane_at, plane_at)))
    assert np.allclose(s.anchors["PLANE_YZ"][0], px_of((0, PLANE_AT * LENGTH)))
    assert s.rings["ROT_Z"].shape == (RING_SEGMENTS, 2)
    assert np.allclose(np.linalg.norm(s.rings["ROT_Z"] - s.pivot, axis=1), RING_RADIUS * 90)


# ---------------------------------------------------------------- hit test


def test_hit_test_each_handle_at_its_anchor_and_none_away():
    from rok.core.gizmo import GIZMO_SKIP_IN_MODEL, GIZMO_SKIP_IN_SCENE

    s = make_screen()
    for handle, pts in s.anchors.items():
        if handle in ("MOVE_Z", "FLIP_Z"):
            continue  # under the CENTER in a top-down view
        # The centre square and the scene's sphere share the pivot: each level
        # shows one of them.
        skip = GIZMO_SKIP_IN_SCENE if handle == "SCALE_ALL" else GIZMO_SKIP_IN_MODEL
        mx, my = pts[0]
        assert hit_test(s, mx, my, 10.0, DOWN, skip=skip) == handle
        assert hit_test(s, mx + 20, my + 20, 10.0, DOWN, skip=skip) != handle
    # 20 px off the pivot toward -y, +x nothing lies (the squares sit at +, +).
    assert hit_test(s, 100 + 20, 100 - 20, 10.0, DOWN) is None
    assert hit_test(s, 100 + 5, 100 + 5, 10.0, DOWN) == "CENTER"
    assert hit_test(s, 10, 190, 10.0, DOWN) is None
    assert hit_test(None, 100, 100, 10.0, DOWN) is None


def test_hit_test_order_and_nearest():
    s = make_screen()
    # Between FLIP_X (161.2) and MOVE_X (181.9): the nearer anchor wins.
    assert hit_test(s, 168, 100, 10.0, DOWN) == "FLIP_X"
    assert hit_test(s, 176, 100, 10.0, DOWN) == "MOVE_X"
    # CENTER before MOVE_Z at the same pixel.
    assert hit_test(s, 100, 100, 10.0, DOWN) == "CENTER"


def test_hit_test_ring_and_faded_ring():
    s = make_screen()
    r = RING_RADIUS * 90
    # A point on the ROT_Z ring, away from the other handles.
    a = math.radians(20)
    mx, my = 100 + r * math.cos(a), 100 + r * math.sin(a)
    assert hit_test(s, mx, my, 10.0, DOWN) == "ROT_Z"
    assert hit_test(s, mx + 8, my + 8, 10.0, DOWN) is None
    # ROT_X is edge-on under the top-down view: its ring projects to a line
    # through the pivot and is skipped.
    y = r * math.cos(math.radians(40))
    assert hit_test(s, 100, 100 + y, 10.0, DOWN) is None
    # The same pixel hits ROT_X when the view is not edge-on.
    oblique = (0.3, 0.0, -1.0)
    assert hit_test(s, 100, 100 + y, 10.0, oblique) == "ROT_X"


# ---------------------------------------------------------------- helpers


def test_snap_pivot():
    assert snap_pivot((3, 4, 1), 2) == (3, 5, 1)
    assert snap_pivot((3, 5, 1), 2) == (3, 5, 1)
    assert snap_pivot((3, 4, 1), 0) == (3, 4, 2)  # y, z: 4 and 1 differ -> z + 1
    assert snap_pivot((3, 4, 2), 0) == (3, 4, 2)
    assert snap_pivot((3, 4, 2), 1) == (3, 4, 3)  # x, z: 3 and 2 differ -> z + 1
    assert snap_pivot((3, 4, 3), 1) == (3, 4, 3)
    assert snap_pivot((4, 4, 3), 1) == (4, 4, 4)


def test_plane_point():
    p = plane_point((1, 2, 10), (0, 0, -1), (0, 0, 0), (0, 0, 1))
    assert np.allclose(p, [1, 2, 0])
    assert plane_point((1, 2, 10), (1, 0, 0), (0, 0, 0), (0, 0, 1)) is None
    assert plane_point((1, 2, -10), (0, 0, -1), (0, 0, 0), (0, 0, 1)) is None
    p = plane_point((2, 3, 50), (0, 0, -1), (0, 0, 0), (1, 1, -1))
    assert np.allclose(p, [2, 3, 5])


# ---------------------------------------------------------------- the drag


def make_drag(handle, target=SELECTION, shift=False, view_dir=DOWN, pivot2=(0, 0, 0)):
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[2, 2, 2] = 3
    sel = Selection(SIZE)
    sel.add_cells([(2, 2, 2)])
    sm = ArraySetMany(voxels)
    g = GizmoDrag(
        handle,
        target,
        pivot2,
        voxels,
        sel,
        sm,
        shift=shift,
        view_dir=view_dir,
        pivot_px=PIVOT_PX,
        size=SIZE,
    )
    return g, voxels, sel, sm


def test_drag_protocol_attributes():
    g, *_ = make_drag("MOVE_X")
    assert g.tool == SELECT and g.hit is None and g.last is None
    with pytest.raises(ValueError):
        make_drag("SCALE_X")


def test_move_x_with_rays_2_4_units_apart_gives_offset_2():
    g, voxels, sel, sm = make_drag("MOVE_X")
    assert g.begin_ray(*down_ray(8.19, 0.0)) is True
    assert g.extend(*down_ray(8.19 + 0.3, 0.0)) is False
    assert g.extend(*down_ray(8.19 + 2.4, 0.0)) is True
    assert g.result() == GizmoResult("MOVE", offset=(2, 0, 0))
    assert g.extend(*down_ray(8.19 + 2.4, 5.0)) is False  # off-axis does not count
    p = g.preview()
    assert p.kind == "MOVE" and p.offset == (2, 0, 0) and p.label == "+2 X"
    assert g.finish() == 2
    assert int(voxels[4, 2, 2]) == 3 and int(voxels[2, 2, 2]) == 0
    assert sel.contains((4, 2, 2)) and sel.count() == 1


def test_move_with_shift_copies():
    g, voxels, sel, sm = make_drag("MOVE_Y", shift=True)
    assert g.begin_ray(*down_ray(0.0, 8.19))
    assert g.extend(*down_ray(0.0, 8.19 - 2.0))
    assert g.result() == GizmoResult("MOVE", offset=(0, -2, 0), copy=True)
    assert g.preview().label == "copy -2 Y"
    assert g.finish() == 1
    assert int(voxels[2, 2, 2]) == 3 and int(voxels[2, 0, 2]) == 3
    assert sel.contains((2, 0, 2)) and not sel.contains((2, 2, 2))


def test_move_zero_offset_is_none_and_writes_nothing():
    g, voxels, sel, sm = make_drag("MOVE_X")
    assert g.begin_ray(*down_ray(8.19, 0.0))
    assert g.result().kind == "NONE"
    assert g.preview().label == ""
    assert g.finish() == 0 and sm.calls == []


def test_move_begin_fails_for_a_ray_along_the_axis():
    g, *_ = make_drag("MOVE_Z")
    assert g.begin_ray((0.0, 0.0, 50.0), DOWN) is False


def test_plane_xy_gives_two_components():
    g, voxels, sel, sm = make_drag("PLANE_XY")
    assert g.begin_ray(*down_ray(2.52, 2.52))
    assert g.extend(*down_ray(2.52 + 2.4, 2.52 - 1.4)) is True
    assert g.result().offset == (2, -1, 0)
    assert g.preview().label == "+2 X -1 Y"
    assert g.finish() == 2
    assert int(voxels[4, 1, 2]) == 3


def test_plane_yz_zeroes_x():
    g, voxels, sel, sm = make_drag("PLANE_YZ")
    # The plane x = 0 is hit by a ray along -x.
    origin = (50.0, 2.52, 2.52)
    assert g.begin_ray(origin, (-1.0, 0.0, 0.0))
    assert g.extend((50.0, 2.52 + 1.0, 2.52 + 2.0), (-1.0, 0.0, 0.0))
    assert g.result().offset == (0, 1, 2)


def test_center_gives_three_components():
    view = (1.0, 1.0, -1.0)
    g, voxels, sel, sm = make_drag("CENTER", view_dir=view)
    # The plane x + y - z = 0 through the pivot: a ray down at (x, y) hits z = x + y.
    assert g.begin_ray(*down_ray(0.0, 0.0))
    assert g.extend(*down_ray(2.0, 3.0)) is True
    assert g.result().offset == (2, 3, 5)
    assert g.preview().label == "+2 X +3 Y +5 Z"


def test_center_with_shift_moves_the_pivot():
    g, voxels, sel, sm = make_drag("CENTER", shift=True, pivot2=(3, 3, 3))
    assert g.begin_ray(*down_ray(1.5, 1.5))
    assert g.extend(*down_ray(2.3, 0.9)) is True
    assert g.result() == GizmoResult("PIVOT", pivot2=(5, 2, 3))
    assert g.preview().label == "pivot"
    assert g.finish() == 0 and sm.calls == []
    # The pivot never leaves the grid box.
    assert g.extend(*down_ray(-4.0, 30.0)) is True
    assert g.result().pivot2 == (0, 16, 3)


def turn_mouse(g, deg):
    a = math.radians(deg)
    g.set_mouse(100 + 50 * math.cos(a), 100 + 50 * math.sin(a))


def test_rot_z_turns_freely_in_tenths_and_snaps_to_quarter_turns():
    g, voxels, sel, sm = make_drag("ROT_Z")
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0)) is True
    assert g.result().turns == 1  # a press alone is one quarter turn
    turn_mouse(g, 37.44)
    assert g.extend(*down_ray(0.0, 0.0)) is True
    res = g.result()
    assert res.kind == "ROTATE" and res.turns == 0 and res.degrees == 37.4 and res.free
    assert g.preview().label == "37.4° Z"
    turn_mouse(g, 93.0)  # within the snap: an exact quarter turn
    g.extend(*down_ray(0.0, 0.0))
    assert g.result() == GizmoResult("ROTATE", axis=2, turns=1, pivot2=(0, 0, 0), degrees=90.0)
    assert not g.result().free and g.preview().label == "90° Z"
    turn_mouse(g, 100.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().degrees == 100.0 and g.result().turns == 0
    turn_mouse(g, 181.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().turns == 2 and g.result().degrees == 180.0
    # Back to the start: the sweep was made, so zero is zero.
    g.set_mouse(150.0, 100.0)
    assert g.extend(*down_ray(0.0, 0.0)) is True
    assert g.result().kind == "NONE"
    assert g.finish() == 0


def test_rot_ctrl_turns_in_15_degree_steps_and_shift_turns_slower():
    g, *_ = make_drag("ROT_Z")
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    g.coarse = True
    turn_mouse(g, 37.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().degrees == 30.0
    g, *_ = make_drag("ROT_Z")
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    g.set_shift(True)
    turn_mouse(g, 50.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().degrees == 5.0
    assert g.result().copy is False  # Shift during the drag never copies


def test_rot_sign_flips_when_the_axis_points_away():
    up = (0.0, 0.0, 1.0)  # the view looks along +z: z points away
    g, *_ = make_drag("ROT_Z", view_dir=up)
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    turn_mouse(g, 100.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().degrees == -100.0
    turn_mouse(g, 90.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().turns == 3


def test_rot_sweep_past_180_keeps_counting():
    g, *_ = make_drag("ROT_Z")
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    for deg in (60, 120, 180, 240, 271):
        turn_mouse(g, deg)
        g.extend(*down_ray(0.0, 0.0))
    assert g.result().turns == 3 and g.result().degrees == -90.0


def test_rot_mouse_after_begin_sets_the_reference():
    g, *_ = make_drag("ROT_Z")
    assert g.begin_ray(*down_ray(5.0, 0.0))
    g.set_mouse(150.0, 100.0)
    assert g.extend(*down_ray(0.0, 0.0)) is False
    turn_mouse(g, 100.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().degrees == 100.0


def test_turning_a_selection_back_to_zero_gives_the_voxels_back():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1:6, 4, 2] = 5
    voxels[1, 5, 2] = 7
    start = voxels.copy()
    sel = Selection(SIZE)
    sel.add_cells([(x, 4, 2) for x in range(1, 6)] + [(1, 5, 2)])
    mask = sel.mask.copy()
    g = GizmoDrag("ROT_Z", SELECTION, (7, 9, 5), voxels, sel, ArraySetMany(voxels), shift=False,
                  view_dir=DOWN, pivot_px=PIVOT_PX, size=SIZE)
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    for deg in (20.0, 45.0, 70.0, 33.0, 10.0):
        turn_mouse(g, deg)
        g.extend(*down_ray(0.0, 0.0))
        assert not np.array_equal(voxels, start)
    g.set_mouse(150.0, 100.0)
    g.extend(*down_ray(0.0, 0.0))
    assert np.array_equal(voxels, start) and np.array_equal(sel.mask, mask)
    assert g.finish() == 0


def test_wrap_snap_and_label():
    assert wrap_degrees(190.0) == -170.0 and wrap_degrees(-180.0) == 180.0
    assert snap_degrees(86.5) == 90.0 and snap_degrees(85.9) == 85.9
    assert snap_degrees(-3.0) == 0.0 and snap_degrees(22.0, coarse=True) == 15.0
    assert degrees_label(12.34, 0) == "12.3° X" and degrees_label(-90.0, 1) == "-90° Y"


def test_free_turn_of_a_selection_writes_the_turned_cells():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1:6, 4, 2] = 5  # a bar of five along x
    sel = Selection(SIZE)
    sel.add_cells([(x, 4, 2) for x in range(1, 6)])
    sm = ArraySetMany(voxels)
    g = GizmoDrag("ROT_Z", SELECTION, (7, 9, 5), voxels, sel, sm, shift=False,
                  view_dir=DOWN, pivot_px=PIVOT_PX, size=SIZE)
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    turn_mouse(g, 45.0)
    g.extend(*down_ray(0.0, 0.0))
    assert g.result().free and g.result().degrees == 45.0
    live = voxels.copy()  # the drag already turned the voxels
    assert not np.array_equal(live[1:6, 4, 2], [5] * 5)
    assert g.finish() > 0
    assert np.array_equal(voxels, live)
    filled = np.argwhere(voxels != 0)
    assert sel.count() == len(filled)
    # A diagonal: both x and y change along the bar.
    assert np.ptp(filled[:, 0]) >= 2 and np.ptp(filled[:, 1]) >= 2
    assert set(np.unique(voxels[voxels != 0]).tolist()) == {5}


def test_rot_on_a_faded_ring_does_not_begin():
    g, *_ = make_drag("ROT_X")
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0)) is False


def test_click_on_a_ring_is_one_turn_on_finish():
    g, voxels, sel, sm = make_drag("ROT_Z", pivot2=(5, 5, 5))
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    p = g.preview()
    assert p.kind == "LABEL" and p.label == "90° Z"
    assert int(voxels[2, 2, 2]) == 3  # a press alone writes nothing yet
    g.finish()  # the cell lands where it was: cleared, then written back
    assert int(voxels[2, 2, 2]) == 3 and int((voxels != 0).sum()) == 1
    assert sel.contains((2, 2, 2)) and sel.count() == 1
    g, voxels, sel, sm = make_drag("ROT_Z", pivot2=(7, 7, 5))
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    # About the centre of cell (3, 3): (2, 2) -> (4, 2).
    assert g.finish() == 2
    assert int(voxels[4, 2, 2]) == 3 and int(voxels[2, 2, 2]) == 0
    assert sel.contains((4, 2, 2)) and sel.count() == 1


def test_half_turn_keeps_a_3x2_block_in_place():
    # A 3 x 2 block: its doubled centre (3, 6) has mixed parity. A half turn
    # about it maps cells to cells, so no snap; the block keeps its bounds.
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[0:3, 2:4, 2] = 3
    sel = Selection(SIZE)
    sel.add_cells([(x, y, 2) for x in range(3) for y in (2, 3)])
    sm = ArraySetMany(voxels)
    g = GizmoDrag("ROT_Z", SELECTION, (3, 6, 5), voxels, sel, sm, shift=False,
                  view_dir=DOWN, pivot_px=PIVOT_PX, size=SIZE)
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    turn_mouse(g, 180.0)
    g.extend(*down_ray(0.0, 0.0))
    res = g.result()
    assert res.turns == 2 and res.pivot2 == (3, 6, 5)
    assert (voxels[0:3, 2:4, 2] == 3).all() and int((voxels != 0).sum()) == 6
    g.finish()
    assert int((voxels != 0).sum()) == 6
    assert (voxels[0:3, 2:4, 2] == 3).all()
    assert sel.count() == 6 and sel.contains((0, 2, 2)) and sel.contains((2, 3, 2))


def test_rot_result_carries_the_snapped_pivot():
    g, *_ = make_drag("ROT_Z", target=MODEL, pivot2=(3, 4, 1))
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    res = g.result()
    assert res.kind == "ROTATE" and res.turns == 1 and res.pivot2 == (3, 5, 1)
    p = g.preview()
    assert p.kind == "LABEL" and p.label == "90° Z" and len(p.cells) == 0
    # Two turns need no snap: the pivot the gizmo draws is the one that turns.
    turn_mouse(g, 180.0)
    g.extend(*down_ray(0.0, 0.0))
    res = g.result()
    assert res.turns == 2 and res.pivot2 == (3, 4, 1)
    assert g.finish() == 0


def test_flip_applies_on_finish():
    g, voxels, sel, sm = make_drag("FLIP_X", pivot2=(6, 5, 5))
    assert g.begin_ray(*down_ray(0.0, 0.0)) is True
    assert g.extend(*down_ray(3.0, 3.0)) is False
    assert g.result() == GizmoResult("FLIP", axis=0, pivot2=(6, 5, 5))
    p = g.preview()
    assert p.kind == "CELLS" and p.label == "flip X"
    assert p.cells.tolist() == [[3, 2, 2]]
    assert g.finish() == 2
    assert int(voxels[3, 2, 2]) == 3 and int(voxels[2, 2, 2]) == 0
    assert sel.contains((3, 2, 2)) and sel.count() == 1


def test_model_target_never_writes():
    g, voxels, sel, sm = make_drag("MOVE_X", target=MODEL)
    assert g.begin_ray(*down_ray(8.19, 0.0))
    assert g.extend(*down_ray(8.19 + 2.4, 0.0))
    assert g.result().offset == (2, 0, 0)
    # The label only: the Empty shows the move, so no cells and no MOVE offset to draw.
    p = g.preview()
    assert p.kind == "LABEL" and p.label == "+2 X" and p.offset == (0, 0, 0)
    assert g.finish() == 0 and sm.calls == []
    assert int(voxels[2, 2, 2]) == 3


def test_model_target_labels_flip_copy_and_pivot():
    g, *_ = make_drag("FLIP_Y", target=MODEL, pivot2=(6, 5, 5))
    assert g.begin_ray(*down_ray(0.0, 0.0))
    assert g.preview().label == "flip Y"
    # Shift copies a model as it copies a selection: the label says so.
    g, *_ = make_drag("MOVE_X", target=MODEL, shift=True)
    assert g.begin_ray(*down_ray(8.19, 0.0))
    assert g.extend(*down_ray(8.19 + 3.4, 0.0))
    assert g.preview().label == "copy +3 X"
    assert g.result().copy is True
    g, *_ = make_drag("CENTER", target=MODEL, shift=True)
    assert g.begin_ray(*down_ray(0.0, 0.0))
    assert g.preview().label == "pivot"


# ---------------------------------------------------------------- golden


def test_golden_bird_selection_turn_through_recorder_and_revert():
    model = read_vox(FIXTURES / "bird.vox")
    grid = VoxelGrid.from_model(model)
    _, sel = bird_slab()
    lo, hi = sel.bounds()
    pivot2 = (lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1)
    rec = StrokeRecorder(grid)
    g = GizmoDrag(
        "ROT_Z",
        SELECTION,
        pivot2,
        grid.voxels,
        sel,
        rec.set_many,
        shift=False,
        view_dir=DOWN,
        pivot_px=PIVOT_PX,
        size=grid.voxels.shape,
    )
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(35.0, 35.0))
    count = sel.count()
    turn_mouse(g, 37.0)  # a free angle on the way, then back to a quarter turn
    g.extend(*down_ray(35.0, 35.0))
    assert not np.array_equal(grid.voxels, model.voxels)  # the voxels turn during the drag
    turn_mouse(g, 91.0)
    g.extend(*down_ray(35.0, 35.0))
    assert g.preview().kind == "LABEL" and g.preview().label == "90° Z"
    assert g.finish() > 0
    assert sel.count() == count
    assert not np.array_equal(grid.voxels, model.voxels)
    record = rec.finish()
    assert record is not None
    rec.revert()
    assert grid.voxels.tobytes() == model.voxels.tobytes()
    record.redo(grid)
    assert not np.array_equal(grid.voxels, model.voxels)
    record.undo(grid)
    assert grid.voxels.tobytes() == model.voxels.tobytes()


def test_golden_bird_selection_flip_through_recorder():
    model = read_vox(FIXTURES / "bird.vox")
    grid = VoxelGrid.from_model(model)
    _, sel = bird_slab()
    lo, hi = sel.bounds()
    pivot2 = (lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1)
    rec = StrokeRecorder(grid)
    g = GizmoDrag(
        "FLIP_Z",
        SELECTION,
        pivot2,
        grid.voxels,
        sel,
        rec.set_many,
        shift=False,
        view_dir=DOWN,
        pivot_px=PIVOT_PX,
        size=grid.voxels.shape,
    )
    assert g.begin_ray(*down_ray(35.0, 35.0))
    count = sel.count()
    assert g.finish() > 0
    assert sel.count() == count
    assert int((grid.voxels != 0).sum()) == int((model.voxels != 0).sum())
    # The flipped cells lie in the same columns, upside down.
    assert np.array_equal(sel.mask.any(axis=2), bird_slab()[1].mask.any(axis=2))
    rec.revert()
    assert grid.voxels.tobytes() == model.voxels.tobytes()


def test_shift_on_an_arrow_copies_a_model_and_a_selection_alike():
    import numpy as np

    from rok.core.gizmo import MODEL, SELECTION, GizmoDrag

    v = np.zeros((4, 4, 4), dtype=np.uint8)

    def drag(handle, target, shift=False):
        return GizmoDrag(handle, target, (4, 4, 4), v, None, None, shift=shift, view_dir=(0, 1, -1),
                         pivot_px=(0, 0), size=(4, 4, 4))

    assert drag("MOVE_X", MODEL, shift=True).copy and drag("MOVE_X", SELECTION, shift=True).copy
    assert not drag("MOVE_X", MODEL).copy
    assert not drag("ROT_Z", MODEL, shift=True).copy  # only moves copy


def test_a_live_turn_is_one_write_a_step_and_matches_a_turn_done_at_once():
    """The live turn of a big selection crawled: four full
    writes a step (the press state back, the old cells cleared, the new ones
    written). Now one write of the cells that differ, with the same result."""
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[1:6, 4, 2] = 5
    voxels[1:4, 3, 2] = 6
    sel = Selection(SIZE)
    sel.add_cells([(x, 4, 2) for x in range(1, 6)] + [(x, 3, 2) for x in range(1, 4)])
    writes = []

    class Counting(ArraySetMany):
        def __call__(self, cells, values):
            writes.append(len(np.asarray(cells).reshape(-1, 3)))
            return super().__call__(cells, values)

    sm = Counting(voxels)
    g = GizmoDrag("ROT_Z", SELECTION, (7, 9, 5), voxels, sel, sm, shift=False,
                  view_dir=DOWN, pivot_px=PIVOT_PX, size=SIZE)
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    for deg in (20.0, 45.0, 70.0):
        turn_mouse(g, deg)
        g.extend(*down_ray(0.0, 0.0))
    assert len(writes) == 3  # one write a step
    live = voxels.copy()
    # The same as the 70 degree turn written once from the press.
    fresh = np.zeros(SIZE, dtype=np.uint8)
    fresh[1:6, 4, 2] = 5
    fresh[1:4, 3, 2] = 6
    fresh_sel = Selection(SIZE)
    fresh_sel.add_cells(np.argwhere(fresh != 0))
    res = g.result()
    free_rotate_cells(fresh, fresh_sel, res.axis, res.degrees, res.pivot2, ArraySetMany(fresh))
    assert np.array_equal(live, fresh)
    assert np.array_equal(sel.mask, fresh != 0)


def test_a_live_turn_through_the_grid_write_matches_a_turn_done_at_once():
    """The mask-based write (StrokeRecorder.set_grid) of a live free turn gives
    the same voxels and selection as free_rotate_cells written cell by cell,
    and its record undoes back to the press."""
    size = (12, 12, 6)
    start = np.zeros(size, dtype=np.uint8)
    start[2:7, 5, 2] = 5
    start[2:5, 4, 2] = 6
    start[9, 9, 3] = 9  # not selected: stays
    grid = VoxelGrid(size, start.copy())
    grid.clear_dirty()
    sel = Selection(size)
    sel.add_cells([(x, 5, 2) for x in range(2, 7)] + [(x, 4, 2) for x in range(2, 5)])
    recorder = StrokeRecorder(grid)
    g = GizmoDrag("ROT_Z", SELECTION, (9, 10, 5), grid.visible, sel, recorder.set_many, shift=False,
                  view_dir=DOWN, pivot_px=PIVOT_PX, size=size, set_grid=recorder.set_grid)
    g.set_mouse(150.0, 100.0)
    assert g.begin_ray(*down_ray(5.0, 0.0))
    for deg in (20.0, 55.0):
        turn_mouse(g, deg)
        g.extend(*down_ray(0.0, 0.0))
    res = g.result()
    assert res.free
    fresh = start.copy()
    fresh_sel = Selection(size)
    fresh_sel.add_cells(np.argwhere((start != 0) & (start != 9)))
    new_mask = free_rotate_cells(fresh, fresh_sel, res.axis, res.degrees, res.pivot2, ArraySetMany(fresh))
    assert np.array_equal(grid.voxels, fresh)
    assert np.array_equal(sel.mask, new_mask)
    assert grid.dirty_chunks()  # the write reached the GPU's dirty list
    record = recorder.finish()
    assert record is not None
    record.undo(grid)
    assert np.array_equal(grid.voxels, start)


def test_a_selection_whose_turned_box_is_wider_than_a_grid_still_turns():
    """A selected 210-cube would not turn; its box turned by
    45 degrees is wider than the grid's limit of 256. The box is clipped to
    the grid when it is pasted back, so the limit does not apply to it."""
    from rok.core.gizmo import free_turned_cells

    size = (200, 200, 3)
    voxels = np.zeros(size, dtype=np.uint8)
    voxels[5:195, 5:195, 1] = 7  # 190 wide: turned by 45 degrees it spans 269
    sel = Selection(size)
    sel.add_mask(voxels != 0)
    cells, values = free_turned_cells(voxels, sel, 2, 45.0, (200, 200, 3))
    assert len(cells) > 0 and set(values.tolist()) == {7}
    assert (cells >= 0).all() and (cells < np.asarray(size)).all()  # clipped to the grid
    assert len(cells) < int((voxels != 0).sum())  # the corners that left the grid are gone
