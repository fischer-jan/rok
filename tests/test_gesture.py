"""Shape and Face gestures, the preview and the factory, with injected setters."""

import numpy as np
import pytest

from rok.core.edit import (
    ATTACH,
    BOX,
    CENTER,
    ERASE,
    FACE,
    LASSO,
    LINE,
    PAINT,
    RECT,
    SELECT,
    VOXEL,
    Stroke,
)
from rok.core.gesture import (
    LASSO_MAX_VERTICES,
    FaceGesture,
    PatternGesture,
    Preview,
    ScreenGesture,
    ShapeGesture,
    make_gesture,
)
from rok.core.raycast import Hit, raycast
from rok.core.select import PickGesture, Selection, footprint_mask
from rok.core.vox_read import read_vox

from helpers import FIXTURES, ArraySetMany, ArraySetter, filled, top_down, top_ray

SIZE = (8, 8, 8)
NO_MIRROR = (False, False, False)


def side_ray(x, z, y=-10.0):
    """A ray along +Y through the centre of cell (x, *, z), perpendicular to +Z."""
    return (x + 0.5, y, z + 0.5), (0.0, 1.0, 0.0)


def cells_set(cells: np.ndarray) -> set:
    return {tuple(c) for c in np.asarray(cells).reshape(-1, 3).tolist()}


def floor_hit(x, y) -> Hit:
    return Hit((x, y, -1), (0, 0, 1), 1.0, wall=True)


def top_hit(x, y, z) -> Hit:
    return Hit((x, y, z), (0, 0, 1), 1.0)


def make_shape(voxels, mode, tool, colour=5, mirror=NO_MIRROR, selection=None, subtract=False):
    sm = ArraySetMany(voxels)
    return ShapeGesture(voxels, mode, tool, colour, mirror, sm, selection, subtract), sm


def make_face(voxels, tool, colour=5, mirror=NO_MIRROR, selection=None, subtract=False):
    sm = ArraySetMany(voxels)
    return FaceGesture(voxels, tool, colour, mirror, sm, selection, subtract), sm


def rect(x0, x1, y0, y1, z) -> set:
    return {(x, y, z) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


# ---------------------------------------------------------------- Preview


def test_preview_defaults():
    p = Preview(kind="CELLS", tool=ATTACH)
    assert p.cells.shape == (0, 3)
    assert p.normal == (0, 0, 1)
    assert p.box is None
    assert p.label == ""
    assert p.points.shape == (0, 2) and p.points.dtype == np.float32
    assert p.offset == (0, 0, 0)
    assert p.subtract is False


# ---------------------------------------------------------------- ShapeGesture on the floor


def test_box_attach_on_floor_writes_the_rectangle_at_z0():
    voxels = np.zeros(SIZE, np.uint8)
    g, sm = make_shape(voxels, BOX, ATTACH)
    assert g.begin(floor_hit(2, 2))
    assert g.hit == floor_hit(2, 2)
    assert g.last == (2, 2, 0)
    assert g.lock.axis == 2 and g.lock.plane == 0.0 and g.lock.layer == 0
    assert sm.calls == []
    assert g.extend(*top_ray(5, 4))
    assert g.last == (5, 4, 0)
    assert sm.calls == []
    assert g.finish() == 12
    assert filled(voxels, 5) == rect(2, 5, 2, 4, 0)
    assert len(sm.calls) == 1


def test_line_attach_on_floor_writes_the_line():
    voxels = np.zeros(SIZE, np.uint8)
    g, sm = make_shape(voxels, LINE, ATTACH)
    assert g.begin(floor_hit(2, 2))
    assert g.extend(*top_ray(5, 4))
    assert g.preview().label == "4"
    assert g.finish() == 4
    cells = filled(voxels, 5)
    assert len(cells) == 4
    assert (2, 2, 0) in cells and (5, 4, 0) in cells
    assert all(z == 0 for _, _, z in cells)


def test_line_attach_follows_the_surface_up_a_wall():
    # A wall of height 3 at x = 4. Press on the floor at (1, 2), drag onto the
    # wall's top: the line climbs from the floor to the cell above the wall.
    voxels = np.zeros(SIZE, np.uint8)
    voxels[4, :, :3] = 1
    g, sm = make_shape(voxels, LINE, ATTACH)
    assert g.begin(floor_hit(1, 2))
    assert g.extend(*top_ray(4, 2))
    assert g.current == (4, 2, 3)
    preview = g.preview()
    assert preview.kind == "CELLS" and preview.label == "4"
    assert g.finish() == 4
    cells = filled(voxels, 5)
    assert (1, 2, 0) in cells and (4, 2, 3) in cells
    assert {z for _, _, z in cells} == {0, 1, 2, 3}


def test_line_attach_on_a_side_face_can_go_sideways():
    # Press on the -Y side of a column, drag along the floor in X: the plane of
    # the pressed face would allow only Y-Z, but the line follows the surface.
    voxels = np.zeros(SIZE, np.uint8)
    voxels[2, 3, :2] = 1
    g, sm = make_shape(voxels, LINE, ATTACH)
    assert g.begin(Hit((2, 3, 0), (0, -1, 0), 1.0))
    assert g.anchor == (2, 2, 0)
    assert g.extend(*top_ray(6, 2))
    assert g.current == (6, 2, 0)
    assert g.finish() == 5
    assert filled(voxels, 5) == {(x, 2, 0) for x in range(2, 7)}


def test_line_keeps_its_end_when_the_ray_leaves_the_grid():
    voxels = np.zeros(SIZE, np.uint8)
    g, sm = make_shape(voxels, LINE, ATTACH)
    assert g.begin(floor_hit(2, 2))
    assert g.extend(*top_ray(5, 4))
    assert not g.extend((20.5, 20.5, 50.0), (0.0, 0.0, -1.0))
    assert g.current == (5, 4, 0)


def test_center_attach_on_floor_writes_the_centred_rectangle():
    voxels = np.zeros(SIZE, np.uint8)
    g, sm = make_shape(voxels, CENTER, ATTACH)
    assert g.begin(floor_hit(2, 2))
    assert g.extend(*top_ray(5, 4))
    # Half extents 3 and 2 around (2, 2); x clips at the border.
    assert g.preview().label == "7 x 5"
    assert g.finish() == 30
    assert filled(voxels, 5) == rect(0, 5, 0, 4, 0)


def test_click_without_drag_writes_one_cell():
    voxels = np.zeros(SIZE, np.uint8)
    for mode in (BOX, LINE, CENTER):
        voxels[...] = 0
        g, sm = make_shape(voxels, mode, ATTACH)
        assert g.begin(floor_hit(3, 3))
        assert g.finish() == 1
        assert filled(voxels) == {(3, 3, 0)}


def test_extend_returns_false_when_the_cell_does_not_change():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_shape(voxels, BOX, ATTACH)
    assert g.begin(floor_hit(2, 2))
    assert not g.extend(*top_ray(2, 2))
    assert g.extend(*top_ray(4, 4))
    assert not g.extend(*top_ray(4, 4))
    assert not g.extend((4.9, 4.1, 50.0), (0.0, 0.0, -1.0))


def test_extend_returns_false_when_the_ray_misses_the_plane():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_shape(voxels, BOX, ATTACH)
    assert g.begin(floor_hit(2, 2))
    # Parallel to the plane.
    assert not g.extend((4.5, -10.0, 3.0), (0.0, 1.0, 0.0))
    # The plane lies behind the origin.
    assert not g.extend((4.5, 4.5, -5.0), (0.0, 0.0, -1.0))
    assert g.current == (2, 2, 0)


def test_extend_clamps_a_drag_off_the_grid_to_the_border():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_shape(voxels, BOX, ATTACH)
    assert g.begin(floor_hit(2, 2))
    assert g.extend(*top_ray(20, -5))
    assert g.current == (7, 0, 0)
    assert g.finish() == 6 * 3


def test_preview_cells_equal_the_cells_finish_writes():
    voxels = np.zeros(SIZE, np.uint8)
    g, sm = make_shape(voxels, BOX, ATTACH)
    assert g.begin(floor_hit(1, 1))
    assert g.extend(*top_ray(3, 6))
    p = g.preview()
    assert p.kind == "PLANE" and p.tool == ATTACH
    assert p.normal == (0, 0, 1)
    assert p.box == ((1, 1, 0), (3, 6, 0))
    assert p.label == "3 x 6"
    assert g.finish() == 18
    assert cells_set(p.cells) == filled(voxels, 5) == cells_set(sm.calls[0][0])


def test_box_attach_with_mirror_x_writes_the_images():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_shape(voxels, BOX, ATTACH, mirror=(True, False, False))
    assert g.begin(floor_hit(1, 1))
    assert g.extend(*top_ray(2, 2))
    assert g.finish() == 8
    assert filled(voxels, 5) == rect(1, 2, 1, 2, 0) | rect(5, 6, 1, 2, 0)


def test_box_attach_with_mirror_does_not_overwrite_a_filled_image():
    voxels = np.zeros(SIZE, np.uint8)
    voxels[6, 1, 0] = 9
    g, _ = make_shape(voxels, BOX, ATTACH, mirror=(True, False, False))
    assert g.begin(floor_hit(1, 1))
    assert g.finish() == 1
    assert voxels[6, 1, 0] == 9 and voxels[1, 1, 0] == 5


def test_box_attach_on_a_side_wall():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_shape(voxels, BOX, ATTACH)
    # The +X wall: the cell just outside with the normal pointing back in.
    hit = Hit((8, 2, 2), (-1, 0, 0), 1.0, wall=True)
    assert g.begin(hit)
    assert g.lock.axis == 0 and g.lock.plane == 8.0 and g.lock.layer == 7
    # A ray along -X that crosses x == 8 at (y, z) == (4, 5).
    assert g.extend((20.0, 4.5, 5.5), (-1.0, 0.0, 0.0))
    assert g.finish() == 3 * 4
    assert filled(voxels, 5) == {(7, y, z) for y in range(2, 5) for z in range(2, 6)}


# ---------------------------------------------------------------- ShapeGesture on a top face


def block(colour=1, height=3):
    voxels = np.zeros(SIZE, np.uint8)
    voxels[:, :, :height] = colour
    return voxels


@pytest.mark.parametrize("mode", [BOX, LINE, CENTER])
def test_erase_and_paint_on_a_wall_begin_false(mode):
    voxels = np.zeros(SIZE, np.uint8)
    for tool in (ERASE, PAINT):
        g, sm = make_shape(voxels, mode, tool)
        assert not g.begin(floor_hit(2, 2))
        assert sm.calls == []


def test_box_erase_on_top_face_acts_in_the_voxel_layer_only():
    voxels = block()
    before = voxels.copy()
    g, sm = make_shape(voxels, BOX, ERASE)
    assert g.begin(top_hit(2, 2, 2))
    assert g.last == (2, 2, 2)
    assert g.lock.axis == 2 and g.lock.plane == 3.0 and g.lock.layer == 2
    assert g.extend(*top_ray(5, 4))
    assert g.finish() == 12
    changed = {tuple(c) for c in np.argwhere(before != voxels).tolist()}
    assert changed == rect(2, 5, 2, 4, 2)
    assert (voxels[:, :, :2] == 1).all()


def test_box_paint_on_top_face_acts_in_the_voxel_layer_only():
    voxels = block()
    g, _ = make_shape(voxels, BOX, PAINT, colour=7)
    assert g.begin(top_hit(2, 2, 2))
    assert g.extend(*top_ray(5, 4))
    assert g.finish() == 12
    assert filled(voxels, 7) == rect(2, 5, 2, 4, 2)
    assert int((voxels == 1).sum()) == 8 * 8 * 3 - 12


def test_line_and_center_erase_on_top_face():
    voxels = block()
    g, _ = make_shape(voxels, LINE, ERASE)
    assert g.begin(top_hit(0, 0, 2))
    assert g.extend(*top_ray(3, 3))
    assert g.finish() == 4
    assert filled(voxels) == filled(block()) - {(i, i, 2) for i in range(4)}

    voxels = block()
    g, _ = make_shape(voxels, CENTER, ERASE)
    assert g.begin(top_hit(4, 4, 2))
    assert g.extend(*top_ray(5, 5))
    assert g.finish() == 9
    assert filled(voxels) == filled(block()) - rect(3, 5, 3, 5, 2)


def test_box_erase_on_a_bumpy_surface_removes_only_the_first_layer():
    voxels = block(height=2)
    voxels[4, 4, 2] = 1
    g, _ = make_shape(voxels, BOX, ERASE)
    assert g.begin(top_hit(2, 2, 1))
    # The drag crosses the bump; the lock keeps the shape in layer 1.
    assert g.extend(*top_ray(5, 5))
    assert g.finish() == 16
    assert voxels[4, 4, 2] == 1
    assert (voxels[2:6, 2:6, 1] == 0).all()
    assert (voxels[:, :, 0] == 1).all()


def test_paint_with_the_same_colour_writes_nothing():
    voxels = block(colour=5)
    g, sm = make_shape(voxels, BOX, PAINT, colour=5)
    assert g.begin(top_hit(1, 1, 2))
    assert g.extend(*top_ray(3, 3))
    assert g.finish() == 0
    assert sm.calls == []


# ---------------------------------------------------------------- ShapeGesture with Select


def columns_grid():
    voxels = np.zeros(SIZE, np.uint8)
    voxels[1, 1, 0:4] = 1
    voxels[2, 2, 0:2] = 2
    voxels[5, 5, 0:3] = 3
    return voxels


def test_box_select_on_the_floor_takes_the_filled_cells_of_the_block():
    voxels = columns_grid()
    before = voxels.copy()
    sel = Selection(SIZE)
    g, sm = make_shape(voxels, BOX, SELECT, selection=sel)
    assert g.begin(floor_hit(1, 1))
    assert g.last == (1, 1, 0)
    assert g.extend(*top_ray(2, 2))
    # Without the height stage the block is one layer: the floor cells that are filled.
    assert g.finish() == 0
    assert sm.calls == []
    assert np.array_equal(voxels, before)
    assert cells_set(sel.cells()) == {(1, 1, 0), (2, 2, 0)}
    # From the floor the height rises; four layers take both columns whole.
    sel.clear()
    g, _ = make_shape(voxels, BOX, SELECT, selection=sel)
    assert g.begin(floor_hit(1, 1)) and g.extend(*top_ray(2, 2)) and g.begin_height()
    assert g.extend_height((1.5, -10.0, 3.5), (0.0, 1.0, 0.0)) and g.height == 4
    assert g.preview().kind == "CELLS" and g.preview().label.endswith("x 4")
    assert g.finish() == 0
    assert cells_set(sel.cells()) == {(1, 1, z) for z in range(4)} | {(2, 2, z) for z in range(2)}


def test_box_select_on_a_top_face_reaches_down_with_the_height():
    voxels = columns_grid()
    sel = Selection(SIZE)
    g, _ = make_shape(voxels, BOX, SELECT, selection=sel)
    assert g.begin(top_hit(1, 1, 3))
    assert g.extend(*top_ray(5, 5))
    assert g.finish() == 0
    assert cells_set(sel.cells()) == {(1, 1, 3)}  # one layer: the top cell only
    sel.clear()
    g, _ = make_shape(voxels, BOX, SELECT, selection=sel)
    assert g.begin(top_hit(1, 1, 3)) and g.extend(*top_ray(5, 5)) and g.begin_height()
    assert g.extend_height((1.5, -10.0, 0.5), (0.0, 1.0, 0.0)) and g.height == 4
    assert g.finish() == 0
    assert sel.count() == 4 + 2 + 3  # the whole columns, now by choice of the height
    # Subtract with the same block drops them again.
    g, _ = make_shape(voxels, BOX, SELECT, selection=sel)
    g.subtract = True
    assert g.begin(top_hit(1, 1, 3)) and g.extend(*top_ray(5, 5)) and g.begin_height()
    g.extend_height((1.5, -10.0, 0.5), (0.0, 1.0, 0.0))
    assert g.finish() == 0 and sel.count() == 0


def test_box_paint_with_the_height_recolours_the_block():
    voxels = columns_grid()
    g, _ = make_shape(voxels, BOX, PAINT, colour=9)
    assert g.begin(top_hit(1, 1, 3)) and g.extend(*top_ray(5, 5))
    assert g.has_height_stage() and g.begin_height()
    assert g.extend_height((1.5, -10.0, 1.5), (0.0, 1.0, 0.0)) and g.height == 3
    changed = g.finish()
    assert changed == 3 + 1 + 2  # z 1..3 of the three columns
    assert (voxels[1, 1, 1:4] == 9).all() and voxels[1, 1, 0] != 9


def test_box_select_subtract_removes_the_block():
    voxels = columns_grid()
    sel = Selection(SIZE)
    sel.add_mask(voxels != 0)
    g, _ = make_shape(voxels, BOX, SELECT, selection=sel)
    g.subtract = True
    assert g.begin(top_hit(1, 1, 3)) and g.extend(*top_ray(2, 2))
    assert g.finish() == 0
    assert cells_set(sel.cells()) == cells_set(np.argwhere(voxels != 0)) - {(1, 1, 3)}  # one layer
    g, _ = make_shape(voxels, BOX, SELECT, selection=sel)
    g.subtract = True
    assert g.begin(top_hit(1, 1, 3)) and g.extend(*top_ray(2, 2)) and g.begin_height()
    g.extend_height((1.5, -10.0, 0.5), (0.0, 1.0, 0.0))
    assert g.finish() == 0
    assert cells_set(sel.cells()) == {(5, 5, z) for z in range(3)}  # the block dropped both columns


def test_select_preview_box_is_the_footprint():
    voxels = columns_grid()
    g, _ = make_shape(voxels, BOX, SELECT, selection=Selection(SIZE))
    assert g.begin(floor_hit(1, 1))
    assert g.extend(*top_ray(2, 3))
    p = g.preview()
    assert p.kind == "PLANE" and p.tool == SELECT
    assert p.box == ((1, 1, 0), (2, 3, 0))  # the footprint; the height stage sets the depth
    assert cells_set(p.cells) == rect(1, 2, 1, 3, 0)
    assert p.subtract is False


def test_box_select_subtract_preview_carries_subtract():
    # So the overlay can draw the outline in the Erase colour, not the plain
    # Select colour, while Ctrl+Shift is held.
    voxels = columns_grid()
    g, _ = make_shape(voxels, BOX, SELECT, selection=Selection(SIZE), subtract=True)
    assert g.begin(floor_hit(1, 1))
    assert g.extend(*top_ray(2, 3))
    assert g.preview().subtract is True


def test_select_finish_without_a_selection_writes_nothing():
    voxels = columns_grid()
    g, sm = make_shape(voxels, BOX, SELECT, selection=None)
    assert g.begin(floor_hit(1, 1))
    assert g.finish() == 0
    assert sm.calls == []


# ---------------------------------------------------------------- FaceGesture


def plate(colour=1, n=3, z=0):
    """An n x n plate of one colour at layer z."""
    voxels = np.zeros(SIZE, np.uint8)
    voxels[:n, :n, z] = colour
    return voxels


def test_face_attach_click_adds_one_layer_on_a_plate():
    voxels = plate()
    g, sm = make_face(voxels, ATTACH, colour=5)
    assert g.begin(top_hit(1, 1, 0))
    assert g.first == 1 and g.steps == 1
    assert g.axis == 2 and g.normal == (0, 0, 1)
    assert np.allclose(g.centre, (1.5, 1.5, 0.5))
    assert g.last == (1, 1, 1)
    assert sm.calls == []
    assert g.preview().label == "+1"
    assert cells_set(g.preview().cells) == rect(0, 2, 0, 2, 1)
    assert g.finish() == 9
    assert filled(voxels, 5) == rect(0, 2, 0, 2, 1)
    assert filled(voxels, 1) == rect(0, 2, 0, 2, 0)


def test_face_attach_extrudes_by_three_with_a_drag_along_the_normal():
    voxels = plate()
    g, _ = make_face(voxels, ATTACH, colour=5)
    assert g.begin(top_hit(1, 1, 0))
    # First move sets the reference: a ray through the centre, closest approach s == 0.
    assert not g.extend(*side_ray(1, 0))
    assert g.s0 == pytest.approx(0.0)
    assert g.steps == 1
    # Same ray shifted +2 along the normal: s == 2, delta == 2, three layers.
    assert g.extend(*side_ray(1, 2))
    assert g.steps == 3
    p = g.preview()
    assert p.kind == "CELLS" and p.tool == ATTACH and p.label == "+3"
    assert cells_set(p.cells) == rect(0, 2, 0, 2, 1) | rect(0, 2, 0, 2, 2) | rect(0, 2, 0, 2, 3)
    assert not g.extend(*side_ray(1, 2))
    assert g.finish() == 27
    assert filled(voxels, 5) == cells_set(p.cells)


def test_face_attach_drag_never_goes_below_one_layer_and_caps_at_the_grid():
    voxels = plate()
    g, _ = make_face(voxels, ATTACH, colour=5)
    assert g.begin(top_hit(1, 1, 0))
    assert not g.extend(*side_ray(1, 0))
    # Dragging against the normal keeps one layer.
    assert not g.extend(*side_ray(1, -3))
    assert g.steps == 1
    # Far along the normal caps at size[axis]; the layers are clipped to the grid.
    assert g.extend(*side_ray(1, 30))
    assert g.steps == 8
    assert cells_set(g.layers()) == {(x, y, z) for x in range(3) for y in range(3) for z in range(1, 8)}
    assert g.finish() == 63


def test_face_attach_ignores_a_ray_along_the_normal():
    voxels = plate()
    g, _ = make_face(voxels, ATTACH, colour=5)
    assert g.begin(top_hit(1, 1, 0))
    assert not g.extend(*top_ray(1, 1))
    assert g.s0 is None


def test_face_erase_click_pushes_one_layer_and_drag_two():
    voxels = block(height=2)
    g, _ = make_face(voxels, ERASE)
    assert g.begin(top_hit(3, 3, 1))
    assert g.first == 1 and g.last == (3, 3, 1)
    assert g.preview().label == "-1"
    assert cells_set(g.layers()) == rect(0, 7, 0, 7, 1)
    assert g.finish() == 64
    assert filled(voxels) == rect(0, 7, 0, 7, 0)

    voxels = block(height=2)
    g, _ = make_face(voxels, ERASE)
    assert g.begin(top_hit(3, 3, 1))
    assert not g.extend(*side_ray(3, 1))
    # Pushing in: the closest approach moves against the normal by one.
    assert g.extend(*side_ray(3, 0))
    assert g.steps == 2
    assert g.preview().label == "-2"
    assert g.finish() == 128
    assert filled(voxels) == set()


def test_face_erase_on_a_side_face():
    voxels = block(height=2)
    g, _ = make_face(voxels, ERASE)
    hit = Hit((7, 3, 1), (1, 0, 0), 1.0)
    assert g.begin(hit)
    assert g.axis == 0
    assert cells_set(g.base) == {(7, y, z) for y in range(8) for z in range(2)}
    assert g.finish() == 16
    assert (voxels[7] == 0).all() and (voxels[:7, :, :2] == 1).all()


def test_face_paint_click_recolours_the_region():
    voxels = plate()
    g, sm = make_face(voxels, PAINT, colour=7)
    assert g.begin(top_hit(1, 1, 0))
    assert filled(voxels, 7) == rect(0, 2, 0, 2, 0)
    assert len(sm.calls) == 1
    assert g.painted == rect(0, 2, 0, 2, 0)
    assert g.preview() is None
    assert g.finish() == 0
    assert len(sm.calls) == 1


def test_face_paint_drag_over_two_regions_recolours_both():
    voxels = np.zeros(SIZE, np.uint8)
    voxels[0:3, 0:3, 0] = 1
    voxels[3:6, 0:3, 0] = 2
    g, sm = make_face(voxels, PAINT, colour=7)
    assert g.begin(top_hit(1, 1, 0))
    assert filled(voxels, 7) == rect(0, 2, 0, 2, 0)
    # Still on the first region: nothing new.
    assert not g.extend(*top_ray(2, 2))
    # Over the second region.
    assert g.extend(*top_ray(4, 1))
    assert g.last == (4, 1, 0)
    assert filled(voxels, 7) == rect(0, 5, 0, 2, 0)
    assert not g.extend(*top_ray(5, 2))
    # A miss changes nothing.
    assert not g.extend(*top_ray(7, 7))
    assert g.finish() == 0
    assert len(sm.calls) == 2


def test_face_select_adds_the_region_and_every_region_crossed():
    voxels = np.zeros(SIZE, np.uint8)
    voxels[0:3, 0:3, 0] = 1
    voxels[3:6, 0:3, 0] = 2
    sel = Selection(SIZE)
    g, sm = make_face(voxels, SELECT, selection=sel)
    assert g.begin(top_hit(1, 1, 0))
    assert cells_set(sel.cells()) == rect(0, 2, 0, 2, 0)
    assert not g.extend(*top_ray(0, 0))
    assert g.extend(*top_ray(4, 1))
    assert cells_set(sel.cells()) == rect(0, 5, 0, 2, 0)
    assert g.preview() is None
    assert g.finish() == 0
    assert sm.calls == []


def test_face_select_subtract_removes_the_region_and_every_region_crossed():
    voxels = np.zeros(SIZE, np.uint8)
    voxels[0:3, 0:3, 0] = 1
    voxels[3:6, 0:3, 0] = 2
    sel = Selection(SIZE)
    sel.add_mask(voxels != 0)
    g, sm = make_face(voxels, SELECT, selection=sel, subtract=True)
    assert g.begin(top_hit(1, 1, 0))
    assert cells_set(sel.cells()) == rect(3, 5, 0, 2, 0)
    assert g.extend(*top_ray(4, 1))
    assert sel.is_empty()
    assert sm.calls == []


def test_face_wall_attach_fills_the_floor_of_an_empty_grid():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_face(voxels, ATTACH, colour=3)
    assert g.begin(floor_hit(3, 3))
    assert g.first == 0
    assert len(g.base) == 64
    assert g.last == (3, 3, 0)
    assert g.preview().label == "+1"
    assert g.finish() == 64
    assert filled(voxels, 3) == rect(0, 7, 0, 7, 0)


def test_face_wall_attach_drag_stacks_layers_from_the_floor():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_face(voxels, ATTACH, colour=3)
    assert g.begin(floor_hit(3, 3))
    assert not g.extend(*side_ray(3, 0))
    assert g.extend(*side_ray(3, 1))
    assert g.steps == 2
    assert g.finish() == 128
    assert filled(voxels, 3) == rect(0, 7, 0, 7, 0) | rect(0, 7, 0, 7, 1)


def test_face_wall_region_stops_at_filled_cells():
    voxels = np.zeros(SIZE, np.uint8)
    voxels[3, :, 0] = 1
    g, _ = make_face(voxels, ATTACH, colour=3)
    assert g.begin(floor_hit(1, 1))
    assert cells_set(g.base) == rect(0, 2, 0, 7, 0)


@pytest.mark.parametrize("tool", [ERASE, PAINT, SELECT])
def test_face_non_attach_on_a_wall_begins_false(tool):
    voxels = np.zeros(SIZE, np.uint8)
    sel = Selection(SIZE)
    g, sm = make_face(voxels, tool, selection=sel)
    assert not g.begin(floor_hit(3, 3))
    assert sm.calls == []
    assert sel.is_empty()


def test_face_begin_false_on_an_empty_cell():
    voxels = np.zeros(SIZE, np.uint8)
    g, _ = make_face(voxels, ATTACH)
    assert not g.begin(top_hit(1, 1, 0))


def test_face_covered_cell_is_not_part_of_the_region():
    voxels = plate()
    voxels[1, 1, 1] = 2
    g, _ = make_face(voxels, ATTACH, colour=5)
    assert g.begin(top_hit(0, 0, 0))
    assert cells_set(g.base) == rect(0, 2, 0, 2, 0) - {(1, 1, 0)}
    assert g.finish() == 8
    assert filled(voxels, 5) == rect(0, 2, 0, 2, 1) - {(1, 1, 1)}
    assert voxels[1, 1, 1] == 2


def test_face_region_stops_at_the_colour_border():
    voxels = plate()
    voxels[2, :3, 0] = 2
    g, _ = make_face(voxels, ATTACH, colour=5)
    assert g.begin(top_hit(0, 0, 0))
    assert cells_set(g.base) == rect(0, 1, 0, 2, 0)


def test_face_attach_with_mirror_x_writes_the_images():
    voxels = plate()
    g, _ = make_face(voxels, ATTACH, colour=5, mirror=(True, False, False))
    assert g.begin(top_hit(1, 1, 0))
    assert g.finish() == 18
    assert filled(voxels, 5) == rect(0, 2, 0, 2, 1) | rect(5, 7, 0, 2, 1)


# ---------------------------------------------------------------- factory


def test_make_gesture_returns_the_right_class():
    voxels = np.zeros(SIZE, np.uint8)
    setter = ArraySetter(voxels)
    sm = ArraySetMany(voxels)
    sel = Selection(SIZE)

    def make(mode, tool):
        return make_gesture(mode, tool, voxels, 5, NO_MIRROR, setter, sm, sel)

    assert type(make(VOXEL, SELECT)) is PickGesture
    assert make(VOXEL, SELECT).radius == 1
    brush = make_gesture(VOXEL, SELECT, voxels, 5, NO_MIRROR, setter, sm, sel, radius=3)
    assert brush.radius == 3
    for mode in (RECT, LASSO):
        assert make(mode, ATTACH) is None
        for tool in (ERASE, PAINT, SELECT):
            g = make(mode, tool)
            assert type(g) is ScreenGesture and g.mode == mode and g.tool == tool
    for tool in (ATTACH, ERASE, PAINT):
        g = make(VOXEL, tool)
        assert type(g) is Stroke and g.tool == tool
    for tool in (ATTACH, ERASE, PAINT, SELECT):
        g = make(FACE, tool)
        assert type(g) is FaceGesture and g.tool == tool
        for mode in (BOX, LINE, CENTER):
            g = make(mode, tool)
            assert type(g) is ShapeGesture and g.mode == mode and g.tool == tool


def test_make_gesture_passes_the_setters_and_selection():
    voxels = np.zeros(SIZE, np.uint8)
    setter = ArraySetter(voxels)
    sm = ArraySetMany(voxels)
    sel = Selection(SIZE)
    stroke = make_gesture(VOXEL, ATTACH, voxels, 5, NO_MIRROR, setter, sm, sel)
    assert stroke.begin(floor_hit(1, 1))
    assert setter.calls == [(1, 1, 0, 5)]
    box = make_gesture(BOX, ATTACH, voxels, 5, (True, False, False), setter, sm, sel)
    assert box.begin(floor_hit(2, 2)) and box.finish() == 2
    assert len(sm.calls) == 1 and box.mirror == (True, False, False)
    face = make_gesture(FACE, SELECT, voxels, 5, NO_MIRROR, setter, sm, sel)
    assert face.begin(top_hit(2, 2, 0))
    assert sel.contains((2, 2, 0))


# ---------------------------------------------------------------- golden


def test_golden_bird_box_erase_top_layer_equals_slicing():
    voxels = read_vox(FIXTURES / "bird.vox").voxels.copy()
    before = voxels.copy()
    first = raycast(voxels, *top_ray(38, 35))
    assert first is not None and not first.wall and first.normal == (0, 0, 1)
    z = first.voxel[2]
    g, sm = make_shape(voxels, BOX, ERASE)
    assert g.begin(first)
    assert g.lock.layer == z
    # The far corner sits on a lower column; the lock keeps the box in layer z.
    assert g.extend(*top_ray(36, 30))
    assert g.current == (36, 30, z)
    expected = before.copy()
    expected[36:39, 30:36, z] = 0
    removed = int(np.count_nonzero(before[36:39, 30:36, z]))
    assert removed > 0
    assert g.finish() == removed
    assert np.array_equal(voxels, expected)
    assert len(sm.calls) == 1


def test_golden_robot_box_attach_on_the_floor_adds_w_times_h():
    voxels = read_vox(FIXTURES / "robot.vox").voxels.copy()
    before = voxels.copy()
    x0, y0, x1, y1 = 0, 0, 3, 2
    assert not before[x0 : x1 + 1, y0 : y1 + 1, 0].any()
    first = raycast(voxels, *top_ray(x0, y0))
    assert first is not None and first.wall and first.target == (x0, y0, 0)
    g, _ = make_shape(voxels, BOX, ATTACH, colour=9)
    assert g.begin(first)
    assert g.extend(*top_ray(x1, y1))
    w, h = x1 - x0 + 1, y1 - y0 + 1
    assert g.preview().label == f"{w} x {h}"
    assert g.finish() == w * h
    assert int(np.count_nonzero(voxels)) == int(np.count_nonzero(before)) + w * h
    assert (voxels[x0 : x1 + 1, y0 : y1 + 1, 0] == 9).all()
    changed = np.argwhere(before != voxels)
    assert len(changed) == w * h and (changed[:, 2] == 0).all()


# ---------------------------------------------------------------- screen gestures


def robot():
    return read_vox(FIXTURES / "robot.vox").voxels.copy()


# A 6 x 6 footprint in the middle of the robot: 36 filled columns, 380 voxels.
X0, X1, Y0, Y1 = 8, 13, 10, 15


def rect_cells(x0=X0, x1=X1, y0=Y0, y1=Y1) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size, np.int64)], axis=1)


def make_screen(voxels, mode, tool, colour=5, mirror=NO_MIRROR, selection=None, subtract=False):
    sm = ArraySetMany(voxels)
    g = make_gesture(
        mode, tool, voxels, colour, mirror, None, sm, selection,
        project=top_down, subtract=subtract,
    )
    return g, sm


def test_screen_gesture_rejects_a_face_mode():
    with pytest.raises(ValueError):
        ScreenGesture(np.zeros(SIZE, np.uint8), BOX, SELECT, 1, None, None, top_down)


def test_rect_select_adds_the_cells_under_the_rectangle():
    voxels = robot()
    sel = Selection(voxels.shape)
    g, sm = make_screen(voxels, RECT, SELECT, selection=sel)
    assert g.begin(None) is True and g.hit is None and g.last is None
    g.begin_screen(X0 * 10, Y0 * 10)
    assert g.extend_screen(X0 * 10, Y0 * 10) is False
    assert g.extend_screen((X1 + 1) * 10, (Y1 + 1) * 10) is True
    assert g.extend(*top_ray(0, 0)) is False
    expected = footprint_mask(voxels, 2, rect_cells())
    assert int(expected.sum()) == 380
    assert g.finish() == 0
    assert np.array_equal(sel.mask, expected)
    assert sm.calls == []
    # A second rectangle adds.
    g2, _ = make_screen(voxels, RECT, SELECT, selection=sel)
    g2.begin_screen(0, 0)
    g2.extend_screen(240, 300)
    g2.finish()
    assert np.array_equal(sel.mask, voxels != 0)


def test_rect_select_subtract_removes_the_cells_under_the_rectangle():
    voxels = robot()
    sel = Selection(voxels.shape)
    sel.add_mask(voxels != 0)  # everything selected
    g, sm = make_screen(voxels, RECT, SELECT, selection=sel, subtract=True)
    g.begin_screen(X0 * 10, Y0 * 10)
    g.extend_screen((X1 + 1) * 10, (Y1 + 1) * 10)
    assert g.finish() == 0
    footprint = footprint_mask(voxels, 2, rect_cells())
    assert np.array_equal(sel.mask, (voxels != 0) & ~footprint)
    assert sm.calls == []
    assert g.preview().subtract is True


def test_pattern_select_subtract_removes_the_stamped_cells():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 7
    selection = Selection(SIZE)
    selection.add_mask(voxels != 0)  # everything selected
    box = np.ones((2, 2, 1), dtype=np.uint8) * 3
    g = PatternGesture(
        voxels, SELECT, box, NO_MIRROR, ArraySetMany(voxels), selection, subtract=True
    )
    assert g.begin(raycast(voxels, (4.5, 4.5, 50.0), (0.0, 0.0, -1.0))) is True
    assert not selection.contains((4, 4, 0)) and not selection.contains((5, 5, 0))
    assert selection.count() == 64 - 4
    assert int(np.count_nonzero(voxels == 3)) == 0  # Select never writes voxels
    assert g.preview().subtract is True


def test_rect_corners_in_any_order_and_preview_polygon():
    voxels = robot()
    sel = Selection(voxels.shape)
    g, _ = make_screen(voxels, RECT, SELECT, selection=sel)
    g.begin_screen((X1 + 1) * 10, (Y1 + 1) * 10)
    g.extend_screen(X0 * 10, Y0 * 10)
    p = g.preview()
    assert p.kind == "SCREEN" and p.tool == SELECT
    assert p.points.dtype == np.float32 and p.points.shape == (4, 2)
    assert p.points.tolist() == [[140, 160], [80, 160], [80, 100], [140, 100]]
    assert p.cells.shape == (0, 3) and p.label == "R to round"  # the hint while a Rect is dragged
    g.finish()
    assert sel.count() == 380


def test_empty_rect_selects_nothing():
    voxels = robot()
    sel = Selection(voxels.shape)
    g, sm = make_screen(voxels, RECT, ERASE, selection=sel)
    g.begin_screen(50, 50)
    assert g.finish() == 0 and sm.calls == []
    g.extend_screen(90, 50)  # zero height
    assert g.finish() == 0 and sm.calls == []


def test_rect_erase_clears_the_cells():
    voxels = robot()
    before = voxels.copy()
    g, sm = make_screen(voxels, RECT, ERASE)
    g.begin_screen(X0 * 10, Y0 * 10)
    g.extend_screen((X1 + 1) * 10, (Y1 + 1) * 10)
    assert g.finish() == 380
    expected = before.copy()
    expected[X0 : X1 + 1, Y0 : Y1 + 1, :] = 0
    assert np.array_equal(voxels, expected)
    assert len(sm.calls) == 1


def test_rect_paint_recolours_the_cells():
    voxels = robot()
    before = voxels.copy()
    g, sm = make_screen(voxels, RECT, PAINT, colour=7)
    g.begin_screen(X0 * 10, Y0 * 10)
    g.extend_screen((X1 + 1) * 10, (Y1 + 1) * 10)
    assert g.finish() == 380
    inside = np.zeros(voxels.shape, dtype=bool)
    inside[X0 : X1 + 1, Y0 : Y1 + 1, :] = True
    assert (voxels[inside & (before != 0)] == 7).all()
    assert np.array_equal(voxels[~inside], before[~inside])
    # Painting the same colour again changes nothing.
    g, sm = make_screen(voxels, RECT, PAINT, colour=7)
    g.begin_screen(X0 * 10, Y0 * 10)
    g.extend_screen((X1 + 1) * 10, (Y1 + 1) * 10)
    assert g.finish() == 0


def test_rect_erase_with_mirror_x():
    voxels = np.zeros(SIZE, np.uint8)
    voxels[1, 1, 0] = 1
    voxels[6, 1, 0] = 2  # the mirror image of (1, 1, 0) in an 8-wide grid
    g, sm = make_screen(voxels, RECT, ERASE, mirror=(True, False, False))
    g.begin_screen(10, 10)
    g.extend_screen(20, 20)
    assert g.finish() == 2
    assert not voxels.any()


def test_lasso_collects_points_three_px_apart():
    voxels = robot()
    g, _ = make_screen(voxels, LASSO, SELECT, selection=Selection(voxels.shape))
    g.begin_screen(0.0, 0.0)
    assert g.extend_screen(1.0, 1.0) is False
    assert g.extend_screen(2.9, 0.0) is False
    assert g.extend_screen(3.0, 0.0) is True
    assert g.extend_screen(3.0, 2.0) is False
    assert g.extend_screen(3.0, 3.0) is True
    assert g.preview().points.tolist() == [[0, 0], [3, 0], [3, 3]]


def test_lasso_select_around_a_rectangle_equals_the_footprint():
    voxels = robot()
    sel = Selection(voxels.shape)
    g, _ = make_screen(voxels, LASSO, SELECT, selection=sel)
    # Trace the rectangle's border in 4 px steps: many points, decimated on finish.
    x0, y0, x1, y1 = X0 * 10, Y0 * 10, (X1 + 1) * 10, (Y1 + 1) * 10
    g.begin_screen(x0, y0)
    for x in np.arange(x0, x1, 4):
        g.extend_screen(x, y0)
    for y in np.arange(y0, y1, 4):
        g.extend_screen(x1, y)
    for x in np.arange(x1, x0, -4):
        g.extend_screen(x, y1)
    for y in np.arange(y1, y0, -4):
        g.extend_screen(x0, y)
    assert len(g.points) > LASSO_MAX_VERTICES / 2
    assert g.finish() == 0
    assert np.array_equal(sel.mask, footprint_mask(voxels, 2, rect_cells()))


def test_lasso_concave_loop_takes_the_interior_only():
    voxels = np.ones(SIZE, np.uint8)
    sel = Selection(SIZE)
    g, _ = make_screen(voxels, LASSO, SELECT, selection=sel)
    # A "U" over the 8 x 8 footprint (10 px per cell): the notch is left out.
    loop = [(0, 0), (80, 0), (80, 80), (50, 80), (50, 30), (30, 30), (30, 80), (0, 80)]
    g.begin_screen(*loop[0])
    for p in loop[1:]:
        g.extend_screen(*p)
    g.finish()
    columns = sel.mask.any(axis=2)
    expected = np.ones((8, 8), dtype=bool)
    expected[3:5, 3:8] = False
    assert np.array_equal(columns, expected)
    assert sel.count() == int(expected.sum()) * 8


def test_lasso_is_decimated_to_96_vertices_on_finish():
    size = (16, 16, 4)
    voxels = np.ones(size, np.uint8)
    sel = Selection(size)
    g, _ = make_screen(voxels, LASSO, SELECT, selection=sel)
    # A circle of radius 6 cells around the grid centre, traced in 400 steps.
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    pts = np.stack([80 + 60 * np.cos(t), 80 + 60 * np.sin(t)], axis=1)
    g.begin_screen(*pts[0])
    for p in pts[1:]:
        g.extend_screen(*p)
    assert len(g.points) > LASSO_MAX_VERTICES
    g.finish()
    centres = np.argwhere(voxels != 0) + 0.5
    inside = np.hypot(centres[:, 0] - 8, centres[:, 1] - 8) < 6
    # The decimated polygon is a 96-gon inside the circle: only cells whose
    # centre is close to the circle can differ.
    idx = centres.astype(int)
    got = sel.mask[idx[:, 0], idx[:, 1], idx[:, 2]]
    differ = got != inside
    assert not differ.any() or (
        np.abs(np.hypot(centres[differ, 0] - 8, centres[differ, 1] - 8) - 6) < 0.1
    ).all()
    assert int(got.sum()) > 0.9 * int(inside.sum())


def test_lasso_with_fewer_than_three_points_selects_nothing():
    voxels = np.ones(SIZE, np.uint8)
    sel = Selection(SIZE)
    g, sm = make_screen(voxels, LASSO, ERASE, selection=sel)
    g.begin_screen(10, 10)
    g.extend_screen(50, 10)
    assert g.finish() == 0 and sm.calls == []


def test_lasso_erase_and_paint():
    voxels = robot()
    before = voxels.copy()
    loop = [(X0 * 10, Y0 * 10), ((X1 + 1) * 10, Y0 * 10), ((X1 + 1) * 10, (Y1 + 1) * 10)]
    # A triangle: the cells above the diagonal of the rectangle.
    g, _ = make_screen(voxels, LASSO, PAINT, colour=9)
    g.begin_screen(*loop[0])
    for p in loop[1:]:
        g.extend_screen(*p)
    painted = g.finish()
    assert 0 < painted < 380
    changed = np.argwhere(voxels != before)
    assert (voxels[changed[:, 0], changed[:, 1], changed[:, 2]] == 9).all()
    assert (changed[:, 0] - X0 >= changed[:, 1] - Y0).all()
    g, _ = make_screen(voxels, LASSO, ERASE)
    g.begin_screen(*loop[0])
    for p in loop[1:]:
        g.extend_screen(*p)
    assert g.finish() == painted
    assert not (voxels == 9).any()


# --- the height stage of Box and Center ------------------------------------------


def test_box_height_stage_stacks_the_footprint_upward_for_attach():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    writer = ArraySetMany(voxels)
    g = ShapeGesture(voxels, BOX, ATTACH, 5, NO_MIRROR, writer, None)
    assert g.begin(raycast(voxels, *top_ray(1, 1)))
    assert g.extend(*top_ray(3, 2))
    assert len(g.footprint()) == 6 and g.height == 1
    assert g.has_height_stage() and g.begin_height()
    assert g.height_stage and g.height == 1
    # A ray from the side at z 2.5 reads three layers; the same again changes nothing.
    assert g.extend_height((2.5, -10.0, 2.5), (0.0, 1.0, 0.0))
    assert g.height == 3 and not g.extend_height((2.5, -10.0, 2.5), (0.0, 1.0, 0.0))
    preview = g.preview()
    assert preview.kind == "CELLS" and preview.label == "3 x 2 x 3 · scroll for hollow"
    # A plain box previews as its box, no cell listed; the cells are still there to write.
    assert preview.box == ((1, 1, 0), (3, 2, 2)) and len(preview.cells) == 0 and len(g.cells()) == 18
    g.wall = 2
    hollow = g.preview()
    assert hollow.label == "3 x 2 x 3 · hollow 2" and hollow.box is None and len(hollow.cells) > 0  # scroll to hollow
    g.wall = 0
    # A ray along the height line gives no reading; a far one clamps to the grid.
    assert not g.extend_height(*top_ray(2, 1))
    assert g.extend_height((2.5, -10.0, 20.5), (0.0, 1.0, 0.0)) and g.height == 8
    assert g.max_height() == 8
    g.extend_height((2.5, -10.0, 2.5), (0.0, 1.0, 0.0))
    assert g.finish() == 18
    assert int(np.count_nonzero(voxels)) == 18
    assert voxels[1:4, 1:3, 0:3].all() and not voxels[:, :, 3:].any()


def test_box_height_stage_digs_into_the_model_for_erase():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0:4] = 7
    g = ShapeGesture(voxels, BOX, ERASE, 5, NO_MIRROR, ArraySetMany(voxels), None)
    assert g.begin(raycast(voxels, *top_ray(1, 1)))
    assert g.extend(*top_ray(3, 2))
    assert g.begin_height()
    # Depth grows downward from the top layer (z 3): a ray at z 1.5 reads three layers.
    assert g.extend_height((2.5, -10.0, 1.5), (0.0, 1.0, 0.0)) and g.height == 3
    assert g.max_height() == 4
    assert g.extend_height((2.5, -10.0, -5.0), (0.0, 1.0, 0.0)) and g.height == 4
    g.extend_height((2.5, -10.0, 1.5), (0.0, 1.0, 0.0))
    assert g.finish() == 18
    assert not voxels[1:4, 1:3, 1:4].any() and voxels[1:4, 1:3, 0].all()


def _rectangle(voxels, tool, hit, selection=None):
    """A 3 by 2 box whose height stage is open. The hit's face is the rectangle."""
    g = ShapeGesture(voxels, BOX, tool, 5, NO_MIRROR, ArraySetMany(voxels), selection)
    assert g.begin(hit) and g.extend(*top_ray(3, 2)) and g.begin_height()
    return g


def test_box_height_can_grow_to_either_side_of_the_rectangle():
    # Attach writes the cell in front of the face. A hit on z 3 puts the rectangle at z 4.
    slab = np.zeros(SIZE, dtype=np.uint8)
    slab[:, :, 4] = 7
    attach = _rectangle(slab, ATTACH, top_hit(1, 1, 3))
    assert attach.extend_height((2.5, -10.0, 6.5), (0.0, 1.0, 0.0))
    assert attach.side == 1 and attach.height == 3
    assert attach.plain_box() == ((1, 1, 4), (3, 2, 6))
    assert attach.preview().level == (2, 7.0)
    up = {int(z) for z in attach.cells()[:, 2]}
    # The same height on the other side is a different block, so the cell cache must miss.
    assert attach.extend_height((2.5, -10.0, 2.5), (0.0, 1.0, 0.0))
    assert attach.side == -1 and attach.height == 3
    assert attach.plain_box() == ((1, 1, 2), (3, 2, 4))
    assert attach.preview().level == (2, 2.0)
    assert {int(z) for z in attach.cells()[:, 2]} == {2, 3, 4} and up == {4, 5, 6}
    assert attach.finish() == 12
    assert (slab[1:4, 1:3, 2:4] == 5).all() and (slab[1:4, 1:3, 4] == 7).all()
    assert not slab[:, :, 5:].any() and not slab[:, :, :2].any()
    # A ray still inside the footprint cell keeps the one layer on the usual face.
    near = _rectangle(np.zeros(SIZE, dtype=np.uint8), ATTACH, top_hit(1, 1, 3))
    assert not near.extend_height((2.5, -10.0, 4.0), (0.0, 1.0, 0.0))
    assert near.side == 1 and near.height == 1 and near.plain_box() == ((1, 1, 4), (3, 2, 4))
    # The floor has no cell below it, so a drag past it stays the one layer.
    floor = np.zeros(SIZE, dtype=np.uint8)
    held = ShapeGesture(floor, BOX, ATTACH, 5, NO_MIRROR, ArraySetMany(floor), None)
    assert held.begin(raycast(floor, *top_ray(1, 1))) and held.extend(*top_ray(3, 2)) and held.begin_height()
    assert not held.extend_height((2.5, -10.0, -5.0), (0.0, 1.0, 0.0))
    assert held.side == 1 and held.height == 1
    # Erase from a middle layer. The usual way is into the model; the other way is out.
    block = np.full(SIZE, 7, dtype=np.uint8)
    erase = _rectangle(block, ERASE, top_hit(1, 1, 3))
    assert erase.extend_height((2.5, -10.0, 1.5), (0.0, 1.0, 0.0))
    assert erase.side == 1 and erase.height == 3 and erase.plain_box() == ((1, 1, 1), (3, 2, 3))
    assert erase.extend_height((2.5, -10.0, 5.5), (0.0, 1.0, 0.0))
    assert erase.side == -1 and erase.height == 3 and erase.plain_box() == ((1, 1, 3), (3, 2, 5))
    assert erase.finish() == 18
    assert not block[1:4, 1:3, 3:6].any() and block[1:4, 1:3, 1:3].all()
    # Select keeps the filled cells of whichever side the drag is on.
    filled = np.full(SIZE, 1, dtype=np.uint8)

    def selected(origin_z):
        sel = Selection(SIZE)
        pick = _rectangle(filled, SELECT, top_hit(1, 1, 3), sel)
        assert pick.extend_height((2.5, -10.0, origin_z), (0.0, 1.0, 0.0))
        pick.finish()
        return {int(z) for z in np.argwhere(sel.mask)[:, 2]}

    assert selected(1.5) == {1, 2, 3}
    assert selected(5.5) == {3, 4, 5}


def test_height_stage_for_box_and_center_with_every_action():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    writer = ArraySetMany(voxels)
    for mode_name, tool, expected in (
        (BOX, ATTACH, True), (CENTER, ATTACH, True), (BOX, ERASE, True),
        (BOX, PAINT, True), (BOX, SELECT, True), (CENTER, SELECT, True), (LINE, ATTACH, False),
    ):
        g = ShapeGesture(voxels, mode_name, tool, 5, NO_MIRROR, writer, Selection(SIZE))
        if not g.begin(raycast(voxels, *top_ray(2, 2))):
            continue  # Paint and Erase on the empty floor have no target
        assert g.begin_height() is expected, (mode_name, tool)
        if not expected:
            assert not g.extend_height((2.5, -10.0, 2.5), (0.0, 1.0, 0.0)) and g.height == 1
    # Select from the floor rises; from a voxel it reaches down.
    slab = np.zeros(SIZE, dtype=np.uint8)
    slab[:, :, 0] = 1
    sel = Selection(SIZE)
    up = ShapeGesture(voxels, BOX, SELECT, 5, NO_MIRROR, writer, sel)
    assert up.begin(raycast(voxels, *top_ray(2, 2))) and up._height_axis()[2] > 0
    down = ShapeGesture(slab, BOX, SELECT, 5, NO_MIRROR, writer, sel)
    assert down.begin(raycast(slab, *top_ray(2, 2))) and down._height_axis()[2] < 0


def test_height_stage_preview_carries_the_level_plane_at_the_top_of_the_block():
    voxels = np.zeros(SIZE, dtype=np.uint8)
    g = ShapeGesture(voxels, BOX, ATTACH, 5, NO_MIRROR, ArraySetMany(voxels), None)
    assert g.begin(raycast(voxels, *top_ray(1, 1)))
    assert g.preview().level is None
    assert g.begin_height()
    g.extend_height((1.5, -10.0, 2.5), (0.0, 1.0, 0.0))
    assert g.preview().level == (2, 3.0)  # three layers on the floor: the plane at z 3
    # Erase from the top of a slab: the level sinks into it.
    slab = np.zeros(SIZE, dtype=np.uint8)
    slab[:, :, 0:4] = 7
    e = ShapeGesture(slab, BOX, ERASE, 5, NO_MIRROR, ArraySetMany(slab), None)
    assert e.begin(raycast(slab, *top_ray(1, 1))) and e.begin_height()
    e.extend_height((1.5, -10.0, 1.5), (0.0, 1.0, 0.0))
    assert e.height == 3 and e.preview().level == (2, 1.0)


def test_solids_grow_from_the_footprint():
    from rok.core import shape as shape_core

    voxels = np.zeros((16, 16, 16), dtype=np.uint8)
    g = ShapeGesture(voxels, BOX, ATTACH, 5, NO_MIRROR, ArraySetMany(voxels), None)
    assert g.begin(raycast(voxels, *top_ray(2, 2)))
    assert g.extend(*top_ray(8, 8))  # a 7 x 7 footprint
    assert g.begin_height()
    g.extend_height((5.5, -10.0, 3.5), (0.0, 1.0, 0.0))
    assert g.height == 4
    g.solid = shape_core.BOX_SOLID
    assert len(g.cells()) == 49 * 4
    g.solid = shape_core.PYRAMID
    layers = [int(np.count_nonzero(g.cells()[:, 2] == z)) for z in range(4)]
    assert layers == [49, 25, 9, 1]  # 7, 5, 3, 1 wide: a point at the top
    g.solid = shape_core.CYLINDER
    per_layer = [int(np.count_nonzero(g.cells()[:, 2] == z)) for z in range(4)]
    assert per_layer[0] == per_layer[3] and 30 <= per_layer[0] < 49  # round, straight sides
    assert (2, 2, 0) not in {tuple(c) for c in g.cells().tolist()}  # the corner is cut
    g.solid = shape_core.CONE
    per_layer = [int(np.count_nonzero(g.cells()[:, 2] == z)) for z in range(4)]
    assert per_layer[0] < 49 and per_layer[3] == 1 and per_layer == sorted(per_layer, reverse=True)
    # The footprint stage shows the round base already.
    g.height_stage = False
    g.height = 1
    assert len(g.cells()) == per_layer[0]
    g.solid = shape_core.BOX_SOLID
    assert len(g.cells()) == 49
    # An even base: the pyramid's top keeps the middle four.
    h = ShapeGesture(voxels, BOX, ATTACH, 5, NO_MIRROR, ArraySetMany(voxels), None)
    assert h.begin(raycast(voxels, *top_ray(2, 2))) and h.extend(*top_ray(7, 7)) and h.begin_height()
    h.extend_height((5.5, -10.0, 2.5), (0.0, 1.0, 0.0))
    h.solid = shape_core.PYRAMID
    assert [int(np.count_nonzero(h.cells()[:, 2] == z)) for z in range(3)] == [36, 16, 4]
    assert h.finish() == 56 and int(np.count_nonzero(voxels)) == 56


def test_a_framed_buckyball_writes_its_bars_in_the_frame_colour():
    from rok.core import shape as shape_core

    voxels = np.zeros((16, 16, 16), dtype=np.uint8)
    g = ShapeGesture(voxels, BOX, ATTACH, 5, NO_MIRROR, ArraySetMany(voxels), None)
    assert g.begin(raycast(voxels, *top_ray(1, 1))) and g.extend(*top_ray(14, 14)) and g.begin_height()
    g.extend_height((7.5, -10.0, 13.5), (0.0, 1.0, 0.0))
    g.solid = shape_core.FRAMED_BUCKYBALL
    g.frame_colour = 9
    assert g.finish() > 0
    assert set(np.unique(voxels)) == {0, 5, 9}
    # Without a frame colour the whole ball takes the colour.
    plain = np.zeros((16, 16, 16), dtype=np.uint8)
    h = ShapeGesture(plain, BOX, ATTACH, 5, NO_MIRROR, ArraySetMany(plain), None)
    assert h.begin(raycast(plain, *top_ray(1, 1))) and h.extend(*top_ray(14, 14)) and h.begin_height()
    h.extend_height((7.5, -10.0, 13.5), (0.0, 1.0, 0.0))
    h.solid = shape_core.FRAMED_BUCKYBALL
    h.finish()
    assert set(np.unique(plain)) == {0, 5} and np.array_equal(plain != 0, voxels != 0)


def test_shift_makes_the_footprint_square():
    from rok.core.shape import square_corner

    assert square_corner((2, 2, 0), (8, 4, 0), 2) == (8, 8, 0)
    assert square_corner((2, 2, 0), (0, 5, 0), 2) == (-1, 5, 0)
    assert square_corner((2, 2, 3), (2, 2, 3), 2) == (2, 2, 3)
    voxels = np.zeros((16, 16, 16), dtype=np.uint8)
    g = ShapeGesture(voxels, BOX, ATTACH, 5, NO_MIRROR, ArraySetMany(voxels), None)
    assert g.begin(raycast(voxels, *top_ray(2, 2)))
    g.square = True
    assert g.extend(*top_ray(8, 4)) and g.current == (8, 8, 0)
    g.square = False
    assert g.extend(*top_ray(8, 4)) and g.current == (8, 4, 0)



def test_a_marquee_with_visible_only_takes_the_near_side():
    from rok.core.edit import RECT, SELECT
    from rok.core.gesture import ScreenGesture
    from rok.core.select import Selection

    voxels = np.zeros((10, 10, 10), dtype=np.uint8)
    voxels[2:8, 2:8, 2:8] = 1
    project = lambda pts: np.asarray(pts, dtype=np.float64)[:, :2] * 10.0  # seen from above

    def run(viewer):
        sel = Selection(voxels.shape)
        g = ScreenGesture(voxels, RECT, SELECT, 1, ArraySetMany(voxels), sel, project)
        g.viewer = viewer
        g.begin(None)
        g.begin_screen(0, 0)
        g.extend_screen(100, 100)
        g.finish()
        return np.argwhere(sel.mask)

    assert len(run(None)) == 6**3  # through the model
    top = run({"toward": (0.0, 0.0, 1.0)})
    assert len(top) == 36 and set(top[:, 2].tolist()) == {7}


def test_a_box_select_with_visible_only_takes_the_near_side():
    from rok.core.edit import BOX, SELECT
    from rok.core.select import Selection

    voxels = np.zeros((16, 16, 16), dtype=np.uint8)
    voxels[2:9, 2:9, 0:4] = 5
    sel = Selection(voxels.shape)
    g = ShapeGesture(voxels, BOX, SELECT, 5, NO_MIRROR, ArraySetMany(voxels), sel)
    g.viewer = {"toward": (0.0, 0.0, 1.0)}
    assert g.begin(raycast(voxels, *top_ray(2, 2)))
    assert g.extend(*top_ray(8, 8)) and g.begin_height()
    g.extend_height((5.5, -10.0, 0.5), (0.0, 1.0, 0.0))
    g.finish()
    picked = np.argwhere(sel.mask)
    assert len(picked) and set(picked[:, 2].tolist()) == {3}  # only the top layer shows


def _wall():
    v = np.zeros((8, 8, 8), np.uint8)
    v[:, :, 0:2] = 3  # a floor two cells thick
    return v


def test_a_stamp_with_holes_cuts_the_patterns_empty_cells():
    """A frame with an empty middle stamped with holes cuts
    the opening; plain Attach only adds the frame on top."""
    from rok.core.gesture import PatternGesture
    from rok.core.raycast import raycast

    frame = np.ones((3, 3, 2), np.uint8) * 5
    frame[1, 1, :] = 0  # the opening
    for holes in (False, True):
        v = _wall()
        writes = []

        def set_many(cells, values, v=v, writes=writes):
            cells = np.asarray(cells).reshape(-1, 3)
            v[cells[:, 0], cells[:, 1], cells[:, 2]] = values
            writes.append(len(cells))
            return len(cells)

        g = PatternGesture(v, ATTACH, frame, (False, False, False), set_many, None, colour=7, holes=holes)
        hit = raycast(v, (4.5, 4.5, 6.0), (0.0, 0.0, -1.0))
        assert g.begin(hit)
        if holes:
            ox, oy, _oz = g.origin
            assert v[ox + 1, oy + 1, 1] == 0 and v[ox + 1, oy + 1, 0] == 0  # the opening goes through the floor
            assert v[ox, oy, 1] == 3  # the frame keeps the voxel that was there
            assert (v[:, :, 0:2] == 0).sum() == 2  # only the opening
            assert (v[:, :, 2:] == 0).all()  # nothing on top: the box sank in
        else:
            assert (v[:, :, 0:2] == 3).all() and (v[:, :, 2] != 0).sum() == 8  # the frame on top, no hole


@pytest.mark.parametrize("tool", [ATTACH, ERASE, PAINT])
@pytest.mark.parametrize("mirror", [NO_MIRROR, (True, False, False), (False, True, True)])
def test_a_plain_box_written_as_a_block_matches_the_cell_path(tool, mirror):
    """A 200 x 200 base felt slow. A plain box writes as a
    block through set_grid; the voxels come out as apply_cells writes them."""
    start = np.zeros(SIZE, dtype=np.uint8)
    start[:, :, 0:3] = 3  # a slab to erase or paint into; Attach builds on top of it
    start[2, 1, 1] = 5  # a cell of the colour already: Paint leaves it
    results = []
    for use_block in (False, True):
        voxels = start.copy()
        writer = ArraySetMany(voxels)

        def set_grid(want, voxels=voxels):
            diff = want != voxels
            n = int(diff.sum())
            voxels[diff] = want[diff]
            return n

        g = ShapeGesture(voxels, BOX, tool, 5, mirror, writer, None)
        g.set_grid = set_grid if use_block else None
        assert g.begin(raycast(voxels, *top_ray(1, 1)))
        g.extend(*top_ray(3, 2))
        g.begin_height()
        for z in (0.5, 6.5):  # whichever side the block grows to
            g.extend_height((2.5, -10.0, z), (0.0, 1.0, 0.0))
            if g.height > 1:
                break
        assert g.plain_box() is not None and g.height > 1
        changed = g.finish()
        results.append((voxels.copy(), changed))
    (cells_voxels, cells_changed), (block_voxels, block_changed) = results
    assert np.array_equal(block_voxels, cells_voxels)
    assert block_changed == cells_changed > 0


def test_a_plain_box_selects_by_mask_as_the_cell_path_selects():
    from rok.core.select import Selection

    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0:3] = 3
    voxels[2, 1, 1] = 0  # a hole: not selected
    masks = []
    for block_path in (True, False):
        sel = Selection(SIZE)
        g = ShapeGesture(voxels, BOX, SELECT, 5, NO_MIRROR, None, sel)
        if not block_path:
            g.plain_box = lambda: None  # the cell path, for comparison
        assert g.begin(raycast(voxels, *top_ray(1, 1)))
        g.extend(*top_ray(3, 2))
        g.begin_height()
        for z in (0.5, 6.5):
            g.extend_height((2.5, -10.0, z), (0.0, 1.0, 0.0))
            if g.height > 1:
                break
        assert g.height > 1
        g.finish()
        masks.append(sel.mask.copy())
    assert masks[0].any() and np.array_equal(masks[0], masks[1])
    assert not masks[0][2, 1, 1]


@pytest.mark.parametrize("mode, expect", [("PATTERN", {9}), ("CLONE", {3, 5})])
def test_clone_stamps_the_copied_colours_and_pattern_the_current_one(mode, expect):
    from rok.core.gesture import make_gesture

    voxels = np.zeros(SIZE, dtype=np.uint8)
    voxels[:, :, 0] = 7  # the floor to stamp on
    box = np.zeros((2, 1, 1), dtype=np.uint8)
    box[0, 0, 0], box[1, 0, 0] = 3, 5  # two copied colours
    g = make_gesture(mode, ATTACH, voxels, 9, NO_MIRROR, ArraySetter(voxels), ArraySetMany(voxels), None,
                     pattern_box=box)
    assert g.begin(raycast(voxels, (4.5, 4.5, 50.0), (0.0, 0.0, -1.0))) is True
    assert set(np.unique(voxels[:, :, 1])) - {0} == expect
