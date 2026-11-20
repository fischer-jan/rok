"""Where a shader may change the model (rok/core/shader_scope.py)."""

import numpy as np

from rok.core import shader_scope as sc
from rok.core.shader_scope import Where


def _block():
    """A 4 x 4 x 3 block standing in a 6 x 6 x 5 grid, colour 2 on top."""
    v = np.zeros((6, 6, 5), np.uint8)
    v[1:5, 1:5, 0:3] = 1
    v[1:5, 1:5, 2] = 2
    return v


def test_no_selection_and_no_filter_means_every_cell():
    assert sc.allowed(_block(), Where()) is None
    assert sc.allowed(_block(), Where(room=3)) is None  # room only widens a limited scope


def test_the_selection_is_the_default_scope_when_there_is_one():
    v = _block()
    sel = np.zeros(v.shape, bool)
    sel[1, 1, 2] = True
    assert np.array_equal(sc.allowed(v, Where(), sel), sel)
    assert Where().summary(True) == "Selection" and Where().summary(False) == "Whole model"
    rest = sc.allowed(v, Where(sc.NOT_SELECTION), sel)
    assert rest.sum() == (v != 0).sum() - 1 and not rest[1, 1, 2]
    assert sc.allowed(v, Where(sc.MODEL), sel) is None  # Model: all of it, selection or not
    assert sc.allowed(v, Where(sc.SELECTION)) is None  # no selection to keep to


def test_faces():
    v = _block()
    top = sc.face_mask(v, "TOP")
    assert top.sum() == 16 and top[:, :, 2].sum() == 16
    assert sc.face_mask(v, "BOTTOM").sum() == 16  # the grid's floor counts as open
    sides = sc.face_mask(v, "SIDES")
    assert sides.sum() == 12 * 3 and not sides[2, 2].any()
    edges = sc.face_mask(v, "EDGES")
    assert edges[1, 2, 2] and edges[1, 1, 1] and not edges[2, 2, 2] and not edges[1, 2, 1]


def test_hollows_are_next_to_narrow_gaps_not_flat_faces():
    v = np.zeros((9, 5, 5), np.uint8)
    v[1:8, 1:4, 0:3] = 1
    v[4, 1:4, 1:3] = 0  # a slot one cell wide, two deep, across the top
    hollow = sc.face_mask(v, "HOLLOWS")
    assert hollow[3, 2, 2] and hollow[5, 2, 2] and hollow[4, 2, 0]  # its walls and its floor
    assert not hollow[1, 2, 2] and not hollow[7, 2, 2]  # the block's far ends


def test_filters_narrow_the_scope_and_room_reaches_out():
    v = _block()
    on_top = sc.allowed(v, Where(faces=("TOP",)))
    assert np.array_equal(on_top, sc.face_mask(v, "TOP"))
    by_colour = sc.allowed(v, Where(colours=(1,)))
    assert by_colour.sum() == 32 and not by_colour[:, :, 2].any()
    both = sc.allowed(v, Where(faces=("SIDES",), colours=(2,)))
    assert both.sum() == 12
    roomy = sc.allowed(v, Where(faces=("TOP",), room=2))
    assert roomy[2, 2, 3] and roomy[2, 2, 4]  # two cells up into the air
    assert not roomy[2, 2, 1]  # but not into the voxels under the top
    assert Where(faces=("TOP", "EDGES"), room=2).summary(False) == "Whole model · top, edges · room 2"


def test_parse():
    assert sc.parse_faces("edges, top,nonsense") == ("TOP", "EDGES")
    assert sc.parse_colours("3, 9,3,0,300,x") == (3, 9)


def test_feather_fades_toward_the_edge_along_the_surface():
    v = np.zeros((24, 24, 4), np.uint8)
    v[:, :, 0:2] = 1  # a slab two thick
    patch = np.zeros(v.shape, bool)
    patch[4:20, 4:20, 1] = True  # a patch of its top, the layer under it not selected
    w = sc.feather_weights(patch, v != 0, 4)
    assert w[12, 12, 1] == 1.0  # the middle keeps it all: the voxels under the patch do not count
    assert w[4, 12, 1] == w[19, 12, 1] == 1 / 5  # one step in from the edge
    assert w[5, 12, 1] == 2 / 5 and w[7, 12, 1] == 4 / 5 and w[8, 12, 1] == 1.0
    after = np.where(patch, 7, v).astype(np.uint8)  # a shader that paints all of the patch
    out = sc.feathered(v, after, patch, 4)
    assert np.all(out[8:16, 8:16, 1] == 7)
    edge_kept = np.mean(out[4, 4:20, 1] == 7)
    assert 0.0 < edge_kept < 0.6  # thinned out at the edge
    assert np.array_equal(sc.feathered(v, after, patch, 4), out)  # the same every run
    assert np.array_equal(sc.feathered(v, after, patch, 0), after)
    assert Where(sc.SELECTION, feather=3).summary(True) == "Selection · feather 3"


def test_a_volume_holds_empty_cells_too_box_or_sphere():
    v = _block()
    vol = (0, 0, 0, 5, 5, 4)
    box = sc.allowed(v, Where(sc.VOLUME, volume=vol))
    assert box.all()  # the whole 6 x 6 x 5 grid, empty cells too
    ball = sc.volume_mask(v.shape, (0, 0, 0, 4, 4, 4), sc.SPHERE)
    assert ball[2, 2, 2] and ball[0, 2, 2] and not ball[0, 0, 0]  # the corners are outside
    assert sc.allowed(v, Where(sc.VOLUME)) is None  # no volume given: the whole model
    on_top = sc.allowed(v, Where(sc.VOLUME, faces=("TOP",), volume=vol))
    assert np.array_equal(on_top, sc.face_mask(v, "TOP"))  # a filter keeps to the volume's voxels
    assert sc.volume_mask((4, 4, 4), (2, 2, 2, 9, 9, 9)).sum() == 8  # cut at the grid's walls
    assert sc.parse_volume("5,0,0, 1,2,3") == (1, 0, 0, 5, 2, 3) and sc.parse_volume("1,2") is None
    assert Where(sc.VOLUME, volume=(1, 0, 0, 5, 2, 3), shape=sc.SPHERE).summary(False) == "Sphere 5 x 3 x 4"


def test_feather_in_a_volume_fades_toward_its_walls():
    before = np.zeros((20, 20, 3), np.uint8)
    walls = sc.volume_mask(before.shape, (2, 2, 0, 17, 17, 2))
    after = np.where(walls, 4, 0).astype(np.uint8)  # a shader that fills the volume
    out = sc.feathered(before, after, walls, 3, walls)
    assert np.all(out[8:12, 8:12] == 4)
    assert 0 < np.count_nonzero(out[2, 2:18, 1]) < 16
