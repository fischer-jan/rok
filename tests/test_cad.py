"""CAD face field, side lengths and the voxel length in a Rok file."""

from __future__ import annotations

import numpy as np
import pytest

from rok.core import rokfile
from rok.core.cad import (
    FOOT_METRES,
    decode_normal,
    dimension_ends,
    display_number,
    field,
    measurement,
    hidden_segments,
    nearest_segment,
    parse_length,
    side_text,
    straight_edges,
    to_metres,
)
from rok.core.grid import VoxelGrid
from rok.core.scene import VoxScene


def test_a_box_has_corner_lines_and_a_flat_face_has_none():
    voxels = np.ones((8, 6, 4), np.uint8)
    got = field(voxels)
    mid = got[3, 2, 3]
    normal = decode_normal(mid)
    assert mid[3] == 0
    assert normal[2] > 0.9 and abs(normal[0]) < 0.2 and abs(normal[1]) < 0.2
    # The fitted corner sits one cell in: the top face meets the side there.
    assert got[5, 2, 3, 3] & 1  # +X, the in-face direction of that corner
    surface = int(np.any(got[:, :, :, :3] != 0, axis=3).sum())
    corners = int(np.count_nonzero(got[:, :, :, 3]))
    assert 0 < corners < surface
    assert not np.any(got[3, 2, 1])  # inside the solid: no face


def test_one_voxel_has_no_corner_line():
    assert not np.any(field(np.ones((1, 1, 1), np.uint8)))


def test_the_middle_of_a_tread_has_no_line():
    voxels = np.zeros((16, 2, 4), np.uint8)
    for step in range(4):
        voxels[step * 4:(step + 1) * 4, :, :step + 1] = 1
    got = field(voxels)
    assert got[9, 0, 2, 3] == 0


def _spans(keys: np.ndarray) -> list[tuple]:
    return sorted(tuple(int(v) for v in row) for row in keys)


def test_a_box_has_twelve_straight_edges():
    _segments, keys = straight_edges(np.ones((8, 6, 4), np.uint8))
    assert _spans(keys) == [
        (0, 0, 0, 0, 8), (0, 0, 4, 0, 8), (0, 6, 0, 0, 8), (0, 6, 4, 0, 8),
        (1, 0, 0, 0, 6), (1, 0, 4, 0, 6), (1, 8, 0, 0, 6), (1, 8, 4, 0, 6),
        (2, 0, 0, 0, 4), (2, 0, 6, 0, 4), (2, 8, 0, 0, 4), (2, 8, 6, 0, 4),
    ]


def test_one_voxel_has_twelve_edges_of_one_cell():
    _segments, keys = straight_edges(np.ones((1, 1, 1), np.uint8))
    assert len(keys) == 12
    assert np.all(keys[:, 4] - keys[:, 3] == 1)
    assert np.all((keys[:, 1] == 0) | (keys[:, 1] == 1))
    assert np.all((keys[:, 2] == 0) | (keys[:, 2] == 1))


def test_an_empty_grid_and_a_flat_run_have_no_false_edge():
    with pytest.raises(ValueError):
        straight_edges(np.ones((4, 4), np.uint8))
    segments, keys = straight_edges(np.zeros((3, 3, 3), np.uint8))
    assert len(segments) == 0 and len(keys) == 0


def test_a_notch_splits_one_edge_and_a_hole_has_its_own_length():
    box = np.ones((8, 4, 4), np.uint8)
    box[3, 0, 3] = 0
    _segments, keys = straight_edges(box)
    line = keys[(keys[:, 0] == 0) & (keys[:, 1] == 0) & (keys[:, 2] == 4)]
    spans = sorted((int(a), int(b)) for a, b in line[:, 3:5])
    assert spans == [(0, 3), (4, 8)]

    hole = np.ones((8, 6, 6), np.uint8)
    hole[2:6, :, 2:4] = 0
    _segments, keys = straight_edges(hole)
    xlen = (keys[keys[:, 0] == 0, 4] - keys[keys[:, 0] == 0, 3]).tolist()
    assert 8 in xlen and 4 in xlen


def test_stair_lips_stay_separate_edges():
    voxels = np.zeros((12, 2, 4), np.uint8)
    for step in range(3):
        voxels[step * 4:(step + 1) * 4, :, :step + 1] = 1
    _segments, keys = straight_edges(voxels)
    front = keys[(keys[:, 0] == 0) & (keys[:, 1] == 0)]
    spans = sorted((int(z), int(a), int(b)) for a, b, z in zip(front[:, 3], front[:, 4], front[:, 2]))
    # The floor is one run. Each tread lip stops where the next riser starts.
    assert spans == [(0, 0, 12), (1, 0, 4), (2, 4, 8), (3, 8, 12)]
    ylen = keys[keys[:, 0] == 1, 4] - keys[keys[:, 0] == 1, 3]
    assert np.any(ylen == 2)


def _ends(segment) -> tuple:
    start = tuple(int(round(float(v))) for v in segment[0])
    end = tuple(int(round(float(v))) for v in segment[1])
    return tuple(sorted((start, end)))


def test_a_model_in_front_hides_the_edges_of_the_model_behind():
    """A cover is another model's grid. It stands apart from the model, so the ray of
    every point starts outside the cover's box and must be followed into it."""
    box = np.ones((4, 4, 4), np.uint8)
    segments, _keys = straight_edges(box)
    toward = np.array([1.0, 0.0, 0.0])
    alone = hidden_segments(box, segments, toward)
    # A wall five cells nearer to the camera, one cell larger on each side: a point p of the
    # model is p + shift in the cover.
    wall = np.ones((4, 6, 6), np.uint8)
    in_front = hidden_segments(box, segments, toward, covers=[((-5.0, 1.0, 1.0), wall)])
    assert len(alone) < 12 and len(in_front) == 12
    length = np.abs(in_front[:, 1] - in_front[:, 0]).sum(axis=1)
    assert np.allclose(length, 4.0)  # each edge is hidden from end to end
    # A cover beside the line of sight hides nothing more.
    beside = hidden_segments(box, segments, toward, covers=[((-5.0, 10.0, 1.0), wall)])
    assert len(beside) == len(alone)


def test_the_far_edges_of_a_box_are_hidden():
    voxels = np.ones((8, 6, 4), np.uint8)
    segments, _keys = straight_edges(voxels)
    hidden = hidden_segments(voxels, segments, np.array([1.0, -1.0, 1.0]))
    assert {_ends(segment) for segment in hidden} == {
        ((0, 6, 0), (8, 6, 0)),
        ((0, 0, 0), (0, 6, 0)),
        ((0, 6, 0), (0, 6, 4)),
    }
    other = hidden_segments(voxels, segments, np.array([-1.0, 1.0, -1.0]))
    assert {_ends(segment) for segment in other} == {
        ((0, 0, 4), (8, 0, 4)),
        ((8, 0, 4), (8, 6, 4)),
        ((8, 0, 0), (8, 0, 4)),
    }
    assert len(hidden_segments(voxels, np.zeros((0, 2, 3)), np.array([1.0, 0.0, 0.0]))) == 0
    assert len(hidden_segments(voxels, segments, np.zeros(3))) == 0

    # A block in front of the middle hides only that part of the edge.
    slab = np.zeros((8, 2, 4), np.uint8)
    slab[:, :, 0] = 1
    slab[3:5, :, 2:4] = 1
    segments, keys = straight_edges(slab)
    hidden = hidden_segments(slab, segments, np.array([0.0, 0.0, 1.0]))
    edge = keys[(keys[:, 0] == 0) & (keys[:, 1] == 0) & (keys[:, 2] == 1) & (keys[:, 3] == 0)]
    assert len(edge) == 1
    covered = [
        segment for segment in hidden
        if abs(float(segment[0, 1])) < 1e-6 and abs(float(segment[0, 2]) - 1.0) < 1e-6
        and abs(float(segment[1, 1])) < 1e-6 and abs(float(segment[1, 2]) - 1.0) < 1e-6
    ]
    assert len(covered) == 1
    xs = sorted(round(float(v), 3) for v in (covered[0][0, 0], covered[0][1, 0]))
    assert xs == [3.0, 5.0]


def test_the_nearest_segment_wins_and_a_miss_is_none():
    starts = np.array([[0.0, 0.0], [0.0, 0.0], [np.nan, 0.0]])
    ends = np.array([[10.0, 0.0], [0.0, 10.0], [1.0, 0.0]])
    assert nearest_segment(starts, ends, np.array([5.0, 1.0]), 2.0) == 0
    assert nearest_segment(starts, ends, np.array([5.0, 5.0]), 2.0) == -1
    # Both segments contain the corner. The nearer one wins.
    depth = np.array([5.0, 1.0, 0.0])
    assert nearest_segment(starts, ends, np.array([0.0, 0.0]), 2.0, depth) == 1
    assert nearest_segment(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(2), 8.0) == -1


def test_lengths_stay_in_cells_until_a_voxel_length_is_set():
    assert measurement(40, None, "m") == "40 cells"
    assert measurement(1, None, "m") == "1 cell"
    assert measurement(40, 0.25, "m") == "10 m"
    assert measurement(40, FOOT_METRES, "ft") == "40 ft"
    assert side_text(0, 40, 0.25, "m") == "X 10 m"


def test_feet_and_metres_are_one_length():
    assert to_metres(1, "ft") == pytest.approx(FOOT_METRES)
    assert to_metres(0.25, "m") == pytest.approx(0.25)
    assert display_number(FOOT_METRES, "ft") == "1"
    assert display_number(0.25, "m") == "0.25"
    assert parse_length("") is None
    assert parse_length("0.25") == pytest.approx(0.25)
    with pytest.raises(ValueError):
        parse_length("0")
    with pytest.raises(ValueError):
        parse_length("nope")


def test_a_side_line_sits_on_the_sides_that_face_the_eye():
    lo = np.zeros(3)
    hi = np.array([8.0, 6.0, 4.0])
    x0, x1 = dimension_ends(lo, hi, np.array([1.0, 1.0, 1.0]))[0]
    assert x0[1] == pytest.approx(6.8) and x0[2] == pytest.approx(4.8)
    assert x1[0] - x0[0] == pytest.approx(8.0)
    x0, _x1 = dimension_ends(lo, hi, np.array([1.0, -1.0, 1.0]))[0]
    assert x0[1] == pytest.approx(-0.8)


def test_the_voxel_length_round_trips_and_a_vox_cannot_hold_it():
    grid = VoxelGrid((2, 2, 2), np.ones((2, 2, 2), np.uint8))
    scene = VoxScene.single(grid)
    assert rokfile.needs_rok(scene) == ()
    scene.cell_metres = 0.25
    scene.cell_unit = "ft"
    assert rokfile.needs_rok(scene) == ("a voxel length",)
    again, doc = rokfile.unpack(rokfile.pack(scene, {"active": 0}))
    assert doc == {"active": 0}
    assert again.cell_metres == pytest.approx(0.25)
    assert again.cell_unit == "ft"
    scene.cell_metres = None
    plain, _doc = rokfile.unpack(rokfile.pack(scene, {}))
    assert plain.cell_metres is None and plain.cell_unit == "m"
