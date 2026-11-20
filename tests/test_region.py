"""Scanline floods: 2D face regions, wall areas and the 3D colour blob."""

import json

import numpy as np
import pytest

import reference
from rok.core.region import (
    components,
    inset_region,
    connected_blobs,
    connected_colour,
    connected_filled,
    face_region,
    flood_2d,
    wall_region,
)
from rok.core.vox_read import read_vox

from helpers import FIXTURES

GOLDEN = FIXTURES / "phase2"


def mask_from(rows: list[str]) -> np.ndarray:
    """A 2D bool mask from strings; '#' is True. rows index axis 0."""
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def cell_set(cells):
    return {tuple(int(c) for c in row) for row in np.asarray(cells).reshape(-1, 3)}


def check_flood(mask, seed):
    fast = flood_2d(mask, seed)
    ref = reference.flood_2d_ref(mask, seed)
    assert fast.shape == mask.shape
    assert fast.dtype == bool
    assert np.array_equal(fast, ref)
    return fast


# ------------------------------------------------------------------ flood_2d


def test_flood_l_shape():
    mask = mask_from([
        "#....",
        "#....",
        "####.",
        ".....",
    ])
    out = check_flood(mask, (0, 0))
    assert out.sum() == 6
    assert np.array_equal(out, mask)


def test_flood_ring_leaves_the_hole():
    mask = mask_from([
        ".....",
        ".###.",
        ".#.#.",
        ".###.",
        ".....",
    ])
    out = check_flood(mask, (1, 1))
    assert out.sum() == 8
    assert not out[2, 2]
    # The hole is not part of the mask, so a seed there is empty.
    assert not check_flood(mask, (2, 2)).any()


def test_flood_diagonal_blobs_are_separate():
    mask = mask_from([
        "##..",
        "##..",
        "..##",
        "..##",
    ])
    out = check_flood(mask, (0, 0))
    assert out.sum() == 4
    assert out[:2, :2].all()
    assert not out[2:, 2:].any()


def test_flood_seed_off_the_mask_is_empty():
    mask = mask_from([
        "##.",
        "...",
    ])
    assert not check_flood(mask, (1, 1)).any()
    assert not check_flood(mask, (5, 5)).any()
    assert not check_flood(mask, (-1, 0)).any()


def test_flood_full_mask_is_full():
    mask = np.ones((64, 64), dtype=bool)
    out = check_flood(mask, (17, 40))
    assert out.all()


def test_flood_serpentine():
    rows = []
    for i in range(9):
        if i % 2 == 0:
            rows.append("#" * 12)
        elif i % 4 == 1:
            rows.append("." * 11 + "#")
        else:
            rows.append("#" + "." * 11)
    mask = mask_from(rows)
    out = check_flood(mask, (0, 0))
    assert np.array_equal(out, mask)
    assert out.sum() == 5 * 12 + 4


def test_flood_random_masks_match_reference():
    rng = np.random.default_rng(7)
    for _ in range(20):
        mask = rng.random((23, 31)) < 0.6
        seed = (int(rng.integers(23)), int(rng.integers(31)))
        check_flood(mask, seed)


def test_flood_does_not_change_the_input():
    mask = np.ones((5, 5), dtype=bool)
    keep = mask.copy()
    flood_2d(mask, (2, 2))
    assert np.array_equal(mask, keep)


# --------------------------------------------------------------- face_region


def three_colour_plate():
    """A 3 x 3 plate at z == 0: colour 1 left, 2 middle column, 3 right.

    One cell of the left column carries a voxel on top.
    """
    voxels = np.zeros((5, 5, 3), dtype=np.uint8)
    voxels[0, 0:3, 0] = 1
    voxels[1, 0:3, 0] = 2
    voxels[2, 0:3, 0] = 3
    voxels[0, 1, 1] = 4  # covers (0, 1, 0)
    return voxels


def test_face_region_stops_at_the_colour_border():
    voxels = three_colour_plate()
    cells = face_region(voxels, (1, 1, 0), (0, 0, 1))
    assert cell_set(cells) == {(1, 0, 0), (1, 1, 0), (1, 2, 0)}
    assert cells.dtype == np.int64


def test_face_region_excludes_a_covered_cell():
    voxels = three_colour_plate()
    cells = face_region(voxels, (0, 0, 0), (0, 0, 1))
    assert cell_set(cells) == {(0, 0, 0)}
    # From the other side of the covered cell the region is the other stub.
    cells = face_region(voxels, (0, 2, 0), (0, 0, 1))
    assert cell_set(cells) == {(0, 2, 0)}
    # On the -Z face nothing covers the column.
    cells = face_region(voxels, (0, 0, 0), (0, 0, -1))
    assert cell_set(cells) == {(0, 0, 0), (0, 1, 0), (0, 2, 0)}


def test_face_region_side_faces():
    voxels = three_colour_plate()
    # +X of the right column: the whole column is exposed.
    cells = face_region(voxels, (2, 1, 0), (1, 0, 0))
    assert cell_set(cells) == {(2, 0, 0), (2, 1, 0), (2, 2, 0)}
    # -X of the middle column is covered by the left column.
    assert face_region(voxels, (1, 1, 0), (-1, 0, 0)).shape == (0, 3)
    # -Y of the front row: one cell per colour, so one cell each.
    assert cell_set(face_region(voxels, (1, 0, 0), (0, -1, 0))) == {(1, 0, 0)}


def test_face_region_on_an_empty_cell():
    voxels = three_colour_plate()
    cells = face_region(voxels, (4, 4, 2), (0, 0, 1))
    assert cells.shape == (0, 3)


def test_face_region_matches_reference_on_random_grid():
    rng = np.random.default_rng(3)
    voxels = (rng.integers(0, 3, size=(12, 10, 6))).astype(np.uint8)
    normals = ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
    for _ in range(30):
        voxel = tuple(int(rng.integers(n)) for n in voxels.shape)
        normal = normals[int(rng.integers(6))]
        fast = face_region(voxels, voxel, normal)
        ref = reference.face_region_ref(voxels, voxel, normal)
        assert np.array_equal(fast, ref)


# --------------------------------------------------------------- wall_region


def test_wall_region_floor_of_empty_grid():
    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    cells = wall_region(voxels, (3, 3, 0), (0, 0, 1))
    assert cells.shape == (64, 3)
    assert set(cells[:, 2].tolist()) == {0}
    assert cell_set(cells) == {(x, y, 0) for x in range(8) for y in range(8)}


def test_wall_region_stops_at_filled_cells():
    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    voxels[3, :, 0] = 1  # a wall across the floor
    cells = wall_region(voxels, (0, 0, 0), (0, 0, 1))
    assert cells.shape == (3 * 8, 3)
    assert cells[:, 0].max() == 2
    cells = wall_region(voxels, (7, 7, 0), (0, 0, 1))
    assert cells.shape == (4 * 8, 3)
    assert cells[:, 0].min() == 4


def test_wall_region_on_a_side_wall():
    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    voxels[7, 2, 2] = 1
    cells = wall_region(voxels, (7, 0, 0), (-1, 0, 0))
    assert cells.shape == (63, 3)
    assert (7, 2, 2) not in cell_set(cells)


# ---------------------------------------------------------- connected_colour


def two_part_grid():
    voxels = np.zeros((10, 6, 6), dtype=np.uint8)
    voxels[0:3, 0:3, 0:3] = 1  # blob A
    voxels[6:9, 0:3, 0:3] = 1  # blob B, same colour, not connected
    voxels[3:6, 1, 1] = 2  # a bridge in another colour
    voxels[9, 5, 5] = 1  # a lone cell, diagonal to nothing
    return voxels


def test_connected_colour_picks_one_part():
    voxels = two_part_grid()
    mask = connected_colour(voxels, (1, 1, 1))
    assert mask.shape == voxels.shape
    assert mask.dtype == bool
    assert mask.sum() == 27
    assert mask[0:3, 0:3, 0:3].all()
    assert not mask[6:9].any()
    assert np.array_equal(mask, reference.connected_colour_ref(voxels, (1, 1, 1)))
    # The bridge is its own blob.
    mask = connected_colour(voxels, (4, 1, 1))
    assert mask.sum() == 3
    assert np.array_equal(mask, reference.connected_colour_ref(voxels, (4, 1, 1)))


def test_connected_colour_empty_seed_is_empty():
    voxels = two_part_grid()
    assert not connected_colour(voxels, (5, 5, 0)).any()
    assert not connected_colour(voxels, (50, 0, 0)).any()


def test_connected_colour_matches_reference_on_random_grid():
    rng = np.random.default_rng(11)
    voxels = (rng.integers(0, 3, size=(14, 9, 7))).astype(np.uint8)
    for _ in range(20):
        seed = tuple(int(rng.integers(n)) for n in voxels.shape)
        fast = connected_colour(voxels, seed)
        ref = reference.connected_colour_ref(voxels, seed)
        assert np.array_equal(fast, ref)


# -------------------------------------------------------------------- golden


def load_golden(name):
    return json.loads((GOLDEN / name).read_text())


_FIXTURE_CACHE: dict[str, np.ndarray] = {}


def fixture_voxels(stem):
    if stem not in _FIXTURE_CACHE:
        _FIXTURE_CACHE[stem] = read_vox(FIXTURES / f"{stem}.vox").voxels
    return _FIXTURE_CACHE[stem]


def check_golden(entry, fast, ref):
    fast = np.asarray(fast).reshape(-1, 3)
    ref = np.asarray(ref).reshape(-1, 3)
    assert fast.shape[0] == entry["count"]
    assert reference.cells_bbox(fast) == entry["bbox"]
    assert reference.cells_sha1(fast) == entry["sha1"]
    assert np.array_equal(fast, ref)


FACE_ENTRIES = load_golden("face_regions.json")
COLOUR_ENTRIES = load_golden("connected_colour.json")


def entry_id(entry):
    kind = "wall" if entry.get("wall") else "face"
    parts = [entry["fixture"], kind, ",".join(map(str, entry["voxel"]))]
    if "normal" in entry:
        parts.append(",".join(map(str, entry["normal"])))
    return "-".join(parts)


@pytest.mark.parametrize("entry", FACE_ENTRIES, ids=[entry_id(e) for e in FACE_ENTRIES])
def test_golden_face_regions(entry):
    voxels = fixture_voxels(entry["fixture"])
    voxel = tuple(entry["voxel"])
    normal = tuple(entry["normal"])
    if entry["wall"]:
        target = tuple(voxel[i] + normal[i] for i in range(3))
        fast = wall_region(voxels, target, normal)
        ref = reference.wall_region_ref(voxels, target, normal)
    else:
        fast = face_region(voxels, voxel, normal)
        ref = reference.face_region_ref(voxels, voxel, normal)
    check_golden(entry, fast, ref)


@pytest.mark.parametrize(
    "entry", COLOUR_ENTRIES, ids=[entry_id(e) for e in COLOUR_ENTRIES]
)
def test_golden_connected_colour(entry):
    voxels = fixture_voxels(entry["fixture"])
    voxel = tuple(entry["voxel"])
    fast = np.argwhere(connected_colour(voxels, voxel))
    ref = np.argwhere(reference.connected_colour_ref(voxels, voxel))
    check_golden(entry, fast, ref)


def test_golden_files_cover_every_fixture():
    stems = {p.stem for p in FIXTURES.glob("*.vox")}
    assert {e["fixture"] for e in FACE_ENTRIES} == stems
    assert {e["fixture"] for e in COLOUR_ENTRIES} == stems
    assert sum(1 for e in FACE_ENTRIES if e["wall"]) == len(stems)
    assert len(COLOUR_ENTRIES) == 3 * len(stems)


# ---------------------------------------------------------- connected_filled


def test_connected_filled_joins_colours():
    voxels = two_part_grid()
    # Blob A, the bridge and blob B are one filled blob.
    mask = connected_filled(voxels, (1, 1, 1))
    assert mask.shape == voxels.shape
    assert mask.dtype == bool
    assert mask.sum() == 27 + 3 + 27
    assert not mask[9, 5, 5]
    assert np.array_equal(mask, reference.connected_filled_ref(voxels, (1, 1, 1)))
    assert np.array_equal(mask, connected_filled(voxels, (4, 1, 1)))
    assert np.array_equal(mask, connected_filled(voxels, (8, 2, 2)))
    lone = connected_filled(voxels, (9, 5, 5))
    assert lone.sum() == 1 and lone[9, 5, 5]


def test_connected_filled_empty_seed_is_empty():
    voxels = two_part_grid()
    assert not connected_filled(voxels, (5, 5, 0)).any()
    assert not connected_filled(voxels, (50, 0, 0)).any()
    assert not connected_filled(voxels, (-1, 0, 0)).any()


def test_connected_filled_matches_reference_on_bird():
    voxels = fixture_voxels("bird")
    filled = np.argwhere(voxels != 0)
    seeds = [tuple(int(c) for c in filled[i]) for i in (0, len(filled) // 2, len(filled) - 1)]
    for seed in seeds:
        fast = connected_filled(voxels, seed)
        ref = reference.connected_filled_ref(voxels, seed)
        assert fast[seed]
        assert np.array_equal(fast, ref)
        # The filled blob holds the colour blob of the same seed.
        assert (connected_colour(voxels, seed) & ~fast).sum() == 0


# ---------------------------------------------------------- connected_blobs


def two_blobs():
    voxels = np.zeros((10, 6, 6), dtype=np.uint8)
    voxels[0:3, 1, 1] = 2  # blob A, three cells
    voxels[6:9, 1, 1] = 5  # blob B, three cells
    voxels[8, 2, 1] = 6  # joins blob B
    return voxels


def test_connected_blobs_is_the_union_over_the_seeds():
    voxels = two_blobs()
    mask = connected_blobs(voxels, [(1, 1, 1), (7, 1, 1)])
    assert np.array_equal(mask, (voxels != 0))
    only_a = connected_blobs(voxels, [(0, 1, 1), (2, 1, 1)])
    assert np.array_equal(only_a, connected_filled(voxels, (1, 1, 1)))


def test_connected_blobs_skips_empty_out_of_grid_and_no_seeds():
    voxels = two_blobs()
    assert not connected_blobs(voxels, [(4, 1, 1), (40, 1, 1), (-1, 0, 0)]).any()
    assert not connected_blobs(voxels, np.zeros((0, 3), dtype=np.int64)).any()
    mask = connected_blobs(voxels, [(4, 1, 1), (8, 2, 1)])
    assert np.array_equal(mask, connected_filled(voxels, (6, 1, 1)))


def test_inset_region_shrinks_a_face_by_its_rim():
    floor = np.array([[x, y, 0] for x in range(2, 8) for y in range(3, 9)])  # 6 x 6 at z 0
    inner = inset_region(floor, (0, 0, 1), 1)
    assert len(inner) == 16 and inner[:, 0].min() == 3 and inner[:, 0].max() == 6 and (inner[:, 2] == 0).all()
    assert len(inset_region(floor, (0, 0, 1), 2)) == 4
    assert len(inset_region(floor, (0, 0, 1), 3)) == 0
    assert np.array_equal(inset_region(floor, (0, 0, 1), 0), floor)
    # A plus shape: the arms are one wide, so only the middle survives one rim.
    plus = np.array([[c, 5, 2] for c in range(2, 9)] + [[5, r, 2] for r in range(2, 9) if r != 5])
    assert inset_region(plus, (0, 0, -1), 1).tolist() == [[5, 5, 2]]
    # A vertical face keeps its fixed x.
    wall = np.array([[4, y, z] for y in range(0, 5) for z in range(0, 5)])
    inner = inset_region(wall, (1, 0, 0), 1)
    assert len(inner) == 9 and (inner[:, 0] == 4).all()


def test_components_splits_a_mask_into_its_parts_largest_first():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[0:2, 0:2, 0:2] = True  # 8 cells
    mask[5:8, 5:8, 5:8] = True  # 27 cells
    mask[3, 3, 3] = True  # 1 cell
    parts = components(mask)
    assert [len(p) for p in parts] == [27, 8, 1]
    assert components(np.zeros((2, 2, 2), dtype=bool)) == []



def _brute_blob(filled: np.ndarray, seed, corners: bool) -> np.ndarray:
    """A plain breadth-first search: the reference for the run flood."""
    steps = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if (dx, dy, dz) != (0, 0, 0) and (corners or abs(dx) + abs(dy) + abs(dz) == 1)]
    out = np.zeros(filled.shape, dtype=bool)
    if not filled[seed]:
        return out
    out[seed] = True
    todo = [seed]
    while todo:
        x, y, z = todo.pop()
        for dx, dy, dz in steps:
            n = (x + dx, y + dy, z + dz)
            if all(0 <= n[i] < filled.shape[i] for i in range(3)) and filled[n] and not out[n]:
                out[n] = True
                todo.append(n)
    return out


@pytest.mark.parametrize("corners", [False, True])
def test_connected_blobs_matches_a_plain_search_by_faces_and_by_corners(corners):
    rng = np.random.default_rng(7)
    for trial in range(40):
        voxels = (rng.random((7, 6, 5)) < 0.35).astype(np.uint8) * rng.integers(1, 9, (7, 6, 5)).astype(np.uint8)
        filled = voxels != 0
        seeds = np.argwhere(filled)
        if len(seeds) == 0:
            continue
        seed = tuple(int(v) for v in seeds[trial % len(seeds)])
        got = connected_blobs(voxels, np.asarray([seed]), corners=corners)
        assert np.array_equal(got, _brute_blob(filled, seed, corners)), (trial, seed)


def test_a_corner_joins_only_with_corners():
    voxels = np.zeros((3, 3, 3), dtype=np.uint8)
    voxels[0, 0, 0] = 1
    voxels[1, 1, 1] = 2  # touches the first at one corner
    voxels[2, 1, 1] = 3  # touches the second face to face
    seed = np.asarray([(0, 0, 0)])
    assert connected_blobs(voxels, seed).sum() == 1
    assert connected_blobs(voxels, seed, corners=True).sum() == 3
    assert connected_blobs(voxels, np.asarray([(1, 1, 1)])).sum() == 2


@pytest.mark.parametrize("corners", [False, True])
def test_connected_colour_matches_a_plain_search_on_one_colour(corners):
    from rok.core.region import connected_colour

    rng = np.random.default_rng(11)
    for trial in range(40):
        voxels = (rng.random((6, 6, 5)) < 0.5).astype(np.uint8) * rng.integers(1, 3, (6, 6, 5)).astype(np.uint8)
        seeds = np.argwhere(voxels != 0)
        if len(seeds) == 0:
            continue
        seed = tuple(int(v) for v in seeds[trial % len(seeds)])
        want = _brute_blob(voxels == voxels[seed], seed, corners)
        assert np.array_equal(connected_colour(voxels, seed, corners), want), (trial, seed)


def test_components_leaves_the_callers_mask_alone():
    # A caller, such as Bridge, passes its live selection mask.
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[0, 0, 0] = mask[4, 4, 4] = True
    keep = mask.copy()
    from rok.core import region as region_mod

    parts = region_mod.components(mask)
    assert len(parts) == 2 and np.array_equal(mask, keep)
