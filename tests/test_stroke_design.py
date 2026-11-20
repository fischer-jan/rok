"""Stroke Designer: the p-norm brush, the curves, and a stroke drawn again."""

from __future__ import annotations

import numpy as np

from rok.core.edit import ATTACH, ERASE, PAINT, sphere_offsets
from rok.core.grid import VoxelGrid
from rok.core.stroke_design import (
    CUBE_EXPONENT,
    Design,
    colour_slot,
    default_design,
    densify,
    flat,
    pack,
    ramp,
    replay,
    shape_offsets,
    sized,
    unpack,
)
from rok.core.undo import StrokeRecorder, UndoStack


def _cells(size: int, pred) -> set[tuple[int, int, int]]:
    r = size - 1
    out = set()
    for x in range(-r, r + 1):
        for y in range(-r, r + 1):
            for z in range(-r, r + 1):
                if pred(abs(x), abs(y), abs(z), r):
                    out.add((x, y, z))
    return out


def _as_set(offsets) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in row) for row in np.asarray(offsets).reshape(-1, 3)}


def test_the_pnorm_brush_is_a_diamond_a_sphere_and_a_cube():
    for size in (1, 2, 3, 5, 8):
        r = size - 1
        diamond = _cells(size, lambda x, y, z, rad: x + y + z <= rad)
        cube = _cells(size, lambda x, y, z, rad: max(x, y, z) <= rad)
        assert _as_set(shape_offsets(size, 1)) == diamond
        assert _as_set(shape_offsets(size, 2)) == _as_set(sphere_offsets(size))
        assert _as_set(shape_offsets(size, CUBE_EXPONENT)) == cube
        assert _as_set(shape_offsets(size, 64)) == cube
        # the cells of an octahedron of radius r: (2r + 1)(2r^2 + 2r + 3) / 3
        assert len(diamond) == (2 * r + 1) * (2 * r * r + 2 * r + 3) // 3


def test_size_one_is_one_voxel_for_every_shape():
    one = {(0, 0, 0)}
    for exponent in (1, 2, 8, CUBE_EXPONENT):
        assert _as_set(shape_offsets(1, exponent)) == one


def test_thickness_never_goes_below_one_voxel():
    assert sized(0.0, 12) == 1
    assert sized(1.0, 12) == 12
    assert sized(0.5, 10) == 5


def test_a_tapering_stroke_is_wide_then_one_voxel_and_one_piece():
    voxels = np.zeros((48, 16, 16), dtype=np.uint8)
    path = [(x, 8, 8) for x in range(4, 44)]

    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return int(len(cells))

    replay(voxels, ATTACH, path, 8, default_design(), 3, set_many)
    filled = np.argwhere(voxels != 0)
    assert len(filled) > 0
    at_start = filled[filled[:, 0] == 4]
    at_end = filled[filled[:, 0] == filled[:, 0].max()]
    assert len(at_start) > 1  # wide at the start
    assert len(at_end) == 1  # one voxel at the end
    assert tuple(int(v) for v in at_end[0]) == (43, 8, 8)
    assert _parts(voxels != 0) == 1


def test_an_off_thickness_row_keeps_the_size_field():
    design = default_design().with_row("thickness", False)
    voxels = np.zeros((20, 12, 12), dtype=np.uint8)

    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return int(len(cells))

    replay(voxels, ATTACH, [(4, 6, 6), (12, 6, 6)], 3, design, 2, set_many)
    # A sphere of size 3 at both ends: the same reach, not a taper to one voxel.
    assert int(np.count_nonzero(voxels[4])) > 1
    assert int(np.count_nonzero(voxels[12])) > 1


def test_the_colour_curve_walks_the_slots_from_one_colour_to_the_other():
    walked = [colour_slot(v, 3, 7) for v in np.linspace(0.0, 1.0, 9)]
    assert walked[0] == 3 and walked[-1] == 7 and walked == sorted(walked)
    assert set(walked) == set(range(3, 8))  # every slot between, no other
    assert {colour_slot(v, 9, 2) for v in np.linspace(0.0, 1.0, 15)} == set(range(2, 10))  # either way round
    assert colour_slot(0.7, 4, 4) == 4


def test_a_replay_never_draws_across_a_jump_of_the_stroke():
    voxels = np.zeros((30, 8, 8), dtype=np.uint8)

    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return int(len(cells))

    plain = Design(False, False, False, flat(1.0), flat(0.5), flat(0.0), 5)
    # two runs: the live stroke jumped from x 8 to x 20 (past an edge)
    replay(voxels, ATTACH, [[(x, 4, 4) for x in range(2, 9)], [(x, 4, 4) for x in range(20, 27)]], 1, plain, 3,
           set_many)
    assert not voxels[9:20].any()  # nothing in the jump
    assert voxels[2:9, 4, 4].all() and voxels[20:27, 4, 4].all()


def test_a_mirrored_stroke_takes_the_colours_of_its_own_image():
    voxels = np.zeros((20, 8, 8), dtype=np.uint8)

    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return int(len(cells))

    ramp_only = Design(False, False, True, flat(1.0), flat(0.5), ramp(), 5)
    # from x 9 (the middle) out to x 2; its mirror image runs from x 10 out to x 17
    replay(voxels, ATTACH, [(x, 4, 4) for x in range(9, 1, -1)], 1, ramp_only, 1, set_many, mirror=(True, False, False))
    left = [int(voxels[x, 4, 4]) for x in range(2, 10)]
    right = [int(voxels[19 - x, 4, 4]) for x in range(2, 10)]
    assert left == right  # the same colours, mirrored
    assert left[-1] == 1 and left[0] == 5  # the current colour at the start, the second at the end


def test_a_painted_stroke_never_invents_a_palette_colour():
    voxels = np.zeros((24, 8, 8), dtype=np.uint8)
    voxels[2:20, 4, 4] = 9  # a row to recolour

    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return int(len(cells))

    design = default_design().with_row("colour", True).with_second(5)
    replay(voxels, PAINT, [(x, 4, 4) for x in range(2, 20)], 1, design, 1, set_many)
    used = set(int(v) for v in np.unique(voxels) if v != 0)
    assert used <= set(range(1, 6))
    assert 9 not in used


def test_erase_ignores_the_colour_curve():
    voxels = np.zeros((16, 8, 8), dtype=np.uint8)
    voxels[2:12, 4, 4] = 4

    def set_many(cells, values):
        voxels[cells[:, 0], cells[:, 1], cells[:, 2]] = values
        return int(len(cells))

    design = default_design().with_row("colour", True).with_row("thickness", False)
    replay(voxels, ERASE, [(x, 4, 4) for x in range(2, 12)], 1, design, 1, set_many)
    assert int(voxels.sum()) == 0


def test_one_undo_removes_the_whole_designed_stroke():
    grid = VoxelGrid((32, 12, 12))
    recorder = StrokeRecorder(grid)
    path = [(x, 6, 6) for x in range(3, 28)]
    replay(grid.voxels, ATTACH, path, 6, default_design(), 3, recorder.set_many)
    assert int(np.count_nonzero(grid.voxels)) > 1
    record = recorder.finish()
    assert record is not None
    stack = UndoStack()
    stack.push(record)
    assert stack.undo(grid)
    assert int(np.count_nonzero(grid.voxels)) == 0


def test_a_design_survives_a_round_trip_through_its_text():
    design = default_design().with_row("roundness", True).with_second(12)
    assert unpack(pack(design)) == design
    assert unpack("broken") == default_design()
    assert unpack(None) == default_design()


def test_the_stroke_button_sits_above_the_size_field_and_lights_up():
    from rok.core.chrome import SIZE_ID, STROKE_ID, ChromeState, chrome_layout
    from rok.core.edit import BOX, VOXEL

    lay = chrome_layout(1.0, ChromeState(mode=VOXEL, stroke_on=True), 400, 900)
    button = lay.by_id(STROKE_ID)
    size = lay.by_id(SIZE_ID)
    assert button is not None and button.active and button.glyph == "STROKE"
    assert button.rect[1] >= size.rect[1] + size.rect[3]
    assert "Stroke" in button.tooltip
    assert chrome_layout(1.0, ChromeState(mode=BOX), 400, 900).by_id(STROKE_ID) is None


def test_the_panel_has_three_curves_presets_and_points():
    from rok.core.stroke_panel import hit, panel_layout

    lay = panel_layout(20, 40, 1.0, default_design())
    assert [i.key for i in lay.items if i.kind == "WELL"] == ["thickness", "roundness", "colour"]
    assert [i.text for i in lay.items if i.kind == "PRESET"] == ["Tentacle", "Horn", "Bead", "Plain"]
    for row in ("thickness", "roundness", "colour"):
        well = lay.find("WELL", row)
        assert len(lay.points(row)) == 4
        for point in lay.points(row):
            assert hit(lay, point.rect[0] + point.rect[2] / 2, point.rect[1] + point.rect[3] / 2).kind == "POINT"
            assert point.rect[0] >= well.rect[0] - 8


def test_the_curves_are_saved_and_read_back(tmp_path, monkeypatch):
    from rok.app import settings, stroke_ops

    monkeypatch.setattr(settings, "_path", lambda: tmp_path / "settings.txt")

    class Prefs:
        rok_stroke_design = False
        rok_stroke_curves = ""

    prefs = Prefs()
    stroke_ops.load(prefs)
    assert not prefs.rok_stroke_design
    prefs.rok_stroke_design = True
    prefs.rok_stroke_curves = pack(default_design().with_second(9).with_row("roundness", True))
    stroke_ops.save(prefs)
    again = Prefs()
    stroke_ops.load(again)
    assert again.rok_stroke_design
    assert unpack(again.rok_stroke_curves).second == 9
    assert unpack(again.rok_stroke_curves).roundness_on


def test_the_replay_path_stays_face_joined():
    line = densify([(0, 0, 0), (4, 2, 0)])
    for a, b in zip(line, line[1:]):
        assert sum(abs(a[k] - b[k]) for k in range(3)) == 1


def _parts(mask: np.ndarray) -> int:
    """How many face-joined pieces the filled cells make."""
    cells = {tuple(int(v) for v in c) for c in np.argwhere(mask)}
    seen: set[tuple[int, int, int]] = set()
    parts = 0
    for start in cells:
        if start in seen:
            continue
        parts += 1
        stack = [start]
        seen.add(start)
        while stack:
            x, y, z = stack.pop()
            for step in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nxt = (x + step[0], y + step[1], z + step[2])
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return parts
