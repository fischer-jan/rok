"""Palette layout, hit testing and overlay geometry."""

import math

import numpy as np
import pytest

from rok.core.hud import (
    STRIP_BUTTONS,
    rounded_rect,
    HUD_ABOVE_HINT_PX,
    HUD_ERASE_RGBA,
    HUD_ERASING_TIP,
    HUD_LEFT_PX,
    HUD_RAIL_PX,
    HUD_TIP_PX,
    HUD_TIP_RGBA,
    HUD_TIPS,
    HUD_TITLE_PX,
    HUD_TITLE_RGBA,
    HUD_TITLE_TIP_GAP_PX,
    HUD_TITLES,
    MARK_INSET,
    OPTION_ERASE_FADE_AFTER,
    MARK_OUTLINE,
    MARK_PX,
    PALETTE_COLUMNS,
    PALETTE_ROWS,
    SLOT0_COLOUR,
    STRIP_BUTTON,
    STRIP_CURRENT,
    STRIP_GAP,
    STRIP_MARGIN,
    STRIP_SWATCH,
    STRIP_CAPTION,
    STRIP_TEXT_BAND,
    STRIP_USED_MAX,
    PaletteLayout,
    StripLayout,
    box_edges,
    cell_rect,
    cells_edges,
    cells_face_quads,
    cells_faces,
    contains,
    cube_edges,
    face_quad,
    hit_test,
    make_layout,
    marker_quads,
    mask_surface,
    mirror_plane_quad,
    palette_marker_rects,
    palette_quads,
    rect_outline,
    slab_edges,
    srgb_to_linear,
    hud_after_option_erase,
    hud_anchor,
    hud_icon_rect,
    hud_text_x,
    hud_erase_held,
    hud_on_tool_change,
    hud_readout,
    hud_title_origin,
    strip_contains,
    strip_hit_test,
    strip_layout,
    strip_marker_rects,
    strip_quads,
)
from rok.core.edit import ATTACH, ERASE, PAINT, SELECT
from rok.core.mesh import _FACE_CORNERS
from rok.core.palette import default_palette

ALL_NORMALS = [
    ((1, 0, 0), 0),
    ((-1, 0, 0), 1),
    ((0, 1, 0), 2),
    ((0, -1, 0), 3),
    ((0, 0, 1), 4),
    ((0, 0, -1), 5),
]


# --- layout -----------------------------------------------------------------


def test_make_layout_scale_1():
    layout = make_layout(1.0)
    assert layout == PaletteLayout(x0=8, y0=8, cell=16)
    assert layout.width == 256
    assert layout.height == 256


def test_make_layout_scale_2():
    layout = make_layout(2.0)
    assert layout == PaletteLayout(x0=16, y0=16, cell=32)
    assert layout.width == 512
    assert layout.height == 512


def test_cell_rect_corners_and_neighbours():
    layout = make_layout(1.0)
    top = layout.y0 + 15 * layout.cell
    assert cell_rect(layout, 0) == (8, top, 16, 16)
    assert cell_rect(layout, 15) == (8 + 15 * 16, top, 16, 16)
    assert cell_rect(layout, 16) == (8, top - 16, 16, 16)
    # 17 is column 1, one row below the top.
    assert cell_rect(layout, 17) == (8 + 16, top - 16, 16, 16)
    assert cell_rect(layout, 255) == (8 + 15 * 16, layout.y0, 16, 16)


# --- hit testing ------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_hit_test_every_cell_centre(scale):
    layout = make_layout(scale)
    for index in range(PALETTE_COLUMNS * PALETTE_ROWS):
        x, y, w, h = cell_rect(layout, index)
        hit = hit_test(layout, x + w / 2, y + h / 2)
        if index == 0:
            assert hit is None
        else:
            assert hit == index


def test_hit_test_on_shared_corner_picks_the_cell_that_starts_there():
    layout = make_layout(1.0)
    # The corner shared by cells 1, 2, 17 and 18 belongs to cell 2:
    # the cell whose left edge and bottom edge meet at that pixel.
    x, y, w, h = cell_rect(layout, 2)
    assert hit_test(layout, x, y) == 2
    # The corner shared by 17, 18, 33 and 34 belongs to 18.
    x, y, w, h = cell_rect(layout, 18)
    assert hit_test(layout, x, y) == 18
    # The right edge of cell 17 belongs to cell 18.
    x, y, w, h = cell_rect(layout, 17)
    assert hit_test(layout, x + w, y + h / 2) == 18


def test_hit_test_outside_is_none():
    layout = make_layout(1.0)
    assert hit_test(layout, layout.x0 - 1, layout.y0 + 10) is None
    assert hit_test(layout, layout.x0 + 10, layout.y0 - 1) is None
    assert hit_test(layout, layout.x0 + layout.width, layout.y0 + 10) is None
    assert hit_test(layout, layout.x0 + 10, layout.y0 + layout.height) is None
    assert hit_test(layout, -100, -100) is None


def test_contains_inside_and_outside():
    layout = make_layout(2.0)
    assert contains(layout, layout.x0, layout.y0)
    assert contains(layout, layout.x0 + layout.width - 1, layout.y0 + layout.height - 1)
    assert contains(layout, layout.x0 + 100, layout.y0 + 100)
    assert not contains(layout, layout.x0 - 1, layout.y0)
    assert not contains(layout, layout.x0, layout.y0 - 1)
    assert not contains(layout, layout.x0 + layout.width, layout.y0)
    assert not contains(layout, layout.x0, layout.y0 + layout.height)
    assert not contains(layout, 5000, 5000)


# --- palette geometry -------------------------------------------------------


def test_palette_quads_shapes_and_slot0():
    layout = make_layout(2.0)
    linear = srgb_to_linear(default_palette())
    positions, colours = palette_quads(layout, linear)
    assert positions.shape == (1536, 2)
    assert positions.dtype == np.float32
    assert colours.shape == (1536, 4)
    assert colours.dtype == np.float32
    np.testing.assert_allclose(colours[:6], np.array([SLOT0_COLOUR] * 6))
    # Slot 1 keeps its own colour on all six vertices.
    np.testing.assert_allclose(colours[6:12], np.tile(linear[1], (6, 1)))
    # The six vertices of slot 0 lie inside its rectangle.
    x, y, w, h = cell_rect(layout, 0)
    assert positions[:6, 0].min() == x
    assert positions[:6, 0].max() == x + w
    assert positions[:6, 1].min() == y
    assert positions[:6, 1].max() == y + h


def test_palette_quads_last_slot_bottom_right():
    layout = make_layout(1.0)
    positions, _ = palette_quads(layout, default_palette() / 255.0)
    x, y, w, h = cell_rect(layout, 255)
    last = positions[-6:]
    assert last[:, 0].min() == x
    assert last[:, 0].max() == x + w
    assert last[:, 1].min() == y
    assert last[:, 1].max() == y + h


def test_rect_outline_shape_and_inset():
    lines = rect_outline(10, 20, 30, 40)
    assert lines.shape == (8, 2)
    assert lines.dtype == np.float32
    assert lines[:, 0].min() == 10
    assert lines[:, 0].max() == 40
    assert lines[:, 1].min() == 20
    assert lines[:, 1].max() == 60
    inset = rect_outline(10, 20, 30, 40, inset=2.0)
    assert inset[:, 0].min() == 12
    assert inset[:, 0].max() == 38
    assert inset[:, 1].min() == 22
    assert inset[:, 1].max() == 58


# --- colour -----------------------------------------------------------------


def test_srgb_to_linear_curve():
    rgba = np.zeros((256, 4), dtype=np.uint8)
    rgba[0] = (0, 0, 0, 255)
    rgba[1] = (255, 255, 255, 255)
    rgba[2] = (128, 128, 128, 255)
    rgba[3] = (10, 10, 10, 128)
    out = srgb_to_linear(rgba)
    assert out.shape == (256, 4)
    assert out.dtype == np.float32
    assert out[0, 0] == 0.0
    assert out[1, 0] == 1.0
    assert abs(out[2, 0] - 0.2158) < 1e-3
    assert out[0, 3] == 1.0
    # Below the knee the curve is linear.
    assert abs(out[3, 0] - (10 / 255) / 12.92) < 1e-6
    # Alpha is not curved.
    assert abs(out[3, 3] - 128 / 255) < 1e-6


# --- 3D overlay -------------------------------------------------------------


@pytest.mark.parametrize("normal,row", ALL_NORMALS)
def test_face_quad_all_normals(normal, row):
    cell = (3, 5, 7)
    quad = face_quad(cell, normal)
    assert quad.shape == (4, 3)
    assert quad.dtype == np.float32
    expected = np.array(cell, dtype=np.float32) + _FACE_CORNERS[row]
    expected = expected + 0.003 * np.array(normal, dtype=np.float32)
    np.testing.assert_allclose(quad, expected, atol=1e-6)


def test_face_quad_floor_cell():
    quad = face_quad((4, 2, -1), (0, 0, 1), offset=0.003)
    expected = np.array((4, 2, -1), dtype=np.float32) + _FACE_CORNERS[4]
    expected[:, 2] += 0.003
    np.testing.assert_allclose(quad, expected, atol=1e-6)
    np.testing.assert_allclose(quad[:, 2], 0.003, atol=1e-6)


def test_cube_edges_twelve_distinct_edges():
    edges = cube_edges((2, 3, 4))
    assert edges.shape == (24, 3)
    assert edges.dtype == np.float32
    pairs = edges.reshape(12, 2, 3)
    # Every edge has unit length and lies on the cube at the cell.
    lengths = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
    np.testing.assert_allclose(lengths, 1.0)
    assert edges.min(axis=0).tolist() == [2, 3, 4]
    assert edges.max(axis=0).tolist() == [3, 4, 5]
    keys = {tuple(sorted((tuple(a), tuple(b)))) for a, b in pairs}
    assert len(keys) == 12
    # Every corner touches exactly three edges.
    corners = {tuple(v) for v in edges}
    assert len(corners) == 8
    for corner in corners:
        assert sum(1 for v in edges if tuple(v) == corner) == 3


def test_box_edges_span_size():
    edges = box_edges((10, 20, 30))
    assert edges.shape == (24, 3)
    assert edges.dtype == np.float32
    assert edges.min(axis=0).tolist() == [0, 0, 0]
    assert edges.max(axis=0).tolist() == [10, 20, 30]
    pairs = edges.reshape(12, 2, 3)
    keys = {tuple(sorted((tuple(a), tuple(b)))) for a, b in pairs}
    assert len(keys) == 12
    # Each edge is parallel to one axis and spans that axis fully.
    for a, b in pairs:
        diff = np.abs(b - a)
        assert np.count_nonzero(diff) == 1
        axis = int(np.argmax(diff))
        assert diff[axis] == (10, 20, 30)[axis]


# --- Phase 2 overlay geometry ----------------------------------------------


def _edge_keys(lines: np.ndarray) -> set:
    pairs = lines.reshape(-1, 2, 3)
    return {tuple(sorted((tuple(a), tuple(b)))) for a, b in pairs}


@pytest.mark.parametrize("normal,row", ALL_NORMALS)
def test_cells_face_quads_matches_face_quad(normal, row):
    cells = np.array([[0, 0, 0], [3, 5, 7], [10, 2, 1]])
    tris = cells_face_quads(cells, normal)
    assert tris.shape == (3 * 6, 3)
    assert tris.dtype == np.float32
    push = 0.003 * np.array(normal, dtype=np.float32)
    for i, cell in enumerate(cells):
        block = tris[i * 6 : (i + 1) * 6]
        quad = np.array(cell, dtype=np.float32) + _FACE_CORNERS[row] + push
        expected = quad[[0, 1, 2, 0, 2, 3]]
        np.testing.assert_allclose(block, expected, atol=1e-6)
        np.testing.assert_allclose(block, face_quad(cell, normal)[[0, 1, 2, 0, 2, 3]])


def test_cells_face_quads_offset_argument_and_empty():
    tris = cells_face_quads([(1, 1, 1)], (0, 0, 1), offset=0.5)
    np.testing.assert_allclose(tris[:, 2], 2.5)
    empty = cells_face_quads(np.zeros((0, 3), dtype=np.int64), (0, 0, 1))
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float32


def test_cells_edges_blocks_equal_cube_edges():
    cells = [(2, 3, 4), (0, 0, 0), (7, 1, 9)]
    edges = cells_edges(cells)
    assert edges.shape == (3 * 24, 3)
    assert edges.dtype == np.float32
    for i, cell in enumerate(cells):
        np.testing.assert_array_equal(edges[i * 24 : (i + 1) * 24], cube_edges(cell))
    assert cells_edges([]).shape == (0, 3)


def test_slab_edges_spans_lo_to_hi_plus_one():
    lo, hi = (2, 3, 4), (5, 3, 10)
    edges = slab_edges(lo, hi)
    assert edges.shape == (24, 3)
    assert edges.dtype == np.float32
    assert edges.min(axis=0).tolist() == [2, 3, 4]
    assert edges.max(axis=0).tolist() == [6, 4, 11]
    assert len(_edge_keys(edges)) == 12
    # A one-cell slab is the cube at that cell.
    np.testing.assert_array_equal(slab_edges((1, 2, 3), (1, 2, 3)), cube_edges((1, 2, 3)))


def _outside_unit_cube(points: np.ndarray, cell, offset: float) -> None:
    """Every point sits exactly `offset` outside the cube at cell on one axis."""
    local = points - np.asarray(cell, dtype=np.float32)
    # Distance outside [0, 1] per axis.
    below = np.clip(-local, 0, None)
    above = np.clip(local - 1.0, 0, None)
    outside = below + above
    np.testing.assert_allclose(outside.max(axis=1), offset, atol=1e-6)
    assert np.all(np.count_nonzero(outside > 1e-7, axis=1) == 1)


def test_mask_surface_single_cell():
    mask = np.zeros((1, 1, 1), dtype=bool)
    mask[0, 0, 0] = True
    tris, lines = mask_surface(mask, (0, 0, 0), (0, 0, 0))
    assert tris.shape == (36, 3)
    assert lines.shape == (48, 3)
    assert tris.dtype == np.float32
    assert lines.dtype == np.float32
    _outside_unit_cube(tris, (0, 0, 0), 0.003)
    _outside_unit_cube(lines, (0, 0, 0), 0.003)
    # Each quad's lines are its four edges: 4 distinct unit edges per quad.
    for q in range(6):
        block = lines[q * 8 : (q + 1) * 8]
        keys = _edge_keys(block)
        assert len(keys) == 4
        pairs = block.reshape(4, 2, 3)
        np.testing.assert_allclose(np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1), 1.0)


def test_mask_surface_cell_in_grid_space():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[3, 1, 4] = True
    tris, lines = mask_surface(mask, (0, 0, 0), (4, 4, 4))
    assert tris.shape == (36, 3)
    assert lines.shape == (48, 3)
    _outside_unit_cube(tris, (3, 1, 4), 0.003)
    _outside_unit_cube(lines, (3, 1, 4), 0.003)


def test_mask_surface_two_adjacent_cells_ten_quads():
    mask = np.zeros((2, 1, 1), dtype=bool)
    mask[:, 0, 0] = True
    tris, lines = mask_surface(mask, (0, 0, 0), (1, 0, 0))
    assert tris.shape == (60, 3)
    assert lines.shape == (80, 3)
    # No face at x == 1 (the shared face), so no point sits on that plane.
    assert not np.any(np.isclose(tris[:, 0], 1.0 + 0.003))
    assert not np.any(np.isclose(tris[:, 0], 1.0 - 0.003))


def test_mask_surface_bounds_restrict_the_output():
    mask = np.ones((4, 4, 4), dtype=bool)
    tris, lines = mask_surface(mask, (1, 1, 1), (2, 2, 2))
    # A 2 x 2 x 2 block has 24 exposed faces.
    assert tris.shape == (24 * 6, 3)
    assert lines.shape == (24 * 8, 3)
    assert tris.min() >= 1 - 0.003 - 1e-6
    assert tris.max() <= 3 + 0.003 + 1e-6
    # The full grid gives 6 * 16 faces.
    full, _ = mask_surface(mask, (0, 0, 0), (3, 3, 3))
    assert full.shape == (96 * 6, 3)


def test_mask_surface_empty_and_clipped_box():
    mask = np.zeros((3, 3, 3), dtype=bool)
    tris, lines = mask_surface(mask, (0, 0, 0), (2, 2, 2))
    assert tris.shape == (0, 3)
    assert lines.shape == (0, 3)
    mask[2, 2, 2] = True
    # A box that reaches beyond the grid is clipped to it.
    tris, _ = mask_surface(mask, (0, 0, 0), (10, 10, 10))
    assert tris.shape == (36, 3)
    tris, _ = mask_surface(mask, (3, 3, 3), (2, 2, 2))
    assert tris.shape == (0, 3)


def test_cells_faces_diagonal_line_eighteen_quads():
    cells = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    tris = cells_faces(cells)
    assert tris.shape == (18 * 6, 3)
    assert tris.dtype == np.float32
    # Same result when the line sits away from the origin.
    shifted = cells_faces([(5, 6, 7), (6, 7, 8), (7, 8, 9)])
    np.testing.assert_allclose(shifted, tris + np.array([5, 6, 7], dtype=np.float32), atol=1e-5)
    # Two touching cells share a face: 10 quads.
    assert cells_faces([(4, 4, 4), (4, 5, 4)]).shape == (60, 3)
    assert cells_faces([]).shape == (0, 3)


def test_cells_faces_single_cell_matches_mask_surface():
    mask = np.zeros((1, 1, 1), dtype=bool)
    mask[0, 0, 0] = True
    expected, _ = mask_surface(mask, (0, 0, 0), (0, 0, 0))
    got = cells_faces([(9, 8, 7)])
    np.testing.assert_allclose(got, expected + np.array([9, 8, 7], dtype=np.float32), atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_mirror_plane_quad_at_half_size(axis):
    size = (10, 20, 30)
    quad = mirror_plane_quad(size, axis)
    assert quad.shape == (4, 3)
    assert quad.dtype == np.float32
    np.testing.assert_allclose(quad[:, axis], size[axis] / 2)
    for other in range(3):
        if other == axis:
            continue
        assert quad[:, other].min() == 0
        assert quad[:, other].max() == size[other]
    assert len({tuple(v) for v in quad}) == 4


# --- Phase 3: the palette strip --------------------------------------------


def _centre(rect):
    x, y, w, h = rect
    return x + w / 2, y + h / 2


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_strip_layout_row_1_only(scale):
    layout = strip_layout(scale, (), False)
    margin = round(STRIP_MARGIN * scale)
    cur = round(STRIP_CURRENT * scale)
    btn = round(STRIP_BUTTON * scale)
    gap = round(STRIP_GAP * scale)
    assert isinstance(layout, StripLayout)
    assert (layout.x0, layout.y0) == (margin, margin)
    assert layout.swatch == round(STRIP_SWATCH * scale)
    assert layout.current == (margin, margin, cur, cur)
    assert [name for name, _ in layout.buttons] == ["NEW", "EYEDROP"]  # no 256 button
    x = margin + cur + gap
    for _, (bx, by, bw, bh) in layout.buttons:
        assert (bw, bh) == (btn, btn)
        assert bx == x
        # Buttons sit inside the height of the current swatch.
        assert margin <= by and by + bh <= margin + cur
        x += btn + gap
    assert layout.used == ()
    assert layout.height == cur
    assert layout.width == cur + len(STRIP_BUTTONS) * (gap + btn)
    assert layout.grid_y0 == margin + cur + round(STRIP_TEXT_BAND * scale)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_strip_layout_rows_stack_from_the_bottom(scale):
    used = tuple(range(1, 41))  # 40 used colours: three rows of 16, 16, 8
    layout = strip_layout(scale, used, False)
    swatch = layout.swatch
    gap = round(STRIP_GAP * scale)
    cur = layout.current[3]
    # Rows 2..: used, sorted, 16 per row, each row above the last.
    assert [i for i, _ in layout.used] == list(range(1, 41))
    used_y = layout.y0 + cur + gap
    caption_w = round(STRIP_CAPTION * scale)
    for k, (_, (x, y, w, h)) in enumerate(layout.used):
        assert (x, y, w, h) == (
            layout.x0 + caption_w + (k % 16) * swatch, used_y + (k // 16) * swatch, swatch, swatch
        )
    top = used_y + 3 * swatch
    # The name stands upright in a column left of the rows, as tall as they are.
    assert layout.captions == (("USED", (layout.x0, used_y, caption_w, 3 * swatch)),)
    assert layout.height == top - layout.y0
    assert layout.width == caption_w + 16 * swatch
    assert layout.grid_y0 == top + round(STRIP_TEXT_BAND * scale)
    # Every rect sits inside the bounding box.
    for _, (x, y, w, h) in layout.buttons + layout.used + layout.captions:
        assert layout.x0 <= x and x + w <= layout.x0 + layout.width
        assert layout.y0 <= y and y + h <= layout.y0 + layout.height


def test_strip_layout_used_sits_on_row_2():
    layout = strip_layout(1.0, (5, 2), False, caption_length=lambda _t: 10)
    assert [i for i, _ in layout.used] == [5, 2]  # your order
    y = layout.y0 + layout.current[3] + round(STRIP_GAP)
    assert layout.used[0][1][1] == y
    assert layout.height == y + layout.swatch - layout.y0


def test_strip_layout_names_its_sections_left_of_their_rows():
    layout = strip_layout(1.0, (5, 2), False, made=(7, 8), favourites=3)
    names = [t for t, _r in layout.captions]
    assert names == ["USED", "UNUSED", "FAV"]
    for (_t, caption), rows in zip(layout.captions, (layout.used, layout.made, layout.favourites)):
        x, y, w, h = caption
        assert x == layout.x0 and x + w == min(r[0] for _k, r in rows)  # the column left of the rows
        assert y == min(r[1] for _k, r in rows)  # level with the first row
    assert strip_layout(1.0, (5, 2), True).captions == ()  # the 256 grid shows every colour instead


def test_strip_layout_a_section_is_as_tall_as_its_name():
    # One row of colours is shorter than the upright name: the section grows to fit it.
    length = {"USED": 22.0, "UNUSED": 33.0, "FAV": 15.0}
    for scale in (1.0, 2.0):
        layout = strip_layout(scale, (5,), False, made=(7,), favourites=1, caption_length=length.get)
        tall = dict(layout.captions)
        assert tall["USED"][3] == math.ceil(22.0 * scale)
        assert tall["UNUSED"][3] == math.ceil(33.0 * scale)
        assert tall["FAV"][3] == layout.swatch  # one row is taller than "FAV"
        # The next section starts above the grown one, so names never overlap.
        assert layout.made[0][1][1] > tall["USED"][1] + tall["USED"][3]
        assert layout.favourites[0][1][1] > tall["UNUSED"][1] + tall["UNUSED"][3]


def test_strip_layout_used_is_capped_in_order_and_without_slot_0():
    used = tuple(range(255, -1, -1))  # every slot, slot 0 included, reversed
    layout = strip_layout(1.0, used, False)
    assert len(layout.used) == STRIP_USED_MAX == 48
    assert [i for i, _ in layout.used] == list(range(255, 207, -1))
    # Duplicates count once.
    layout = strip_layout(1.0, (4, 4, 4), False)
    assert [i for i, _ in layout.used] == [4]


def test_strip_layout_with_grid_shown_is_row_1_only():
    layout = strip_layout(1.0, (3, 4), True)
    assert layout.used == ()
    assert layout.height == layout.current[3]


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_strip_hit_test_returns_each_kind(scale):
    layout = strip_layout(scale, (20, 21), False)
    assert strip_hit_test(layout, *_centre(layout.current)) == ("CURRENT", 0)
    for name, rect in layout.buttons:
        assert strip_hit_test(layout, *_centre(rect)) == (name, 0)
    for index, rect in layout.used:
        assert strip_hit_test(layout, *_centre(rect)) == ("SWATCH", index)
    # A rect starts at its bottom-left pixel and ends before its top-right one.
    x, y, w, h = layout.used[0][1]
    assert strip_hit_test(layout, x, y) == ("SWATCH", 20)
    assert strip_hit_test(layout, x + w, y) == ("SWATCH", 21)
    assert strip_hit_test(layout, x, y + h) is None
    # The gap between the current swatch and the first button hits nothing.
    gx = layout.current[0] + layout.current[2]
    assert strip_hit_test(layout, gx, layout.y0) is None
    assert strip_hit_test(layout, layout.x0 - 1, layout.y0) is None
    assert strip_hit_test(layout, -100, -100) is None


def test_strip_contains_bounding_box():
    layout = strip_layout(2.0, tuple(range(1, 20)), False)
    assert strip_contains(layout, layout.x0, layout.y0)
    assert strip_contains(layout, layout.x0 + layout.width - 1, layout.y0 + layout.height - 1)
    # The gap between rows counts as inside, so a click there is swallowed.
    assert strip_contains(layout, layout.x0 + 1, layout.y0 + layout.current[3] + 1)
    assert not strip_contains(layout, layout.x0 - 1, layout.y0)
    assert not strip_contains(layout, layout.x0, layout.y0 - 1)
    assert not strip_contains(layout, layout.x0 + layout.width, layout.y0)
    assert not strip_contains(layout, layout.x0, layout.y0 + layout.height)


def test_strip_quads_used_in_order():
    layout = strip_layout(1.0, (3, 1, 2), False)
    linear = srgb_to_linear(default_palette())
    positions, colours = strip_quads(layout, linear)
    assert positions.shape == (3 * 6, 2)
    assert positions.dtype == np.float32
    assert colours.shape == (3 * 6, 4)
    assert colours.dtype == np.float32
    order = [3, 1, 2]
    rects = [rect for _, rect in layout.used]
    for q, (index, (x, y, w, h)) in enumerate(zip(order, rects)):
        block = positions[q * 6 : (q + 1) * 6]
        assert block[:, 0].min() == x
        assert block[:, 0].max() == x + w
        assert block[:, 1].min() == y
        assert block[:, 1].max() == y + h
        np.testing.assert_allclose(colours[q * 6 : (q + 1) * 6], np.tile(linear[index], (6, 1)))


def test_strip_quads_empty():
    layout = strip_layout(1.0, (), False)
    positions, colours = strip_quads(layout, srgb_to_linear(default_palette()))
    assert positions.shape == (0, 2)
    assert colours.shape == (0, 4)
    assert positions.dtype == np.float32
    assert colours.dtype == np.float32


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_grid_with_y0_sits_above_the_strip(scale):
    strip = strip_layout(scale, tuple(range(1, 30)), False)
    grid = make_layout(scale, strip.grid_y0)
    assert grid.cell == round(16 * scale)
    assert grid.x0 == round(8 * scale)
    assert grid.y0 == strip.grid_y0
    assert grid.y0 >= strip.y0 + strip.height + round(STRIP_TEXT_BAND * scale)
    # The grid's hit test still works at the new place, and slot 0 stays dead.
    x, y, w, h = cell_rect(grid, 17)
    assert hit_test(grid, x + w / 2, y + h / 2) == 17
    x, y, w, h = cell_rect(grid, 0)
    assert hit_test(grid, x + w / 2, y + h / 2) is None
    assert not contains(grid, grid.x0, strip.y0 + strip.height)
    # y0=None is the Phase 1 default.
    assert make_layout(scale, None) == make_layout(scale)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_strip_and_grid_x0_clears_the_rail_only_in_new_ui(scale):
    """Classic keeps the old left margin. New UI starts past the rail."""
    used = (1, 2, 3)
    classic_strip = strip_layout(scale, used, False)
    new_strip = strip_layout(scale, used, False, rail_width=HUD_RAIL_PX)
    classic_grid = make_layout(scale)
    new_grid = make_layout(scale, new_strip.grid_y0, rail_width=HUD_RAIL_PX)
    margin = round(STRIP_MARGIN * scale)
    rail = round(HUD_RAIL_PX * scale)
    past_rail = round((HUD_RAIL_PX + STRIP_MARGIN) * scale)

    assert classic_strip.x0 == margin
    assert classic_grid.x0 == margin
    assert classic_strip.current[0] == margin
    assert new_strip.x0 == past_rail
    assert new_grid.x0 == past_rail
    assert new_strip.current[0] == past_rail
    assert new_strip.x0 > rail
    assert new_grid.x0 > rail
    # Hit tests follow x0: a click under the rail misses the new-UI strip
    # and grid, and still hits classic.
    under_rail_x = rail - 1
    strip_y = new_strip.y0 + 1
    assert strip_hit_test(new_strip, under_rail_x, strip_y) is None
    assert not strip_contains(new_strip, under_rail_x, strip_y)
    assert strip_hit_test(classic_strip, classic_strip.x0 + 1, classic_strip.y0 + 1) == (
        "CURRENT",
        0,
    )
    assert strip_hit_test(new_strip, new_strip.x0 + 1, new_strip.y0 + 1) == (
        "CURRENT",
        0,
    )
    assert not contains(new_grid, under_rail_x, new_grid.y0 + 1)
    assert contains(new_grid, new_grid.x0, new_grid.y0)
    assert contains(classic_grid, classic_grid.x0, classic_grid.y0)


# --- Phase 3 materials: the strip marker ------------------------------------


def _codes(*marked: int) -> np.ndarray:
    codes = np.zeros(256, dtype=np.uint8)
    for k, index in enumerate(marked):
        codes[index] = 1 + k % 5  # every type code 1..5 gets the same dot
    return codes


def _assert_dot_top_right(dot, swatch, scale, big=False):
    """The dot sits inside the swatch, at its top-right corner, MARK_INSET in."""
    x, y, w, h = swatch
    dx, dy, dw, dh = dot
    side = round(MARK_PX * scale) * (2 if big else 1)
    inset = round(MARK_INSET * scale)
    assert (dw, dh) == (side, side)
    assert dx + dw == x + w - inset
    assert dy + dh == y + h - inset
    assert x <= dx and y <= dy


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_strip_marker_rects_marks_used_non_plain_swatches(scale):
    layout = strip_layout(scale, (1, 2, 3, 4, 5, 6), False)
    codes = _codes(2, 5, 6, 40)  # 40 is not used, so it has no swatch
    rects = strip_marker_rects(layout, codes, colour=1)  # slot 1 is Plain: no big dot
    swatches = dict(layout.used)
    assert len(rects) == 3
    for dot, index in zip(rects, (2, 5, 6)):
        _assert_dot_top_right(dot, swatches[index], scale)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_strip_marker_rects_adds_the_current_swatch_dot(scale):
    layout = strip_layout(scale, (1, 2, 3), False)
    codes = _codes(3)
    rects = strip_marker_rects(layout, codes, colour=3)
    assert len(rects) == 2
    _assert_dot_top_right(rects[0], dict(layout.used)[3], scale)
    _assert_dot_top_right(rects[1], layout.current, scale, big=True)
    # The current slot need not be in the used rows.
    rects = strip_marker_rects(layout, _codes(77), colour=77)
    assert len(rects) == 1
    _assert_dot_top_right(rects[0], layout.current, scale, big=True)
    # Type code 5 (Other) gets the dot too.
    codes = np.zeros(256, dtype=np.uint8)
    codes[2] = 5
    assert len(strip_marker_rects(layout, codes, colour=1)) == 1


def test_strip_marker_rects_empty():
    layout = strip_layout(1.0, (1, 2, 3), False)
    assert strip_marker_rects(layout, np.zeros(256, dtype=np.uint8), 1) == ()
    layout = strip_layout(1.0, (), False)
    assert strip_marker_rects(layout, _codes(1, 2, 3), colour=9) == ()


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_palette_marker_rects_on_the_grid(scale):
    layout = make_layout(scale)
    codes = _codes(1, 17, 255)
    codes[0] = 3  # slot 0 never gets a dot
    rects = palette_marker_rects(layout, codes)
    assert len(rects) == 3
    for dot, index in zip(rects, (1, 17, 255)):
        _assert_dot_top_right(dot, cell_rect(layout, index), scale)
    assert palette_marker_rects(layout, np.zeros(256, dtype=np.uint8)) == ()
    every = np.ones(256, dtype=np.uint8)
    assert len(palette_marker_rects(layout, every)) == 255


def test_marker_quads_white_is_the_rect_and_dark_is_one_outline_larger():
    rects = ((10, 20, 3, 3), (50, 60, 6, 6))
    dark, white = marker_quads(rects, MARK_OUTLINE)
    assert dark.shape == white.shape == (2 * 6, 2)
    assert dark.dtype == white.dtype == np.float32
    for q, (x, y, w, h) in enumerate(rects):
        block = white[q * 6 : (q + 1) * 6]
        assert (block[:, 0].min(), block[:, 0].max()) == (x, x + w)
        assert (block[:, 1].min(), block[:, 1].max()) == (y, y + h)
        block = dark[q * 6 : (q + 1) * 6]
        assert (block[:, 0].min(), block[:, 0].max()) == (x - 1, x + w + 1)
        assert (block[:, 1].min(), block[:, 1].max()) == (y - 1, y + h + 1)
    dark, _ = marker_quads(rects, 2)
    assert (dark[:6, 0].min(), dark[:6, 0].max()) == (8, 15)


def test_marker_quads_empty():
    dark, white = marker_quads((), MARK_OUTLINE)
    assert dark.shape == white.shape == (0, 2)
    assert dark.dtype == white.dtype == np.float32


def test_rounded_rect_stays_inside_the_rect_and_touches_its_sides():
    poly = rounded_rect(10, 20, 30, 40, 5, segments=4)
    assert poly.shape == (20, 2) and poly.dtype == np.float32
    assert poly[:, 0].min() >= 10 - 1e-4 and poly[:, 0].max() <= 40 + 1e-4
    assert poly[:, 1].min() >= 20 - 1e-4 and poly[:, 1].max() <= 60 + 1e-4
    assert abs(poly[:, 0].min() - 10) < 1e-4 and abs(poly[:, 1].max() - 60) < 1e-4
    small = rounded_rect(0, 0, 10, 4, 50, segments=2)
    assert abs(small[:, 1].max() - 4) < 1e-4
    sharp = rounded_rect(0, 0, 10, 4, 0, segments=1)
    assert {tuple(p) for p in np.round(sharp).tolist()} == {(10, 0), (10, 4), (0, 4), (0, 0)}


# --- new UI: the tool HUD ----------------------------------------------------


def test_hud_sizes_are_small():
    """One line at about the status bar's weight; the mode and the hint under it."""
    assert 14 <= HUD_TITLE_PX <= 18
    assert 11 <= HUD_TIP_PX <= 13
    assert HUD_LEFT_PX == 16
    assert HUD_RAIL_PX == 56
    from rok.core.chrome import RAIL_WIDTH

    assert HUD_RAIL_PX == RAIL_WIDTH
    assert HUD_ABOVE_HINT_PX == 12
    assert OPTION_ERASE_FADE_AFTER == 3


def test_hud_readout_idle_titles_and_tips():
    attach = hud_readout(ATTACH, False, False)
    assert attach.title == "Attach" and attach.tip == "⌥ click or drag to erase"
    assert attach.title_rgba == HUD_TITLE_RGBA and attach.tip_rgba == HUD_TIP_RGBA
    assert not attach.erase_held
    assert hud_readout(ERASE, False, False).title == "Erase"
    assert hud_readout(ERASE, False, False).tip == "Click or drag to erase"
    assert hud_readout(PAINT, False, False).tip == "Click or drag to paint"
    assert hud_readout(SELECT, False, False).tip == "Click or drag to select · ⌥ or right-click to deselect"
    assert HUD_TITLES[ATTACH] == "Attach"
    assert HUD_TIPS[ATTACH].startswith("⌥")
    # With a mode, the tip names it first; alone once the tip has faded.
    boxed = hud_readout(ATTACH, False, False, mode="BOX")
    assert boxed.title == "Attach" and boxed.tip == "Box · ⌥ click or drag to erase"
    assert hud_readout(ATTACH, False, False, mode="BOX", tip_faded=True).tip == "Box"
    assert hud_readout(ATTACH, True, False, mode="BOX").tip == HUD_ERASING_TIP


def test_hud_readout_option_held_erase_flips_title_not_the_strip_tool():
    """Strip ember stays on Attach; the HUD title becomes Erase in coral."""
    held = hud_readout(ATTACH, True, False)
    assert held.title == "Erase"
    assert held.title_rgba == HUD_ERASE_RGBA
    assert held.tip == HUD_ERASING_TIP
    assert held.tip_rgba == HUD_ERASE_RGBA
    assert held.erase_held
    assert hud_erase_held(ATTACH, True, False)
    assert not hud_erase_held(ATTACH, False, False)
    # Paint + Option is erase too; Select + Option (with or without Shift) deselects, never erases.
    assert hud_readout(PAINT, True, False).title == "Erase"
    select_sub = hud_readout(SELECT, True, True)
    assert select_sub.title == "Deselect" and not select_sub.erase_held


def test_hud_idle_tip_fades_after_three_option_erases_until_tool_changes():
    count, faded = 0, False
    for _ in range(OPTION_ERASE_FADE_AFTER):
        count, faded = hud_after_option_erase(count, faded)
    assert count == 3 and faded
    idle = hud_readout(ATTACH, False, False, tip_faded=True)
    assert idle.title == "Attach" and idle.tip is None
    # While Option is held the coral Erasing… tip still shows.
    held = hud_readout(ATTACH, True, False, tip_faded=True)
    assert held.tip == HUD_ERASING_TIP
    tool, faded = hud_on_tool_change(ATTACH, PAINT, True)
    assert tool == PAINT and faded is False
    assert hud_readout(PAINT, False, False, tip_faded=False).tip == HUD_TIPS[PAINT]


@pytest.mark.parametrize("scale, hint_top", [(1.0, 0), (2.0, 0), (1.5, 40), (1.0, 72)])
def test_hud_anchor_is_pixel_snapped_past_the_rail_twelve_above_hint(scale, hint_top):
    x, y = hud_anchor(scale, hint_top)
    assert (x, y) == (
        round((HUD_RAIL_PX + HUD_LEFT_PX) * scale),
        round(hint_top + 12 * scale),
    )
    assert isinstance(x, int) and isinstance(y, int)
    tx, ty = hud_title_origin(scale, hint_top)
    assert tx == x
    assert ty == y + round(HUD_TIP_PX * scale) + round(HUD_TITLE_TIP_GAP_PX * scale)
    assert isinstance(ty, int)


@pytest.mark.parametrize("scale, hint_top", [(1.0, 0), (2.0, 40)])
def test_the_mode_icon_sits_at_the_anchor_and_the_text_starts_right_of_it(scale, hint_top):
    ax, ay = hud_anchor(scale, hint_top)
    x, y, w, h = hud_icon_rect(scale, hint_top)
    assert (x, w, h) == (ax, round(28 * scale), round(28 * scale))
    _tx, ty = hud_title_origin(scale, hint_top)
    assert ay <= y and y + h <= ty + round(16 * scale) + 1  # inside the two text lines
    assert hud_text_x(scale, hint_top) == x + w + round(8 * scale)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_hud_clears_the_rail_and_sits_above_the_new_ui_strip(scale):
    """The tool HUD is past the rail plate and above the colour-strip band."""
    strip = strip_layout(scale, (1, 2, 3), False, rail_width=HUD_RAIL_PX)
    x, y = hud_anchor(scale, strip.grid_y0)
    rail = round(HUD_RAIL_PX * scale)
    assert x == round((HUD_RAIL_PX + HUD_LEFT_PX) * scale)
    assert x >= rail + round(HUD_LEFT_PX * scale)
    assert y >= strip.y0 + strip.height
    tx, ty = hud_title_origin(scale, strip.grid_y0)
    assert tx == x
    assert ty > y



def test_the_scene_readout_names_the_picked_model_and_how_to_step_in():
    from rok.core.hud import SCENE_TIP, scene_readout

    readout = scene_readout("Recon")
    assert readout.title == "Recon picked" and SCENE_TIP in readout.tip
    assert scene_readout("").title == "Scene"


def test_inside_a_model_the_tip_names_it_first():
    from rok.core.hud import hud_readout, inside_readout

    base = hud_readout("ATTACH", False, False, mode="VOXEL")
    inside = inside_readout(base, "Recon")
    assert inside.tip == f"Inside Recon · {base.tip}" and inside.title == base.title
    assert inside_readout(base, "") == base


def test_a_linked_copy_says_that_edits_change_every_copy():
    from rok.core.hud import hud_readout, inside_readout, linked_note, scene_readout

    base = hud_readout("ATTACH", False, False, mode="VOXEL")
    assert linked_note(3) in inside_readout(base, "Crate", 3).tip
    assert linked_note(3) in scene_readout("Crate", 3).tip
    assert linked_note(1) == "" and inside_readout(base, "Crate", 1) == inside_readout(base, "Crate")


def test_option_in_select_reads_deselect_never_erase():
    from rok.core.edit import SELECT
    from rok.core.hud import HUD_DESELECT_TIP, hud_readout

    for shift in (False, True):
        readout = hud_readout(SELECT, True, shift)
        assert readout.title == "Deselect" and readout.tip == HUD_DESELECT_TIP
        assert not readout.erase_held


def test_colours_made_and_not_used_yet_sit_in_their_own_section_above():
    from rok.core import hud

    lay = hud.strip_layout(1.0, used=(3, 7, 9), show_grid=False, made=(12, 7, 40))
    assert [i for i, _r in lay.used] == [3, 7, 9]
    assert [i for i, _r in lay.made] == [12, 40]  # 7 is used: it sits with the used ones
    top_used = max(r[1] + r[3] for _i, r in lay.used)
    assert lay.divider is not None and top_used < lay.divider < min(r[1] for _i, r in lay.made)
    assert lay.section_of(12) == "MADE" and lay.section_of(3) == "USED" and lay.section_of(99) is None
    assert hud.strip_hit_test(lay, lay.made[0][1][0] + 1, lay.made[0][1][1] + 1) == ("SWATCH", 12)
    assert lay.height >= max(r[1] + r[3] for _i, r in lay.made) - lay.y0
    plain = hud.strip_layout(1.0, used=(3,), show_grid=False)
    assert plain.made == () and plain.divider is None


def test_a_swatch_that_changes_section_flies_there_in_an_arc_and_pops():
    from rok.core import hud

    before = hud.strip_layout(1.0, used=(3,), show_grid=False, made=(12,))
    last = {i: ("USED", r) for i, r in before.used}
    last.update({i: ("MADE", r) for i, r in before.made})
    after = hud.strip_layout(1.0, used=(3, 12), show_grid=False)  # 12 got used
    moves = hud.swatch_moves(last, after)
    assert set(moves) == {12} and moves[12] == dict(before.made)[12]
    start, end = moves[12], dict(after.used)[12]
    assert hud.flying_rect(start, end, 0.0) == pytest.approx(start)
    assert hud.flying_rect(start, end, 1.0) == pytest.approx(end)
    mid = hud.flying_rect(start, end, 0.5)
    straight_y = (start[1] + end[1]) / 2
    assert mid[1] > straight_y + end[3]  # an arc, up and over
    assert mid[2] > end[2]  # bigger on the way
    assert hud.flying_rect(start, end, 0.9)[2] > end[2]  # the pop as it lands
    positions, colours = hud.strip_quads(after, np.zeros((256, 4), np.float32), {12: mid})
    assert positions.shape[0] == 6 * len(after.swatches)


def test_a_skin_built_in_blocks_is_the_skin_built_whole():
    """The selection skin is drawn in blocks (overlay._Skin); with `neighbours`
    a block looks across its edges, so the blocks add up to the whole skin."""
    from rok.core import hud

    rng = np.random.default_rng(3)
    mask = rng.random((40, 37, 33)) < 0.5
    whole, _ = hud.mask_surface(mask, (0, 0, 0), (39, 36, 32))
    parts = [
        hud.mask_surface(mask, (x, y, z), (x + 15, y + 15, z + 15), neighbours=True)[0]
        for x in range(0, 40, 16) for y in range(0, 37, 16) for z in range(0, 33, 16)
    ]
    blocks = np.concatenate(parts)

    def faces(a):
        return sorted(map(tuple, np.round(a.reshape(-1, 18), 4).tolist()))

    assert faces(blocks) == faces(whole)
    # Without neighbours a block shows faces at its edges too.
    edged = hud.mask_surface(mask, (0, 0, 0), (15, 15, 15))[0]
    inner = hud.mask_surface(mask, (0, 0, 0), (15, 15, 15), neighbours=True)[0]
    assert len(edged) > len(inner)


@pytest.mark.parametrize("normal", [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, -1, 0)])
def test_box_face_covers_what_the_cell_quads_cover(normal):
    from rok.core.hud import box_face, cells_face_quads

    axis = [abs(v) for v in normal].index(1)
    lo, hi = [3, 4, 5], [7, 9, 12]
    lo[axis] = hi[axis] = 6  # one cell thick along the normal, as a footprint is
    grid = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    per_cell = cells_face_quads(grid, normal)
    one = box_face(lo, hi, normal)
    assert one.shape == (6, 3)
    assert np.allclose(one.min(axis=0), per_cell.min(axis=0)) and np.allclose(one.max(axis=0), per_cell.max(axis=0))
    t = one.reshape(2, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
    assert np.isclose(area, len(grid))  # one unit of area per cell
    n = np.cross(t[0, 1] - t[0, 0], t[0, 2] - t[0, 0])
    q = per_cell.reshape(-1, 3, 3)[0]
    assert np.dot(n, np.cross(q[1] - q[0], q[2] - q[0])) > 0  # the same winding: it faces the same way
