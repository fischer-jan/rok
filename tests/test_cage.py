"""The selection cage: the AABB outline, its projection and the pill strip."""

from pathlib import Path

import numpy as np

from rok.core.cage import (
    CAGE_IDLE_HINT,
    CORNERS,
    EDGES,
    FACES,
    GLYPHS,
    GROUPS,
    GROUP_GAP,
    BAR_PAD,
    PILL_GAP,
    PILL_HOVER_S,
    PILL_SIZE,
    PILL_STATUS,
    PILL_TIPS,
    PILL_TITLES,
    CageGeometry,
    CageScreen,
    aabb,
    face_axis_side,
    geometry,
    glyph_lines,
    glyph_rect,
    is_glyph,
    pill_hint,
    pill_hud,
    pill_tip_due,
    screen_aabb,
    screen_geometry,
    strip_contains,
    strip_hit,
    strip_layout,
)
from rok.core.gizmo import project

from helpers import ORTHO, px_of

SIZE = (8, 8, 8)
WIDTH = HEIGHT = 200
BOUNDS = ((1, 2, 3), (4, 5, 6))  # lo, hi inclusive → box 1..5, 2..6, 3..7


def make_screen(bounds=BOUNDS) -> CageScreen:
    return screen_geometry(geometry(bounds), ORTHO, WIDTH, HEIGHT)


# ---------------------------------------------------------------- geometry


def test_handle_names_are_unique_and_complete():
    assert len(FACES) == 6 and len(EDGES) == 12 and len(CORNERS) == 8
    # Eight buttons in four groups.
    assert GLYPHS == ("PAINT", "MASK", "SEL_GROW", "SEL_SHRINK", "INVERT", "BEVEL", "SMOOTH", "SPLIT")
    assert [len(g) for g in GROUPS] == [2, 3, 2, 1]
    assert len(set(FACES + EDGES + CORNERS)) == 6 + 12 + 8
    assert all(is_glyph(g) for g in GLYPHS)
    assert not is_glyph("FACE_X0")
    assert face_axis_side("FACE_Z0") == (2, 0)
    assert face_axis_side("FACE_Y1") == (1, 1)


def test_aabb_is_hi_plus_one():
    lo, hi = aabb(BOUNDS)
    assert lo.tolist() == [1.0, 2.0, 3.0]
    assert hi.tolist() == [5.0, 6.0, 7.0]


def test_geometry_layout():
    g = geometry(BOUNDS)
    assert isinstance(g, CageGeometry)
    assert g.lo.tolist() == [1.0, 2.0, 3.0] and g.hi.tolist() == [5.0, 6.0, 7.0]
    assert g.lines.shape == (24, 3)
    assert set(g.faces) == set(FACES)
    assert set(g.edges) == set(EDGES)
    assert set(g.corners) == set(CORNERS)
    assert g.faces["FACE_X0"].shape == (4, 3)
    assert np.allclose(g.faces["FACE_X0"][:, 0], 1.0)
    assert np.allclose(g.faces["FACE_X1"][:, 0], 5.0)
    assert np.allclose(g.corners["CORNER_000"], [1.0, 2.0, 3.0])
    assert np.allclose(g.corners["CORNER_111"], [5.0, 6.0, 7.0])
    # Twelve unique edge midpoints.
    mids = [tuple(np.round(e.mean(axis=0), 5)) for e in g.edges.values()]
    assert len(set(mids)) == 12


def test_screen_geometry_projects_the_top_face():
    s = make_screen()
    # FACE_Z1 is z = 7, so it projects to the same xy as the box top.
    pts = s.faces["FACE_Z1"]
    assert pts.shape == (4, 2)
    xs, ys = pts[:, 0], pts[:, 1]
    assert np.allclose(sorted(xs), sorted([px_of((1, 0))[0], px_of((5, 0))[0]] * 2))
    assert np.allclose(sorted(ys), sorted([px_of((0, 2))[1], px_of((0, 6))[1]] * 2))
    assert np.allclose(s.corners["CORNER_000"], px_of((1, 2)))
    assert np.allclose(s.corners["CORNER_110"], px_of((5, 6)))
    assert s.strip is not None
    assert [name for name, _rect in s.strip.pills] == list(GLYPHS)


def test_glyph_lines_land_in_the_rect():
    rect = glyph_rect(40.0, 80.0, 16.0)
    for name in ("BEVEL", "SPLIT", "INVERT", "MASK", "PAINT"):  # the others have icon files
        lines = glyph_lines(name, rect)
        assert lines.ndim == 2 and lines.shape[1] == 2 and len(lines) >= 2
        assert lines[:, 0].min() >= rect[0] - 0.5
        assert lines[:, 0].max() <= rect[0] + rect[2] + 0.5
        assert lines[:, 1].min() >= rect[1] - 0.5
        assert lines[:, 1].max() <= rect[1] + rect[3] + 0.5
    assert glyph_lines("NOPE", rect).shape == (0, 2)


def test_pill_hints_name_the_ops():
    assert pill_hint("SPLIT") == PILL_STATUS["SPLIT"]
    assert "Split" in pill_hint("SPLIT")
    assert "Grow" in pill_hint("SEL_GROW")
    assert pill_hint("FACE_X0") == ""
    assert CAGE_IDLE_HINT == "Drag a selected block to move it · the bar edits the selection"
    title, tip = pill_hud("SPLIT")
    assert title == "Split off" and "new model" in tip
    assert PILL_HOVER_S == 0.2
    assert pill_tip_due(None) is False
    assert pill_tip_due(10.0, now=10.1) is False
    assert pill_tip_due(10.0, now=10.2 + 1e-9) is True


def test_strip_layout_sits_above_the_aabb():
    # Room above: the bar and an 8 px gap fit under a 400 px region.
    aabb = (160.0, 80.0, 240.0, 140.0)
    lay = strip_layout(aabb, 400, 400, scale=1.0)
    assert lay is not None and lay.above is True
    assert [name for name, _ in lay.pills] == list(GLYPHS)
    size, gap, group_gap, pad = int(PILL_SIZE), int(PILL_GAP), int(GROUP_GAP), int(BAR_PAD)
    x, y, w, h = lay.rect
    assert h == 2 * pad + size
    assert w == 2 * pad + len(GLYPHS) * size + (len(GLYPHS) - len(GROUPS)) * gap + (len(GROUPS) - 1) * group_gap
    assert y == int(round(140.0 + 8))
    assert abs((x + w / 2.0) - 0.5 * (160.0 + 240.0)) <= 1.0
    assert lay.pills[0][1] == (x + pad, y + pad, size, size)  # buttons sit next to the box
    assert lay.pills[1][1][0] == x + pad + size + gap  # Mask follows Paint closely
    assert lay.pills[2][1][0] == lay.pills[1][1][0] + size + group_gap  # Grow starts a new group
    assert len(lay.wells) == len(GROUPS)
    for (wx, wy, ww, wh), group in zip(lay.wells, GROUPS):
        first = next(rect for name, rect in lay.pills if name == group[0])
        assert (wx, wy, wh) == (first[0], first[1], first[3])
        assert ww == len(group) * size + (len(group) - 1) * gap
    for _name, (px, py, pw, ph) in lay.pills:  # every button is on the plate
        assert x <= px and px + pw <= x + w and y <= py and py + ph <= y + h


def test_every_bar_button_has_an_icon_file_or_a_line_glyph():
    icons = Path(__file__).resolve().parents[1] / "assets" / "icons" / "chrome" / "1x"
    from rok.app.overlay import PILL_ICONS

    for name in GLYPHS:
        icon = PILL_ICONS.get(name, name)
        assert (icons / f"{icon}.png").exists() or glyph_lines(name, (0, 0, 10, 10)).shape[0] > 0, name
        assert name in PILL_TITLES and name in PILL_TIPS and name in PILL_STATUS, name


def test_strip_layout_drops_below_when_near_the_top():
    aabb = (40.0, 150.0, 120.0, 190.0)
    lay = strip_layout(aabb, 200, 200, scale=1.0)
    assert lay is not None and lay.above is False
    assert lay.rect[1] == int(round(150.0 - 8 - lay.rect[3]))
    _name, (_px, py, _pw, ph) = lay.pills[0]
    assert py == lay.rect[1] + int(BAR_PAD)  # the plate is only the buttons and their pad
    assert lay.rect[1] + lay.rect[3] == py + ph + int(BAR_PAD)
    # A tall region has room above the same box.
    high = strip_layout(aabb, 200, 400, scale=1.0)
    assert high is not None and high.above is True


def test_strip_layout_scales_and_rejects_a_bad_aabb():
    lay = strip_layout((10.0, 10.0, 50.0, 50.0), 800, 600, scale=2.0)
    assert lay is not None
    one = strip_layout((10.0, 10.0, 50.0, 50.0), 800, 600, scale=1.0)
    assert lay.rect[2:] == (one.rect[2] * 2, one.rect[3] * 2)
    assert strip_layout(None, 200, 200) is None
    assert strip_layout((float("nan"), 0, 1, 1), 200, 200) is None


def test_strip_hit_and_contains():
    lay = strip_layout((100.0, 80.0, 180.0, 140.0), 400, 400, scale=1.0)
    assert lay is not None
    hx, hy, hw, hh = lay.pills[0][1]
    assert strip_hit(lay, hx + hw / 2.0, hy + hh / 2.0) == "PAINT"
    dx, dy, dw, dh = lay.pills[-1][1]
    assert strip_hit(lay, dx + 1.0, dy + 1.0) == "SPLIT"  # Split is the last button
    # The gap between two groups is on the plate, but hits no button.
    wx, _wy, ww, _wh = lay.wells[0]
    gap_x = wx + ww + int(GROUP_GAP) / 2.0
    assert strip_hit(lay, gap_x, hy + hh / 2.0) is None
    assert strip_contains(lay, gap_x, hy + hh / 2.0) is True
    assert strip_contains(lay, 0.0, 0.0) is False
    assert strip_hit(None, 0.0, 0.0) is None
    assert strip_contains(None, 0.0, 0.0) is False


def test_screen_geometry_lays_the_strip_on_the_projected_aabb():
    s = make_screen()
    box = screen_aabb(s)
    assert box is not None
    assert s.strip is not None
    rebuilt = strip_layout(box, WIDTH, HEIGHT, scale=1.0)
    assert rebuilt is not None
    assert s.strip.rect == rebuilt.rect
    assert s.strip.pills == rebuilt.pills


def test_behind_the_eye_is_nan():
    persp = ORTHO.copy()
    persp[3] = [0.0, 0.0, -1.0, 0.0]
    px = project(persp, [(1, 0, -2), (1, 0, 5)], WIDTH, HEIGHT)
    assert np.allclose(px[0], [105.0, 100.0])
    assert np.isnan(px[1]).all()
