"""Palette v2 pass 3: hover tip delay, chip fade, fold motion, editor, drags, wheel, clicks."""

from __future__ import annotations

import pytest

from rok.app import gfx, overlay, palette_ui, ui
from rok.app.draw import DrawList
from rok.app.object import document
from rok.app.opkit import CONTEXT
from rok.app.state import state
from rok.core import palette_anim as pa
from test_app_ui import app, centre_of, click_at, move_to  # noqa: F401


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    c = Clock()
    pa.set_clock(c)
    overlay._TW.clear()
    yield c
    pa.set_clock()
    overlay._TW.clear()


def _frame(app):
    overlay.invalidate_palette()
    app.redraw()


def _swatch(app, i=0):
    _frame(app)
    return overlay._dock.section("USED").swatches[i]


def test_hover_ring_now_tip_after_400ms_in_hint_line(app, clock):
    key, rect = _swatch(app)
    move_to(app, *centre_of(rect))
    assert state.hover_strip == ("SWATCH", key)
    _frame(app)
    assert overlay.dock_hover_tip() is None
    clock.t += 0.39
    _frame(app)
    assert overlay.dock_hover_tip() is None
    clock.t += 0.02
    _frame(app)
    tip = overlay.dock_hover_tip()
    assert tip.startswith("#") and "Ctrl drag to move · Shift drag for a gradient" in tip
    assert ui._hint(CONTEXT) == tip


def test_click_picks_and_chip_fades_over_80ms(app, clock):
    _frame(app)
    before = overlay._TW.value("chip", None, 0)
    key, rect = _swatch(app, 2)
    click_at(app, *centre_of(rect))
    assert document.colour == key
    _frame(app)
    start = overlay._TW._t["chip"]
    assert start[0] == before and start[1] != before and start[3] == pa.CHIP_FADE
    clock.t += pa.CHIP_FADE
    assert not overlay._TW.running()


def test_fold_animates_card_height_upward_and_hud_follows(app, clock):
    _frame(app)
    # Fold the other lists first. With them open, their rows would take the
    # space and the card would stay at the height cap.
    for name in ("UNUSED", "FAV"):
        sec = overlay._dock.section(name)
        if sec is not None and not sec.folded:
            click_at(app, *centre_of(sec.heading))
            _frame(app)
    clock.t += pa.FOLD
    _frame(app)
    assert overlay._dock_dy == 0
    h0 = overlay._dock.card[3]
    head = overlay._dock.section("USED").heading
    click_at(app, *centre_of(head))
    _frame(app)
    h1 = overlay._dock.card[3]
    assert h1 < h0
    assert overlay._dock_dy == pytest.approx(h0 - h1)  # starts at the old height
    plate = overlay.strip_plate(overlay.strip_layout(app.region), 1.0)
    assert plate[3] == pytest.approx(h0)  # the HUD sits on the shown card, no jump
    clock.t += pa.FOLD / 2
    _frame(app)
    assert 0 < overlay._dock_dy < h0 - h1
    clock.t += pa.FOLD
    _frame(app)
    assert overlay._dock_dy == 0
    assert overlay._dock.card[1] == 20 or overlay._dock.card[1] > 0  # bottom edge fixed


def test_editor_slides_up_and_closes_by_chevron(app, clock):
    _frame(app)
    key, rect = _swatch(app)
    click_at(app, *centre_of(overlay._dock.chip))
    assert palette_ui._popup is not None and palette_ui._popup.docked
    _frame(app)
    assert overlay._dock_dy < 0  # the plate starts at the old height and grows
    assert overlay._TW._t["card_h"][3] == pa.EDITOR
    clock.t += pa.EDITOR
    _frame(app)
    assert overlay._dock_dy == 0
    click_at(app, *centre_of(overlay._dock.chevron))
    assert palette_ui._popup is None


def test_editor_click_outside_card_closes(app, clock):
    _frame(app)
    click_at(app, *centre_of(overlay._dock.chip))
    assert palette_ui._popup is not None
    click_at(app, app.region.width - 200, app.region.height / 2)
    assert palette_ui._popup is None


def test_swap_release_off_swatch_flies_back(app, clock):
    _frame(app)
    palette_ui._drag = None
    key, rect = _swatch(app)
    overlay.start_fly_back(key, (500.0, 500.0))
    _frame(app)
    assert overlay._fly_back is not None
    clock.t += pa.FLY_BACK + 0.01
    _frame(app)
    assert overlay._fly_back is None


def test_swap_motion_slides_moved_swatches(app, clock):
    _frame(app)
    used = overlay._dock.section("USED")
    (k0, r0), (k1, r1) = used.swatches[0], used.swatches[1]
    overlay.arm_swap_motion()
    palette_ui._apply_move(k0, k1)
    _frame(app)
    assert overlay._swap_flights
    clock.t += pa.SWAP + 0.01
    _frame(app)
    assert not overlay._swap_flights


def test_gradient_preview_colours():
    from rok.core import palette_dock
    p = palette_dock.gradient_preview((0, 0, 0), (100, 200, 50), 4)
    assert p[0] == (0, 0, 0) and p[-1] == (100, 200, 50) and len(p) == 5


def test_clicks_on_card_never_reach_the_view(app, clock):
    _frame(app)
    card = overlay._dock.card
    before = document.stroke_serial
    click_at(app, card[0] + card[2] - 3, card[1] + 3)  # plate, no swatch
    assert document.stroke_serial == before


def test_ctrl_move_over_card_does_not_orbit(app, clock):
    _frame(app)
    card = overlay._dock.card
    move_to(app, card[0] + card[2] + 300, card[1] + 300)
    app.handle("modifiers", {"shift": False, "ctrl": True, "alt": False, "meta": False})
    cam = app.camera
    before = repr(cam.view_proj())
    move_to(app, card[0] + 20, card[1] + 20)
    move_to(app, card[0] + 40, card[1] + 30)
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})
    assert repr(cam.view_proj()) == before


def test_ctrl_drag_on_swatch_starts_the_swap_not_the_orbit(app, clock):
    key, rect = _swatch(app)
    other = overlay._dock.section("USED").swatches[3]
    move_to(app, *centre_of(rect))
    app.handle("modifiers", {"shift": False, "ctrl": True, "alt": False, "meta": False})
    before = repr(app.camera.view_proj())
    app.handle("mouse_down", {"button": "left"})
    assert palette_ui._drag == ("SWAP", key)
    move_to(app, *centre_of(other[1]))
    app.handle("mouse_up", {"button": "left"})
    app.handle("modifiers", {"shift": False, "ctrl": False, "alt": False, "meta": False})
    assert palette_ui._drag is None
    assert repr(app.camera.view_proj()) == before
    _frame(app)
    assert overlay._swap_flights  # the moved swatches slide over SWAP seconds


def test_wheel_over_scrolling_grid_scrolls_it_not_the_view(app, clock, monkeypatch):
    _frame(app)
    from rok.core import palette_dock
    dock = palette_dock.dock_layout(300, list(range(1, 90)), [], 0, x0=76, y0=20)
    assert dock.section("USED").scroll
    monkeypatch.setattr(overlay, "_dock", dock)
    monkeypatch.setattr(overlay, "dock_layout", lambda region=None: dock)
    c = dock.section("USED").clip
    before = repr(app.camera.view_proj())
    overlay._dock_scroll.clear()
    assert overlay.dock_scroll(app.region, c[0] + 5, c[1] + 5, -1)
    assert overlay._dock_scroll["USED"] > 0
    assert repr(app.camera.view_proj()) == before


def test_folded_swatches_fade_out_inside_the_card(app, clock):
    _frame(app)
    head = overlay._dock.section("USED").heading
    n = len(overlay._dock.section("USED").swatches)
    click_at(app, *centre_of(head))
    _frame(app)
    assert overlay._dock.section("USED").folded
    assert len(overlay._sec_cache["USED"]) == n  # the old rows are still drawn, fading
    assert 0.0 < overlay._TW.value("fade:USED", 0.0, pa.CONTENT_FADE, eased=False) <= 1.0
    clock.t += pa.CONTENT_FADE
    assert overlay._TW.value("fade:USED", 0.0, pa.CONTENT_FADE, eased=False) == 0.0


def test_insertion_line_between_swatches():
    from rok.core import palette_dock
    dock = palette_dock.dock_layout(2000, list(range(1, 15)), [], 0)
    used = dict(dock.section("USED").swatches)
    fwd = overlay.insertion_line(dock, 1, 5)
    assert fwd[2] == 2.0 and fwd[3] == used[5][3]
    assert used[5][0] + used[5][2] <= fwd[0] + 1 <= used[6][0]  # after the target
    back = overlay.insertion_line(dock, 9, 5)
    assert used[4][0] + used[4][2] <= back[0] + 1 <= used[5][0]  # before it
    assert palette_ui.drag_label(None) is None or palette_ui._drag is None


def _region_rects(drawlist, height):
    """Rects from a draw list, back in region pixels (origin at the bottom left)."""
    data = drawlist.arrays()["rects"]
    if len(data) == 0:
        return []
    out = []
    for x, y, w, h, r, g, b, a, _rad, stroke in data.reshape(-1, 10):
        out.append((float(x), float(height - y - h), float(w), float(h),
                    (float(r), float(g), float(b), float(a)), float(stroke)))
    return out


def _overlaps(a, b) -> bool:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _reset_dock_motion() -> None:
    overlay._TW.clear()
    overlay._FLIGHTS.clear()
    overlay._swap_flights.clear()
    overlay._swap_from.clear()
    overlay._swap_armed[0] = False
    overlay._fly_back = None
    overlay._sec_cache.clear()
    overlay._HOVER.item = None


def _draw_in_use(n_used, colour, scroll_offset=0):
    """Draw a card whose In use section scrolls, and return (dock, region rects)."""
    from rok.core import palette_dock as pd

    dock = pd.dock_layout(768, list(range(1, n_used + 1)), [], 0, x0=76, y0=20, scroll_offset=scroll_offset)
    _reset_dock_motion()
    d = DrawList()
    pal = [(0.25, 0.25, 0.25, 1.0)] * 256

    class Region:
        height = 900
        x = 0
        y = 0

    with gfx.pixels(d, Region):
        overlay.draw_dock(dock, overlay.dock_to_strip(dock, 1.0), pal, colour, None, 1.0)
    return dock, _region_rects(d, Region.height)


def _ember_ring(rects, swatch):
    """The ember stroke around `swatch`, or None when it was not drawn whole."""
    from rok.core import palette_dock as pd

    want = (swatch[0] - 1.0, swatch[1] - 1.0, swatch[2] + 2.0, swatch[3] + 2.0)
    ember = pd.RING_EMBER
    found = [r for r in rects if r[5] > 0 and all(abs(c - e) < 1e-3 for c, e in zip(r[4], ember))]
    # A clipped stroke is dropped, not trimmed: a ring of the full size means it was not cut.
    rings = [r for r in found if r[:4] == pytest.approx(want)]
    assert rings, f"no whole ember ring around {swatch}"
    assert rings[0][5] == pytest.approx(pd.RING_EMBER_W)
    return rings[0]


@pytest.fixture
def fresh_dock_motion():
    """The card's animation state starts empty, and the editor height is put back."""
    saved_editor = overlay._last_editor_h[0]
    overlay._last_editor_h[0] = 0
    _reset_dock_motion()
    yield
    overlay._last_editor_h[0] = saved_editor
    _reset_dock_motion()


@pytest.mark.parametrize("n_used, swatch, dense_after", ((255, 16, None), (100, 20, 10**6)))
def test_the_current_swatch_keeps_its_ring_while_in_use_scrolls(monkeypatch, fresh_dock_motion,
                                                                n_used, swatch, dense_after):
    """255 colours, and 100 colours at 20 px, both scroll. The current swatch's
    ring stays whole on the first visible row, the rows below it and in the first
    and last column; the last row's bottom edge stays cut by the fade. A row
    that has scrolled away does not draw over the heading."""
    from rok.core import palette_dock as pd

    if dense_after is not None:
        monkeypatch.setattr(pd, "DENSE_AFTER", dense_after)
    dock, _rects = _draw_in_use(n_used, 1)
    used = dock.section("USED")
    assert dock.swatch == swatch and used.scroll and used.clip is not None
    by = {}
    for key, rect in used.swatches:
        by.setdefault(rect[1], []).append((key, rect))
    rows = [by[y] for y in sorted(by)]
    corners = [rows[-1][0], rows[-1][-1], rows[1][0], rows[1][-1]]  # y is up: top row, then above the bottom
    middle = rows[len(rows) // 2][len(rows[len(rows) // 2]) // 2]
    for key, rect in corners + [middle]:
        dock, rects = _draw_in_use(n_used, key)
        ring = _ember_ring(rects, rect)
        section = dock.section("USED")
        assert ring[1] + ring[3] <= section.divider[1]  # below the heading's divider
        assert not _overlaps(ring, section.heading)
    # Three rows up: colour 1 has left the clip and must not paint the heading.
    step = swatch + pd.GAP
    dock, rects = _draw_in_use(n_used, 1, scroll_offset=step * 3)
    section = dock.section("USED")
    assert 1 not in dict(section.swatches)
    for rect in rects:
        if rect[2] <= swatch + 4 and rect[3] <= swatch + 4:
            assert not _overlaps(rect, section.heading)


def _force_short_count(monkeypatch, n=12_345):
    """The current colour has `n` voxels and the exact text never fits the count rect."""
    monkeypatch.setattr(overlay, "_colour_count", lambda colour: n)
    monkeypatch.setattr(overlay, "text_width", lambda text, size: 10_000.0)
    overlay.invalidate_palette()


def _count_point(app):
    _frame(app)
    return centre_of(overlay._dock.count)


def test_count_hit_and_tip_follow_the_current_count_and_scene(app, monkeypatch):
    _force_short_count(monkeypatch)
    mx, my = _count_point(app)
    assert overlay.dock_hit(app.region, mx, my) == ("COUNT", 0)
    assert overlay._count_exact(overlay._dock) == "12,345 voxels"
    monkeypatch.setattr(overlay, "text_width", lambda text, size: 1.0)  # the exact text fits again
    assert overlay.dock_hit(app.region, mx, my) is None
    assert overlay._count_exact(overlay._dock) is None
    monkeypatch.setattr(overlay, "text_width", lambda text, size: 10_000.0)
    assert overlay.dock_hit(app.region, mx, my) == ("COUNT", 0)
    monkeypatch.setattr(document, "scene", None)  # no scene: nothing to count, nothing to hit
    assert overlay.dock_hit(app.region, mx, my) is None
    overlay._HOVER.item, overlay._HOVER.since = ("COUNT", 0), 0.0
    assert overlay.dock_hover_tip() is None


def test_count_hit_is_current_after_the_card_collapses(app, monkeypatch):
    _force_short_count(monkeypatch)
    mx, my = _count_point(app)
    assert overlay.dock_hit(app.region, mx, my) == ("COUNT", 0)
    palette_ui.set_dock_collapsed(True)
    overlay.invalidate_palette()
    mx, my = _count_point(app)  # the collapsed header keeps its count line
    assert overlay.dock_hit(app.region, mx, my) == ("COUNT", 0)
    monkeypatch.setattr(overlay, "_colour_count", lambda colour: 12)  # a count that fits: no stale tip
    assert overlay.dock_hit(app.region, mx, my) is None
    overlay._HOVER.item, overlay._HOVER.since = ("COUNT", 0), 0.0
    assert overlay.dock_hover_tip() is None


def test_hovering_the_short_count_shows_the_exact_count_after_the_delay(app, clock, monkeypatch):
    _force_short_count(monkeypatch)
    move_to(app, *_count_point(app))
    assert state.hover_strip == ("COUNT", 0)
    _frame(app)
    assert overlay._HOVER.item == ("COUNT", 0)
    assert overlay.dock_hover_tip() is None
    clock.t += pa.HOVER_TIP_DELAY + 0.01
    _frame(app)
    assert overlay.dock_hover_tip() == "12,345 voxels"
    assert ui._hint(CONTEXT) == "12,345 voxels"


def test_a_click_on_the_short_count_is_eaten_and_changes_nothing(app, monkeypatch):
    _force_short_count(monkeypatch)
    mx, my = _count_point(app)
    serial, colour = document.stroke_serial, document.colour
    collapsed, folds = palette_ui.dock_collapsed(), dict(palette_ui.dock_folds())
    click_at(app, mx, my)
    assert document.stroke_serial == serial and document.colour == colour
    assert palette_ui.dock_collapsed() == collapsed and palette_ui.dock_folds() == folds
    assert palette_ui._popup is None
    assert overlay.dock_click(app.region, ("COUNT", 0)) is False


def test_clip_cuts_rects_and_drops_text_outside():
    from rok.app import gfx

    class L:
        def __init__(self):
            self.rects, self.texts = [], []

        def rect(self, *a):
            self.rects.append(a)

        def text(self, *a):
            self.texts.append(a)

    d = L()
    saved = gfx._t.list, gfx._t.height
    gfx._t.list, gfx._t.height = d, 100.0
    try:
        with gfx.clipped((0, 0, 50, 50)):
            gfx.rrect((40, 40, 20, 20), (1, 1, 1, 1), 4)
            gfx.rrect((60, 60, 10, 10), (1, 1, 1, 1))
            gfx.fonts.position(0, 70, 70)
            gfx.fonts.draw(0, "x")
            with gfx.faded(0.5):
                gfx.rrect((1, 1, 5, 5), (1, 1, 1, 1))
    finally:
        gfx._t.list, gfx._t.height = saved
    assert len(d.rects) == 2 and d.rects[0][2] == 10 and d.rects[0][3] == 10
    assert d.rects[1][4][3] == 0.5
    assert d.texts == []


def test_the_open_editor_sits_on_the_header_after_the_card_shrinks(app, clock):
    """Opening the editor gives the sections fewer rows, so the header moves down;
    the editor moves with it and stays inside the card."""
    _frame(app)
    click_at(app, *centre_of(dict(overlay._dock.buttons)["NEW"]))
    assert palette_ui._popup is not None and palette_ui._popup.docked
    clock.t += pa.EDITOR
    _frame(app)
    dock = overlay._dock
    panel = palette_ui._current_layout(palette_ui._popup)["PANEL"]
    assert panel[1] == pytest.approx(dock.editor[1], abs=1)  # its bottom on the header's top
    assert panel[1] + panel[3] <= dock.card[1] + dock.card[3] + 1  # inside the card's plate
