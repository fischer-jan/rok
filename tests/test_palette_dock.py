"""Palette v2 dock card layout (rok.core.palette_dock)."""

from rok.core import palette_dock as pd


def _lay(n_used=12, n_unused=5, favs=3, vh=2000, **kw):
    return pd.dock_layout(vh, list(range(1, n_used + 1)), list(range(100, 100 + n_unused)), favs, **kw)


def test_sizes():
    lay = _lay()
    assert lay.swatch == 20
    assert lay.chip[2:] == (40, 40)
    assert [n for n, _ in lay.buttons] == ["NEW", "EYEDROP", "FAVOURITE"]
    assert all(r[2:] == (24, 24) for _, r in lay.buttons)
    a, b = lay.section("USED").swatches[:2]
    assert b[1][0] - a[1][0] == 20 + 2  # 2 px gap
    assert len({r[1] for _, r in lay.section("USED").swatches[:10]}) == 1  # 10 per row


COUNTS = (10, 40, 41, 100, 255)  # colours in the card, split over In use and Unused


def _n(total, **kw):
    return _lay(total - total // 4, total // 4, 0, **kw)


def test_card_width_never_changes_with_colour_count():
    for scale in (1.0, 1.5):
        widths = {_n(n, scale=scale).card[2] for n in COUNTS}
        assert widths == {round(234 * scale)}


def test_swatch_size_20_until_40_then_16():
    for scale in (1.0, 1.5):
        for n in COUNTS:
            lay = _n(n, scale=scale)
            expect = round((pd.SWATCH_DENSE if n > pd.DENSE_AFTER else pd.SWATCH) * scale)
            assert lay.swatch == expect
            used = lay.section("USED")
            assert used and all(r[2] == expect and r[3] == expect for _, r in used.swatches)


def test_dense_twelve_per_row():
    for scale in (1.0, 1.5):
        lay = _n(41, scale=scale)
        used = lay.section("USED").swatches
        assert len(used) >= 12
        assert len({r[1] for _, r in used[:12]}) == 1
        assert used[11][1][1] == used[0][1][1]
        if len(used) > 12:
            assert used[12][1][1] != used[0][1][1]


def test_header_rects_identical_at_40_and_41():
    for scale in (1.0, 1.5):
        a, b = _n(40, scale=scale), _n(41, scale=scale)
        dy = (a.card[1] + a.card[3]) - (b.card[1] + b.card[3])
        b = pd.shifted(b, dy)
        assert a.chip == b.chip
        assert a.label == b.label
        assert a.count == b.count
        assert a.buttons == b.buttons
        assert a.chevron == b.chevron


def _all_cells(lay):
    cells = [(f"{sec.name}:{k}", r) for sec in lay.sections for k, r in sec.hits]
    return cells + [(f"RECENT:{k}", r) for k, r in lay.recent_hits]


def test_hit_cell_is_swatch_plus_gap_centred_in_both_modes():
    for n, size in ((40, 20), (41, 16)):
        for scale in (1.0, 1.5, 2.0):
            lay = _n(n, scale=scale)
            sw, gap = round(size * scale), round(pd.GAP * scale)
            assert lay.swatch == sw and lay.gap == gap
            for sec in lay.sections:
                assert len(sec.hits) == len(sec.swatches)
                for (k, r), (hk, c) in zip(sec.swatches, sec.hits):
                    assert hk == k
                    assert c[2:] == (sw + gap, sw + gap)
                    assert (r[0] - c[0], r[1] - c[1]) == (gap // 2, gap // 2)
                    assert (c[0] + c[2] - r[0] - r[2], c[1] + c[3] - r[1] - r[3]) == (gap - gap // 2,) * 2


def test_dense_gap_hits_the_nearest_swatch_on_both_sides():
    lay = _n(41)
    assert lay.swatch == 16
    (ka, a), (kb, b) = lay.section("USED").swatches[:2]
    assert a[1] == b[1]  # same row
    my = a[1] + a[3] / 2
    # the 2 px gap between them: its left pixel belongs to a, its right pixel to b
    assert pd.hit(lay, a[0] + a[2], my) == f"USED:{ka}"
    assert pd.hit(lay, a[0] + a[2] + 1, my) == f"USED:{kb}"
    # the 1 px half-gap above the first row is still the swatch's
    assert pd.hit(lay, a[0] + 5, a[1] + a[3]) == f"USED:{ka}"


def _tiles(rects, step):
    """Cells on one grid with pitch `step`: no overlap, no gap between neighbours."""
    xs = sorted({r[0] for r in rects})
    ys = sorted({r[1] for r in rects})
    assert all(b - a == step for a, b in zip(xs, xs[1:])), xs
    assert all(b - a == step for a, b in zip(ys, ys[1:])), ys
    assert len({(r[0], r[1]) for r in rects}) == len(rects)


def test_cells_tile_and_stay_in_the_grid_area():
    for n, size in ((30, 20), (41, 16), (100, 16)):
        for scale in (1.0, 1.5):
            lay = _n(n, scale=scale)
            sw, gap = round(size * scale), round(pd.GAP * scale)
            pad, below = round(pd.PAD * scale), round(pd.BELOW_DIVIDER * scale)
            lo, hi = gap // 2, gap - gap // 2
            for sec in lay.sections:
                cells = [c for _k, c in sec.hits]
                _tiles(cells, sw + gap)
                assert all(c[2] == c[3] == sw + gap for c in cells)
                sw_rects = [r for _k, r in sec.swatches]
                # the grid area is the swatches' grid grown by half a gap on each side
                assert min(c[0] for c in cells) == min(r[0] for r in sw_rects) - lo
                assert max(c[0] + c[2] for c in cells) == max(r[0] + r[2] for r in sw_rects) + hi
                assert max(c[1] + c[3] for c in cells) == sec.divider[1] - below + hi
                assert hi < below  # the divider band stays clear
                for c in cells:
                    assert lay.card[0] + pad - hi <= c[0] and c[0] + c[2] <= lay.card[0] + lay.card[2] - pad + hi
            for a, ca in _all_cells(lay):
                for b, cb in _all_cells(lay):
                    if a < b:
                        assert (ca[0] + ca[2] <= cb[0] or cb[0] + cb[2] <= ca[0]
                                or ca[1] + ca[3] <= cb[1] or cb[1] + cb[3] <= ca[1]), (a, b)


def test_collapsed_recent_cells_are_swatch_plus_gap():
    lay = _lay(collapsed=True, recent=list(range(1, 9)))
    cells = [c for _k, c in lay.recent_hits]
    assert all(c[2:] == (22, 22) for c in cells)
    _tiles(cells, 22)


def test_swatch_cells_exact_and_other_targets_at_least_20px():
    for n in COUNTS:
        for scale in (1.0, 1.5):
            for lay in (_n(n, scale=scale), _n(n, scale=scale, vh=500),
                        _lay(collapsed=True, recent=list(range(1, 13)), scale=scale)):
                cell = lay.swatch + lay.gap
                for name, r in lay.targets():
                    if ":" in name and not name.startswith("HEADING:"):
                        assert r[2:] == (cell, cell), (n, scale, name, r)
                    else:
                        assert r[2] >= 20 * scale and r[3] >= 20 * scale, (n, scale, name, r)


def test_40_41_boundary_counts_favourites():
    at40 = pd.dock_layout(2000, list(range(1, 39)), [], 2)  # 38 used + 2 favourites
    at41 = pd.dock_layout(2000, list(range(1, 40)), [], 2)  # 39 used + 2 favourites
    assert at40.swatch == 20 and at41.swatch == 16
    assert at40.section("FAV").swatches[0][1][2] == 20
    assert at41.section("FAV").swatches[0][1][2] == 16
    assert at41.section("USED").swatches[0][1][2] == 16
    assert at40.card[2] == at41.card[2] == 234


def test_scale_sweep_grid_fits_and_is_centred():
    for scale in (0.75, 1.0, 1.1, 1.25, 1.3, 1.5, 2.0):
        for n, cols in ((30, 10), (100, 12)):
            lay = _n(n, scale=scale)
            pad = round(pd.PAD * scale)
            inner = lay.card[2] - 2 * pad
            for sec in lay.sections:
                rects = [r for _k, r in sec.swatches]
                if len(rects) < cols:
                    continue
                left = min(r[0] for r in rects) - (lay.card[0] + pad)
                right = lay.card[0] + pad + inner - max(r[0] + r[2] for r in rects)
                grid_w = max(r[0] + r[2] for r in rects) - min(r[0] for r in rects)
                assert grid_w <= inner, (scale, n, grid_w, inner)
                assert left >= 0 and right >= 0 and abs(left - right) <= 1, (scale, n, left, right)


def test_short_count():
    cases = {0: "0", 1: "1", 999: "999", 1000: "1.0k", 1240: "1.2k", 9999: "9.9k", 10000: "10k",
             999_999: "999k", 1_000_000: "1.0M", 12_345_678: "12M", 999_999_999: "999M",
             1_000_000_000: "1.0B", 1_200_000_000: "1.2B", 1_999_999_999: "1.9B", 12_000_000_000: "12B"}
    for n, text in cases.items():
        assert pd.short_count(n) == text, n
    assert not any(pd.short_count(n) == "1000M" for n in cases)


def test_count_text_exact_when_it_fits_else_short():
    assert pd.count_text(1240) == "1,240 voxels"
    assert pd.count_text(1240, lambda t: len(t) <= 12) == "1,240 voxels"
    assert pd.count_text(1_234_567, lambda t: len(t) <= 12) == "1.2M voxels"


def test_count_rect_fits_text_and_clears_buttons():
    for scale in (1.0, 1.5, 2.0):
        for n in COUNTS:
            for lay in (_n(n, scale=scale), _lay(collapsed=True, scale=scale)):
                cx, cy, cw, ch = lay.count
                first = lay.buttons[0][1]
                assert cx + cw <= first[0] - round(pd.LABEL_CLEAR * scale)
                for _name, b in lay.buttons:
                    assert cx + cw <= b[0] or cy + ch <= b[1] or b[1] + b[3] <= cy
                assert cx >= lay.chip[0] + lay.chip[2]
                for voxels in (100, 9999, 999_999):
                    est = len(pd.count_text(voxels, lambda _t: False)) * 0.65 * 10 * scale  # a conservative width at 10 pt
                    assert est <= cw, (scale, voxels, est, cw)


def test_scale():
    lay = _lay(scale=2.0)
    assert lay.swatch == 40 and lay.chip[2] == 80


def test_order_top_to_bottom_and_header_on_top():
    lay = _lay()
    assert [s.name for s in lay.sections] == ["USED", "UNUSED", "FAV"]
    ys = [s.heading[1] for s in lay.sections]
    assert ys == sorted(ys, reverse=True)  # y up: In use highest
    assert lay.chip[1] > lay.sections[0].heading[1]
    assert [s.title for s in lay.sections] == ["In use", "Unused", "Favourites"]
    assert [s.count for s in lay.sections] == [12, 5, 3]


def test_heading_spacing_and_divider():
    lay = _lay()
    s = lay.section("USED")
    assert s.divider[3] == 1
    assert s.divider[1] + 1 == s.heading[1]
    header_bottom = lay.chip[1] - pd.PAD
    assert header_bottom - (s.heading[1] + s.heading[3]) == 8


def test_empty_section_has_no_heading():
    lay = _lay(n_unused=0, favs=0)
    assert [s.name for s in lay.sections] == ["USED"]


def test_fold_hides_swatches_and_shrinks():
    open_ = _lay()
    shut = _lay(folds={"UNUSED": True})
    sp = shut.section("UNUSED")
    assert sp.folded and sp.swatches == ()
    assert shut.card[3] < open_.card[3]


def test_fold_state_round_trip():
    f = {"USED": False, "UNUSED": True, "FAV": False}
    assert pd.fold_state(pd.fold_text(f)) == f
    assert pd.fold_state("junk,UNUSED=1") == {"UNUSED": True}


def test_height_cap_scrolls_used_and_unused_together():
    big = dict(n_used=30, n_unused=30, favs=30)
    vh = 700
    lay = _lay(vh=vh, **big)
    assert lay.card[3] <= 0.4 * vh
    used, unused = lay.section("USED"), lay.section("UNUSED")
    assert not used.folded and not unused.folded
    assert used.swatches and unused.swatches
    assert used.scroll and unused.scroll
    # Each list moves on its own.
    moved = _lay(vh=vh, scroll_offsets={"UNUSED": 10**6}, **big)
    assert moved.section("UNUSED").swatches[0][0] != unused.swatches[0][0]
    assert moved.section("USED").swatches[0][0] == used.swatches[0][0]
    # A short viewport still shows both. The header stays on the card.
    tight = _lay(vh=520, **big)
    assert tight.section("USED").swatches and tight.section("UNUSED").swatches
    assert not tight.section("UNUSED").folded and not tight.section("FAV").folded
    assert tight.chip[1] + tight.chip[3] <= tight.card[1] + tight.card[3]


def test_a_tight_card_keeps_unused_open():
    # A viewport in which the card fits without folding Unused.
    full = _lay(n_used=10, n_unused=30, favs=10, vh=10000)
    unused_rows = full.section("UNUSED").content_height
    vh = int((full.card[3] - unused_rows) / 0.4) + 5
    lay = _lay(n_used=10, n_unused=30, favs=10, vh=vh)
    assert not lay.section("UNUSED").folded and lay.section("UNUSED").swatches
    assert lay.section("USED").swatches and not lay.section("FAV").folded


def test_collapsed_is_chip_plus_8_recent():
    lay = _lay(collapsed=True, recent=list(range(1, 13)))
    assert lay.collapsed and lay.sections == ()
    assert len(lay.recent) == 8
    assert lay.chip[2] == 40
    many = _n(100, collapsed=True, recent=list(range(1, 13)))
    assert many.swatch == 20 and many.card[2] == 234


def test_inline_editor_unfolds_upward():
    shut = _lay()
    lay = _lay(editor_height=200)
    assert lay.editor is not None
    assert lay.editor[1] + lay.editor[3] == lay.card[1] + lay.card[3]  # inside the card, at its top
    assert lay.card[3] == shut.card[3] + 200  # the same plate grows upward
    assert lay.chip == shut.chip  # the header stays put
    assert lay.editor[2] == lay.card[2]


def test_hit():
    lay = _lay()
    cx, cy = lay.chip[0] + 5, lay.chip[1] + 5
    assert pd.hit(lay, cx, cy) == "CHIP"
    k, r = lay.section("UNUSED").swatches[0]
    assert pd.hit(lay, r[0] + 1, r[1] + 1) == f"UNUSED:{k}"
    h = lay.section("FAV").heading
    assert pd.hit(lay, h[0] + 30, h[1] + 5) == "HEADING:FAV"
    assert pd.hit(lay, -5, -5) is None


def test_dock_prefs_persist(tmp_path, monkeypatch):
    from rok.app import palette_ui, settings
    monkeypatch.setattr(settings, "_path", lambda: tmp_path / "settings.txt")
    assert palette_ui.dock_folds() == {}
    palette_ui.toggle_dock_fold("UNUSED")
    assert palette_ui.dock_folds() == {"UNUSED": True}
    palette_ui.toggle_dock_fold("UNUSED")
    assert palette_ui.dock_folds() == {"UNUSED": False}
    assert not palette_ui.dock_collapsed()
    palette_ui.set_dock_collapsed(True)
    assert palette_ui.dock_collapsed()


def test_toggle_palette_command_registered():
    from rok.app import palette_ui
    assert palette_ui.TogglePaletteAction.title == "Toggle Palette"
    assert palette_ui.TogglePaletteAction.command_id == "rok.toggle_palette"


def test_scroll_keeps_swatches_inside_clip():
    big = dict(n_used=60, n_unused=30, favs=30)
    a = _lay(vh=600, **big)
    b = _lay(vh=600, scroll_offset=44, **big)
    clip = b.section("USED").clip
    for _k, r in b.section("USED").swatches:
        assert r[1] >= clip[1] and r[1] + r[3] <= clip[1] + clip[3]
    assert a.section("USED").swatches[0][0] != b.section("USED").swatches[0][0]


def test_strip_hit_test_uses_the_centred_cells():
    """The real app path: dock_to_strip, then hud.strip_hit_test, in both modes."""
    from rok.app import overlay
    from rok.core import hud
    for used, favs, size in ((38, 2, 20), (39, 2, 16)):
        lay = pd.dock_layout(2000, list(range(1, used + 1)), [], favs, x0=76, y0=20)
        strip = overlay.dock_to_strip(lay, 1.0)
        assert lay.swatch == size
        sws = lay.section("USED").swatches
        (k0, _r0), (k1, r1), (k2, _r2) = sws[:3]
        my = r1[1] + r1[3] // 2
        assert hud.strip_hit_test(strip, r1[0] - 1, my) == ("SWATCH", k1)  # the half gap on its left
        assert hud.strip_hit_test(strip, r1[0] - 2, my) == ("SWATCH", k0)  # belongs to the neighbour
        assert hud.strip_hit_test(strip, r1[0] + r1[2], my) == ("SWATCH", k1)  # the half gap on its right
        assert hud.strip_hit_test(strip, r1[0] + r1[2] + 1, my) == ("SWATCH", k2)
        (fk, fr), (fk1, _fr1) = lay.section("FAV").swatches[:2]
        assert hud.strip_hit_test(strip, fr[0] - 1, fr[1] + 1) == ("FAV", fk)
        assert hud.strip_hit_test(strip, fr[0] + fr[2] + 1, fr[1] + 1) == ("FAV", fk1)
        top = sws[0][1]
        assert hud.strip_hit_test(strip, top[0] + 3, top[1] + top[3] + 1) is None  # the band above the grid


def test_dock_to_strip_cells_move_with_the_layout():
    from rok.app import overlay
    lay = _lay(x0=10, y0=10)
    strip = overlay.dock_to_strip(lay, 1.0)
    assert len(strip.cells) == len(strip.used) + len(strip.made) + len(strip.favourites)
    assert {kind for kind, _k, _r in strip.cells} == {"SWATCH", "FAV"}
    raised = overlay._raised(strip, 7)
    assert all(a[2][1] - b[2][1] == 7 for a, b in zip(raised.cells, strip.cells))
    shifted = overlay._shifted(strip, 3, 4)
    assert all(a[2][0] - b[2][0] == 3 and a[2][1] - b[2][1] == 4 for a, b in zip(shifted.cells, strip.cells))


def test_dock_to_strip_has_no_captions_and_dock_rects():
    from rok.app import overlay
    lay = _lay()
    strip = overlay.dock_to_strip(lay, 1.0)
    assert strip.captions == ()
    assert strip.current == lay.chip
    assert [n for n, _ in strip.buttons] == ["NEW", "EYEDROP", "FAVOURITE"]
    assert strip.used == lay.section("USED").swatches
    assert strip.made == lay.section("UNUSED").swatches
    assert (strip.x0, strip.y0, strip.width, strip.height) == lay.card


def test_inline_editor_anchor_sits_on_card(monkeypatch):
    from rok.app import overlay, palette_ui
    lay = _lay(x0=76, y0=20)
    monkeypatch.setattr(overlay, "_dock", lay)
    x, top = palette_ui.dock_editor_anchor(300.0, 2000.0)
    assert x == 76
    monkeypatch.setattr(overlay, "_dock", None)
    assert palette_ui.dock_editor_anchor(300.0, 2000.0) is None


def test_dock_tip():
    from rok.app import overlay
    assert overlay.dock_tip(("SWATCH", 3)) == pd.HOVER_TIP
    assert "Ctrl drag to move" in pd.HOVER_TIP and "Shift drag for a gradient" in pd.HOVER_TIP
    assert overlay.dock_tip(None) is None


def test_header_label_clears_buttons():
    for kw in ({}, {"n_used": 50}):
        lay = _lay(**kw)
        right = lay.card[0] + lay.card[2] - pd.PAD
        assert lay.label[0] + lay.label[2] <= lay.chevron[0]  # line 1: hex, then the chevron
        assert lay.label[2] >= 140
        assert lay.label[1] > lay.buttons[0][1][1]  # line 2 is below line 1
        assert lay.buttons[-1][1][0] + 24 == right and lay.chevron[0] + lay.chevron[2] == right
        xs = [r[0] for _, r in lay.buttons]
        assert [b - a for a, b in zip(xs, xs[1:])] == [28, 28]  # 24 px, 4 px apart


def test_swatches_20_at_1x_and_card_fits_rows():
    lay = _lay(10, 5, 3)
    assert all(r[2] == 20 for _, r in lay.section("USED").swatches)
    row = lay.section("USED").swatches[9][1]
    inner_right = lay.card[0] + lay.card[2] - pd.PAD
    assert row[0] + row[2] == inner_right
    assert lay.card[2] == 10 * pd.SWATCH + 9 * pd.GAP + 2 * pd.PAD
    assert lay.chevron[0] + lay.chevron[2] == inner_right == lay.buttons[-1][1][0] + lay.buttons[-1][1][2]
    assert lay.section("UNUSED").chevron[0] + lay.section("UNUSED").chevron[2] == inner_right
    dense = _lay(30, 20, 5)
    assert dense.swatch == 16
    assert all(r[2] == 16 for _, r in dense.section("USED").swatches)
    assert dense.card[2] == lay.card[2]
    first, last = dense.section("USED").swatches[0][1], dense.section("USED").swatches[11][1]
    assert first[0] - (dense.card[0] + pd.PAD) == 2  # 214 px grid centred in 218
    assert inner_right - (last[0] + last[2]) == 2
    assert dense.chevron[0] + dense.chevron[2] == inner_right
    assert dense.section("UNUSED").chevron[0] + dense.section("UNUSED").chevron[2] == inner_right


def test_no_cap_unused_lists_all():
    lay = _lay(n_used=5, n_unused=120, favs=0, vh=100000)
    assert lay.section("UNUSED").count == 120 and len(lay.section("UNUSED").swatches) == 120


def test_editor_cap_scrolls_used_and_unused_together():
    big = dict(n_used=72, n_unused=30, favs=0)  # 6 rows of 12, enough to hit the 60% editor cap
    lay = _lay(vh=1000, editor_height=300, **big)
    assert lay.card[3] <= 600
    used, unused = lay.section("USED"), lay.section("UNUSED")
    assert used.scroll
    assert lay.editor[1] + lay.editor[3] == lay.card[1] + lay.card[3]  # editor whole, on top
    assert not unused.folded and unused.swatches
    tight = _lay(vh=800, editor_height=300, **big)  # little room left: both lists scroll, neither folds
    assert tight.section("USED").scroll and tight.section("UNUSED").scroll
    assert tight.section("USED").swatches and tight.section("UNUSED").swatches


def test_width_is_grid_plus_8_and_right_edges_line_up():
    for kw in ({}, {"n_used": 50}):
        lay = _lay(**kw)
        row_w = 10 * pd.SWATCH + 9 * pd.GAP  # comfortable inner width, also when dense
        assert lay.card[2] == row_w + 2 * pd.PAD
        right = lay.card[0] + pd.PAD + row_w
        assert lay.chevron[0] + lay.chevron[2] == right
        for s in lay.sections:
            assert s.chevron[0] + s.chevron[2] == right


def test_headings_on_8px_grid():
    lay = _lay()
    for s in lay.sections:
        assert s.heading[3] % 8 == 0
    used, unused = lay.section("USED"), lay.section("UNUSED")
    first = min(r[1] for _, r in used.swatches)
    assert (used.divider[1] - (max(r[1] + r[3] for _, r in used.swatches))) == 8  # divider to swatches
    assert first - (unused.heading[1] + unused.heading[3]) == 8  # last row to the next heading


def test_reveal_scrolls_current_row_into_view():
    big = dict(n_used=80, n_unused=30, favs=30)
    lay = _lay(vh=600, reveal=80, **big)
    used = lay.section("USED")
    assert used.scroll and 80 in dict(used.swatches)


def test_header_lines_do_not_overlap_and_card_not_taller():
    lay = _lay()
    b = lay.buttons[0][1]
    assert lay.label[1] >= b[1] + b[3]  # line 1 above the buttons' line
    assert lay.label[1] + lay.label[3] <= lay.chip[1] + lay.chip[3]
    assert lay.chip[1] - lay.card[1] >= 0 and (lay.card[1] + lay.card[3]) - (lay.chip[1] + lay.chip[3]) == pd.PAD


def test_right_edges_on_one_line_at_1x_and_2x():
    for scale in (1.0, 2.0):
        for kw in ({}, {"n_used": 50}):
            lay = _lay(scale=scale, **kw)
            edge = lay.chevron[0] + lay.chevron[2]
            assert lay.buttons[-1][1][0] + lay.buttons[-1][1][2] == edge
            assert all(s.chevron[0] + s.chevron[2] == edge for s in lay.sections)


def test_section_keys_listed_when_folded():
    lay = _lay(folds={"UNUSED": True})
    assert lay.section("UNUSED").swatches == () and lay.section("UNUSED").keys == tuple(range(100, 105))
