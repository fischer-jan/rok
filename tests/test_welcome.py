"""The welcome screen's layout."""

import pytest

from rok.core import welcome


def _inside(inner, outer):
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return ix >= ox - 1e-6 and iy >= oy - 1e-6 and ix + iw <= ox + ow + 1e-6 and iy + ih <= oy + oh + 1e-6


@pytest.mark.parametrize("size", [(1600, 900), (1000, 700), (700, 560), (1920, 1080)])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_everything_sits_inside_the_panel_and_the_view(size, scale):
    recent = [f"/a/b/model{k}.vox" for k in range(8)]
    lay = welcome.welcome_layout(scale, size[0] * scale if scale > 1 and size[0] < 1200 else size[0],
                                 size[1] * scale if scale > 1 and size[0] < 1200 else size[1], recent)
    view = (0, 0, size[0] * scale if scale > 1 and size[0] < 1200 else size[0],
            size[1] * scale if scale > 1 and size[0] < 1200 else size[1])
    assert _inside(lay.rect, view)
    for rect in (lay.header, lay.logo, lay.new_button, lay.open_button, lay.start_caption, lay.recent_caption, lay.hint,
                 *[r.rect for r in lay.rows]):
        assert _inside(rect, lay.rect), rect


def test_the_logo_sits_in_the_header_left_of_the_wordmark():
    lay = welcome.welcome_layout(1.0, 1600, 900, [])
    assert _inside(lay.logo, lay.header)
    assert lay.logo[0] + lay.logo[2] <= lay.title[0]  # the wordmark starts right of the logo
    assert lay.title[1] == pytest.approx(lay.logo[1] + lay.logo[3] / 2)  # and its middle is the logo's


def test_the_panel_is_the_header_and_the_part_under_it():
    lay = welcome.welcome_layout(1.0, 1600, 900, [])
    assert lay.rect[3] == pytest.approx(welcome.HEADER_H + welcome.LOWER_H)
    assert lay.header[1] + lay.header[3] == pytest.approx(lay.rect[1] + lay.rect[3])  # the header is the top
    assert lay.logo[2] == welcome.LOGO


def test_a_low_view_makes_the_header_lower_and_keeps_the_buttons():
    tall = welcome.welcome_layout(1.0, 1600, 900, [])
    low = welcome.welcome_layout(1.0, 1600, 300, [])
    assert low.header[3] < tall.header[3] and low.logo[2] < tall.logo[2]
    assert low.new_button[3] == tall.new_button[3] and low.open_button[3] == tall.open_button[3]
    assert _inside(low.logo, low.header)


def test_recent_rows_are_capped_and_named():
    lay = welcome.welcome_layout(1.0, 1600, 900, [f"/x/dir/m{k}.vox" for k in range(9)])
    assert len(lay.rows) == welcome.MAX_RECENT
    assert lay.rows[0].name == "m0.vox" and lay.rows[0].folder == "dir"


def test_clicks_find_their_targets():
    lay = welcome.welcome_layout(1.0, 1600, 900, ["/x/a.vox", "/x/b.vox"])
    cx = lambda r: (r[0] + r[2] / 2, r[1] + r[3] / 2)  # noqa: E731
    assert welcome.target_at(lay, *cx(lay.new_button)) == welcome.NEW
    assert welcome.target_at(lay, *cx(lay.open_button)) == welcome.OPEN
    assert welcome.target_at(lay, *cx(lay.rows[1].rect)) is lay.rows[1]
    assert welcome.target_at(lay, *cx(lay.header)) is None


def test_no_recent_files_gives_no_rows():
    lay = welcome.welcome_layout(1.0, 1600, 900, [])
    assert lay.rows == ()


def test_the_buttons_do_not_overlap_and_rows_do_not_overlap():
    lay = welcome.welcome_layout(1.0, 1600, 900, [f"/x/{k}.vox" for k in range(5)])
    nb, ob = lay.new_button, lay.open_button
    assert nb[1] >= ob[1] + ob[3]
    for a, b in zip(lay.rows, lay.rows[1:]):
        assert a.rect[1] >= b.rect[1] + b.rect[3] - 1e-6
