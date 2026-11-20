"""Tooltips for shader arguments: the rok_tip lines and where the panel shows them."""

from rok.core import shader_panel as sp
from rok.core import xs

SOURCE = """// xs demo [Size] [Fir Share]
// xs_begin
// arg : { name = 'Size'      var = 'm_size' range = '6 60'  value = '18' step = '1' precision = '0' }
// arg : { name = 'Fir Share' var = 'm_mix'  range = '0 100' value = '30' step = '1' precision = '0' }
// xs_end
// rok_tip Size : A tree's height in cells.
// rok_tip Size : Below 8 the firs have one tier.
// rok_tip fir share: How many of the trees are firs: 0 none, 100 all.
"""


def test_tip_lines_join_per_argument_and_match_names_in_any_case():
    args = xs.parse_args(SOURCE)
    assert args[0].tip == "A tree's height in cells. Below 8 the firs have one tier."
    assert args[1].tip == "How many of the trees are firs: 0 none, 100 all."


def test_an_argument_without_a_tip_line_has_an_empty_tip():
    args = xs.parse_args(SOURCE.replace("// rok_tip Size", "// size"))
    assert args[0].tip == "" and args[1].tip


def test_the_label_of_a_row_is_where_its_tip_opens():
    specs = [sp.Spec("a0", "Size", 6, 60, 1, 0, tip="A tree's height in cells.")]
    lay = sp.panel_layout(1.0, 1400, 900, specs)
    c = lay.control("a0")
    lx, ly, lw, lh = c.label_rect
    assert sp.hit(lay, lx + 2, ly + lh / 2) == ("LABEL", "a0")
    tx, _ty, tw, th = c.track
    assert sp.hit(lay, tx + tw / 2, ly + lh / 2) == ("TRACK", "a0")
    assert c.tip == "A tree's height in cells."


def test_the_tip_box_stands_left_of_the_panel_inside_the_region():
    specs = [sp.Spec("a0", "Size", 6, 60, 1, 0, tip="x")]
    lay = sp.panel_layout(1.0, 1400, 900, specs)
    c = lay.control("a0")
    x, y, w, h = sp.tip_rect(lay, c, 6, 1400)
    assert x + w <= lay.rect[0] and x >= 0 and y >= 0
    assert abs((y + h) - (c.row[1] + c.row[3])) < 1e-6  # its top at the row's top
    tall = sp.tip_rect(lay, c, 200, 1400)
    assert tall[1] >= 0  # pushed inside the region
