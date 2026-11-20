"""Every 2D drawing path follows gfx.shifted, gfx.clipped and gfx.faded:
a sliding, cut or fading panel draws as one piece."""

from types import SimpleNamespace

import numpy as np

from rok.app import gfx, paint
from rok.app.draw import DrawList
from rok.app.opkit import CONTEXT

REGION = SimpleNamespace(width=400, height=300)


def _rects(d: DrawList) -> np.ndarray:
    return np.array(d.parts["rects"]).reshape(-1, 10)


def _tris(d: DrawList) -> np.ndarray:
    return np.concatenate(d.parts["tris2"]).reshape(-1, 6) if d.parts["tris2"] else np.zeros((0, 6))


def test_a_round_rect_moves_is_cut_and_fades_like_the_rest(monkeypatch):
    monkeypatch.setattr(CONTEXT, "region", REGION, raising=False)
    d = DrawList()
    with gfx.pixels(d, REGION), gfx.clipped((0, 0, 100, 100)), gfx.shifted(10, -20), gfx.faded(0.5):
        paint.round_rect((20, 30, 40, 40), 5, (1.0, 0.0, 0.0, 1.0))  # moved to (30, 10): inside
        paint.round_rect((20, 90, 40, 40), 5, (1.0, 0.0, 0.0, 1.0))  # moved to y 70..110: cut at 100
        paint.round_outline((20, 90, 40, 40), 5, (1.0, 1.0, 1.0, 1.0), 1.0)  # a cut outline is left out
        paint.round_rect((300, 30, 40, 40), 5, (1.0, 0.0, 0.0, 1.0))  # outside the clip
    r = _rects(d)
    assert len(r) == 2
    x, y, w, h = r[0, :4]
    assert (x, REGION.height - y - h, w, h) == (30, 10, 40, 40)
    assert r[0, 7] == 0.5  # faded
    x, y, w, h = r[1, :4]
    assert (x, REGION.height - y - h, w, h) == (30, 70, 40, 30)


def test_flat_triangles_move_are_cut_with_their_colours_and_fade():
    d = DrawList()
    inside = [(10, 10), (50, 10), (10, 50)]
    outside = [(10, 10), (500, 10), (10, 50)]
    colours = [(1, 0, 0, 1)] * 3 + [(0, 1, 0, 1)] * 3
    with gfx.pixels(d, REGION), gfx.clipped((0, 0, 100, 100)), gfx.shifted(5, 0), gfx.faded(0.5):
        gfx.pen.flat(np.array(inside + outside, np.float32), colours)
    t = _tris(d)
    assert len(t) == 3  # the triangle that leaves the clip is gone, and its colours with it
    assert (t[:, 0] == np.array([15, 55, 15])).all()
    assert (t[:, 2:6] == (1, 0, 0, 0.5)).all()
