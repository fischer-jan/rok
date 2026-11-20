"""The palette card's star is filled amber while the current colour is a favourite."""

from types import SimpleNamespace

import numpy as np

from rok.app import gfx, overlay, palette_ui
from rok.app.draw import DrawList
from rok.app.opkit import CONTEXT

REGION = SimpleNamespace(width=400, height=300)
RECT = (100, 100, 28, 28)


def _star_lines(monkeypatch, on_colour) -> np.ndarray:
    monkeypatch.setattr(CONTEXT, "region", REGION, raising=False)
    d = DrawList()
    with gfx.pixels(d, REGION):
        overlay._draw_button("FAVOURITE", RECT, False, 1.0, on_colour=on_colour)
    assert not d.parts["tris2"]  # a layer draws triangles under its rounded rects: the card's plate would hide them
    return np.concatenate(d.parts["lines2"]).reshape(-1, 9)


def test_the_star_is_filled_amber_for_a_favourite(monkeypatch):
    plain = _star_lines(monkeypatch, None)
    filled = _star_lines(monkeypatch, palette_ui.STAR_ON)
    assert len(filled) > 5 * len(plain)  # a dense fan of lines fills the star
    assert np.allclose(filled[:, 4:8], palette_ui.STAR_ON)
