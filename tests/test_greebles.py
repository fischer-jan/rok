"""The greebles shader (rok/shaders/hard_surface/greebles.txt) on the GPU: it
adds parts to flat hull, in its own colours, and never breaks the hull."""

from __future__ import annotations

import numpy as np
import pytest

from rok.app import xs_ops
from rok.core import xs
from rok.core.palette import default_palette

rok_native = pytest.importorskip("rok_native")

GREY = 250
SHADER = xs.load(xs_ops.bundled_folder() / "hard_surface/greebles.txt", "hard_surface/greebles")
DARK, LIGHT, HAZARD = (58, 63, 71), (127, 231, 255), (242, 194, 48)


@pytest.fixture(scope="module", autouse=True)
def gpu_ok():
    try:
        rok_native.ShaderRunner()
    except Exception as exc:  # no GPU here
        pytest.skip(f"no GPU: {exc}")


def hull() -> np.ndarray:
    v = np.zeros((72, 48, 30), np.uint8)
    v[4:68, 6:42, 4:18] = GREY
    v[20:52, 14:34, 18:22] = GREY
    return v


def run(v, **named):
    args = [float(named.get(a.name, a.value)) for a in SHADER.args]
    return xs_ops.run_coloured(SHADER, v, colour=40, palette=default_palette(), args=args)


def colours_of(new, pal) -> set:
    return {tuple(int(c) for c in pal[i][:3]) for i in np.unique(new[new > 0])}


def test_parts_grow_on_the_hull_in_the_shaders_colours():
    v = hull()
    new, pal = run(v, Size=12, Density=90, Lights=100)
    added = (new > 0) & (v == 0)
    assert added.sum() > 500  # pipes, vents, tanks, boxes, masts
    assert {DARK, LIGHT, HAZARD} <= colours_of(new, pal)


def test_the_inside_stays_and_nothing_floats():
    v = hull()
    new, _pal = run(v, Size=12, Density=90)
    inner = np.zeros_like(v, dtype=bool)
    inner[6:66, 8:40, 6:16] = True  # two cells in from every side of the main hull
    assert np.array_equal(new[inner], v[inner])
    # Every added cell touches the model: parts stand on the hull.
    filled = new > 0
    p = np.pad(filled, 1)
    touches = (p[2:, 1:-1, 1:-1] | p[:-2, 1:-1, 1:-1] | p[1:-1, 2:, 1:-1]
               | p[1:-1, :-2, 1:-1] | p[1:-1, 1:-1, 2:] | p[1:-1, 1:-1, :-2])
    added = filled & (v == 0)
    assert added.any() and touches[added].all()


def test_a_thin_wall_gets_no_holes():
    v = np.zeros((64, 64, 8), np.uint8)
    v[2:62, 2:62, 3] = GREY  # one cell thick
    new, _pal = run(v, Size=12, Density=100, Seams=100)
    assert (new[2:62, 2:62, 3] > 0).all()


def test_density_zero_and_seams_zero_leave_the_model_alone():
    v = hull()
    new, _pal = run(v, Density=0, Seams=0)
    assert np.array_equal(new, v)


def test_colour_zero_keeps_the_hull_colour_but_the_lights():
    new, pal = run(hull(), Size=12, Density=90, Lights=100, Colour=0)
    assert DARK not in colours_of(new, pal) and HAZARD not in colours_of(new, pal)
    assert LIGHT in colours_of(new, pal)


def test_the_seed_changes_the_parts():
    a, _ = run(hull(), Seed=1)
    b, _ = run(hull(), Seed=2)
    assert not np.array_equal(a, b)


def test_extras_add_a_dish_a_long_antenna_an_airlock_and_a_hangar_door():
    # The occasional satellite dish, very long antenna,
    # airlock, and hangar door where there is a lot of room.
    v = np.zeros((96, 64, 60), np.uint8)
    v[4:92, 6:58, 4:20] = GREY
    new, pal = run(v, Size=16, Density=100, Extras=100, Lights=100)
    added = (new > 0) & (v == 0)
    assert added[:, :, 32:].any()  # a long antenna: the old parts reach 11 cells
    assert added[:, :, 20:].sum() > 500
    assert HAZARD in colours_of(new, pal)  # a hangar door's frame (no hatches with Extras 100)
    filled = new > 0
    p = np.pad(filled, 1)
    touches = (p[2:, 1:-1, 1:-1] | p[:-2, 1:-1, 1:-1] | p[1:-1, 2:, 1:-1]
               | p[1:-1, :-2, 1:-1] | p[1:-1, 1:-1, 2:] | p[1:-1, 1:-1, :-2])
    assert touches[added].all()  # nothing floats: the dish's rings overlap
    inner = np.zeros_like(v, dtype=bool)
    inner[6:90, 8:56, 6:18] = True
    assert np.array_equal(new[inner], v[inner])
    plain, _ = run(v, Size=16, Density=100, Extras=0)
    assert not ((plain > 0) & (v == 0))[:, :, 32:].any()  # without Extras, no long antenna
