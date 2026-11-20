"""The hard surface shaders on the GPU (rok/shaders/hard_surface): city
tiles meet without a seam, and the hull shaders keep the hull's shape.
greebles has its own file (test_greebles.py)."""

from __future__ import annotations

import numpy as np
import pytest

from rok.app import xs_ops
from rok.core import xs
from rok.core.palette import default_palette

rok_native = pytest.importorskip("rok_native")

GREY = 250


@pytest.fixture(scope="module", autouse=True)
def gpu_ok():
    try:
        rok_native.ShaderRunner()
    except Exception as exc:  # no GPU here
        pytest.skip(f"no GPU: {exc}")


def shader(name: str) -> xs.Shader:
    return xs.load(xs_ops.bundled_folder() / f"hard_surface/{name}.txt", f"hard_surface/{name}")


def run(name: str, v: np.ndarray, **named):
    s = shader(name)
    args = [float(named.get(a.name, a.value)) for a in s.args]
    return xs_ops.run_coloured(s, v, colour=40, palette=default_palette(), args=args)


def colours_of(new, pal) -> set:
    return {tuple(int(c) for c in pal[i][:3]) for i in np.unique(new[new > 0])}


def rgb(new, pal) -> np.ndarray:
    """The colour of every cell, black where empty: runs that put the
    shader's colours in other slots compare by colour."""
    return pal[new][..., :3] * (new[..., None] > 0)


def hull() -> np.ndarray:
    v = np.zeros((60, 40, 26), np.uint8)
    v[2:58, 4:36, 2:14] = GREY
    v[14:46, 10:30, 14:19] = GREY
    return v


def station() -> np.ndarray:
    v = np.zeros((48, 36, 32), np.uint8)
    v[4:44, 4:32, 1:22] = GREY
    return v


def test_city_tiles_meet_without_a_seam():
    # Tile 0 of a model twice as wide is tiles 0 and 1 of the narrow one, side by side.
    wide, wpal = run("city_blocks", np.zeros((96, 48, 40), np.uint8))
    left, lpal = run("city_blocks", np.zeros((48, 48, 40), np.uint8))
    right, rpal = run("city_blocks", np.zeros((48, 48, 40), np.uint8), **{"Tile X": 1})
    assert np.array_equal(rgb(wide, wpal), np.concatenate([rgb(left, lpal), rgb(right, rpal)]))


def test_the_city_has_streets_buildings_and_lit_windows():
    new, pal = run("city_blocks", np.zeros((72, 72, 40), np.uint8), Lights=60)
    colours = colours_of(new, pal)
    asphalt, lit, roof = (0x34, 0x37, 0x3D), (0xFF, 0xD2, 0x7A), (0x63, 0x66, 0x6C)
    assert {asphalt, lit, roof} <= colours
    heights = [np.flatnonzero(col).max() for col in new.reshape(-1, 40) if col.any()]
    assert max(heights) > 20 and min(heights) == 0  # towers and flat streets


def touches_the_model(new) -> np.ndarray:
    """True where a cell has a filled neighbour (6 sides)."""
    p = np.pad(new > 0, 1)
    return (p[2:, 1:-1, 1:-1] | p[:-2, 1:-1, 1:-1] | p[1:-1, 2:, 1:-1]
            | p[1:-1, :-2, 1:-1] | p[1:-1, 1:-1, 2:] | p[1:-1, 1:-1, :-2])


def test_hull_plating_with_relief_0_only_paints():
    v = hull()
    new, pal = run("hull_plating", v, Tones=100, Seams=100, Details=100, Relief=0)
    assert np.array_equal(new > 0, v > 0)
    assert (new != v).sum() > 1000
    # Panel lines and tones are shades of the hull, never black (black
    # seams look like tape); no hazard by default.
    colours = colours_of(new, pal)
    assert len(colours) >= 3 and (0x2A, 0x2D, 0x33) not in colours and (0xF2, 0xC2, 0x30) not in colours
    assert all(abs(int(r) - int(b)) < 30 for r, g, b in colours)  # greys on a grey hull
    inner = np.zeros_like(v, dtype=bool)
    inner[3:57, 5:35, 3:13] = True  # one cell in from every side of the main hull
    assert np.array_equal(new[inner], v[inner])


def test_hull_plating_raises_plates_and_sinks_details():
    # As good as greebles, so the plates stand out for real.
    v = hull()
    new, pal = run("hull_plating", v, Details=100, Relief=2)
    added = (new > 0) & (v == 0)
    assert added.sum() > 1000  # plates, rivets, fins, bosses, rails
    assert ((new == 0) & (v > 0)).sum() > 20  # grilles, hatches and insets cut in
    assert touches_the_model(new)[added].all()  # nothing floats
    # Plates are greys on a grey hull, never the hazard colours.
    colours = colours_of(new, pal)
    assert (0x2A, 0x2D, 0x33) not in colours and (0xF2, 0xC2, 0x30) not in colours
    assert all(abs(int(r) - int(b)) < 30 for r, g, b in colours)
    inner = np.zeros_like(v, dtype=bool)
    inner[4:56, 6:34, 4:12] = True  # two cells in from every side of the main hull
    assert np.array_equal(new[inner], v[inner])


def test_hull_plating_keeps_a_thin_wall_whole():
    v = np.zeros((64, 64, 10), np.uint8)
    v[2:62, 2:62, 4] = GREY  # one cell thick
    new, _pal = run("hull_plating", v, Details=100, Relief=3)
    assert (new[2:62, 2:62, 4] > 0).all()
    assert ((new > 0) & (v == 0)).any()


def test_hull_plating_with_walls_0_leaves_the_sides_for_windows():
    # On one ship, plating covered the windows.
    v = station()
    plated, pal = run("hull_plating", v, Details=100, Relief=2, Walls=0)
    side = np.zeros_like(v, dtype=bool)
    side[4:44, 4:32, 3:20] = True
    side[6:42, 6:30, :] = False  # the side walls' cells, clear of the bottom's cuts and the roof edge
    assert np.array_equal(plated[side], v[side])
    assert not np.array_equal(plated, v)  # the roof is plated
    # Windows after the plating still find flat walls.
    s = shader("hull_windows")
    args = [float({"Lit": 100}.get(a.name, a.value)) for a in s.args]
    both, pal2 = xs_ops.run_coloured(s, plated, colour=40, palette=pal, args=args)
    assert {(0xFF, 0xD9, 0x8E), (0xBF, 0xE9, 0xFF)} <= colours_of(both, pal2)
    assert (rgb(both, pal2)[:, :, 21:] == rgb(plated, pal)[:, :, 21:]).all()  # the plated roof stays


def test_hull_windows_get_sills_ribs_hangars_and_airlocks():
    v = station()
    new, pal = run("hull_windows", v, Lit=100, Features=100, Gap=0)
    added = (new > 0) & (v == 0)
    assert added.sum() > 200  # sills, ribs and lights stand out of the walls
    assert ((new == 0) & (v > 0)).sum() > 100  # hangar bays, doors and glass bands sink in
    assert touches_the_model(new)[added].all()
    assert not (new[:, :, 21] != v[:, :, 21]).any()  # the roof stays plain
    assert not added[:, :, 21:].any()
    # All seven of the shader's own colours reach it: the hazard stripes and the
    # red lights are its sixth and seventh (the run once gave a shader four).
    colours = colours_of(new, pal)
    assert {(0xF2, 0xC2, 0x30), (0xFF, 0x4A, 0x3A), (0x1B, 0x21, 0x29)} <= colours
    assert {(0xFF, 0xD9, 0x8E), (0xBF, 0xE9, 0xFF)} <= colours  # lit windows, warm and cool


def test_hull_windows_light_picks_warm_or_cool():
    warm, wpal = run("hull_windows", station(), Lit=100, Light=0)
    cool, cpal = run("hull_windows", station(), Lit=100, Light=1)
    assert (0xFF, 0xD9, 0x8E) in colours_of(warm, wpal) and (0xBF, 0xE9, 0xFF) not in colours_of(warm, wpal)
    assert (0xBF, 0xE9, 0xFF) in colours_of(cool, cpal) and (0xFF, 0xD9, 0x8E) not in colours_of(cool, cpal)


def test_hull_windows_keep_a_thin_wall_whole():
    v = np.zeros((48, 12, 30), np.uint8)
    v[2:46, 5, 1:28] = GREY  # one cell thick
    new, _pal = run("hull_windows", v, Lit=100, Features=100, Gap=0)
    assert (new[2:46, 5, 1:28] > 0).all()
    assert (new != v).any()


def test_battle_damage_cuts_dents_but_no_holes_in_a_thin_wall():
    v = np.zeros((64, 64, 12), np.uint8)
    v[2:62, 2:62, 5] = GREY  # one cell thick
    new, _pal = run("battle_damage", v, Amount=100, Size=4, Holes=0)
    assert np.array_equal(new > 0, v > 0)
    assert (new != v).sum() > 100  # soot
    holed, _pal = run("battle_damage", v, Amount=100, Size=4, Holes=1)
    assert ((holed == 0) & (v > 0)).sum() > 10
    assert not ((holed > 0) & (v == 0)).any()


def test_battle_damage_dents_a_solid_hull():
    v = hull()
    new, pal = run("battle_damage", v, Amount=100, Glow=100)
    assert not ((new > 0) & (v == 0)).any()
    assert ((new == 0) & (v > 0)).sum() > 50
    assert (0xFF, 0x8A, 0x2A) in colours_of(new, pal)
