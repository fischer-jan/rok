"""The palette a shader sees for one run, and the way back to your colours."""

from __future__ import annotations

import numpy as np
import pytest

from rok.core import shader_palette as shp
from rok.core.colour_store import oklab, rgba
from rok.core.palette import DEFAULT_PALETTE

BRICK = (176, 92, 64, 255)


def _palette(**slots):
    """A palette with only the given slots (name s<k>) set; the rest empty."""
    pal = np.zeros((256, 4), dtype=np.uint8)
    for name, colour in slots.items():
        pal[int(name[1:])] = colour
    return pal


def _model(*slots):
    v = np.zeros((4, 4, 4), dtype=np.uint8)
    for k, slot in enumerate(slots):
        v[k, 0, 0] = slot
    return v


def test_the_model_keeps_its_slots_and_every_other_slot_is_the_default_palette():
    pal = _palette(s40=BRICK, s41=(10, 20, 30, 255))
    sp = shp.build(pal, _model(40, 41), 40)
    assert rgba(sp.palette[40]) == BRICK and rgba(sp.palette[41]) == (10, 20, 30, 255)
    others = [k for k in range(1, 256) if k not in (40, 41, *sp.run)]
    assert np.array_equal(sp.palette[others], DEFAULT_PALETTE[others])
    assert sp.to_shader[40] == 40 or sp.to_shader[40] in sp.run


def test_a_colour_list_without_own_colours_is_a_ramp_of_the_current_colour_in_a_row():
    pal = _palette(s40=BRICK)
    sp = shp.build(pal, _model(40), 40, count=5)
    assert len(sp.run) == 5 and sp.run == list(range(sp.run[0], sp.run[0] + 5))
    shades = [rgba(sp.palette[k]) for k in sp.run]
    assert BRICK in shades
    lightness = [oklab(c)[0] for c in shades]
    assert lightness == sorted(lightness)
    assert sp.colour in sp.run and sp.to_shader[40] == sp.colour  # the brick voxels read as the current colour


def test_own_colours_follow_the_marker():
    pal = _palette(s7=(255, 0, 0, 255))
    flame = [(255, 214, 60), (255, 140, 30)]
    sp = shp.build(pal, _model(7), 7, own=flame, marker=True)
    assert [rgba(sp.palette[k]) for k in sp.run] == [(255, 0, 0, 255), *(rgba(c) for c in flame)]
    assert sp.colour == sp.run[0] and sp.to_shader[7] == sp.run[0]


def test_the_way_back_keeps_your_colours_and_asks_for_new_ones():
    pal = _palette(s40=BRICK)
    sp = shp.build(pal, _model(40), 40)
    result = np.array([sp.to_shader[40], 9, 9, 0], dtype=np.uint8)  # brick, a default colour, empty
    asked = []

    def free_slot(colour, hint):
        asked.append((colour, hint))
        return 200

    back, writes = sp.back(result, free_slot)
    assert back.tolist() == [40, 200, 200, 0]
    assert writes == {200: rgba(DEFAULT_PALETTE[9])}
    assert asked == [(rgba(DEFAULT_PALETTE[9]), 9)]  # asked once, with its own number as the hint


def test_a_colour_given_twice_comes_back_on_one_slot():
    pal = _palette(s40=BRICK)
    sp = shp.build(pal, _model(40), 40, own=[(9, 9, 9), (9, 9, 9)])
    given = {}

    def free_slot(colour, hint):
        return given.setdefault(colour, 100 + len(given))

    back, writes = sp.back(np.array(sp.run, dtype=np.uint8), free_slot)
    assert back[0] == back[1] and len(writes) == 1


def test_a_shader_that_searches_the_palette_finds_a_shade_in_a_bare_palette():
    rok_native = pytest.importorskip("rok_native")
    try:
        rok_native.ShaderRunner()
    except Exception as exc:  # no GPU here
        pytest.skip(f"no GPU: {exc}")
    from rok.app import xs_ops

    grime = xs_ops.shader_by_name("patterns/grime")
    pal = _palette(s40=BRICK)  # your palette holds only the wall colour
    wall = np.zeros((16, 16, 16), dtype=np.uint8)
    wall[2:14, 2:14, 0:10] = 40
    new, pal2 = xs_ops.run_coloured(grime, wall, colour=40, palette=pal, args=[a.value for a in grime.args])
    added = [int(k) for k in np.unique(new) if k not in (0, 40)]
    assert added, "grime found no shade"
    assert all(rgba(pal2[k]) in {rgba(c) for c in DEFAULT_PALETTE} for k in added)
    assert rgba(pal2[40]) == BRICK


def test_a_model_with_every_colour_leaves_no_room_for_a_colour_list():
    full = np.arange(256, dtype=np.uint8).reshape(4, 4, 16)
    with pytest.raises(shp.NoRoom):
        shp.build(DEFAULT_PALETTE, full, 5, count=4)


def test_a_full_palette_stops_new_colours_or_takes_the_nearest_ones():
    from rok.app import xs_ops
    from rok.app.object import document
    pytest.importorskip("rok_native")  # Session needs the native module
    from rok.headless import Session

    s = Session()
    try:
        s.run("rok.new", size=tuple([16, 16, 16]), discard=True)
        flat = document.grid.voxels.reshape(-1)
        flat[:255] = np.arange(1, 256, dtype=np.uint8)  # every slot holds voxels
        document.stroke_serial += 1
        with pytest.raises(xs_ops.PaletteFull):
            xs_ops.store_slots()((250, 1, 1, 255), 0)
        slot = xs_ops.store_slots(merge=True)((250, 1, 1, 255), 0)
        assert tuple(document.scene.palette[slot][:3]) == (255, 0, 0)  # the nearest colour in use
    finally:
        s.close()


def test_your_unused_colours_are_in_the_shader_palette_on_their_own_slots():
    pal = _palette(s40=BRICK)
    sp = shp.build(pal, _model(40), 40, unused=[((90, 40, 30, 255), 41), ((1, 2, 3, 255), None)])
    assert rgba(sp.palette[41]) == (90, 40, 30, 255)
    assert (1, 2, 3, 255) in {rgba(c) for c in sp.palette}
