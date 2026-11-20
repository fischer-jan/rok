"""water/stream and water/sea: what they make, and that their water is glass."""

from __future__ import annotations

import numpy as np
import pytest

from rok.core import xs

rok_native = pytest.importorskip("rok_native")


@pytest.fixture(scope="module")
def session():
    try:
        rok_native.ShaderRunner()
    except Exception as exc:  # no GPU here
        pytest.skip(f"no GPU: {exc}")
    from rok.headless import Session

    s = Session()
    yield s
    s.close()


def _ok(session, tool: str, args: dict) -> str:
    reply, _png = session.tool(tool, args)
    assert not reply.error, reply.text
    return reply.text


def _slab(session, size, top: int) -> None:
    """Earth (colour 20) with grass (21) as its top layer, `top` its height."""
    session.run("rok.new", size=tuple(list(size)), discard=True)
    _ok(session, "rok_edit", {"do": "set_colour", "index": 20, "hex": "#7a5232"})
    _ok(session, "rok_edit", {"do": "set_colour", "index": 21, "hex": "#5fae3a"})
    _ok(session, "rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [size[0] - 1, size[1] - 1, top - 1],
                                "colour": 20})
    _ok(session, "rok_shape", {"kind": "box", "low": [0, 0, top], "high": [size[0] - 1, size[1] - 1, top],
                                "colour": 21})


def _glass_slots(session) -> set[int]:
    scene = session.document.scene
    return {k for k in range(1, 256) if scene.materials.get(k).type == "GLASS"}


def test_the_water_colours_of_both_are_glass():
    from rok.app.xs_ops import shader_by_name

    for name, water in (("water/stream", range(5)), ("water/sea", range(1, 8))):
        need = shader_by_name(name).colours
        assert all(need.material(k) is not None and need.material(k).type == "GLASS" for k in water), name
    assert shader_by_name("water/sea").colours.material(0) is None  # foam is plain


def test_a_stream_cuts_a_channel_of_clear_water_into_the_ground(session):
    _slab(session, (48, 48, 16), 7)
    before = session.document.grid.voxels.copy()
    _ok(session, "rok_edit", {"do": "shader", "shader": "water/stream", "scope": "model", "args": {"Bends": 0}})
    v = session.document.grid.voxels
    glass = _glass_slots(session)
    water = np.isin(v, list(glass))
    assert water.any()
    assert not water[:, :, 7:].any()  # the surface lies below the grass
    assert (before[:, :, 7] != 0).all() and (v[:, :, 7] == 0).any()  # the cut opens the top
    across = np.flatnonzero(water.any(axis=(1, 2)))  # Bends 0: straight along y, in the middle
    assert 18 <= across.min() and across.max() <= 29


def test_a_sea_fills_below_its_level_with_foam_at_the_shore(session):
    session.run("rok.new", size=tuple([40, 40, 20]), discard=True)
    _ok(session, "rok_edit", {"do": "set_colour", "index": 20, "hex": "#efe6cf"})
    _ok(session, "rok_shape", {"kind": "box", "low": [0, 0, 0], "high": [39, 39, 1], "colour": 20})
    _ok(session, "rok_shape", {"kind": "box", "low": [26, 0, 2], "high": [39, 39, 12], "colour": 20})  # the land
    _ok(session, "rok_edit", {"do": "shader", "shader": "water/sea", "scope": "model", "args": {"Level": 40}})
    v = session.document.grid.voxels
    level = int(20 * 40 / 100)
    assert not (v[:26, :, level:] != 0).any()  # never above the level
    assert (v[:26, :, 2:level] != 0).all()  # the sea fills every empty cell below it
    palette = session.document.scene.palette
    foam = {k for k in np.unique(v[:26, :, level - 1]) if tuple(palette[k][:3]) == (0xF7, 0xFC, 0xFF)}
    assert foam and (v[25, :, level - 1] == next(iter(foam))).mean() > 0.5  # along the shore


def test_glass_in_a_colours_line_is_read(tmp_path):
    line = "// rok_colours : { start = '0'  rgb = '112233 445566 778899'  glass = '0.8 0' }"
    need = xs.parse_colours(line)
    assert need.glass == (0.8, 0.0)
    assert need.material(0).type == "GLASS" and need.material(0).weight == pytest.approx(0.8)
    assert need.material(1) is None and need.material(2) is None
