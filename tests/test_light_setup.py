"""A file's light (rok/core/light.py): presets as setups, the tracer's
numbers, changes, and the light kept in a .vox and a Rok file."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rok.core import light, rokfile
from rok.core import render_presets as rp
from rok.core.scene import VoxScene
from rok.core.undo import LightRecord, UndoStack
from rok.core.vox_read import loads_scene, read_scene
from rok.core.vox_write import dumps_scene

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "light" / "preset_trace_settings.json"


def _box(text: str):
    lo, hi = text.split(") (")
    return tuple(int(v) for v in lo.strip("()").split(", ")), tuple(int(v) for v in hi.strip("()").split(", "))


def test_every_preset_as_a_setup_gives_the_tracer_the_presets_numbers():
    """The golden file holds what each preset gave the tracer before presets
    turned into setups; Night's lantern and Sunset's sky included."""
    golden = json.loads(GOLDEN.read_text())
    assert {key.split(" ")[0] for key in golden} == {p.id for p in rp.PRESETS}
    for key, want in golden.items():
        pid, box = key.split(" ", 1)
        got = light.trace_settings(light.from_preset(pid), *_box(box))
        for name, value in want.items():
            if name == "max_samples":
                continue  # the view's samples are yours, not the light's
            if value is None:
                assert got[name] is None, (key, name)
            elif isinstance(value, list):
                assert np.allclose(np.array(got[name], float), np.array(value, float), rtol=1e-9), (key, name)
            else:
                assert got[name] == pytest.approx(value), (key, name)


def test_a_lamp_lights_the_scene_the_same_at_every_size_and_distance():
    for r_box in ((0, 0, 0), (7, 7, 7)), ((0, 0, 0), (199, 199, 99)):
        r = rp.scene_radius(*r_box)
        for distance in (1.5, 2.5, 6.0):
            lamp = light.Light(light.LAMP, 30, 40, (1.0, 0.5, 0.25), 1.0, 0.4, distance)
            kind, x, y, z, red, _g, _b, size = light.lamp(lamp, r)
            dist = math.sqrt(x * x + y * y + z * z)
            assert kind == rp.SPHERE and dist == pytest.approx(distance * r)
            assert red / dist**2 == pytest.approx(light.LAMP_W / math.pi)  # irradiance at the middle
            assert size == pytest.approx(0.4 * r)


def test_a_sky_brings_its_own_sun_low_and_warm():
    sunset = light.from_preset("SUNSET")
    sun = sunset.lights[0]
    assert sun.kind == light.SUN and sun.elevation < 15
    assert sun.colour[0] > sun.colour[2]
    assert sunset.sky_horizon[0] > sunset.sky_top[0]


def test_turn_all_turns_every_lamp_around_the_model():
    studio = light.from_preset("STUDIO")
    box = ((0, 0, 0), (31, 31, 15))
    before = light.trace_settings(studio, *box)["lamps"]
    after = light.trace_settings(replace(studio, turn=90.0), *box)["lamps"]
    for a, b in zip(before, after):
        az0, az1 = math.degrees(math.atan2(a[2], a[1])), math.degrees(math.atan2(b[2], b[1]))
        assert light.wrap(az1 - az0) == pytest.approx(90.0)
        assert b[3] == pytest.approx(a[3]) and b[4:] == a[4:]  # the same height, colour and size


def test_adding_and_removing_lights():
    one = light.from_preset("STANDARD")
    more = light.added(light.added(light.added(one)))
    assert len(more.lights) == light.MAX_LIGHTS
    assert light.added(more) is more  # the tracer takes four
    assert more.lights[1].azimuth != one.lights[0].azimuth and more.lights[1].power < one.lights[0].power
    assert light.removed(more, 0).lights == more.lights[1:]
    muted = light.with_light(more, 1, on=False)
    assert len(light.trace_settings(muted, (0, 0, 0), (9, 9, 9))["lamps"]) == 3


def test_a_setup_knows_whether_it_is_still_its_preset():
    night = light.from_preset("NIGHT")
    assert light.same_as_preset(night)
    assert not light.same_as_preset(replace(night, bloom=0.0))


def test_json_round_trip_keeps_unknown_fields_and_fixes_bad_values():
    night = light.from_preset("NIGHT")
    assert light.from_chunk(light.to_chunk(night)) == night
    data = light.to_json(night)
    data["fog"] = 0.5  # a field a newer Rok wrote
    data["tone"] = "SEPIA"
    data["lights"][0]["elevation"] = 400
    back = light.from_json(data)
    assert back.extra == {"fog": 0.5} and light.to_json(back)["fog"] == 0.5
    assert back.tone == rp.CLIP and back.lights[0].elevation == light.HIGHEST
    for bad in (b"not json", b"[1, 2]", b'{"lights": [{"kind": "LASER"}]}'):
        with pytest.raises(ValueError):
            light.from_chunk(bad)


def test_a_vox_keeps_its_light_and_a_file_without_one_has_none():
    scene = VoxScene.from_data(read_scene(FIXTURES / "car.vox"))
    assert scene.light is None
    scene.light = replace(light.from_preset("SUNSET"), turn=30.0)
    data = dumps_scene(scene.to_vox())
    back = VoxScene.from_data(loads_scene(data))
    assert back.light == scene.light
    assert all(cid != light.CHUNK for cid, _content in back.extras)  # the chunk is the light, not an extra
    # Written again, there is still one light chunk.
    again = loads_scene(dumps_scene(back.to_vox()))
    assert [cid for cid, _c in again.extras].count(light.CHUNK) == 1


def test_a_broken_light_chunk_opens_with_the_default_light():
    scene = VoxScene.from_data(read_scene(FIXTURES / "car.vox"))
    scene.extras.append((light.CHUNK, b"{broken"))
    back = VoxScene.from_data(loads_scene(dumps_scene(scene.to_vox())))
    assert back.light is None and back.models


def test_a_rok_file_keeps_its_light():
    scene = VoxScene.from_data(read_scene(FIXTURES / "car.vox"))
    scene.light = light.from_preset("DIORAMA_NIGHT")
    back, _doc = rokfile.loads(rokfile.dumps(scene, {}))
    assert back.light == scene.light


def test_a_light_change_undoes_and_redoes():
    scene = VoxScene.from_data(read_scene(FIXTURES / "car.vox"))
    stack = UndoStack()
    night = light.from_preset("NIGHT")
    scene.light = night
    stack.push(LightRecord(None, night))
    stack.undo(scene)
    assert scene.light is None
    stack.redo(scene)
    assert scene.light == night
