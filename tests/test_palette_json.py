"""Rok's JSON palette file (core/palette_json.py) and its place in palette_io."""

import json

import numpy as np
import pytest

from rok.core import palette_io
from rok.core.material import DEFAULT, Material
from rok.core.palette_json import Record, dumps, loads

EMIT = Material(((b"_type", b"_emit"), (b"_emit", b"1"), (b"_flux", b"2")))


def test_round_trip_keeps_every_field():
    records = [
        Record(rgba=(255, 136, 0, 255), slot=12, name="ember", group="warm", used=40, favourite=True, material=EMIT),
        Record(rgba=(10, 20, 30, 128), slot=13),
        Record(rgba=(1, 2, 3, 255), slot=None, name="sky", favourite=True),
    ]
    back = loads(dumps(records))
    assert back == records


def test_the_file_is_plain_and_leaves_plain_things_out():
    data = json.loads(dumps([Record(rgba=(255, 0, 0, 255), slot=1), Record(rgba=(0, 0, 255, 128), slot=2, material=EMIT)]))
    assert data["rok_palette"] == 1
    assert data["colours"][0] == {"hex": "#FF0000", "slot": 1}
    assert data["colours"][1]["hex"] == "#0000FF80"
    assert data["colours"][1]["material"] == {"_type": "_emit", "_emit": "1", "_flux": "2"}


@pytest.mark.parametrize("text", [
    "not json",
    "[]",
    '{"colours": []}',
    '{"rok_palette": 1, "colours": [{"name": "no hex"}]}',
    '{"rok_palette": 1, "colours": [{"hex": "#12"}]}',
    '{"rok_palette": 1, "colours": [{"hex": "#123456", "slot": 300}]}',
    '{"rok_palette": 1, "colours": [{"hex": "#123456", "material": "emit"}]}',
])
def test_a_file_that_is_not_a_palette_is_a_value_error(text):
    with pytest.raises(ValueError):
        loads(text)
    with pytest.raises(palette_io.PaletteError):
        palette_io.read_palette_bytes(text.encode(), ".json")


def test_optional_keys_default():
    (record,) = loads('{"rok_palette": 1, "colours": [{"hex": "ff8800"}]}')
    assert record == Record(rgba=(255, 136, 0, 255))
    assert record.material == DEFAULT


def test_palette_io_reads_and_writes_json(tmp_path):
    palette = palette_io.default_palette()
    palette[1] = (1, 2, 3, 200)
    path = tmp_path / "p.json"
    palette_io.write_palette(path, palette)
    assert np.array_equal(palette_io.read_palette(path), palette)
    short = palette_io.palette_bytes(palette, ".json", count=3)
    assert [c["slot"] for c in json.loads(short)["colours"]] == [1, 2, 3]


def test_text_writers_take_a_count():
    palette = palette_io.default_palette()
    assert palette_io.write_hex(palette, 2).count("\n") == 2
    assert palette_io.write_gpl(palette, 2).count("\n") == 6
    assert palette_io.write_jasc(palette, 2).splitlines()[2] == "2"
    assert palette_io.write_pdn_txt(palette, 2).count("\n") == 3
