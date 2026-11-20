"""Round-trip .vox files: read, write, read again."""


import numpy as np
import pytest

from rok.core.palette import default_palette
from rok.core.vox_read import VoxModel, read_vox, loads
from rok.core.vox_write import dumps, write_vox

from helpers import FIXTURES, FIXTURE_META, vox_file, vox_size, vox_xyzi


@pytest.mark.parametrize("stem", FIXTURE_META)
def test_fixture_roundtrip(stem, tmp_path):
    original = read_vox(FIXTURES / f"{stem}.vox")
    out = tmp_path / f"{stem}.vox"
    write_vox(out, original)
    again = read_vox(out)
    assert again.size == original.size
    assert np.array_equal(again.voxels, original.voxels)
    assert np.array_equal(again.palette, original.palette)


def test_roundtrip_empty_grid(tmp_path):
    model = VoxModel(
        size=(32, 32, 32),
        voxels=np.zeros((32, 32, 32), dtype=np.uint8),
        palette=default_palette(),
    )
    path = tmp_path / "empty.vox"
    write_vox(path, model)
    again = read_vox(path)
    assert again.size == (32, 32, 32)
    assert int(again.voxels.sum()) == 0
    assert np.array_equal(again.palette, model.palette)


def test_roundtrip_no_rgba_source():
    """A file written without RGBA still round-trips after we add the default palette."""
    data = vox_file(
        [
            vox_size(2, 1, 1),
            vox_xyzi([(0, 0, 0, 4), (1, 0, 0, 8)]),
        ]
    )
    original = loads(data)
    again = loads(dumps(original))
    assert again.size == original.size
    assert np.array_equal(again.voxels, original.voxels)
    assert np.array_equal(again.palette, original.palette)


def test_written_file_is_one_model_vox(tmp_path):
    original = read_vox(FIXTURES / "robot.vox")
    path = tmp_path / "out.vox"
    write_vox(path, original)
    data = path.read_bytes()
    assert data[:4] == b"VOX "
    assert data[8:12] == b"MAIN"
    assert b"SIZE" in data
    assert b"XYZI" in data
    assert b"RGBA" in data
    assert b"nTRN" in data
    assert b"nGRP" in data
    assert b"nSHP" in data
    assert b"LAYR" in data
    assert b"MATL" not in data
