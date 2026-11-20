"""Read the golden .vox fixtures and a few constructed edge cases."""

import struct

import numpy as np
import pytest

from rok.core.palette import DEFAULT_PALETTE
from rok.core.vox_read import VoxError, read_vox, loads

from helpers import FIXTURES, FIXTURE_META, vox_chunk, vox_file, vox_size, vox_xyzi


@pytest.mark.parametrize("stem", FIXTURE_META)
def test_fixture_size_and_count(stem):
    size, count = FIXTURE_META[stem]
    model = read_vox(FIXTURES / f"{stem}.vox")
    assert model.size == size
    assert model.voxels.shape == size
    assert model.voxels.dtype == np.uint8
    assert int((model.voxels != 0).sum()) == count
    assert model.palette.shape == (256, 4)
    assert model.palette.dtype == np.uint8
    assert model.palette[0].tolist() == [0, 0, 0, 0]


def test_fixtures_with_scene_chunks_load():
    """MATL, nTRN, nGRP, nSHP and the rest must not crash the reader."""
    for path in sorted(FIXTURES.glob("*.vox")):
        model = read_vox(path)
        assert model.voxels.shape == model.size


def test_palette_index_mapping_robot():
    """Voxel colour index 185 uses RGBA entry 184 (spec off-by-one)."""
    data = (FIXTURES / "robot.vox").read_bytes()
    model = read_vox(FIXTURES / "robot.vox")
    assert model.voxels[10, 10, 16] == 185  # an eye
    assert model.palette[185].tolist() == [255, 60, 30, 255]
    rgba = data.index(b"RGBA") + 12
    assert data[rgba + 4 * 184 : rgba + 4 * 185] == bytes([255, 60, 30, 255])


def test_missing_rgba_uses_default_palette():
    data = vox_file(
        [
            vox_size(2, 2, 2),
            vox_xyzi([(0, 0, 0, 1), (1, 1, 1, 2)]),
        ]
    )
    model = loads(data)
    assert model.size == (2, 2, 2)
    assert model.voxels[0, 0, 0] == 1
    assert model.voxels[1, 1, 1] == 2
    assert np.array_equal(model.palette, DEFAULT_PALETTE)


def test_first_of_several_models():
    data = vox_file(
        [
            vox_size(1, 1, 1),
            vox_xyzi([(0, 0, 0, 7)]),
            vox_size(3, 3, 3),
            vox_xyzi([(1, 1, 1, 9)]),
        ]
    )
    model = loads(data)
    assert model.size == (1, 1, 1)
    assert model.voxels[0, 0, 0] == 7


def test_empty_xyzi():
    data = vox_file([vox_size(4, 4, 4), vox_xyzi([])])
    model = loads(data)
    assert model.size == (4, 4, 4)
    assert int(model.voxels.sum()) == 0


def test_rejects_bad_header():
    with pytest.raises(VoxError):
        loads(b"not a vox file at all!!!!")
    with pytest.raises(VoxError):
        loads(b"VOX " + struct.pack("<i", 150))
    with pytest.raises(VoxError):
        loads(b"VOX " + struct.pack("<i", 99) + vox_chunk(b"MAIN", b"", b""))
