"""The golden .vox fixtures must be valid .vox files."""
import struct
from pathlib import Path

import pytest

FIXTURES = sorted(Path(__file__).parent.joinpath("fixtures").glob("*.vox"))


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_fixture_has_vox_header(path):
    data = path.read_bytes()
    assert data[:4] == b"VOX "
    (version,) = struct.unpack("<i", data[4:8])
    assert version in (150, 200)
    assert data[8:12] == b"MAIN"


def test_three_fixtures_present():
    assert len(FIXTURES) == 3
