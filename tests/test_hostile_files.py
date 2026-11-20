"""Small hostile files fail fast with a clear error instead of filling memory.

The bytes are built inline: each test is an attack shape, not a real asset.
"""

import base64
import io
import json
import struct
import tracemalloc
import zipfile
import zlib

import pytest

from rok.core import rokfile, png
from rok.core.vox_read import MAX_TOTAL_CELLS, VoxError, loads

from helpers import vox_file, vox_size


def _gray_png(width: int, height: int, idat: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png._chunk(b"IHDR", ihdr) + png._chunk(b"IDAT", idat) + png._chunk(b"IEND", b"")


def test_a_png_whose_pixels_inflate_far_past_its_size_reads_only_the_rows(monkeypatch):
    monkeypatch.setattr(png, "_NATIVE", [None])  # the Python decoder
    bomb = zlib.compress(b"\0" * (64 << 20), 9)  # 64 MB of zeros in about 64 KB
    data = _gray_png(1, 1, bomb)
    tracemalloc.start()
    try:
        h, w, rgba = png.decode(data)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (h, w) == (1, 1) and tuple(rgba[0, 0]) == (0, 0, 0, 255)
    assert peak < 8 << 20


def test_a_vox_whose_models_add_up_past_the_cap_is_refused():
    side = 256
    count = MAX_TOTAL_CELLS // side**3 + 1
    data = vox_file([vox_size(side, side, side) for _ in range(count)])
    assert len(data) < 4096
    with pytest.raises(VoxError, match="too large"):
        loads(data)


def test_a_rok_zip_that_claims_a_huge_scene_is_refused(monkeypatch):
    monkeypatch.setattr(rokfile, "MAX_SCENE_BYTES", 1 << 20)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(rokfile.SCENE_NAME, b"\0" * (2 << 20))
        z.writestr(rokfile.META_NAME, "{}")
    with pytest.raises(rokfile.RokFileError, match="too large"):
        rokfile.loads(out.getvalue())


def test_a_packed_rok_text_that_inflates_past_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(rokfile, "MAX_SCENE_BYTES", 1 << 20)
    scene = base64.b64encode(zlib.compress(b"\0" * (2 << 20))).decode("ascii")
    text = json.dumps({"format": "rok", "version": 1, "scene": scene})
    with pytest.raises(rokfile.RokFileError, match="too large"):
        rokfile.unpack(text)
