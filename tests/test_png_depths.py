"""PNG bit depths 1, 2 and 4 decode the same with the native and the Python decoder."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from rok.core import png


def _chunk(kind: bytes, content: bytes) -> bytes:
    crc = zlib.crc32(kind + content) & 0xFFFFFFFF
    return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", crc)


def _packed_png(samples: np.ndarray, depth: int, colour_type: int, plte=None, trns=None, filters=None) -> bytes:
    """A PNG of (h, w) samples packed `depth` bits each, first sample in the high bits."""
    height, width = samples.shape
    bits = ((samples[:, :, None] >> np.arange(depth - 1, -1, -1)) & 1).astype(np.uint8).reshape(height, -1)
    rows = np.packbits(bits, axis=1)
    filters = filters or [0] * height
    body = bytearray()
    prev = np.zeros(rows.shape[1], dtype=np.uint8)
    for y in range(height):
        body.append(filters[y])
        line = rows[y] if filters[y] == 0 else ((rows[y].astype(np.int16) - prev) & 0xFF).astype(np.uint8)
        body += line.tobytes()
        prev = rows[y]
    ihdr = struct.pack(">IIBBBBB", width, height, depth, colour_type, 0, 0, 0)
    out = png.SIGNATURE + _chunk(b"IHDR", ihdr)
    if plte is not None:
        out += _chunk(b"PLTE", plte)
    if trns is not None:
        out += _chunk(b"tRNS", trns)
    return out + _chunk(b"IDAT", zlib.compress(bytes(body))) + _chunk(b"IEND", b"")


@pytest.fixture(params=["native", "python"])
def decoder(request, monkeypatch):
    if request.param == "native":
        pytest.importorskip("rok_native")
        monkeypatch.setattr(png, "_NATIVE", [])
    else:
        monkeypatch.setattr(png, "_NATIVE", [None])
    return request.param


def test_four_bit_indexed_pixels_take_their_palette_colours(decoder):
    samples = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.uint8)
    plte = bytes([10, 0, 0, 0, 20, 0, 0, 0, 30, 40, 40, 40, 50, 50, 50])
    h, w, rgba = png.decode(_packed_png(samples, 4, 3, plte=plte))
    assert (h, w) == (2, 5)
    expected = np.frombuffer(plte, dtype=np.uint8).reshape(5, 3)[samples]
    assert np.array_equal(rgba[:, :, :3], expected)
    assert np.all(rgba[:, :, 3] == 255)


def test_one_bit_grey_with_a_padded_row_and_a_transparent_key(decoder):
    samples = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1]], dtype=np.uint8)  # nine pixels: two bytes a row
    h, w, rgba = png.decode(_packed_png(samples, 1, 0, trns=struct.pack(">H", 1)))
    assert (h, w) == (1, 9)
    assert np.array_equal(rgba[0, :, 0], samples[0] * 255)
    assert np.array_equal(rgba[0, :, 3], np.where(samples[0] == 1, 0, 255))


def test_two_bit_grey_scales_to_eight_bits_through_an_up_filter(decoder):
    samples = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
    h, w, rgba = png.decode(_packed_png(samples, 2, 0, filters=[0, 2]))
    assert np.array_equal(rgba[:, :, 0], samples * 85)
    assert np.array_equal(rgba[:, :, 1], samples * 85)


def test_a_depth_the_colour_type_does_not_allow_is_a_png_error(decoder):
    samples = np.zeros((1, 4), dtype=np.uint16)
    raw = samples.astype(">u2").tobytes()
    body = b"\0" + raw
    ihdr = struct.pack(">IIBBBBB", 4, 1, 16, 3, 0, 0, 0)
    data = png.SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"PLTE", bytes(3)) + _chunk(b"IDAT", zlib.compress(body))
    with pytest.raises(png.PngError):
        png.decode(data + _chunk(b"IEND", b""))
