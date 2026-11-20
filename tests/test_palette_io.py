"""Palette import and export, and the PNG codec under it."""

import struct
import zlib

import numpy as np
import pytest

from rok.core import palette_io, png
from rok.core.palette import default_palette
from rok.core.palette_io import (
    PaletteError,
    colours_to_palette,
    palette_bytes,
    parse_gpl,
    parse_hex,
    parse_jasc,
    parse_pdn_txt,
    read_palette,
    read_palette_bytes,
    write_gpl,
    write_hex,
    write_jasc,
    write_palette,
    write_pdn_txt,
)

from helpers import FIXTURES

PALETTES = FIXTURES / "phase3" / "palettes"
TEXT_SUFFIXES = (".gpl", ".txt", ".pal", ".hex")
ALL_SUFFIXES = (".png",) + TEXT_SUFFIXES

# Rok's own 16 sample colours of the fixtures, slots 1..16.
SAMPLE = np.array(
    [
        (12, 14, 20, 255), (38, 52, 92, 255), (70, 110, 170, 255), (120, 190, 230, 255),
        (40, 200, 150, 255), (30, 110, 70, 255), (150, 200, 60, 255), (240, 220, 110, 255),
        (235, 140, 50, 255), (200, 70, 40, 255), (150, 40, 90, 255), (230, 120, 170, 255),
        (110, 80, 60, 255), (120, 120, 130, 255), (190, 190, 200, 255), (250, 248, 240, 255),
    ],
    dtype=np.uint8,
)


# --- PNG building blocks (a reference encoder for the tests) --------------------


def _chunk(kind: bytes, content: bytes) -> bytes:
    crc = zlib.crc32(kind + content) & 0xFFFFFFFF
    return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", crc)


def _filter_row(kind: int, line: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    """Apply PNG filter `kind` to one raw row. Pure Python, the slow reference."""
    n = len(line)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        x = int(line[i])
        a = int(line[i - bpp]) if i >= bpp else 0
        b = int(prev[i])
        c = int(prev[i - bpp]) if i >= bpp else 0
        if kind == 0:
            pred = 0
        elif kind == 1:
            pred = a
        elif kind == 2:
            pred = b
        elif kind == 3:
            pred = (a + b) // 2
        else:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            if pa <= pb and pa <= pc:
                pred = a
            elif pb <= pc:
                pred = b
            else:
                pred = c
        out[i] = (x - pred) & 0xFF
    return out


def _build_png(
    samples: np.ndarray,
    colour_type: int,
    depth: int = 8,
    filters=None,
    plte: bytes | None = None,
    trns: bytes | None = None,
    interlace: int = 0,
    idat_split: int = 1,
) -> bytes:
    """PNG bytes from (h, w, channels) samples at the given depth."""
    height, width, channels = samples.shape
    if depth == 16:
        raw = np.asarray(samples, dtype=">u2").reshape(height, -1).view(np.uint8)
        raw = raw.reshape(height, width * channels * 2)
    else:
        raw = np.asarray(samples, dtype=np.uint8).reshape(height, width * channels)
    bpp = channels * (2 if depth == 16 else 1)
    if filters is None:
        filters = [0] * height
    body = bytearray()
    prev = np.zeros(raw.shape[1], dtype=np.uint8)
    for y in range(height):
        body.append(filters[y])
        body += _filter_row(filters[y], raw[y], prev, bpp).tobytes()
        prev = raw[y]
    ihdr = struct.pack(">IIBBBBB", width, height, depth, colour_type, 0, 0, interlace)
    out = png.SIGNATURE + _chunk(b"IHDR", ihdr)
    if plte is not None:
        out += _chunk(b"PLTE", plte)
    if trns is not None:
        out += _chunk(b"tRNS", trns)
    packed = zlib.compress(bytes(body))
    step = max(1, len(packed) // idat_split)
    for start in range(0, len(packed), step):
        out += _chunk(b"IDAT", packed[start : start + step])
    return out + _chunk(b"IEND", b"")


def _rgba_from(samples: np.ndarray, colour_type: int) -> np.ndarray:
    """Expected RGBA for 8-bit samples of the given colour type (no tRNS)."""
    h, w = samples.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    if colour_type == 0:
        out[:, :, :3] = samples
        out[:, :, 3] = 255
    elif colour_type == 2:
        out[:, :, :3] = samples
        out[:, :, 3] = 255
    elif colour_type == 4:
        out[:, :, :3] = samples[:, :, :1]
        out[:, :, 3] = samples[:, :, 1]
    else:
        out[:] = samples
    return out


# --- png.decode ---------------------------------------------------------------


def test_decode_one_row_per_filter_type():
    rng = np.random.default_rng(1)
    samples = rng.integers(0, 256, size=(5, 7, 4), dtype=np.uint8)
    data = _build_png(samples, 6, filters=[0, 1, 2, 3, 4])
    h, w, rgba = png.decode(data)
    assert (h, w) == (5, 7)
    np.testing.assert_array_equal(rgba, samples)


def test_decode_every_filter_on_every_row():
    # Each filter after a row of every other filter, so `prev` is always real data.
    rng = np.random.default_rng(2)
    samples = rng.integers(0, 256, size=(25, 9, 3), dtype=np.uint8)
    filters = [(i * 7 + i // 5) % 5 for i in range(25)]
    assert set(filters) == {0, 1, 2, 3, 4}
    h, w, rgba = png.decode(_build_png(samples, 2, filters=filters))
    np.testing.assert_array_equal(rgba, _rgba_from(samples, 2))


@pytest.mark.parametrize("colour_type,channels", [(0, 1), (2, 3), (4, 2), (6, 4)])
def test_decode_colour_types_8_bit(colour_type, channels):
    rng = np.random.default_rng(colour_type)
    samples = rng.integers(0, 256, size=(3, 4, channels), dtype=np.uint8)
    h, w, rgba = png.decode(_build_png(samples, colour_type, filters=[4, 3, 1]))
    assert rgba.shape == (3, 4, 4)
    assert rgba.dtype == np.uint8
    np.testing.assert_array_equal(rgba, _rgba_from(samples, colour_type))


@pytest.mark.parametrize("colour_type,channels", [(0, 1), (2, 3), (4, 2), (6, 4)])
def test_decode_16_bit_keeps_the_high_byte(colour_type, channels):
    rng = np.random.default_rng(10 + colour_type)
    samples = rng.integers(0, 65536, size=(4, 5, channels), dtype=np.uint16)
    data = _build_png(samples, colour_type, depth=16, filters=[1, 4, 3, 2])
    h, w, rgba = png.decode(data)
    high = (samples >> 8).astype(np.uint8)
    np.testing.assert_array_equal(rgba, _rgba_from(high, colour_type))


def test_decode_indexed_with_plte_and_trns():
    plte = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30])
    trns = bytes([255, 128])  # entries 2 and 3 keep alpha 255
    index = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)[:, :, None]
    h, w, rgba = png.decode(_build_png(index, 3, plte=plte, trns=trns, filters=[1, 4]))
    expected = np.array(
        [
            [(255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 255), (10, 20, 30, 255)],
            [(10, 20, 30, 255), (0, 0, 255, 255), (0, 255, 0, 128), (255, 0, 0, 255)],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(rgba, expected)


def test_decode_indexed_without_plte_raises():
    index = np.zeros((1, 2, 1), dtype=np.uint8)
    with pytest.raises(png.PngError):
        png.decode(_build_png(index, 3))


def test_decode_idat_split_over_chunks():
    rng = np.random.default_rng(3)
    samples = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
    data = _build_png(samples, 6, filters=[4] * 8, idat_split=5)
    assert data.count(b"IDAT") >= 3
    _, _, rgba = png.decode(data)
    np.testing.assert_array_equal(rgba, samples)


def test_decode_rgb_colour_key_trns():
    samples = np.array([[(1, 2, 3), (4, 5, 6)]], dtype=np.uint8)
    trns = struct.pack(">HHH", 4, 5, 6)
    _, _, rgba = png.decode(_build_png(samples, 2, trns=trns))
    assert rgba[0, 0].tolist() == [1, 2, 3, 255]
    assert rgba[0, 1].tolist() == [4, 5, 6, 0]


def test_decode_interlaced_raises():
    samples = np.zeros((2, 2, 4), dtype=np.uint8)
    data = _build_png(samples, 6, interlace=1)
    with pytest.raises(png.PngError, match="interlaced PNG"):
        png.decode(data)
    with pytest.raises(PaletteError, match="interlaced PNG"):
        read_palette_bytes(data, ".png")


def test_decode_rejects_junk_and_huge_images():
    with pytest.raises(png.PngError):
        png.decode(b"not a png at all")
    with pytest.raises(PaletteError):
        read_palette_bytes(b"not a png at all", ".png")
    # 257 x 256 is one pixel over the limit; the IDAT is never looked at.
    ihdr = struct.pack(">IIBBBBB", 257, 256, 8, 6, 0, 0, 0)
    data = png.SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"") + _chunk(b"IEND", b"")
    with pytest.raises(png.PngError):
        png.decode(data)


# --- png.encode ---------------------------------------------------------------


def test_encode_then_decode_is_identity_on_random_rgba():
    rng = np.random.default_rng(4)
    rgba = rng.integers(0, 256, size=(13, 29, 4), dtype=np.uint8)
    data = png.encode(rgba)
    assert data.startswith(png.SIGNATURE)
    assert data.count(b"IDAT") == 1
    assert data.endswith(_chunk(b"IEND", b""))
    h, w, back = png.decode(data)
    assert (h, w) == (13, 29)
    np.testing.assert_array_equal(back, rgba)


def test_encode_header_is_8_bit_rgba_no_interlace():
    data = png.encode(np.zeros((2, 3, 4), dtype=np.uint8))
    assert data[12:16] == b"IHDR"
    width, height, depth, colour_type, comp, filt, interlace = struct.unpack_from(
        ">IIBBBBB", data, 16
    )
    assert (width, height, depth, colour_type, comp, filt, interlace) == (3, 2, 8, 6, 0, 0, 0)
    # The chunk CRC is zlib.crc32 over type and content.
    assert struct.unpack_from(">I", data, 29)[0] == zlib.crc32(data[12:29]) & 0xFFFFFFFF


def test_encode_rejects_wrong_shape():
    with pytest.raises(ValueError):
        png.encode(np.zeros((2, 3, 3), dtype=np.uint8))


# --- the fixtures ---------------------------------------------------------------


@pytest.mark.parametrize("suffix", TEXT_SUFFIXES)
def test_sample_fixture_parses_to_slots_1_to_16(suffix):
    palette = read_palette(PALETTES / f"sample{suffix}")
    assert palette.shape == (256, 4)
    assert palette.dtype == np.uint8
    assert palette[0].tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(palette[1:17], SAMPLE)
    np.testing.assert_array_equal(palette[17:], default_palette()[17:])


def test_all_four_text_fixtures_are_identical():
    palettes = [read_palette(PALETTES / f"sample{s}") for s in TEXT_SUFFIXES]
    for other in palettes[1:]:
        np.testing.assert_array_equal(other, palettes[0])


def test_vox_default_png_decodes_to_the_default_palette():
    data = (PALETTES / "vox_default.png").read_bytes()
    h, w, rgba = png.decode(data)
    assert (h, w) == (1, 256)
    assert rgba[0, 255].tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(read_palette(PALETTES / "vox_default.png"), default_palette())


def test_vox_fixture_palette():
    palette = read_palette(FIXTURES / "robot.vox")
    assert palette.shape == (256, 4)
    assert palette[0].tolist() == [0, 0, 0, 0]
    assert not np.array_equal(palette, default_palette())
    with pytest.raises(PaletteError):
        read_palette_bytes(b"VOX \x96\x00\x00\x00MAIN" + b"\0" * 8, ".vox")


# --- the PNG mapping ------------------------------------------------------------


def test_png_pixel_0_is_slot_1_and_the_256th_pixel_is_dropped():
    rng = np.random.default_rng(5)
    row = rng.integers(0, 256, size=(1, 256, 4), dtype=np.uint8)
    palette = read_palette_bytes(png.encode(row), ".png")
    np.testing.assert_array_equal(palette[1:256], row[0, :255])
    assert palette[0].tolist() == [0, 0, 0, 0]


def test_png_row_major_top_row_first_and_alpha_kept():
    grid = np.zeros((2, 3, 4), dtype=np.uint8)
    for k in range(6):
        grid[k // 3, k % 3] = (k + 1, 0, 0, 100 + k)
    palette = read_palette_bytes(png.encode(grid), ".png")
    assert palette[1:7, 0].tolist() == [1, 2, 3, 4, 5, 6]
    assert palette[1:7, 3].tolist() == [100, 101, 102, 103, 104, 105]
    np.testing.assert_array_equal(palette[7:], default_palette()[7:])


def test_export_png_is_256_x_1_with_empty_last_pixel():
    palette = default_palette()
    palette[1] = (1, 2, 3, 4)
    palette[255] = (9, 8, 7, 6)
    data = palette_bytes(palette, ".png")
    h, w, rgba = png.decode(data)
    assert (h, w) == (1, 256)
    assert rgba[0, 0].tolist() == [1, 2, 3, 4]
    assert rgba[0, 254].tolist() == [9, 8, 7, 6]
    assert rgba[0, 255].tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(rgba[0, :255], palette[1:])


# --- parsers ----------------------------------------------------------------------


def test_parse_gpl_skips_headers_comments_and_names():
    text = "GIMP Palette\nName: X\nColumns: 4\n#\n# comment\n\n 1 2 3 one\n255 255 255\n"
    assert parse_gpl(text) == [(1, 2, 3, 255), (255, 255, 255, 255)]
    with pytest.raises(PaletteError):
        parse_gpl("GIMP Palette\n1 2\n")


def test_parse_pdn_txt_alpha_and_six_digits():
    text = "; comment\n\n80FF0000\n00ff00\n#0000ff\n"
    assert parse_pdn_txt(text) == [(255, 0, 0, 128), (0, 255, 0, 255), (0, 0, 255, 255)]
    with pytest.raises(PaletteError):
        parse_pdn_txt("zz0000\n")


def test_parse_jasc_count_and_riff():
    text = "JASC-PAL\r\n0100\r\n2\r\n1 2 3\r\n4 5 6\r\n7 8 9\r\n"
    assert parse_jasc(text) == [(1, 2, 3, 255), (4, 5, 6, 255)]
    with pytest.raises(PaletteError, match="not a JASC-PAL file"):
        parse_jasc("RIFF\x00\x00\x00\x00PAL data")
    with pytest.raises(PaletteError, match="not a JASC-PAL file"):
        read_palette_bytes(b"RIFF\x10\x00\x00\x00PAL data", ".pal")
    with pytest.raises(PaletteError):
        parse_jasc("hello\n0100\n1\n1 2 3\n")


def test_parse_hex_optional_hash():
    assert parse_hex("#ff0000\n00FF00\n\n0000ff\n") == [
        (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)
    ]
    with pytest.raises(PaletteError):
        parse_hex("ff00\n")


def test_colours_to_palette_caps_at_255_and_keeps_base():
    colours = [(i, i, i, 255) for i in range(300)]
    palette = colours_to_palette(colours)
    assert palette[255].tolist() == [254, 254, 254, 255]
    assert palette[0].tolist() == [0, 0, 0, 0]
    base = np.full((256, 4), 7, dtype=np.uint8)
    palette = colours_to_palette([(1, 2, 3, 4)], base=base)
    assert palette[1].tolist() == [1, 2, 3, 4]
    assert palette[2].tolist() == [7, 7, 7, 7]
    assert palette[0].tolist() == [0, 0, 0, 0]
    assert base[0].tolist() == [7, 7, 7, 7]  # base is not changed
    np.testing.assert_array_equal(colours_to_palette([]), default_palette())


def test_read_palette_bytes_unknown_suffix_and_case():
    with pytest.raises(PaletteError):
        read_palette_bytes(b"", ".bmp")
    upper = read_palette_bytes((PALETTES / "sample.hex").read_bytes(), ".HEX")
    np.testing.assert_array_equal(upper[1:17], SAMPLE)


# --- writers ------------------------------------------------------------------------


def _random_palette(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(256, 4), dtype=np.uint8)
    palette[0] = (0, 0, 0, 0)
    return palette


@pytest.mark.parametrize("suffix", ALL_SUFFIXES)
def test_write_then_read_every_format(suffix, tmp_path):
    palette = _random_palette(6)
    path = tmp_path / f"out{suffix}"
    write_palette(path, palette)
    back = read_palette(path)
    if suffix in (".png", ".txt"):
        np.testing.assert_array_equal(back, palette)
    else:
        np.testing.assert_array_equal(back[:, :3], palette[:, :3])
        assert np.all(back[1:, 3] == 255)
    assert back[0].tolist() == [0, 0, 0, 0]


def test_write_gpl_columns_16_and_255_entries():
    text = write_gpl(default_palette())
    lines = text.splitlines()
    assert lines[0] == "GIMP Palette"
    assert "Columns: 16" in lines[:4]
    assert len(parse_gpl(text)) == 255


def test_write_pdn_txt_is_aarrggbb():
    palette = default_palette()
    palette[1] = (0x11, 0x22, 0x33, 0x44)
    lines = [line for line in write_pdn_txt(palette).splitlines() if not line.startswith(";")]
    assert lines[0] == "44112233"
    assert len(lines) == 255


def test_write_jasc_255_entries():
    text = write_jasc(default_palette())
    lines = text.splitlines()
    assert lines[:3] == ["JASC-PAL", "0100", "255"]
    assert len(lines) == 258
    assert len(parse_jasc(text)) == 255


def test_write_hex_255_lines():
    lines = write_hex(default_palette()).splitlines()
    assert len(lines) == 255
    assert lines[0] == "ffffff"


def test_palette_bytes_rejects_bad_shape_and_suffix():
    with pytest.raises(PaletteError):
        palette_bytes(np.zeros((10, 4), dtype=np.uint8), ".png")
    with pytest.raises(PaletteError):
        palette_bytes(default_palette(), ".vox")
    assert palette_io.SUFFIXES == (".vox", ".png", ".gpl", ".txt", ".pal", ".hex", ".json")


def test_the_native_decoder_gives_the_python_decoders_pixels():
    from pathlib import Path

    pytest.importorskip("rok_native")
    from rok.core import png as png_mod

    files = sorted((Path(__file__).parent / "fixtures").rglob("*.png"))[:30]
    files.append(Path(__file__).parents[1] / "assets" / "icons" / "welcome" / "logo.png")
    saved = list(png_mod._NATIVE)
    try:
        for f in files:
            data = f.read_bytes()
            png_mod._NATIVE[:] = []
            native = png_mod.decode(data, max_pixels=1 << 24)
            png_mod._NATIVE[:] = [None]
            python = png_mod.decode(data, max_pixels=1 << 24)
            assert native[:2] == python[:2] and np.array_equal(native[2], python[2]), f
    finally:
        png_mod._NATIVE[:] = saved
