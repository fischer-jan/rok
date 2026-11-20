"""A small PNG decoder and encoder for palette images. numpy and zlib only.

decode handles every bit depth the format allows for colour types 0, 2, 3, 4
and 6 (16 keeps the high byte; 1, 2 and 4 are unpacked, and grey is scaled to
8 bits), and the five row filters. Adam7 interlace is refused; the images
here are palettes of at most 65536 pixels. encode writes 8-bit RGBA with filter
0 on every row.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PIXELS = 65536

# Samples per pixel for each colour type, and the bit depths it allows.
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}


class PngError(Exception):
    """A PNG that cannot be read."""


_NATIVE: list = []  # [module or None] once looked up


def _native():
    """rok_native's PNG decoder, or None (no native module, or ROK_PURE_PNG set)."""
    if not _NATIVE:
        import os

        mod = None
        if not os.environ.get("ROK_PURE_PNG"):
            try:
                import rok_native

                mod = rok_native if hasattr(rok_native, "decode_png") else None
            except ImportError:
                mod = None
        _NATIVE.append(mod)
    return _NATIVE[0]


def decode(data: bytes, max_pixels: int = MAX_PIXELS) -> tuple[int, int, np.ndarray]:
    """Decode PNG bytes to (height, width, rgba (h, w, 4) uint8). A heightmap
    passes a larger `max_pixels` than a palette needs."""
    if data[:8] != SIGNATURE:
        raise PngError("not a PNG file")
    header = None
    plte = None
    trns = None
    idat: list[bytes] = []
    offset = 8
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, offset)
        start = offset + 8
        end = start + length
        if end + 4 > len(data):
            raise PngError("truncated chunk")
        chunk = data[start:end]
        if kind == b"IHDR":
            header = _parse_ihdr(chunk)
        elif kind == b"PLTE":
            plte = np.frombuffer(chunk[: len(chunk) - len(chunk) % 3], dtype=np.uint8).reshape(-1, 3)
        elif kind == b"tRNS":
            trns = chunk
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
        offset = end + 4
    if header is None:
        raise PngError("missing IHDR")
    if not idat:
        raise PngError("missing IDAT")
    width, height, depth, colour_type, interlace = header
    if interlace != 0:
        raise PngError("interlaced PNG")
    if width * height > max_pixels:
        raise PngError("PNG too large")
    native = _native()
    if native is not None:
        # The pixels in Rust: a Paeth-filtered 1280 x 720 picture took 2.4 s in Python.
        try:
            return native.decode_png(bytes(data), max_pixels)
        except ValueError as exc:
            raise PngError(str(exc)) from exc

    channels = _CHANNELS[colour_type]
    sample_bytes = 2 if depth == 16 else 1
    bpp = channels * sample_bytes  # bytes per pixel for the filters; one for the packed depths
    row_bytes = (width * channels * depth + 7) // 8
    try:
        # Inflate no more than the rows need: a small file must not inflate to gigabytes.
        raw = zlib.decompressobj().decompress(b"".join(idat), height * (row_bytes + 1))
    except zlib.error as exc:
        raise PngError(f"bad IDAT: {exc}") from exc
    if len(raw) < height * (row_bytes + 1):
        raise PngError("IDAT too short")
    rows = np.frombuffer(raw[: height * (row_bytes + 1)], dtype=np.uint8)
    rows = rows.reshape(height, row_bytes + 1)
    pixels = _unfilter(rows, bpp)  # (h, row_bytes)
    if depth == 16:
        pixels = pixels.reshape(height, width, channels, 2)[:, :, :, 0]
    elif depth < 8:
        pixels = _unpack(pixels, width, depth)
    else:
        pixels = pixels.reshape(height, width, channels)
    return height, width, _to_rgba(pixels, colour_type, plte, trns, depth)


def _unpack(rows: np.ndarray, width: int, depth: int) -> np.ndarray:
    """(h, row_bytes) rows of 1, 2 or 4-bit samples to (h, w, 1), first sample
    in the high bits; the padding at a row's end is dropped."""
    height = rows.shape[0]
    bits = np.unpackbits(rows, axis=1)[:, : width * depth].reshape(height, width, depth)
    weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint8)
    return (bits * weights).sum(axis=2, dtype=np.uint8).reshape(height, width, 1)


def encode(rgba: np.ndarray) -> bytes:
    """Encode an (h, w, 4) uint8 array as an 8-bit RGBA PNG."""
    arr = np.ascontiguousarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("encode needs an (h, w, 4) array")
    height, width = arr.shape[:2]
    rows = np.zeros((height, width * 4 + 1), dtype=np.uint8)
    rows[:, 1:] = arr.reshape(height, width * 4)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(rows.tobytes(), 9)
    return SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _chunk(kind: bytes, content: bytes) -> bytes:
    crc = zlib.crc32(kind + content) & 0xFFFFFFFF
    return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", crc)


def _parse_ihdr(chunk: bytes) -> tuple[int, int, int, int, int]:
    if len(chunk) < 13:
        raise PngError("IHDR too short")
    width, height, depth, colour_type, compression, filt, interlace = struct.unpack_from(
        ">IIBBBBB", chunk
    )
    if colour_type not in _CHANNELS:
        raise PngError(f"unsupported colour type {colour_type}")
    if depth not in _DEPTHS[colour_type]:
        raise PngError(f"unsupported bit depth {depth} for colour type {colour_type}")
    if compression != 0 or filt != 0:
        raise PngError("unsupported compression or filter method")
    if width == 0 or height == 0:
        raise PngError("empty image")
    return width, height, depth, colour_type, interlace


def _unfilter(rows: np.ndarray, bpp: int) -> np.ndarray:
    """Undo the row filters. rows is (h, 1 + row_bytes); returns (h, row_bytes)."""
    height, stride = rows.shape
    out = np.zeros((height, stride - 1), dtype=np.uint8)
    prev = np.zeros(stride - 1, dtype=np.uint8)
    for y in range(height):
        kind = int(rows[y, 0])
        line = rows[y, 1:].copy()
        if kind == 0:
            pass
        elif kind == 1:
            _unfilter_sub(line, bpp)
        elif kind == 2:
            line = (line.astype(np.uint16) + prev).astype(np.uint8)
        elif kind == 3:
            _unfilter_average(line, prev, bpp)
        elif kind == 4:
            _unfilter_paeth(line, prev, bpp)
        else:
            raise PngError(f"bad filter type {kind}")
        out[y] = line
        prev = line
    return out


def _unfilter_sub(line: np.ndarray, bpp: int) -> None:
    # Each byte adds the byte one pixel to its left; a running sum per byte lane.
    for lane in range(bpp):
        lane_bytes = line[lane::bpp]
        line[lane::bpp] = np.cumsum(lane_bytes, dtype=np.uint64).astype(np.uint8)


def _unfilter_average(line: np.ndarray, prev: np.ndarray, bpp: int) -> None:
    # Average depends on the left pixel, so walk the row pixel by pixel; each
    # step handles one pixel's bytes at once.
    n = line.shape[0]
    work = line.astype(np.int32)
    up = prev.astype(np.int32)
    for i in range(0, n, bpp):
        left = work[i - bpp : i] if i >= bpp else np.zeros(bpp, dtype=np.int32)
        seg = slice(i, i + bpp)
        work[seg] = (work[seg] + (left + up[seg]) // 2) & 0xFF
    line[:] = work.astype(np.uint8)


def _unfilter_paeth(line: np.ndarray, prev: np.ndarray, bpp: int) -> None:
    n = line.shape[0]
    work = line.astype(np.int32)
    up = prev.astype(np.int32)
    zero = np.zeros(bpp, dtype=np.int32)
    for i in range(0, n, bpp):
        seg = slice(i, i + bpp)
        a = work[i - bpp : i] if i >= bpp else zero
        b = up[seg]
        c = up[i - bpp : i] if i >= bpp else zero
        p = a + b - c
        pa = np.abs(p - a)
        pb = np.abs(p - b)
        pc = np.abs(p - c)
        pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
        work[seg] = (work[seg] + pred) & 0xFF
    line[:] = work.astype(np.uint8)


def _to_rgba(pixels: np.ndarray, colour_type: int, plte, trns, depth: int) -> np.ndarray:
    """Expand (h, w, channels) samples (high byte only for 16 bit) to RGBA."""
    height, width = pixels.shape[:2]
    out = np.empty((height, width, 4), dtype=np.uint8)
    if colour_type == 0:
        grey = pixels[:, :, 0]
        # A 1, 2 or 4-bit grey spans 0..255 once scaled; the key compares unscaled.
        out[:, :, :3] = (grey * (255 // ((1 << depth) - 1)) if depth < 8 else grey)[:, :, None]
        out[:, :, 3] = 255
        if trns is not None and len(trns) >= 2:
            key = struct.unpack(">H", trns[:2])[0]
            key = key >> 8 if depth == 16 else key & 0xFF
            out[:, :, 3] = np.where(grey == key, 0, 255)
    elif colour_type == 2:
        out[:, :, :3] = pixels
        out[:, :, 3] = 255
        if trns is not None and len(trns) >= 6:
            key = np.array(struct.unpack(">HHH", trns[:6]))
            key = key >> 8 if depth == 16 else key & 0xFF
            match = np.all(pixels == key.astype(np.uint8), axis=2)
            out[:, :, 3] = np.where(match, 0, 255)
    elif colour_type == 3:
        if plte is None:
            raise PngError("indexed PNG without PLTE")
        index = pixels[:, :, 0].astype(np.int64)
        if index.max() >= plte.shape[0]:
            raise PngError("palette index out of range")
        out[:, :, :3] = plte[index]
        alpha = np.full(plte.shape[0], 255, dtype=np.uint8)
        if trns is not None:
            n = min(len(trns), plte.shape[0])
            alpha[:n] = np.frombuffer(trns[:n], dtype=np.uint8)
        out[:, :, 3] = alpha[index]
    elif colour_type == 4:
        out[:, :, :3] = pixels[:, :, :1]
        out[:, :, 3] = pixels[:, :, 1]
    else:  # 6
        out[:, :, :] = pixels
    return out
