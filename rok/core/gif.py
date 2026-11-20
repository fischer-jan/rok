"""Animated GIF files, written without an image library.

`encode` takes RGB frames of one size and gives the bytes of a GIF89a with
one palette for all frames. A picture with 256 colours or fewer keeps them
exactly; more are cut down to 255 by median cut, and every pixel takes the
nearest of them. After the first frame, only the box that changed is
written, and its unchanged pixels are transparent: a small part that moves
costs little. numpy and stdlib only.
"""

from __future__ import annotations

import struct

import numpy as np

_LUT_BITS = 5  # the colour cube the nearest-colour table indexes: 32 steps per channel


def _median_cut(pixels: np.ndarray, count: int) -> np.ndarray:
    """`count` colours that stand for `pixels` (N x 3 uint8): split the box
    with the widest channel at its median until there are enough boxes."""
    boxes = [pixels]
    while len(boxes) < count:
        spans = [int(np.ptp(b, axis=0).max()) if len(b) > 1 else -1 for b in boxes]
        i = int(np.argmax(spans))
        if spans[i] <= 0:
            break
        box = boxes.pop(i)
        channel = int(np.argmax(np.ptp(box, axis=0)))
        order = np.argsort(box[:, channel], kind="stable")
        half = len(box) // 2
        boxes += [box[order[:half]], box[order[half:]]]
    return np.array([b.mean(axis=0) for b in boxes if len(b)]).round().clip(0, 255).astype(np.uint8)


def palette_of(frames: list[np.ndarray], limit: int = 255) -> tuple[np.ndarray, list[np.ndarray]]:
    """(palette (K x 3 uint8, K <= `limit`), one index picture per frame)."""
    flat = np.concatenate([f.reshape(-1, 3) for f in frames])
    keys = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    unique = np.unique(keys)
    if len(unique) <= limit:
        palette = np.stack([(unique >> 16) & 255, (unique >> 8) & 255, unique & 255], axis=1).astype(np.uint8)
        out = []
        for f in frames:
            k = (f[..., 0].astype(np.uint32) << 16) | (f[..., 1].astype(np.uint32) << 8) | f[..., 2]
            out.append(np.searchsorted(unique, k).astype(np.uint8))
        return palette, out
    step = max(1, len(flat) // 250_000)
    palette = _median_cut(flat[::step], limit)
    # The nearest colour for every cell of a 32 x 32 x 32 cube, then a lookup per pixel.
    n = 1 << _LUT_BITS
    shift = 8 - _LUT_BITS
    axis = (np.arange(n) << shift) + (1 << (shift - 1))
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.int32)
    pal = palette.astype(np.int32)
    lut = np.empty(len(cube), dtype=np.uint8)
    for start in range(0, len(cube), 4096):
        d = ((cube[start:start + 4096, None, :] - pal[None]) ** 2).sum(axis=2)
        lut[start:start + 4096] = d.argmin(axis=1)
    out = []
    for f in frames:
        q = (f >> shift).astype(np.int32)
        out.append(lut[(q[..., 0] << (2 * _LUT_BITS)) | (q[..., 1] << _LUT_BITS) | q[..., 2]])
    return palette, out


def _lzw(indices: np.ndarray, min_size: int) -> bytes:
    """GIF's LZW for one picture's indices (flat uint8), as data sub-blocks."""
    clear, end = 1 << min_size, (1 << min_size) + 1
    out = bytearray()
    bits = 0
    nbits = 0
    size = min_size + 1

    def emit(code: int) -> None:
        nonlocal bits, nbits
        bits |= code << nbits
        nbits += size
        while nbits >= 8:
            out.append(bits & 255)
            bits >>= 8
            nbits -= 8

    table: dict[tuple[int, int], int] = {}
    next_code = end + 1
    emit(clear)
    data = indices.tobytes()
    if data:
        prefix = data[0]
        for byte in data[1:]:
            key = (prefix, byte)
            code = table.get(key)
            if code is not None:
                prefix = code
                continue
            emit(prefix)
            if next_code < 4096:
                table[key] = next_code
                if next_code == (1 << size) and size < 12:
                    size += 1
                next_code += 1
            else:
                emit(clear)
                table.clear()
                next_code = end + 1
                size = min_size + 1
            prefix = byte
        emit(prefix)
    emit(end)
    if nbits:
        out.append(bits & 255)
    blocks = bytearray()
    for i in range(0, len(out), 255):
        chunk = out[i:i + 255]
        blocks.append(len(chunk))
        blocks += chunk
    return bytes(blocks) + b"\0"


def delays(count: int, fps: float) -> list[int]:
    """Each frame's delay in hundredths of a second (GIF's unit); the
    rounding carries over, so the whole loop keeps the frame rate."""
    return [round(100 * (i + 1) / fps) - round(100 * i / fps) for i in range(count)]


def encode(frames: list[np.ndarray], fps: float, loop: bool = True) -> bytes:
    """An animated GIF of `frames` (H x W x 3 or 4 uint8, alpha ignored) at
    `fps`; `loop` plays it again and again."""
    if not frames:
        raise ValueError("a GIF needs at least one frame")
    rgb = [np.ascontiguousarray(np.asarray(f)[..., :3], dtype=np.uint8) for f in frames]
    h, w = rgb[0].shape[:2]
    if any(f.shape[:2] != (h, w) for f in rgb):
        raise ValueError("every frame of a GIF needs the same size")
    palette, pictures = palette_of(rgb)
    transparent = len(palette)  # one free index: the pixels a frame leaves as they were
    colours = max(2, 1 << int(np.ceil(np.log2(transparent + 1))))
    bits = int(np.log2(colours))
    table = np.zeros((colours, 3), dtype=np.uint8)
    table[:len(palette)] = palette
    out = bytearray(b"GIF89a")
    out += struct.pack("<HHBBB", w, h, 0x80 | 0x70 | (bits - 1), 0, 0)
    out += table.tobytes()
    if loop:
        out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
    min_size = max(2, bits)
    previous = None
    for picture, delay in zip(pictures, delays(len(pictures), fps)):
        x0, y0, x1, y1 = 0, 0, w, h
        data = picture
        if previous is not None:
            changed = picture != previous
            if changed.any():
                ys, xs = np.nonzero(changed)
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            else:
                x0, x1, y0, y1 = 0, 1, 0, 1
            data = picture[y0:y1, x0:x1].copy()
            data[~changed[y0:y1, x0:x1]] = transparent
        # Graphic control: keep the frame (disposal 1), the transparent index on after the first.
        flags = (1 << 2) | (1 if previous is not None else 0)
        out += struct.pack("<BBBBHBB", 0x21, 0xF9, 4, flags, delay, transparent, 0)
        out += struct.pack("<BHHHHB", 0x2C, x0, y0, x1 - x0, y1 - y0, 0)
        out.append(min_size)
        out += _lzw(np.ascontiguousarray(data).reshape(-1), min_size)
        previous = picture
    out.append(0x3B)
    return bytes(out)


__all__ = ["delays", "encode", "palette_of"]
