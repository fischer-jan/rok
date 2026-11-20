"""A model from a heightmap picture (New > Heightmap in the scene).

Each pixel becomes a column of voxels: the brighter the pixel, the taller the
column. The picture is seen from above, its top edge at the far (+y) side.
"""

from __future__ import annotations

import numpy as np

from . import palette as palette_core

MAX_SIDE = 256  # the widest model a .vox holds
MAX_PIXELS = 4096 * 4096  # the largest picture Rok reads for a heightmap


def shrink(rgba: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """The picture made smaller by a whole factor, averaging blocks of pixels,
    until neither side is longer than `max_side`."""
    rgba = np.asarray(rgba)
    h, w = rgba.shape[:2]
    factor = int(np.ceil(max(h, w) / max_side))
    if factor <= 1:
        return rgba.astype(np.uint8)
    nh, nw = -(-h // factor), -(-w // factor)
    padded = np.zeros((nh * factor, nw * factor, 4), dtype=np.float64)
    padded[:h, :w] = rgba
    weight = np.zeros((nh * factor, nw * factor), dtype=np.float64)
    weight[:h, :w] = 1.0
    blocks = padded.reshape(nh, factor, nw, factor, 4).sum(axis=(1, 3))
    counts = weight.reshape(nh, factor, nw, factor).sum(axis=(1, 3))
    return np.round(blocks / counts[..., None]).astype(np.uint8)


def heights(rgba: np.ndarray, height: int) -> np.ndarray:
    """(w, h) column heights, 1..height, from the pixels' brightness; 0 where
    a pixel is transparent. Index (x, y) is the pixel at column x, counted
    from the picture's bottom row up."""
    rgba = np.asarray(rgba, dtype=np.float64)
    luma = (0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2]) / 255.0
    columns = 1 + np.round(luma * (int(height) - 1)).astype(np.int64)
    columns[rgba[..., 3] < 128] = 0
    return columns[::-1].T  # rows run down the picture; y runs up the model


def colours(rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(w, h) palette slots 1..255 nearest to each pixel's colour, laid out as
    `heights` lays out the columns."""
    rgb = np.asarray(rgba)[..., :3].reshape(-1, 3)
    unique, inverse = np.unique(rgb, axis=0, return_inverse=True)
    candidates = np.ones(256, dtype=bool)
    candidates[0] = False
    slots = np.array([palette_core.nearest(palette, c, candidates) for c in unique], dtype=np.uint8)
    h, w = np.asarray(rgba).shape[:2]
    return slots[inverse.ravel()].reshape(h, w)[::-1].T


def heightmap_voxels(rgba: np.ndarray, height: int, colour: int | np.ndarray) -> np.ndarray:
    """The (w, h, height) model: each column filled from the floor up to its
    height, in `colour` (one slot, or a (w, h) array of slots from `colours`)."""
    columns = heights(rgba, height)
    w, h = columns.shape
    z = np.arange(int(height))
    filled = z[None, None, :] < columns[:, :, None]
    fill = np.broadcast_to(np.asarray(colour, dtype=np.uint8), (w, h))
    return np.where(filled, fill[:, :, None], 0).astype(np.uint8)
