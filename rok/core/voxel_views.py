"""Flat views of a voxel grid, drawn cell by cell: the pictures a voxel artist
judges a piece by. Numpy only.

A view looks along one axis and shows, for each column of cells, the first
filled cell the camera meets. The views and their orientation, as the quad
picture of rok_look shows them:

- front (camera on the -y side): columns x low to high, rows z high to low;
- right (camera on the +x side): columns y low to high, rows z high to low;
- top (camera above): columns x low to high, rows y high to low. A row of the
  top view reads like a row of a rok_voxels_set layer.
"""

from __future__ import annotations

import numpy as np

SIDES = ("front", "right", "top")
EMPTY = "."
# Characters for colours in text views, most used colour first. Not '.', ' ' or
# '0': those mean empty in rok_voxels_set layers.
CHARS = "#ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789@%&*+=?"
BACKGROUND = (34, 34, 38)


def project(voxels: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(colours, depth) of a view, both (rows, columns): the colour number of the
    first filled cell along the view (0 for none) and how many cells deep it lies
    behind the nearest face of the grid (-1 for none)."""
    if side not in SIDES:
        raise ValueError(f"no side {side!r}; the sides are {', '.join(SIDES)}")
    v = np.asarray(voxels)
    if side == "front":
        a = np.transpose(v, (2, 0, 1))[::-1]  # rows z high..low, columns x, depth along +y
    elif side == "right":
        a = np.transpose(v, (2, 1, 0))[::-1, :, ::-1]  # rows z high..low, columns y, depth along -x
    else:
        a = np.transpose(v, (1, 0, 2))[::-1, :, ::-1]  # rows y high..low, columns x, depth along -z
    filled = a != 0
    hit = filled.any(axis=2)
    depth = np.where(hit, filled.argmax(axis=2), -1)
    rows, cols = np.indices(hit.shape)
    colours = np.where(hit, a[rows, cols, np.maximum(depth, 0)], 0)
    return colours.astype(np.int64), depth.astype(np.int64)


def rgb(colours: np.ndarray, depth: np.ndarray, palette: np.ndarray, shade: float = 0.25,
        background=BACKGROUND) -> np.ndarray:
    """(rows, columns, 3) uint8: each cell in its palette colour, a little darker
    the deeper it lies (`shade` at the far side), so steps in depth read."""
    pal = np.asarray(palette)[:, :3].astype(np.float64)
    out = pal[np.clip(colours, 0, 255)]
    if shade and (depth >= 0).any():
        far = max(int(depth.max()), 1)
        out = out * (1.0 - shade * np.clip(depth, 0, None) / far)[..., None]
    out[colours == 0] = background
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def scaled(image: np.ndarray, pixels: int) -> np.ndarray:
    """Every cell as a square of `pixels` x `pixels`: crisp, no smoothing."""
    pixels = max(1, int(pixels))
    return np.repeat(np.repeat(image, pixels, axis=0), pixels, axis=1)


def icon(image: np.ndarray, colours: np.ndarray, side_px: int, background=BACKGROUND) -> np.ndarray:
    """The view shrunk so its longer side is `side_px` pixels, as an icon shows it:
    each pixel the mean of the cells it covers (empty cells count as background).
    A view already that small is kept cell for cell."""
    h, w = colours.shape
    long_side = max(h, w)
    if long_side <= side_px:
        return image.copy()
    out_h = max(1, round(h * side_px / long_side))
    out_w = max(1, round(w * side_px / long_side))
    img = image.astype(np.float64)
    img[colours == 0] = background
    ys = np.linspace(0, h, out_h + 1)
    xs = np.linspace(0, w, out_w + 1)
    out = np.zeros((out_h, out_w, 3))
    for i in range(out_h):
        for j in range(out_w):
            y0, y1 = int(ys[i]), max(int(np.ceil(ys[i + 1])), int(ys[i]) + 1)
            x0, x1 = int(xs[j]), max(int(np.ceil(xs[j + 1])), int(xs[j]) + 1)
            out[i, j] = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def legend_for(voxels: np.ndarray) -> dict[int, str]:
    """A character for each colour in use, the most used colour first."""
    v = np.asarray(voxels)
    values, counts = np.unique(v[v != 0], return_counts=True)
    order = [int(values[i]) for i in np.argsort(-counts, kind="stable")]
    return {c: CHARS[k] if k < len(CHARS) else "?" for k, c in enumerate(order)}


def text(colours: np.ndarray, legend: dict[int, str]) -> list[str]:
    """The view as rows of characters: '.' empty, else the colour's character."""
    lut = np.array([legend.get(c, "?") if c else EMPTY for c in range(256)])
    return ["".join(row) for row in lut[np.clip(colours, 0, 255)]]


def box(voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The low and high cells of the filled voxels, or None for an empty grid."""
    filled = np.argwhere(np.asarray(voxels) != 0)
    if len(filled) == 0:
        return None
    return filled.min(axis=0), filled.max(axis=0)


def cropped(voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The voxels cut to their filled box, and the box's low corner."""
    b = box(voxels)
    if b is None:
        return np.zeros((1, 1, 1), np.uint8), np.zeros(3, np.int64)
    lo, hi = b
    return np.asarray(voxels)[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1], lo
