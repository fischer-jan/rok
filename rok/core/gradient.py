"""Paint's Gradient mode: a ramp of colours across the model, as data. Pure.

A drag from one point to another paints the voxels in bands: the first band
in the start colour, the last in the end colour, `n` colours in all. The
colours between are blended in OKLab, so the middle stays clean (sRGB
blending turns red to green through mud). With dither, neighbouring bands mix
where they meet in an ordered 4 x 4 pattern.
"""

from __future__ import annotations

import numpy as np

MIN_STEPS, MAX_STEPS, DEFAULT_STEPS = 2, 16, 5

# The 4 x 4 ordered threshold: a value from 0 to 1 per cell.
BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float64) + 0.5) / 16.0


def _to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def oklab(rgb) -> np.ndarray:
    """sRGB 0..255 to OKLab (L, a, b)."""
    r, g, b = _to_linear(np.asarray(rgb, dtype=np.float64))
    lms = np.array([
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    ])
    l_, m_, s_ = np.cbrt(lms)
    return np.array([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ])


def from_oklab(lab) -> tuple[int, int, int]:
    L, a, b = (float(v) for v in lab)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    lin = np.array([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ])
    return tuple(int(round(v)) for v in _to_srgb(lin))


def ramp(start, end, n: int) -> list[tuple[int, int, int]]:
    """`n` colours from `start` to `end` (sRGB 0..255), the ends exactly as given."""
    n = max(MIN_STEPS, min(MAX_STEPS, int(n)))
    a, b = oklab(start), oklab(end)
    out = [tuple(int(c) for c in start)]
    for k in range(1, n - 1):
        out.append(from_oklab(a + (b - a) * (k / (n - 1))))
    out.append(tuple(int(c) for c in end))
    return out


def darker(rgb, amount: float = 0.6) -> tuple[int, int, int]:
    """A darker shade of `rgb`: its OKLab lightness lowered by `amount` of itself."""
    L, a, b = oklab(rgb)
    return from_oklab((L * (1.0 - amount), a, b))


def bands(cells, start, end, n: int, dither: bool = False) -> np.ndarray:
    """(N,) the band 0..n-1 of each cell (N, 3) for a gradient from the point
    `start` to the point `end` (cell units; a cell's middle is +0.5). Cells
    before the start take band 0, past the end band n-1."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    n = max(MIN_STEPS, min(MAX_STEPS, int(n)))
    a = np.asarray(start, dtype=np.float64)
    d = np.asarray(end, dtype=np.float64) - a
    length2 = float(d @ d)
    if length2 < 1e-9 or len(cells) == 0:
        return np.zeros(len(cells), dtype=np.int64)
    t = np.clip(((cells + 0.5) - a) @ d / length2, 0.0, 1.0)
    if not dither:
        return np.minimum((t * n).astype(np.int64), n - 1)
    s = t * (n - 1)
    lo = np.floor(s).astype(np.int64)
    x, y, z = cells[:, 0], cells[:, 1], cells[:, 2]
    threshold = BAYER4[(x + z) % 4, (y + 2 * z) % 4]
    return np.minimum(lo + ((s - lo) > threshold), n - 1)
