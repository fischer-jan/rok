"""The colours a shader run puts into the model, and your changes to them.

The shader panel shows the colours a shader will use and
lets you change any of them. Those are the colours of the cells the run
changes: the shader's own colours (its rok_colours line), the shades it
makes of your model's colours, the palette colours it takes. A change is a
map from a colour of the result to the colour you want; it applies after
the run, to the changed cells only, so the rest of the model keeps its
colours. It is kept by colour, not by palette slot, so it holds when the
shader's settings move its colours to other slots.

Text form (a live shader's settings, rok.xs_run's recolour):
"3a3f47:ff0000,7fe7ff:00ff00".
"""

from __future__ import annotations

import numpy as np

from . import colours as colours_core

RGB = tuple[int, int, int]
MAX_SHOWN = 24  # swatches in the panel


def hexes(rgb: RGB) -> str:
    return "{:02x}{:02x}{:02x}".format(*rgb)


def _rgb(text: str) -> RGB | None:
    text = text.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        value = int(text, 16)
    except ValueError:
        return None
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def parse(text: str) -> dict[RGB, RGB]:
    """The map from its text form; a broken pair is left out."""
    out: dict[RGB, RGB] = {}
    for pair in (text or "").split(","):
        src, _sep, dst = pair.partition(":")
        a, b = _rgb(src), _rgb(dst)
        if a is not None and b is not None and a != b:
            out[a] = b
    return out


def encode(mapping: dict[RGB, RGB]) -> str:
    return ",".join(f"{hexes(a)}:{hexes(b)}" for a, b in mapping.items() if a != b)


def changed_cells(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """The cells the run filled or gave another colour."""
    return (after > 0) & (after != before)


def shown(before: np.ndarray, after: np.ndarray, palette: np.ndarray, first=()) -> list[RGB]:
    """The colours of the changed cells, for the panel: the colours in
    `first` (the shader's own, in their order) that the result uses, then
    the others, most cells first. `palette` is the palette after the run."""
    cells = after[changed_cells(before, after)]
    if not cells.size:
        return []
    counts = np.bincount(cells, minlength=256)
    by_colour: dict[RGB, int] = {}
    for slot in np.flatnonzero(counts):
        rgb = tuple(int(c) for c in palette[slot][:3])
        by_colour[rgb] = by_colour.get(rgb, 0) + int(counts[slot])
    out = [tuple(c) for c in first if tuple(c) in by_colour]
    rest = sorted((c for c in by_colour if c not in out), key=lambda c: -by_colour[c])
    return (out + rest)[:MAX_SHOWN]


def apply(before: np.ndarray, after: np.ndarray, palette: np.ndarray, mapping: dict[RGB, RGB],
          used: np.ndarray, keep=(), materials=None) -> tuple[np.ndarray, dict[int, RGB]]:
    """The result with your colours: each changed cell whose colour is a key
    of `mapping` gets that key's colour. The colours go where `colours.fit`
    puts them: the slot that already holds exactly the colour, or a free one
    (no voxel of any model uses it, per `used`, the result does not, and it
    is not in `keep`). `palette` is the palette after the run. Returns the
    new voxels and the palette writes they need. Raises ValueError when the
    palette has no free slot left."""
    if not mapping:
        return after, {}
    out = after.copy()
    changed = changed_cells(before, after)
    sources = [int(s) for s in np.flatnonzero(np.bincount(after[changed], minlength=256))]
    sources = [s for s in sources if tuple(int(c) for c in palette[s][:3]) in mapping]
    if not sources:
        return after, {}
    in_result = np.bincount(after.ravel(), minlength=256)
    try:
        placed = colours_core.fit(palette, np.asarray(used) + in_result,
                                  [(mapping[tuple(int(c) for c in palette[s][:3])], None) for s in sources],
                                  pinned=keep, materials=materials)
    except colours_core.Full:
        raise ValueError("the palette has no free slot for your colour") from None
    for slot, target in zip(sources, placed.slots):
        out[changed & (after == slot)] = target
    return out, {slot: tuple(c[:3]) for slot, (c, _m) in placed.writes.items()}
