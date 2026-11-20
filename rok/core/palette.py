"""The .vox default palette (256 RGBA colours) and palette helpers."""

from __future__ import annotations

import numpy as np

# Stored as 0xAABBGGRR, as the .vox format has it.
_DEFAULT_U32 = (
    0x00000000, 0xffffffff, 0xffccffff, 0xff99ffff, 0xff66ffff, 0xff33ffff, 0xff00ffff, 0xffffccff,
    0xffccccff, 0xff99ccff, 0xff66ccff, 0xff33ccff, 0xff00ccff, 0xffff99ff, 0xffcc99ff, 0xff9999ff,
    0xff6699ff, 0xff3399ff, 0xff0099ff, 0xffff66ff, 0xffcc66ff, 0xff9966ff, 0xff6666ff, 0xff3366ff,
    0xff0066ff, 0xffff33ff, 0xffcc33ff, 0xff9933ff, 0xff6633ff, 0xff3333ff, 0xff0033ff, 0xffff00ff,
    0xffcc00ff, 0xff9900ff, 0xff6600ff, 0xff3300ff, 0xff0000ff, 0xffffffcc, 0xffccffcc, 0xff99ffcc,
    0xff66ffcc, 0xff33ffcc, 0xff00ffcc, 0xffffcccc, 0xffcccccc, 0xff99cccc, 0xff66cccc, 0xff33cccc,
    0xff00cccc, 0xffff99cc, 0xffcc99cc, 0xff9999cc, 0xff6699cc, 0xff3399cc, 0xff0099cc, 0xffff66cc,
    0xffcc66cc, 0xff9966cc, 0xff6666cc, 0xff3366cc, 0xff0066cc, 0xffff33cc, 0xffcc33cc, 0xff9933cc,
    0xff6633cc, 0xff3333cc, 0xff0033cc, 0xffff00cc, 0xffcc00cc, 0xff9900cc, 0xff6600cc, 0xff3300cc,
    0xff0000cc, 0xffffff99, 0xffccff99, 0xff99ff99, 0xff66ff99, 0xff33ff99, 0xff00ff99, 0xffffcc99,
    0xffcccc99, 0xff99cc99, 0xff66cc99, 0xff33cc99, 0xff00cc99, 0xffff9999, 0xffcc9999, 0xff999999,
    0xff669999, 0xff339999, 0xff009999, 0xffff6699, 0xffcc6699, 0xff996699, 0xff666699, 0xff336699,
    0xff006699, 0xffff3399, 0xffcc3399, 0xff993399, 0xff663399, 0xff333399, 0xff003399, 0xffff0099,
    0xffcc0099, 0xff990099, 0xff660099, 0xff330099, 0xff000099, 0xffffff66, 0xffccff66, 0xff99ff66,
    0xff66ff66, 0xff33ff66, 0xff00ff66, 0xffffcc66, 0xffcccc66, 0xff99cc66, 0xff66cc66, 0xff33cc66,
    0xff00cc66, 0xffff9966, 0xffcc9966, 0xff999966, 0xff669966, 0xff339966, 0xff009966, 0xffff6666,
    0xffcc6666, 0xff996666, 0xff666666, 0xff336666, 0xff006666, 0xffff3366, 0xffcc3366, 0xff993366,
    0xff663366, 0xff333366, 0xff003366, 0xffff0066, 0xffcc0066, 0xff990066, 0xff660066, 0xff330066,
    0xff000066, 0xffffff33, 0xffccff33, 0xff99ff33, 0xff66ff33, 0xff33ff33, 0xff00ff33, 0xffffcc33,
    0xffcccc33, 0xff99cc33, 0xff66cc33, 0xff33cc33, 0xff00cc33, 0xffff9933, 0xffcc9933, 0xff999933,
    0xff669933, 0xff339933, 0xff009933, 0xffff6633, 0xffcc6633, 0xff996633, 0xff666633, 0xff336633,
    0xff006633, 0xffff3333, 0xffcc3333, 0xff993333, 0xff663333, 0xff333333, 0xff003333, 0xffff0033,
    0xffcc0033, 0xff990033, 0xff660033, 0xff330033, 0xff000033, 0xffffff00, 0xffccff00, 0xff99ff00,
    0xff66ff00, 0xff33ff00, 0xff00ff00, 0xffffcc00, 0xffcccc00, 0xff99cc00, 0xff66cc00, 0xff33cc00,
    0xff00cc00, 0xffff9900, 0xffcc9900, 0xff999900, 0xff669900, 0xff339900, 0xff009900, 0xffff6600,
    0xffcc6600, 0xff996600, 0xff666600, 0xff336600, 0xff006600, 0xffff3300, 0xffcc3300, 0xff993300,
    0xff663300, 0xff333300, 0xff003300, 0xffff0000, 0xffcc0000, 0xff990000, 0xff660000, 0xff330000,
    0xff0000ee, 0xff0000dd, 0xff0000bb, 0xff0000aa, 0xff000088, 0xff000077, 0xff000055, 0xff000044,
    0xff000022, 0xff000011, 0xff00ee00, 0xff00dd00, 0xff00bb00, 0xff00aa00, 0xff008800, 0xff007700,
    0xff005500, 0xff004400, 0xff002200, 0xff001100, 0xffee0000, 0xffdd0000, 0xffbb0000, 0xffaa0000,
    0xff880000, 0xff770000, 0xff550000, 0xff440000, 0xff220000, 0xff110000, 0xffeeeeee, 0xffdddddd,
    0xffbbbbbb, 0xffaaaaaa, 0xff888888, 0xff777777, 0xff555555, 0xff444444, 0xff222222, 0xff111111,
)


def _u32_to_rgba(values: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint32)
    rgba = np.empty((len(values), 4), dtype=np.uint8)
    rgba[:, 0] = arr & 0xFF
    rgba[:, 1] = (arr >> 8) & 0xFF
    rgba[:, 2] = (arr >> 16) & 0xFF
    rgba[:, 3] = (arr >> 24) & 0xFF
    return rgba


DEFAULT_PALETTE = _u32_to_rgba(_DEFAULT_U32)


def default_palette() -> np.ndarray:
    """Return a copy of the .vox default palette, shape (256, 4) uint8."""
    return DEFAULT_PALETTE.copy()


def first_unused(counts: np.ndarray, exclude: int = 0) -> int | None:
    """The lowest index 1..255 with counts == 0 that is not `exclude`, or None."""
    counts = np.asarray(counts)
    for i in range(1, 256):
        if counts[i] == 0 and i != exclude:
            return i
    return None


def new_colour_slot(
    palette: np.ndarray, counts: np.ndarray, plain: np.ndarray, rgb, exclude=()
) -> int | None:
    """The slot a new colour `rgb` goes into, or None when the palette is full.

    The slot holds no voxels, is not in `exclude` and does not hold `rgb` already.
    A slot that still has its default colour and is Plain (`plain[i]`) comes first,
    so a colour that a file or you set up stays. Then any other unused slot.
    """
    palette = np.asarray(palette)
    counts = np.asarray(counts)
    rgb = np.asarray(rgb, dtype=np.uint8)[:3]
    skip = set(int(i) for i in exclude)
    free = [i for i in range(1, 256)
            if counts[i] == 0 and i not in skip and not np.array_equal(palette[i, :3], rgb)]
    for i in free:
        if plain[i] and np.array_equal(palette[i, :3], DEFAULT_PALETTE[i, :3]):
            return i
    return free[0] if free else None


def nearest(palette: np.ndarray, rgb, candidates: np.ndarray) -> int:
    """The candidate slot with the smallest squared RGB distance to `rgb`."""
    candidates = np.asarray(candidates, dtype=bool)
    if not candidates.any():
        raise ValueError("nearest: no candidates")
    diff = np.asarray(palette, dtype=np.int64)[:, :3] - np.asarray(rgb, dtype=np.int64)[:3]
    dist = (diff * diff).sum(axis=1)
    dist[~candidates] = np.iinfo(np.int64).max
    return int(np.argmin(dist))


def merge(
    palette, used_counts, incoming, incoming_used, *, protect: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the used colours of `incoming` into `palette`. Returns (palette', remap).

    The first two values of `merge_claimed`; see there for the rules.
    """
    out, remap, _claimed_by = merge_claimed(
        palette, used_counts, incoming, incoming_used, protect=protect
    )
    return out, remap


def merge_claimed(
    palette, used_counts, incoming, incoming_used, *, protect: int = 0, keep=(),
    materials=None, incoming_materials=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the used colours of `incoming` into `palette`, through `colours.fit`.

    Returns (palette', remap, claimed_by). `claimed_by[k]` is the incoming
    index that wrote its colour into slot k, 0 when slot k was not written.

    Per used incoming index i: a slot that holds exactly its colour (and its
    material, when the tables are given) is reused; else a free slot, slot i
    itself first, so a file from the same pack keeps its indices; else the
    nearest colour in use. A free slot holds no voxel (`used_counts`, which
    must count every model and frame) and is not `protect` (the colour you
    paint with) or in `keep` (the colours you made and have not used yet).
    """
    from . import colours

    out = np.array(palette, dtype=np.uint8, copy=True)
    incoming = np.asarray(incoming, dtype=np.uint8)
    indices = [i for i in range(1, 256) if incoming_used[i] > 0]
    wants = [(incoming[i], None if incoming_materials is None else incoming_materials.get(i)) for i in indices]
    pinned = set(int(k) for k in keep)
    if 0 < protect < 256:
        pinned.add(int(protect))
    placed = colours.fit(out, used_counts, wants, pinned=pinned, materials=materials, hints=indices, merge=True)
    remap = np.arange(256, dtype=np.uint8)
    claimed_by = np.zeros(256, dtype=np.uint8)
    for i, slot in zip(indices, placed.slots):
        remap[i] = slot
    for slot, (colour, _material) in placed.writes.items():
        out[slot] = colour
        claimed_by[slot] = indices[[s for s in placed.slots].index(slot)]
    remap[0] = 0
    return out, remap, claimed_by


def swap_slots(palette: np.ndarray, a: int, b: int) -> np.ndarray:
    """Palette with slots a and b exchanged. a == b returns an unchanged copy."""
    out = np.array(palette, dtype=np.uint8, copy=True)
    if a != b:
        out[[a, b]] = out[[b, a]]
    return out


def swap_remap(a: int, b: int) -> np.ndarray:
    """(256,) uint8 identity remap with a and b swapped.

    Applied to a grid's voxels after `swap_slots`, it keeps every model looking
    the way it did: a voxel painted a becomes b and the other way round.
    """
    remap = np.arange(256, dtype=np.uint8)
    if a != b:
        remap[a], remap[b] = b, a
    return remap


def gradient(palette: np.ndarray, a: int, b: int) -> np.ndarray:
    """Palette with the slots strictly between a and b (index order) set to a
    straight interpolation of colour a and colour b. a and b keep their own
    colour; equal or adjacent indices leave the palette unchanged.
    """
    out = np.array(palette, dtype=np.uint8, copy=True)
    lo, hi = (int(a), int(b)) if a <= b else (int(b), int(a))
    span = hi - lo
    if span < 2:
        return out
    t = (np.arange(lo + 1, hi, dtype=np.float64) - lo) / span
    start = np.asarray(palette[lo], dtype=np.float64)
    end = np.asarray(palette[hi], dtype=np.float64)
    out[lo + 1 : hi] = np.round(start + (end - start) * t[:, None]).astype(np.uint8)
    return out


def darker_slot(palette: np.ndarray, colour: int, amount: float = 0.45) -> int:
    """The slot whose colour is nearest a darker shade of slot `colour`, among
    the slots that really are darker; `colour` itself when none is."""
    from .gradient import darker, oklab

    colour = int(colour)
    rgb = np.asarray(palette[colour, :3], dtype=np.int64)
    light = float(oklab(rgb)[0])
    candidates = np.zeros(len(palette), dtype=bool)
    for i in range(1, len(palette)):
        candidates[i] = i != colour and float(oklab(palette[i, :3])[0]) < light - 0.04
    if not candidates.any():
        return colour
    return nearest(palette, darker(rgb, amount), candidates)
