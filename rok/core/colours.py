"""Which palette slot a colour goes to: the one rule every command uses.

A voxel holds a slot number (1 to 255) and the palette holds the slot's
colour, so a command that adds a colour must put it on a slot. It may only
take a slot that is free:

- no voxel uses it: no model, in its own grid or in any of its frames;
- it is not pinned: the colour you paint with, and the colours you made and
  have not used yet.

A colour that the palette already holds exactly (the same RGBA and the same
material) goes to that slot, and no other colour of the same command may
take it. `fit` places all the colours of one command at once, so two of them
never land on one slot. Free slots that still hold their default colour and
no material come first: a colour that a file or you set up stays longest.

numpy only; the app passes in what it knows (counts, pins, materials).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .material import DEFAULT, Material, MaterialTable
from .palette import DEFAULT_PALETTE

RGBA = tuple[int, int, int, int]


class Full(ValueError):
    """The palette has fewer free slots than a command needs."""

    def __init__(self, missing: int, needed: int) -> None:
        self.missing = missing
        self.needed = needed
        super().__init__(f"the palette has no free slot for {missing} of {needed} colours")


def counts(grids) -> np.ndarray:
    """(256,) int64: the voxels per slot over `grids`, any arrays of slots."""
    out = np.zeros(256, dtype=np.int64)
    for voxels in grids:
        out += np.bincount(np.asarray(voxels, dtype=np.uint8).ravel(), minlength=256)
    return out


def rgba(colour) -> RGBA:
    """A colour as four ints; three values are opaque."""
    values = [int(c) for c in colour]
    if len(values) == 3:
        values.append(255)
    return (values[0], values[1], values[2], values[3])


def free_slots(palette: np.ndarray, used: np.ndarray, pinned=(), materials: MaterialTable | None = None) -> list[int]:
    """The free slots in the order they are given out: those still at their
    default colour and plain first, then the rest, each from low to high."""
    used = np.asarray(used)
    pins = {int(p) for p in pinned}
    free = [s for s in range(1, 256) if int(used[s]) == 0 and s not in pins]

    def untouched(s: int) -> bool:
        plain = materials is None or materials.get(s) == DEFAULT
        return plain and np.array_equal(palette[s], DEFAULT_PALETTE[s])

    return [s for s in free if untouched(s)] + [s for s in free if not untouched(s)]


@dataclass
class Fit:
    """Where the colours of one command go. `slots[k]` is colour k's slot;
    `writes` the slots that get a new colour, with its material; `merged`
    the colours (by index k) that went to the nearest colour for want of room."""

    slots: list[int] = field(default_factory=list)
    writes: dict[int, tuple[RGBA, Material]] = field(default_factory=dict)
    merged: list[int] = field(default_factory=list)


def fit(palette: np.ndarray, used: np.ndarray, wants, *, pinned=(), materials: MaterialTable | None = None,
        hints=None, merge: bool = False) -> Fit:
    """Place `wants`, a list of (colour, material) pairs (material None:
    plain), on slots. A colour equal to one in the palette reuses that slot;
    any other takes a free slot, its hint (`hints[k]`, a slot number) first
    when that slot is free. Raises Full when free slots run out, unless
    `merge`: then each colour left over goes to the nearest colour in use."""
    palette = np.asarray(palette)
    wants = [(rgba(c), DEFAULT if m is None else m) for c, m in wants]
    hints = list(hints) if hints is not None else [None] * len(wants)

    def material(s: int) -> Material:
        return DEFAULT if materials is None else materials.get(s)

    used_now = np.asarray(used) > 0
    exact: dict[tuple[RGBA, Material], int] = {}
    for s in range(255, 0, -1):  # the lowest slot wins; a used one before an unused one
        key = (rgba(palette[s]), material(s))
        if used_now[s] or key not in exact or not used_now[exact[key]]:
            exact[key] = s
    out = Fit()
    taken: set[int] = set()
    free = free_slots(palette, used, pinned, materials)
    left: list[int] = []
    for k, want in enumerate(wants):
        slot = exact.get(want)
        if slot is not None:
            out.slots.append(slot)
            taken.add(slot)
            continue
        out.slots.append(0)
        left.append(k)
    for k in left:
        want = wants[k]
        same = next((s for s, w in out.writes.items() if w == want), None)
        if same is not None:
            out.slots[k] = same
            continue
        hint = hints[k]
        options = [s for s in free if s not in taken]
        slot = hint if hint in options else (options[0] if options else None)
        if slot is None:
            out.merged.append(k)
            continue
        out.slots[k] = slot
        out.writes[slot] = want
        taken.add(slot)
    if out.merged:
        if not merge:
            raise Full(len(out.merged), len(wants))
        candidates = np.zeros(256, dtype=bool)
        candidates[1:] = used_now[1:]
        for s in taken:
            candidates[s] = True
        if not candidates.any():
            candidates[1:] = True
        after = np.array(palette, copy=True)
        for s, (c, _m) in out.writes.items():
            after[s] = c
        for k in out.merged:
            out.slots[k] = nearest(after, wants[k][0], candidates)
    return out


def nearest(palette: np.ndarray, colour, candidates: np.ndarray) -> int:
    """The candidate slot with the smallest squared RGB distance to `colour`."""
    diff = np.asarray(palette, dtype=np.int64)[:, :3] - np.asarray(colour, dtype=np.int64)[:3]
    dist = (diff * diff).sum(axis=1)
    dist[~np.asarray(candidates, dtype=bool)] = np.iinfo(np.int64).max
    dist[0] = np.iinfo(np.int64).max
    return int(np.argmin(dist))


def write(palette: np.ndarray, materials: MaterialTable, writes) -> bool:
    """Put `writes` ({slot: colour} or {slot: (colour, material)}) into the
    palette and the table, in place. A bare colour is opaque and plain, so a
    slot never keeps the glass or metal of the colour it held before.
    True when the material table changed."""
    changed = False
    for slot, value in writes.items():
        if len(value) == 2 and not isinstance(value[0], (int, np.integer)):
            colour, material = value
        else:
            colour, material = value, DEFAULT
        palette[slot] = rgba(colour)
        if materials.get(slot) != material:
            materials.set(slot, material)
            changed = True
    return changed
