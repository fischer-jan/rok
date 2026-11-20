"""Stroke Designer: curves that shape a stroke along its length.

A curve runs from the stroke's start (left) to its end (right). Its height is
a value from 0 at the bottom to 1 at the top. Three curves:

- Thickness: the brush size as a fraction of the Size field. 0 is one voxel.
- Roundness: the brush shape. Bottom is a diamond, the middle a sphere, the
  top a cube. The shape is a p-norm whose exponent runs from 1 through 2 to
  a very large one.
- Colour: from the current colour (bottom) to a second palette slot (top).
  When the two slots are a run of neighbours, the stroke walks the slots
  between them. Otherwise it takes the nearer of the two. It never makes a
  new palette colour.

While a stroke is drawn it uses each curve's start value. When the button
comes up the stroke is drawn again with the curves spread over the whole path.
A row that is off leaves that part of the stroke as it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .edit import ATTACH, BRUSH_MAX, ERASE, PAINT, sphere_mask, sphere_offsets

MASK_CACHE = 96  # brush boxes kept: a taper with a roundness curve touches many size and shape pairs
CUBE_EXPONENT = 32.0  # large enough that the p-norm is the cube
ROWS = ("thickness", "roundness", "colour")


# ---------------------------------------------------------------- curve


@dataclass(frozen=True)
class Curve:
    """A cubic bezier. x is the stroke (0 at the start, 1 at the end) and y is
    the value (0 at the bottom, 1 at the top). The ends stay at x 0 and x 1.
    The handles stay in order, so the curve is a function of x."""

    y0: float
    x1: float
    y1: float
    x2: float
    y2: float
    y3: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "y0", _unit(self.y0))
        object.__setattr__(self, "y3", _unit(self.y3))
        object.__setattr__(self, "y1", _unit(self.y1))
        object.__setattr__(self, "y2", _unit(self.y2))
        x1 = min(0.98, max(0.02, float(self.x1)))
        x2 = min(0.98, max(x1, float(self.x2)))
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "x2", x2)

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.y0, self.x1, self.y1, self.x2, self.y2, self.y3)

    def with_point(self, which: str, x: float, y: float) -> Curve:
        """A copy with one point or handle moved. Ends keep their x."""
        y = _unit(y)
        if which == "start":
            return Curve(y, self.x1, self.y1, self.x2, self.y2, self.y3)
        if which == "end":
            return Curve(self.y0, self.x1, self.y1, self.x2, self.y2, y)
        if which == "h1":
            return Curve(self.y0, x, y, self.x2, self.y2, self.y3)
        if which == "h2":
            return Curve(self.y0, self.x1, self.y1, x, y, self.y3)
        return self

    def value_at(self, u: float) -> float:
        """The curve's height at stroke fraction `u` (0 at the start)."""
        u = _unit(u)
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) * 0.5
            if _bezier_x(self, mid) < u:
                lo = mid
            else:
                hi = mid
        return _unit(_bezier_y(self, (lo + hi) * 0.5))

    def samples(self, n: int = 32) -> list[tuple[float, float]]:
        """(x, y) along the curve, for drawing."""
        out = []
        for i in range(max(2, n)):
            t = i / (n - 1)
            out.append((_bezier_x(self, t), _bezier_y(self, t)))
        return out


def _unit(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _bezier_x(curve: Curve, t: float) -> float:
    s = 1.0 - t
    return 3.0 * s * s * t * curve.x1 + 3.0 * s * t * t * curve.x2 + t * t * t


def _bezier_y(curve: Curve, t: float) -> float:
    s = 1.0 - t
    return (
        s * s * s * curve.y0
        + 3.0 * s * s * t * curve.y1
        + 3.0 * s * t * t * curve.y2
        + t * t * t * curve.y3
    )


def flat(y: float) -> Curve:
    """A level curve at height `y`."""
    y = _unit(y)
    return Curve(y, 0.35, y, 0.65, y, y)


def taper() -> Curve:
    """Full at the start, down to nothing at the end: a tentacle."""
    return Curve(1.0, 0.38, 0.78, 0.70, 0.16, 0.0)


def ramp() -> Curve:
    """From the bottom at the start to the top at the end."""
    return Curve(0.0, 0.35, 0.22, 0.65, 0.78, 1.0)


# ---------------------------------------------------------------- shape


def sized(fraction: float, brush: int) -> int:
    """Brush size for a thickness fraction of the Size field. Never below 1,
    never above the Size field."""
    brush = max(1, min(BRUSH_MAX, int(brush)))
    return max(1, min(brush, int(round(_unit(fraction) * brush))))


def exponent_of(value: float) -> float:
    """The p-norm exponent for a roundness value: 1 at the bottom (diamond),
    2 in the middle (sphere), and a very large one at the top (cube)."""
    v = _unit(value)
    if v <= 0.5:
        return 1.0 + 2.0 * v
    if v >= 0.999:
        return CUBE_EXPONENT
    return min(CUBE_EXPONENT, 1.0 / (1.0 - v))


def _shape_key(exponent: float) -> int:
    if exponent <= 1.0:
        return 1
    if abs(exponent - 2.0) < 1e-6:
        return 2
    if exponent >= CUBE_EXPONENT:
        return 10_000
    return int(round(float(exponent) * 4.0))


_MASKS: dict[tuple[int, int], np.ndarray] = {}


def shape_mask(size: int, exponent: float) -> np.ndarray:
    """The brush as a bool box, the centre in the middle.

    Exponent 1 is the diamond (|x| + |y| + |z| <= r), 2 is the sphere of
    `sphere_offsets`, and a large exponent is the cube (max <= r). Size 1 is
    the one centre cell for every exponent. r is size - 1, the same reach the
    sphere has along an axis.
    """
    s = max(1, min(BRUSH_MAX, int(size)))
    key = (s, _shape_key(exponent))
    found = _MASKS.get(key)
    if found is not None:
        return found
    if key[1] == 2:
        found = sphere_mask(s)
    else:
        found = _build_mask(s, 1.0 if key[1] == 1 else (CUBE_EXPONENT if key[1] == 10_000 else key[1] / 4.0))
    if len(_MASKS) >= MASK_CACHE:
        _MASKS.pop(next(iter(_MASKS)))  # the oldest box goes first
    _MASKS[key] = found
    return found


def shape_offsets(size: int, exponent: float) -> np.ndarray:
    """(N, 3) int64 offsets of the brush. See `shape_mask`."""
    s = max(1, min(BRUSH_MAX, int(size)))
    if _shape_key(exponent) == 2:
        return sphere_offsets(s)
    mask = shape_mask(s, exponent)
    r = s - 1
    return np.argwhere(mask).astype(np.int64) - r


def _build_mask(size: int, exponent: float) -> np.ndarray:
    r = size - 1
    if r <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    coords = np.arange(-r, r + 1)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    return _inside(ax, ay, az, float(exponent), r)


def _inside(ax, ay, az, p: float, r: int):
    if p <= 1.0:
        return ax + ay + az <= r
    if p >= CUBE_EXPONENT:
        return np.maximum(np.maximum(ax, ay), az) <= r
    if abs(p - 2.0) < 1e-6:
        return ax * ax + ay * ay + az * az <= r * r + r
    if p < 2.0:
        t = p - 1.0
        limit = r + (math.sqrt(r * r + r) - r) * t
        norm = (ax ** p + ay ** p + az ** p) ** (1.0 / p)
        return norm <= limit + 1e-6
    # From the sphere toward the cube: cells outside the sphere come in as the
    # exponent grows, the corners last. Nothing past the cube's box.
    t = (p - 2.0) / (CUBE_EXPONENT - 2.0)
    sphere = ax * ax + ay * ay + az * az <= r * r + r
    cube = np.maximum(np.maximum(ax, ay), az) <= r
    limit = float(r * r + r)
    span = max(3.0 * r * r - limit, 1.0)
    frac = np.clip((ax * ax + ay * ay + az * az - limit) / span, 0.0, 1.0)
    return sphere | (cube & (frac <= t))


# ---------------------------------------------------------------- colour


def colour_slot(value: float, current: int, second: int) -> int:
    """The palette slot for a colour-curve value.

    0 is the current colour, 1 is `second`. The value walks the slots from one
    to the other, so two neighbouring slots blend through the slots between
    them. The empty slot 0 is never taken. The result is always an existing
    palette slot.
    """
    a = int(current) % 256
    b = int(second) % 256
    v = _unit(value)
    if a < 1 or b < 1 or a == b:
        return a if a >= 1 else b
    return int(round(a + (b - a) * v))


# ---------------------------------------------------------------- the design


@dataclass(frozen=True)
class Sample:
    """What the brush is at one point of a stroke."""

    size: int
    exponent: float | None  # None: the sphere, the brush as it is now
    colour: int


@dataclass(frozen=True)
class Design:
    """The three curves, which of them are on, and the second colour."""

    thickness_on: bool = True
    roundness_on: bool = False
    colour_on: bool = False
    thickness: Curve = taper()
    roundness: Curve = flat(0.5)
    colour: Curve = ramp()
    second: int = 5

    def applies(self) -> bool:
        """True when a row is on, so a stroke is drawn again on release."""
        return self.thickness_on or self.roundness_on or self.colour_on

    def curve(self, row: str) -> Curve:
        return getattr(self, row)

    def row_on(self, row: str) -> bool:
        return bool(getattr(self, f"{row}_on"))

    def with_curve(self, row: str, curve: Curve) -> Design:
        return replace(self, **{row: curve})

    def with_row(self, row: str, on: bool) -> Design:
        return replace(self, **{f"{row}_on": bool(on)})

    def with_second(self, slot: int) -> Design:
        return replace(self, second=max(1, int(slot) % 256))

    def sample(self, t: float, brush: int, current: int, tool: str) -> Sample:
        """The brush at stroke fraction `t`. An off row leaves that part as it is."""
        size = sized(self.thickness.value_at(t), brush) if self.thickness_on else max(1, int(brush))
        exponent = exponent_of(self.roundness.value_at(t)) if self.roundness_on else None
        if self.colour_on and tool != ERASE:
            colour = colour_slot(self.colour.value_at(t), current, self.second)
        else:
            colour = int(current) % 256
        return Sample(size, exponent, colour)


def default_design() -> Design:
    """Thickness on, tapering to one voxel: a tentacle. The other rows off."""
    return Design()


def reset_curve(row: str) -> Curve:
    """The curve a row goes back to."""
    if row == "thickness":
        return taper()
    if row == "roundness":
        return flat(0.5)
    return ramp()


# A few shapes to start from. Plain turns every row off.
PRESETS: dict[str, Design] = {
    "tentacle": Design(True, False, False, taper(), flat(0.5), ramp(), 5),
    "horn": Design(
        True, True, False,
        Curve(1.0, 0.42, 0.96, 0.78, 0.42, 0.0),
        Curve(0.55, 0.35, 0.42, 0.68, 0.12, 0.0),
        ramp(), 5,
    ),
    "bead": Design(
        True, False, False,
        Curve(0.12, 0.32, 1.0, 0.68, 1.0, 0.12),
        flat(0.5), ramp(), 5,
    ),
    "plain": Design(False, False, False, flat(1.0), flat(0.5), flat(0.0), 5),
}
PRESET_ORDER = ("tentacle", "horn", "bead", "plain")
PRESET_LABELS = {"tentacle": "Tentacle", "horn": "Horn", "bead": "Bead", "plain": "Plain"}


def same_preset(design: Design, name: str) -> bool:
    """True when `design` is that preset, apart from the second colour."""
    other = PRESETS[name]
    return (
        design.thickness_on == other.thickness_on
        and design.roundness_on == other.roundness_on
        and design.colour_on == other.colour_on
        and design.thickness == other.thickness
        and design.roundness == other.roundness
        and design.colour == other.colour
    )


# ---------------------------------------------------------------- save


def pack(design: Design) -> str:
    """One line for the settings file."""
    flags = f"{int(design.thickness_on)}{int(design.roundness_on)}{int(design.colour_on)}"

    def part(curve: Curve) -> str:
        return ",".join(f"{v:.4f}" for v in curve.as_tuple())

    return f"{flags}|{int(design.second)}|{part(design.thickness)}|{part(design.roundness)}|{part(design.colour)}"


def unpack(text: str | None) -> Design:
    """The design in `text`, or the default when the text is missing or broken."""
    base = default_design()
    if not text or text.count("|") != 4:
        return base
    flags, second, thick, roundness, colour = text.split("|")
    if len(flags) != 3 or any(c not in "01" for c in flags):
        return base
    try:
        slot = int(second)
        curves = tuple(_curve_of(part) for part in (thick, roundness, colour))
    except ValueError:
        return base
    if any(c is None for c in curves) or slot < 1 or slot > 255:
        return base
    return Design(flags[0] == "1", flags[1] == "1", flags[2] == "1", *curves, slot)


def _curve_of(text: str) -> Curve | None:
    bits = text.split(",")
    if len(bits) != 6:
        return None
    return Curve(*(float(b) for b in bits))


# ---------------------------------------------------------------- replay


def face_line(a, b) -> list[tuple[int, int, int]]:
    """Cells from a to b, each sharing a face with the last. Both ends included."""
    a = (int(a[0]), int(a[1]), int(a[2]))
    b = (int(b[0]), int(b[1]), int(b[2]))
    if a == b:
        return [a]
    cur = list(a)
    out = [a]
    guard = sum(abs(b[i] - a[i]) for i in range(3)) + 1
    while tuple(cur) != b and guard > 0:
        guard -= 1
        axis = max(range(3), key=lambda i: abs(b[i] - cur[i]))
        if cur[axis] == b[axis]:
            break
        cur[axis] += 1 if b[axis] > cur[axis] else -1
        out.append((cur[0], cur[1], cur[2]))
    return out


def densify(path) -> list[tuple[int, int, int]]:
    """The path as a face-joined line, so a taper cannot break into pieces."""
    cells = [tuple(int(v) for v in c) for c in path]
    if not cells:
        return []
    out = [cells[0]]
    for cell in cells[1:]:
        if cell == out[-1]:
            continue
        out.extend(face_line(out[-1], cell)[1:])
    return out


def parameters(runs) -> list[list[float]]:
    """Stroke fraction at each cell of each run, by distance along the runs. A
    jump between two runs adds no length."""
    walked, total = [], 0.0
    for cells in runs:
        out = [total]
        for a, b in zip(cells, cells[1:]):
            total += math.dist(a, b)
            out.append(total)
        walked.append(out)
    total = total or 1.0
    return [[d / total for d in run] for run in walked]


def _runs(path) -> list[list[tuple[int, int, int]]]:
    """The path as runs of face-joined cells. `path` is one list of cells or a
    list of runs: the live stroke starts a new run where it jumped (a gap past
    an edge, another face), and a replay must not draw across that jump."""
    path = list(path)
    if path and len(path[0]) and not np.isscalar(path[0][0]):
        runs = path
    else:
        runs = [path]
    return [d for d in (densify(r) for r in runs) if d]


def replay(voxels, tool: str, path, brush: int, design: Design, colour: int, set_many,
           mirror=(False, False, False), write_brush=None) -> int:
    """Draw `path` again with the curves spread over its length.

    `path` is a list of cells, or a list of runs of cells. `set_many(cells,
    values)` writes voxels and is what the undo step records. `write_brush`,
    when given, writes a brush box the same way. A big brush stamps every few
    cells, not at every one (see the spacing below). Returns how many voxels
    the stamps changed; the colour pass may change more.
    """
    runs = _runs(path)
    if not runs or tool not in (ATTACH, ERASE, PAINT):
        return 0
    times = parameters(runs)
    mirror = (bool(mirror[0]), bool(mirror[1]), bool(mirror[2]))
    want_colour = design.colour_on and tool != ERASE
    before = None
    if want_colour:
        cells = [c for run in runs for c in run]
        lo, hi = _bounds(cells, brush, voxels.shape)
        before = (lo, hi, voxels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy())
    changed = 0
    for run, ts in zip(runs, times):
        since = math.inf  # the first cell of a run always takes a stamp
        last = len(run) - 1
        for k, (cell, t) in enumerate(zip(run, ts)):
            spec = design.sample(t, brush, colour, tool)
            r = spec.size - 1
            # A step of s over a sphere of radius r leaves a ridge s * s / (8 r)
            # high; half of sqrt(r) keeps it at 1/32 of a cell, so the side of
            # a stroke on the floor shows no beads, and a big brush still writes
            # several times less than one stamp per cell.
            spacing = max(1.0, 0.5 * math.sqrt(r))
            if k:
                since += math.dist(run[k - 1], cell)
            if since >= spacing or k == last:
                changed += _stamp(voxels, tool, cell, spec, set_many, mirror, write_brush)
                since = 0.0
    if before is not None:
        _paint_colours(voxels, before, runs, times, design, colour, set_many, mirror)
    return changed


def _bounds(cells, brush: int, shape) -> tuple[list[int], list[int]]:
    r = max(1, min(BRUSH_MAX, int(brush))) - 1
    lo = [max(0, min(c[k] for c in cells) - r) for k in range(3)]
    hi = [min(shape[k], max(c[k] for c in cells) + r + 1) for k in range(3)]
    return lo, hi


def _stamp(voxels, tool, cell, spec: Sample, set_many, mirror, write_brush) -> int:
    if write_brush is not None:  # one box write, size 1 too: the cell-list write scans the whole grid
        n = _stamp_box(voxels, tool, cell, spec, mirror, write_brush)
        if n >= 0:
            return n
    from .edit import apply_cells

    cells = shape_offsets(spec.size, 2.0 if spec.exponent is None else spec.exponent) + np.asarray(cell, dtype=np.int64)
    return apply_cells(voxels, tool, spec.colour, cells, set_many, mirror)


def _stamp_box(voxels, tool, cell, spec: Sample, mirror, write_brush) -> int:
    from .edit import mirror_images
    from .grid import BRUSH_ATTACH, BRUSH_ERASE, BRUSH_PAINT

    mode = {ATTACH: BRUSH_ATTACH, ERASE: BRUSH_ERASE, PAINT: BRUSH_PAINT}.get(tool)
    if mode is None:
        return -1
    full = shape_mask(spec.size, 2.0 if spec.exponent is None else spec.exponent)
    r = spec.size - 1
    grid = voxels.shape
    changed = 0
    for image in mirror_images(cell, grid, mirror):
        lo = [int(image[k]) - r for k in range(3)]
        hi = [int(image[k]) + r for k in range(3)]
        clip_lo = [max(lo[k], 0) for k in range(3)]
        clip_hi = [min(hi[k], grid[k] - 1) for k in range(3)]
        if any(clip_lo[k] > clip_hi[k] for k in range(3)):
            continue
        mask = full[tuple(slice(clip_lo[k] - lo[k], clip_hi[k] - lo[k] + 1) for k in range(3))]
        changed += int(write_brush(tuple(clip_lo), mask, mode, spec.colour, None))
    return changed


def _paint_colours(voxels, before, runs, times, design: Design, colour: int, set_many, mirror) -> None:
    """Give each voxel this stroke wrote the colour of the nearest path cell.

    The colour is worked out once per path cell. With Mirror on, the path's
    mirror images count as path cells too, so a mirrored voxel takes the
    colour of its own image of the path, not of the nearest source cell.
    """
    from .edit import mirror_images

    lo, hi, was = before
    now = voxels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    diff = now != was
    if not diff.any():
        return
    world = np.argwhere(diff) + np.asarray(lo, dtype=np.int64)
    points, slots = [], []
    for run, ts in zip(runs, times):
        for cell, t in zip(run, ts):
            slot = colour_slot(design.colour.value_at(t), colour, design.second)
            for image in mirror_images(cell, voxels.shape, mirror):
                points.append(image)
                slots.append(slot)
    points = np.asarray(points, dtype=np.float64)
    slots = np.asarray(slots, dtype=np.uint8)
    # Squared distances as |p|^2 - 2 p.q + |q|^2, a block of voxels at a time,
    # so the block's table stays near a million numbers whatever the path length.
    qq = (points * points).sum(axis=1)
    step = max(64, 1_000_000 // max(1, len(points)))
    out = np.empty(len(world), dtype=np.uint8)
    for i in range(0, len(world), step):
        part = world[i:i + step].astype(np.float64)
        dist = (part * part).sum(axis=1)[:, None] - 2.0 * part @ points.T + qq[None, :]
        out[i:i + step] = slots[dist.argmin(axis=1)]
    set_many(world, out)
