"""Where a shader may change the model: its scope.

A run changes the whole model, the selection, everything but the
selection, or a volume you place: a box or a sphere of cells, empty ones
too, so a shader can build in it. Filters narrow that down to voxels on some faces (top, sides,
bottom, edges, hollows) or of some colours. Room lets a shader that adds
voxels build that many cells out from what is left, and Feather fades the
shader out toward the edge of the scope. numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .morph import dilate, erode

MODEL = "MODEL"
SELECTION = "SELECTION"
NOT_SELECTION = "NOT_SELECTION"
VOLUME = "VOLUME"
SCOPES = (MODEL, SELECTION, NOT_SELECTION, VOLUME)
SCOPE_LABELS = {MODEL: "Model", SELECTION: "Selection", NOT_SELECTION: "Unselected", VOLUME: "Volume"}
BOX = "BOX"
SPHERE = "SPHERE"
SHAPES = (BOX, SPHERE)
SHAPE_LABELS = {BOX: "Box", SPHERE: "Sphere"}

FACES = ("TOP", "SIDES", "BOTTOM", "EDGES", "HOLLOWS")
FACE_LABELS = {"TOP": "Top", "SIDES": "Sides", "BOTTOM": "Bottom", "EDGES": "Edges", "HOLLOWS": "Hollows"}

ROOM_MAX = 8
FEATHER_MAX = 8


@dataclass(frozen=True)
class Where:
    """A run's scope. `scope` "" picks the selection when there is one."""

    scope: str = ""
    faces: tuple[str, ...] = ()
    colours: tuple[int, ...] = ()
    room: int = 0
    feather: int = 0
    volume: tuple[int, int, int, int, int, int] | None = None  # (x0, y0, z0, x1, y1, z1), inclusive
    shape: str = BOX

    def resolved(self, has_selection: bool) -> str:
        if self.scope == VOLUME and self.volume is not None:
            return VOLUME
        if self.scope in (SELECTION, NOT_SELECTION) and has_selection:
            return self.scope
        if self.scope == "" and has_selection:
            return SELECTION
        return MODEL

    def limits(self, has_selection: bool) -> bool:
        """True when the run cannot change every cell of the model."""
        return self.resolved(has_selection) != MODEL or bool(self.faces) or bool(self.colours)

    def summary(self, has_selection: bool) -> str:
        """One line for the panel: "Whole model", "Selection · top, edges", ..."""
        scope = self.resolved(has_selection)
        if scope == VOLUME:
            x0, y0, z0, x1, y1, z1 = self.volume
            parts = [f"{SHAPE_LABELS.get(self.shape, 'Box')} {x1 - x0 + 1} x {y1 - y0 + 1} x {z1 - z0 + 1}"]
        else:
            parts = ["Whole model" if scope == MODEL else SCOPE_LABELS[scope]]
        if self.faces:
            parts.append(", ".join(FACE_LABELS[f].lower() for f in FACES if f in self.faces))
        if self.colours:
            n = len(self.colours)
            parts.append(f"{n} colour{'s' if n != 1 else ''}")
        if self.room and self.limits(has_selection):
            parts.append(f"room {self.room}")
        if self.feather and self.limits(has_selection):
            parts.append(f"feather {self.feather}")
        return " · ".join(parts)


def parse_faces(text: str) -> tuple[str, ...]:
    """"TOP,EDGES" -> ("TOP", "EDGES"), in FACES order; unknown names are dropped."""
    names = {t.strip().upper() for t in (text or "").split(",")}
    return tuple(f for f in FACES if f in names)


def parse_volume(text: str) -> tuple[int, int, int, int, int, int] | None:
    """"x0,y0,z0,x1,y1,z1" -> the volume with each low end first, or None."""
    try:
        v = [int(float(t)) for t in (text or "").split(",")]
    except ValueError:
        return None
    if len(v) != 6:
        return None
    lo = [min(v[i], v[i + 3]) for i in range(3)]
    hi = [max(v[i], v[i + 3]) for i in range(3)]
    return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])


def volume_mask(shape, volume, kind: str = BOX) -> np.ndarray:
    """The cells of a box or a sphere (the ellipsoid inside the box) in a grid
    of `shape`. A volume past the grid is cut at its walls."""
    out = np.zeros(tuple(shape), dtype=bool)
    lo = np.maximum(np.asarray(volume[:3]), 0)
    hi = np.minimum(np.asarray(volume[3:]), np.asarray(shape) - 1)
    if np.any(hi < lo):
        return out
    box = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
    if kind != SPHERE:
        out[box] = True
        return out
    centre = (np.asarray(volume[:3], dtype=np.float64) + np.asarray(volume[3:])) / 2.0
    radius = (np.asarray(volume[3:], dtype=np.float64) - np.asarray(volume[:3]) + 1.0) / 2.0
    x, y, z = np.ogrid[box]
    d = ((x - centre[0]) / radius[0]) ** 2 + ((y - centre[1]) / radius[1]) ** 2 + ((z - centre[2]) / radius[2]) ** 2
    out[box] = d <= 1.0
    return out


def parse_colours(text: str) -> tuple[int, ...]:
    """"3, 9" -> (3, 9): palette indices 1-255, each once, in order."""
    out: list[int] = []
    for t in (text or "").split(","):
        t = t.strip()
        if t.isdigit() and 0 < int(t) < 256 and int(t) not in out:
            out.append(int(t))
    return tuple(out)


def _open(filled: np.ndarray, axis: int, step: int) -> np.ndarray:
    """Filled cells whose neighbour along `axis` by `step` is empty; the grid's
    wall counts as empty."""
    neighbour = np.zeros_like(filled)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if step > 0:
        dst[axis] = slice(0, -1)
        src[axis] = slice(1, None)
    else:
        dst[axis] = slice(1, None)
        src[axis] = slice(0, -1)
    neighbour[tuple(dst)] = filled[tuple(src)]
    return filled & ~neighbour


def face_mask(voxels: np.ndarray, face: str) -> np.ndarray:
    """The filled voxels of one kind of face.

    TOP, SIDES and BOTTOM: open above, toward x or y, below. EDGES: open on two
    sides at a right angle (an outer edge or corner). HOLLOWS: next to a gap or
    pit at most two cells wide, the cells a closing fills.
    """
    filled = np.asarray(voxels) != 0
    if face == "TOP":
        return _open(filled, 2, 1)
    if face == "BOTTOM":
        return _open(filled, 2, -1)
    if face == "SIDES":
        return _open(filled, 0, 1) | _open(filled, 0, -1) | _open(filled, 1, 1) | _open(filled, 1, -1)
    if face == "EDGES":
        opens = [(_open(filled, a, 1), _open(filled, a, -1)) for a in range(3)]
        out = np.zeros_like(filled)
        for i in range(3):
            for j in range(i + 1, 3):
                out |= (opens[i][0] | opens[i][1]) & (opens[j][0] | opens[j][1])
        return out
    if face == "HOLLOWS":
        gaps = erode(dilate(filled, 1, round_corners=True), 1, round_corners=True) & ~filled
        return filled & dilate(gaps, 1)
    raise ValueError(f"no face {face!r}")


def allowed(voxels: np.ndarray, where: Where, selection: np.ndarray | None = None) -> np.ndarray | None:
    """The cells a run may change, or None for every cell (the whole model, no
    filter). Room adds the empty cells up to that many steps from the rest; with
    Top as the only face it adds only the cells straight above them, so grass or
    snow on the tops of one colour does not spread over its neighbours."""
    voxels = np.asarray(voxels)
    has_selection = selection is not None and bool(np.any(selection))
    if not where.limits(has_selection):
        return None
    filled = voxels != 0
    scope = where.resolved(has_selection)
    if scope == VOLUME:
        # Every cell of the volume, empty ones too; a filter keeps to its voxels.
        region = volume_mask(voxels.shape, where.volume, where.shape)
        if where.faces or where.colours:
            region &= filled
    elif scope == SELECTION:
        region = filled & selection
    elif scope == NOT_SELECTION:
        region = filled & ~selection
    else:
        region = filled.copy()
    if where.faces:
        on = np.zeros_like(filled)
        for face in where.faces:
            on |= face_mask(voxels, face)
        region &= on
    if where.colours:
        region &= np.isin(voxels, np.asarray(where.colours, dtype=voxels.dtype))
    if where.room > 0 and region.any():
        if tuple(where.faces) == ("TOP",):
            region |= _above(region, int(where.room)) & ~filled
        else:
            region |= dilate(region, int(where.room)) & ~filled
    return region


def _above(region: np.ndarray, room: int) -> np.ndarray:
    """The cells up to `room` steps straight above `region` (+z)."""
    out = np.zeros_like(region)
    for k in range(1, room + 1):
        out[:, :, k:] |= region[:, :, :-k]
    return out


def _surface(filled: np.ndarray) -> np.ndarray:
    """Filled cells with an empty neighbour by a face. The grid's wall is not
    empty: a model standing on the floor has no surface under it."""
    p = np.pad(filled, 1, constant_values=True)
    n = p.shape
    closed = np.ones(filled.shape, dtype=bool)
    for axis in range(3):
        for step in (-1, 1):
            index = [slice(1, n[k] - 1) for k in range(3)]
            index[axis] = slice(1 + step, n[axis] - 1 + step)
            closed &= p[tuple(index)]
    return filled & ~closed


def feather_weights(allowed: np.ndarray, filled: np.ndarray, feather: int) -> np.ndarray:
    """How much of a shader's change a cell keeps, 0..1: 1 at `feather` steps
    or more from the edge of the scope, less nearer to it. The edge is the
    surface outside the scope: the voxels just under a selected patch do not
    count, or a patch on a hull would be all edge."""
    weights = np.ones(allowed.shape, dtype=np.float32)
    feather = int(feather)
    if feather <= 0:
        return weights
    outside = _surface(filled) & ~allowed
    if not outside.any():
        return weights
    for k in range(feather, 0, -1):
        weights[dilate(outside, k)] = k / (feather + 1.0)
    return weights


def _noise(shape) -> np.ndarray:
    """A fixed value 0..1 per cell, from its coordinates: a fade dithers the
    same way every run, so the live preview does not flicker."""
    x, y, z = np.indices(shape, dtype=np.float64)
    return np.modf(np.abs(np.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453))[0]


def feathered(before: np.ndarray, after: np.ndarray, allowed: np.ndarray, feather: int,
              walls: np.ndarray | None = None) -> np.ndarray:
    """`after` with its changes thinned out toward the edge of `allowed`; with
    `walls` (a volume), toward the volume's walls instead."""
    if int(feather) <= 0:
        return after
    if walls is not None:
        weights = np.ones(allowed.shape, dtype=np.float32)
        outside = ~walls
        for k in range(int(feather), 0, -1):
            weights[dilate(outside, k)] = k / (int(feather) + 1.0)
    else:
        weights = feather_weights(allowed, np.asarray(before) != 0, feather)
    drop = (after != before) & (_noise(before.shape) >= weights)
    return np.where(drop, before, after).astype(np.uint8)
