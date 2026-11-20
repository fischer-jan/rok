"""What the test files share: the fixture folder, the .vox byte builders, the test
doubles for the setters, the camera helpers and the loaders for the three models
that scripts/make_models.py writes."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from rok.core.select import Selection, footprint_mask
from rok.core.vox_read import read_scene, read_vox

FIXTURES = Path(__file__).parent / "fixtures"

# (stem, size, filled voxel count) from the fixture files themselves.
FIXTURE_META = {
    "robot": ((24, 30, 20), 1716),
    "bird": ((80, 68, 22), 9016),
    "car": ((40, 60, 14), 8252),
}


# ---------------------------------------------------------------- .vox bytes


def vox_chunk(cid: bytes, content: bytes, children: bytes = b"") -> bytes:
    return cid + struct.pack("<ii", len(content), len(children)) + content + children


def vox_file(chunks: list[bytes], version: int = 150) -> bytes:
    children = b"".join(chunks)
    return b"VOX " + struct.pack("<i", version) + vox_chunk(b"MAIN", b"", children)


def vox_size(sx: int, sy: int, sz: int) -> bytes:
    return vox_chunk(b"SIZE", struct.pack("<iii", sx, sy, sz))


def vox_xyzi(voxels: list[tuple[int, int, int, int]]) -> bytes:
    body = struct.pack("<i", len(voxels))
    body += bytes(b for vox in voxels for b in vox)
    return vox_chunk(b"XYZI", body)


# ------------------------------------------------------------ setter doubles


class ArraySetter:
    """Scalar test double: writes the array and counts the calls."""

    def __init__(self, voxels: np.ndarray) -> None:
        self.voxels = voxels
        self.calls: list[tuple[int, int, int, int]] = []

    def __call__(self, x: int, y: int, z: int, value: int) -> bool:
        self.calls.append((x, y, z, value))
        if int(self.voxels[x, y, z]) == value:
            return False
        self.voxels[x, y, z] = value
        return True


class ArraySetMany:
    """Array test double: writes the array, keeps the calls, returns the changed count."""

    def __init__(self, voxels: np.ndarray) -> None:
        self.voxels = voxels
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, cells: np.ndarray, values: np.ndarray) -> int:
        cells = np.asarray(cells).reshape(-1, 3)
        values = np.asarray(values, dtype=np.uint8)
        self.calls.append((cells.copy(), values.copy()))
        xs, ys, zs = cells[:, 0], cells[:, 1], cells[:, 2]
        changed = int(np.count_nonzero(self.voxels[xs, ys, zs] != values))
        self.voxels[xs, ys, zs] = values
        return changed

    @property
    def cells(self) -> set:
        return {tuple(c) for call in self.calls for c in call[0].tolist()}


def filled(voxels, colour=None) -> set:
    """The filled cells as a set, or only those of one colour."""
    solid = voxels != 0 if colour is None else voxels == colour
    return {tuple(c) for c in np.argwhere(solid).tolist()}


# ------------------------------------------------------------------- cameras

DOWN = (0.0, 0.0, -1.0)  # a top-down view looks along -z

# Orthographic top-down: clip = world / 10, so px = 100 + 10 * world (a 200 px view).
ORTHO = np.array(
    [
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, -0.1, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def px_of(world) -> tuple[float, float]:
    return (100.0 + 10.0 * world[0], 100.0 + 10.0 * world[1])


def down_ray(x, y, z=50.0):
    return (float(x), float(y), z), DOWN


def top_ray(x, y, z=50.0):
    """A ray straight down through the centre of column (x, y)."""
    return (x + 0.5, y + 0.5, z), DOWN


def top_down(points):
    """Orthographic top-down projection: 10 px per cell."""
    return np.asarray(points, dtype=np.float64)[:, :2] * 10.0


# --------------------------------------------------------------------- models


def model_voxels(stem: str) -> np.ndarray:
    return read_vox(FIXTURES / f"{stem}.vox").voxels


def model_palette_counts(stem: str):
    """(palette, how many voxels of each index) of a model."""
    model = read_vox(FIXTURES / f"{stem}.vox")
    return model.palette, np.bincount(model.voxels.ravel(), minlength=256)


def model_with_materials(stem: str):
    """(voxels copy, palette, materials) of a fixture's first model."""
    data = read_scene(FIXTURES / f"{stem}.vox")
    model = data.models[0]
    return model.voxels.copy(), np.asarray(data.palette, dtype=np.uint8), data.materials


def bird_slab():
    """A slab of the bird: the full columns under a top rectangle."""
    voxels = read_vox(FIXTURES / "bird.vox").voxels.copy()
    xs, ys = np.meshgrid(np.arange(30, 40), np.arange(30, 40), indexing="ij")
    cells = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 21)], axis=1)
    mask = footprint_mask(voxels, 2, cells)
    assert mask.sum() > 0
    sel = Selection(voxels.shape)
    sel.replace(mask)
    return voxels, sel
