"""Face-culled mesh generation. Vectorized numpy, no Python loop over voxels.

`build_mesh` writes one quad for each visible voxel face. `build_mesh_merged`
joins neighbouring faces of one colour into larger rectangles (greedy
meshing) for files that go to a game engine: far fewer triangles, the same
picture. Only exports use it; the viewport ray-marches the voxels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import CHUNK_SIZE

# Four corners of each outward face, relative to the voxel origin.
# Winding is counter-clockwise when seen from outside.
_FACE_CORNERS = np.array(
    [
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],  # +X
        [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],  # -X
        [[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],  # +Y
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # -Y
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],  # +Z
        [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]],  # -Z
    ],
    dtype=np.float32,
)

# Neighbor cell in the padded solid array for each face direction.
_NEIGHBOR_SLICES = (
    (slice(2, None), slice(1, -1), slice(1, -1)),  # +X
    (slice(0, -2), slice(1, -1), slice(1, -1)),  # -X
    (slice(1, -1), slice(2, None), slice(1, -1)),  # +Y
    (slice(1, -1), slice(0, -2), slice(1, -1)),  # -Y
    (slice(1, -1), slice(1, -1), slice(2, None)),  # +Z
    (slice(1, -1), slice(1, -1), slice(0, -2)),  # -Z
)

_CORE = (slice(1, -1), slice(1, -1), slice(1, -1))

# The palette texture is a 16 x 16 atlas of square tiles, one tile per palette index
# (index i at column i % 16, row i // 16, row 0 at the bottom). A face's four UVs span
# its tile, inset by half a texel so linear filtering never samples the next tile. The
# atlas exists so that View > Grid can darken each tile's outer texel ring: the grid
# is then part of the surface, with correct depth and fading with distance.
ATLAS_TILES = 16
TILE_PX = 32
ATLAS_PX = ATLAS_TILES * TILE_PX
_UV_INSET = 0.5 / ATLAS_PX
_TILE_UV = 1.0 / ATLAS_TILES
_TILE_CORNERS = np.array(
    [[_UV_INSET, _UV_INSET], [_TILE_UV - _UV_INSET, _UV_INSET],
     [_TILE_UV - _UV_INSET, _TILE_UV - _UV_INSET], [_UV_INSET, _TILE_UV - _UV_INSET]],
    dtype=np.float32,
)


def tile_uvs(index: np.ndarray) -> np.ndarray:
    """The four UV corners of the palette tile of each index, (N, 4, 2) float32."""
    index = np.asarray(index).astype(np.int64).reshape(-1)
    origin = np.stack((index % ATLAS_TILES, index // ATLAS_TILES), axis=1)
    origin = origin.astype(np.float32) * np.float32(_TILE_UV)
    return origin[:, None, :] + _TILE_CORNERS[None, :, :]


@dataclass
class MeshData:
    """Flat vertex and loop buffers of the mesh."""

    vertices: np.ndarray  # (N, 3) float32
    loop_vertex_index: np.ndarray  # (N,) int32
    loop_start: np.ndarray  # (Q,) int32
    loop_total: np.ndarray  # (Q,) int32
    uvs: np.ndarray  # (N, 2) float32

    @property
    def quad_count(self) -> int:
        return int(self.loop_start.shape[0])


def build_mesh(voxels: np.ndarray) -> MeshData:
    """Face-culled mesh for a whole dense grid."""
    return _build(voxels, emit=None)


def build_chunk_mesh(
    voxels: np.ndarray,
    chunk: tuple[int, int, int],
    chunk_size: int = CHUNK_SIZE,
    occluder: np.ndarray | None = None,
    skip: np.ndarray | None = None,
) -> MeshData:
    """Face-culled mesh for one chunk. Neighbours outside the chunk still cull faces.

    For a render of overlapping models: `occluder`
    marks cells other models fill, which hide a face against them as an own
    voxel does, over the whole grid and a one-cell border (index i + 1 is
    cell i); `skip` marks own cells another model draws, over the grid."""
    cx, cy, cz = chunk
    sx, sy, sz = voxels.shape
    x0 = cx * chunk_size
    y0 = cy * chunk_size
    z0 = cz * chunk_size
    x1 = min(x0 + chunk_size, sx)
    y1 = min(y0 + chunk_size, sy)
    z1 = min(z0 + chunk_size, sz)
    hx0, hy0, hz0 = max(x0 - 1, 0), max(y0 - 1, 0), max(z0 - 1, 0)
    hx1, hy1, hz1 = min(x1 + 1, sx), min(y1 + 1, sy), min(z1 + 1, sz)
    window = voxels[hx0:hx1, hy0:hy1, hz0:hz1]
    emit = np.zeros(window.shape, dtype=bool)
    emit[x0 - hx0 : x1 - hx0, y0 - hy0 : y1 - hy0, z0 - hz0 : z1 - hz0] = True
    if skip is not None:
        emit &= ~skip[hx0:hx1, hy0:hy1, hz0:hz1]
    # The occluder around the window, one cell wider on each side, as _build pads.
    extra = occluder[hx0 : hx1 + 2, hy0 : hy1 + 2, hz0 : hz1 + 2] if occluder is not None else None
    mesh = _build(window, emit=emit, occluder=extra)
    if mesh.vertices.shape[0]:
        mesh.vertices += np.array([hx0, hy0, hz0], dtype=np.float32)
    return mesh


def _build(voxels: np.ndarray, emit: np.ndarray | None, occluder: np.ndarray | None = None) -> MeshData:
    solid = voxels != 0
    padded_solid = np.pad(solid, 1, constant_values=False)
    if occluder is not None:
        padded_solid |= occluder  # other models' cells hide a face as an own voxel does
    if emit is None:
        emit_core = solid
    else:
        emit_core = solid & emit
    padded_emit = np.pad(emit_core, 1, constant_values=False)
    core_emit = padded_emit[_CORE]

    vert_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []

    for corners, neighbor in zip(_FACE_CORNERS, _NEIGHBOR_SLICES):
        mask = core_emit & ~padded_solid[neighbor]
        xs, ys, zs = np.nonzero(mask)
        if xs.size == 0:
            continue
        origins = np.stack(
            (xs.astype(np.float32), ys.astype(np.float32), zs.astype(np.float32)),
            axis=1,
        )
        quads = origins[:, None, :] + corners[None, :, :]
        vert_parts.append(quads.reshape(-1, 3))
        uv_parts.append(tile_uvs(voxels[xs, ys, zs]).reshape(-1, 2))

    if not vert_parts:
        return _empty_mesh()

    return _assemble(np.concatenate(vert_parts, axis=0), np.concatenate(uv_parts, axis=0))


def _assemble(vertices: np.ndarray, uvs: np.ndarray) -> MeshData:
    """A MeshData of whole quads: four vertices each, in order."""
    n_verts = int(vertices.shape[0])
    n_quads = n_verts // 4
    return MeshData(
        vertices=vertices,
        loop_vertex_index=np.arange(n_verts, dtype=np.int32),
        loop_start=np.arange(n_quads, dtype=np.int32) * 4,
        loop_total=np.full(n_quads, 4, dtype=np.int32),
        uvs=uvs,
    )


def build_mesh_merged(voxels: np.ndarray) -> MeshData:
    """The same surface as `build_mesh` with neighbouring faces of one colour
    joined into rectangles. Every quad still has four vertices, wound as in
    `build_mesh`, and keeps one palette tile's UVs, so a flat-colour atlas
    with nearest sampling looks the same. Faces of different colours never
    join. Where a large quad meets smaller ones its edge carries no vertex of
    theirs (a T-junction), which some engines show as a crack of one pixel."""
    voxels = np.asarray(voxels)
    solid = voxels != 0
    padded = np.pad(solid, 1, constant_values=False)
    vert_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    for d, (corners, neighbor) in enumerate(zip(_FACE_CORNERS, _NEIGHBOR_SLICES)):
        face = np.where(solid & ~padded[neighbor], voxels, 0)
        if not face.any():
            continue
        axis = d // 2
        u, v = (a for a in range(3) if a != axis)
        rects = _merge_rects(np.moveaxis(face, (axis, u, v), (0, 1, 2)))  # (R, 6): slice, i, j, w, h, colour
        origin = np.zeros((rects.shape[0], 3), dtype=np.float32)
        extent = np.ones((rects.shape[0], 3), dtype=np.float32)
        origin[:, axis], origin[:, u], origin[:, v] = rects[:, 0], rects[:, 1], rects[:, 2]
        extent[:, u], extent[:, v] = rects[:, 3], rects[:, 4]
        vert_parts.append((origin[:, None, :] + corners[None, :, :] * extent[:, None, :]).reshape(-1, 3))
        uv_parts.append(tile_uvs(rects[:, 5]).reshape(-1, 2))
    if not vert_parts:
        return _empty_mesh()
    return _assemble(np.concatenate(vert_parts, axis=0), np.concatenate(uv_parts, axis=0))


def _merge_rects(face: np.ndarray) -> np.ndarray:
    """Rectangles that tile the non-zero cells of `face` (slice, u, v), each of
    one colour: rows of (slice, u, v, width along u, height along v, colour).
    Greedy: from the first free cell, grow along u, then along v."""
    out: list[tuple[int, int, int, int, int, int]] = []
    _s, nu, nv = face.shape
    for s in range(face.shape[0]):
        free = face[s].copy()  # a cell leaves (becomes 0) once a rectangle holds it
        for i, j in zip(*np.nonzero(free)):
            colour = free[i, j]
            if colour == 0:
                continue
            w = 1
            while i + w < nu and free[i + w, j] == colour:
                w += 1
            h = 1
            while j + h < nv and (free[i:i + w, j + h] == colour).all():
                h += 1
            free[i:i + w, j:j + h] = 0
            out.append((s, int(i), int(j), w, h, int(colour)))
    return np.array(out, dtype=np.int64).reshape(-1, 6)


def _empty_mesh() -> MeshData:
    return MeshData(
        vertices=np.zeros((0, 3), dtype=np.float32),
        loop_vertex_index=np.zeros((0,), dtype=np.int32),
        loop_start=np.zeros((0,), dtype=np.int32),
        loop_total=np.zeros((0,), dtype=np.int32),
        uvs=np.zeros((0, 2), dtype=np.float32),
    )
