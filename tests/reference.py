"""Slow, obviously-correct floods, and hand-built .vox graph chunks that
several test files share. The fast code in rok.core.region is checked
against the floods; the chunk builders let a test lay out an nTRN/nGRP/nSHP
graph, or an animated file, without going through vox_write.
"""

from helpers import vox_chunk, vox_file, vox_size, vox_xyzi

import hashlib
import struct

import numpy as np


def flood_2d_ref(mask, seed):
    """4-connected component of seed in a 2D bool mask, one cell per step."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    n_u, n_v = mask.shape
    u, v = int(seed[0]), int(seed[1])
    if not (0 <= u < n_u and 0 <= v < n_v) or not mask[u, v]:
        return out
    stack = [(u, v)]
    out[u, v] = True
    while stack:
        u, v = stack.pop()
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            un, vn = u + du, v + dv
            if 0 <= un < n_u and 0 <= vn < n_v and mask[un, vn] and not out[un, vn]:
                out[un, vn] = True
                stack.append((un, vn))
    return out


def connected_colour_ref(voxels, seed):
    """6-connected same-colour blob that contains seed, one cell per step."""
    out = np.zeros(voxels.shape, dtype=bool)
    nx, ny, nz = voxels.shape
    x, y, z = (int(seed[i]) for i in range(3))
    if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
        return out
    colour = int(voxels[x, y, z])
    if colour == 0:
        return out
    stack = [(x, y, z)]
    out[x, y, z] = True
    steps = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in steps:
            xn, yn, zn = x + dx, y + dy, z + dz
            if not (0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz):
                continue
            if out[xn, yn, zn] or int(voxels[xn, yn, zn]) != colour:
                continue
            out[xn, yn, zn] = True
            stack.append((xn, yn, zn))
    return out


def connected_filled_ref(voxels, seed):
    """6-connected filled blob of any colour that contains seed, one cell per step."""
    filled = np.asarray(voxels) != 0
    return connected_colour_ref(filled.astype(np.uint8), seed)


def _neighbour_steps(connectivity):
    if connectivity == 6:
        return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    if connectivity == 26:
        return [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
    raise ValueError(f"connectivity {connectivity} is not 6 or 26")


def dilate_ref(mask, n, connectivity=6):
    """Grow a 3D bool mask n times over 6 or 26 neighbours, one cell per step.

    The grid border counts as empty: nothing grows outside the array.
    """
    mask = np.asarray(mask, dtype=bool)
    nx, ny, nz = mask.shape
    steps = _neighbour_steps(connectivity)
    for _ in range(n):
        out = mask.copy()
        for x, y, z in np.argwhere(mask):
            for dx, dy, dz in steps:
                xn, yn, zn = x + dx, y + dy, z + dz
                if 0 <= xn < nx and 0 <= yn < ny and 0 <= zn < nz:
                    out[xn, yn, zn] = True
        mask = out
    return mask


def erode_ref(mask, n, connectivity=6):
    """Shrink a 3D bool mask n times; a cell on the grid wall counts as exposed."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.zeros(tuple(d + 2 for d in mask.shape), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = mask
    grown = dilate_ref(~padded, n, connectivity)
    return ~grown[1:-1, 1:-1, 1:-1]


def face_region_ref(voxels, voxel, normal):
    """(N, 3) face region with the same mask rule as region.face_region."""
    a = next(i for i in range(3) if int(normal[i]) != 0)
    s = 1 if int(normal[a]) > 0 else -1
    k = int(voxel[a])
    layer = np.take(voxels, k, axis=a)
    seed = tuple(int(voxel[i]) for i in range(3) if i != a)
    colour = int(layer[seed])
    if colour == 0:
        return np.zeros((0, 3), dtype=np.int64)
    kn = k + s
    if 0 <= kn < voxels.shape[a]:
        exposed = np.take(voxels, kn, axis=a) == 0
    else:
        exposed = np.ones(layer.shape, dtype=bool)
    region = flood_2d_ref((layer == colour) & exposed, seed)
    return np.insert(np.argwhere(region).astype(np.int64), a, k, axis=1)


def wall_region_ref(voxels, target, normal):
    """(N, 3) empty area of the layer target[a] that contains target."""
    a = next(i for i in range(3) if int(normal[i]) != 0)
    k = int(target[a])
    layer = np.take(voxels, k, axis=a)
    seed = tuple(int(target[i]) for i in range(3) if i != a)
    region = flood_2d_ref(layer == 0, seed)
    return np.insert(np.argwhere(region).astype(np.int64), a, k, axis=1)


def cells_sha1(cells):
    """sha1 of the sorted cell list serialised as "x,y,z;" strings."""
    rows = sorted(tuple(int(c) for c in row) for row in np.asarray(cells).reshape(-1, 3))
    text = "".join(f"{x},{y},{z};" for x, y, z in rows)
    return hashlib.sha1(text.encode("ascii")).hexdigest()


def cells_bbox(cells):
    """[lo, hi] of an (N, 3) cell list as plain int lists; [[], []] when empty."""
    cells = np.asarray(cells).reshape(-1, 3)
    if cells.shape[0] == 0:
        return [[], []]
    return [cells.min(axis=0).tolist(), cells.max(axis=0).tolist()]


# --- hand-built graph chunks (independent of vox_write) ----------------------


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<i", len(raw)) + raw


def _d(pairs) -> bytes:
    return struct.pack("<i", len(pairs)) + b"".join(_s(k) + _s(v) for k, v in pairs)


def _ntrn(node_id, child, layer=0, name="", frames=None) -> bytes:
    frames = frames if frames is not None else [[]]
    attrs = [("_name", name)] if name else []
    body = struct.pack("<i", node_id) + _d(attrs)
    body += struct.pack("<iiii", child, -1, layer, len(frames))
    for frame in frames:
        body += _d(frame)
    return vox_chunk(b"nTRN", body)


def _ngrp(node_id, children) -> bytes:
    body = struct.pack("<i", node_id) + _d([]) + struct.pack("<i", len(children))
    body += b"".join(struct.pack("<i", c) for c in children)
    return vox_chunk(b"nGRP", body)


def _nshp(node_id, model_ids) -> bytes:
    body = struct.pack("<i", node_id) + _d([]) + struct.pack("<i", len(model_ids))
    body += b"".join(struct.pack("<i", m) + _d([]) for m in model_ids)
    return vox_chunk(b"nSHP", body)


def _nshp_frames(node_id, entries) -> bytes:
    """An nSHP listing (model id, _f) entries, as a .vox file holds animation."""
    body = struct.pack("<i", node_id) + _d([]) + struct.pack("<i", len(entries))
    body += b"".join(struct.pack("<i", m) + _d([("_f", str(f))]) for m, f in entries)
    return vox_chunk(b"nSHP", body)


def _raw_cube() -> tuple[tuple[int, int, int], np.ndarray, list]:
    """A 3 x 4 x 5 model with six distinct voxels, no symmetry."""
    size = (3, 4, 5)
    voxels = [(0, 0, 0, 1), (2, 0, 0, 2), (0, 3, 0, 3), (0, 0, 4, 4), (1, 2, 3, 5), (2, 3, 4, 6)]
    arr = np.zeros(size, dtype=np.uint8)
    for x, y, z, i in voxels:
        arr[x, y, z] = i
    return size, arr, voxels


def _spin_file() -> bytes:
    """_raw_cube() as one plain, unanimated model: a fresh scene to bake a
    LINEAR rotation onto (Task 2.3's spin golden)."""
    size, arr, voxels = _raw_cube()
    return vox_file(
        [vox_size(*size), vox_xyzi(voxels), _ntrn(0, 1, layer=-1), _ngrp(1, [2]),
         _ntrn(2, 3, name="cube"), _nshp(3, [0])]
    )


def _animated_file() -> bytes:
    """The Phase 0 golden's bytes: a spinning part with two states, and a still part."""
    size, arr, voxels = _raw_cube()
    other = [(1, 1, 1, 9)]
    return vox_file(
        [vox_size(*size), vox_xyzi(voxels), vox_size(*size), vox_xyzi(other),
         vox_size(2, 2, 2), vox_xyzi([(0, 0, 0, 3)]),
         _ntrn(0, 1, layer=-1), _ngrp(1, [2, 4]),
         _ntrn(2, 3, name="spinner", frames=[[("_t", "0 0 4")], [("_r", "17"), ("_t", "0 0 4"), ("_f", "5")],
                                               [("_r", "98"), ("_t", "1 0 4"), ("_f", "10")]]),
         _nshp_frames(3, [(0, 0), (1, 5)]),
         _ntrn(4, 5, name="still", frames=[[("_t", "8 0 0")]]), _nshp(5, [2])]
    )
