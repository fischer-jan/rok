"""Read .vox files: every model plus the nTRN/nGRP/nSHP graph.

The graph is flattened: each nSHP entry becomes one model with its rotation baked
into the array and its world position kept. nSHP entries that show the same
model (the .vox instances) become linked copies: they share a `link` id
and keep their rotation in `orient` (see core/link.py). MATL goes into a MaterialTable; a
MATL that does not parse passes through untouched with LAYR, rOBJ, rCAM and NOTE.
IMAP and unknown chunks are dropped. A file without RGBA gets the .vox
default palette.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct

import numpy as np

from . import grid_ops
from .material import MaterialTable, parse_matl
from .palette import default_palette

MAX_DIM = 256
# All models of one file together; each SIZE costs its full box in memory.
MAX_TOTAL_CELLS = 1 << 29
# The scene graph: a group may list one child many times, and each listing is
# one more model to bake, so a small file can name a huge scene.
MAX_SHAPES = 1 << 14
MAX_VISITS = 1 << 16
EXTRA_IDS = (b"MATL", b"rOBJ", b"rCAM", b"NOTE", b"MLGT")  # MLGT: Rok's light (core/light.py)

Cell = tuple[int, int, int]


class VoxError(Exception):
    """The file is not a readable .vox document."""


@dataclass
class VoxModel:
    """One dense voxel model plus its 256-colour palette."""

    size: Cell
    voxels: np.ndarray  # (x, y, z) uint8, 0 = empty
    palette: np.ndarray  # (256, 4) uint8 RGBA


@dataclass
class ModelData:
    """One model of a file: the array with its rotation baked and its world position.

    The .vox animation comes along. `frames` are the nTRN's frames after the
    first: (frame number, rotation byte, (dx, dy, dz)), the rotation relative to
    the baked voxels and the move relative to the first frame, both as Rok
    writes them back. `states` are the nSHP's models after the first: (frame
    number, voxels baked like `voxels`); `state_frame` is the first model's.
    """

    size: Cell
    voxels: np.ndarray  # (x, y, z) uint8
    name: str
    position: Cell
    layer: int = 0  # layer id of the nTRN above the nSHP
    frames: tuple[tuple[int, int, Cell], ...] = ()
    states: list[tuple[int, np.ndarray]] = field(default_factory=list)
    state_frame: int = 0
    link: int = 0  # the same non-zero id: linked copies of one shape
    orient: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))  # the shape's turn in this copy (link.py)


@dataclass
class SceneData:
    """Everything Rok keeps from a .vox file."""

    models: list[ModelData]
    palette: np.ndarray  # (256, 4) uint8, slot 0 = (0, 0, 0, 0)
    palette_tail: bytes = b"\0\0\0\0"
    layers: list[bytes] = field(default_factory=list)
    extras: list[tuple[bytes, bytes]] = field(default_factory=list)
    materials: MaterialTable = field(default_factory=MaterialTable)


def read_vox(path: str | Path) -> VoxModel:
    """Read the first model of a .vox file from disk."""
    return loads(Path(path).read_bytes())


def loads(data: bytes) -> VoxModel:
    """Parse .vox bytes and return the first model as a VoxModel."""
    scene = loads_scene(data)
    first = scene.models[0]
    return VoxModel(size=first.size, voxels=first.voxels, palette=scene.palette)


def read_scene(path: str | Path) -> SceneData:
    """Read a whole .vox file from disk."""
    return loads_scene(Path(path).read_bytes())


def loads_scene(data: bytes) -> SceneData:
    """Parse .vox bytes into a SceneData with the graph flattened.

    Every failure leaves this module as a VoxError, so a caller needs one clause.
    """
    try:
        return _parse_scene(data)
    except VoxError:
        raise
    except (struct.error, ValueError, IndexError, MemoryError, RecursionError) as exc:
        raise VoxError(f"cannot read this file: {exc}") from exc


def _parse_scene(data: bytes) -> SceneData:
    if len(data) < 20:
        raise VoxError("file too short")
    if data[:4] != b"VOX ":
        raise VoxError("not a VOX file")
    (version,) = struct.unpack_from("<i", data, 4)
    if version not in (150, 200):
        raise VoxError(f"unsupported VOX version {version}")
    if data[8:12] != b"MAIN":
        raise VoxError("missing MAIN chunk")
    content_size, children_size = struct.unpack_from("<ii", data, 12)
    if content_size < 0 or children_size < 0:
        raise VoxError("negative size in MAIN chunk")
    children_start = 20 + content_size
    children_end = children_start + children_size
    if children_end > len(data):
        raise VoxError("truncated MAIN chunk")

    raw: list[tuple[Cell, np.ndarray | None]] = []
    nodes: dict[int, tuple] = {}
    palette: np.ndarray | None = None
    palette_tail = b"\0\0\0\0"
    layers: list[bytes] = []
    extras: list[tuple[bytes, bytes]] = []
    materials: dict = {}
    total_cells = 0

    for cid, content in _iter_chunks(data, children_start, children_end):
        if cid == b"SIZE":
            size = _parse_size(content)
            total_cells += size[0] * size[1] * size[2]
            if total_cells > MAX_TOTAL_CELLS:
                raise VoxError(f"models too large together: more than {MAX_TOTAL_CELLS} cells")
            raw.append((size, None))
        elif cid == b"XYZI":
            if not raw:
                raise VoxError("XYZI before SIZE")
            size, voxels = raw[-1]
            if voxels is None:
                raw[-1] = (size, _parse_xyzi(content, size))
        elif cid == b"RGBA" and palette is None:
            palette = _parse_rgba(content)
            palette_tail = bytes(content[1020:1024])
        elif cid == b"nTRN":
            node = _parse_ntrn(content)
            nodes[node[0]] = node
        elif cid == b"nGRP":
            node = _parse_ngrp(content)
            nodes[node[0]] = node
        elif cid == b"nSHP":
            node = _parse_nshp(content)
            nodes[node[0]] = node
        elif cid == b"LAYR":
            layers.append(bytes(content))
        elif cid == b"MATL":
            # A MATL that does not parse stays an extra, verbatim and in place.
            try:
                material_id, material = parse_matl(content)
            except ValueError:
                extras.append((bytes(cid), bytes(content)))
            else:
                materials[material_id] = material
        elif cid in EXTRA_IDS:
            extras.append((bytes(cid), bytes(content)))

    if not raw:
        raise VoxError("no SIZE chunk")
    raw_models = [
        (size, voxels if voxels is not None else np.zeros(size, dtype=np.uint8))
        for size, voxels in raw
    ]
    if palette is None:
        palette = default_palette()

    models: list[ModelData] = []
    listed: set[int] = set()
    root = _find_root(nodes)
    if root is not None:
        entries = _walk(nodes, root)
        # Every entry bakes its own copy of the model it shows, so the copies
        # count against the same limit as the models.
        baked_cells = 0
        for shape_models, _n, _l, _R, _t, _a in entries:
            for model_id, _frame in shape_models:
                if not 0 <= model_id < len(raw_models):
                    raise VoxError(f"nSHP names model {model_id}, file has {len(raw_models)}")
                size = raw_models[model_id][0]
                baked_cells += size[0] * size[1] * size[2]
        if baked_cells > MAX_TOTAL_CELLS:
            raise VoxError(f"models shown too often together: more than {MAX_TOTAL_CELLS} cells")
        # A model that several plain nSHP entries show is one shape, linked.
        shown: dict[int, int] = {}
        for shape_models, _n, _l, _R, _t, above in entries:
            if len(shape_models) == 1 and not _later_frames_present(above):
                shown[shape_models[0][0]] = shown.get(shape_models[0][0], 0) + 1
        for shape_models, name, layer, R, t, above in entries:
            model_id, state_frame = shape_models[0]
            size, voxels = raw_models[model_id]
            baked, s = grid_ops.apply_rotation(voxels, R)
            pivot = np.asarray(size, dtype=np.int64) // 2
            position = t - s - R @ pivot
            # A model the shape names twice is one frame shown twice: the
            # entries share one array (VoxScene.from_data keys it once).
            states = []
            baked_of = {model_id: baked}
            for other_id, frame in shape_models[1:]:
                if other_id not in baked_of:
                    baked_of[other_id] = grid_ops.apply_rotation(raw_models[other_id][1], R)[0]
                states.append((frame, baked_of[other_id]))
                listed.add(other_id)
            models.append(
                ModelData(
                    size=tuple(int(d) for d in baked.shape),
                    voxels=baked,
                    name=name,
                    position=tuple(int(v) for v in position),
                    layer=layer,
                    frames=_later_frames(above, R, s, pivot, baked.shape, position),
                    states=states,
                    state_frame=state_frame,
                )
            )
            if shown.get(model_id, 0) > 1 and len(shape_models) == 1 and not _later_frames_present(above):
                models[-1].link = model_id + 1
                models[-1].orient = tuple(tuple(int(v) for v in row) for row in R)
            listed.add(model_id)
    for model_id, (size, voxels) in enumerate(raw_models):
        if model_id in listed:
            continue
        position = tuple(-(int(d) // 2) for d in size)
        models.append(ModelData(size=size, voxels=voxels, name="", position=position))

    return SceneData(
        models=models,
        palette=palette,
        palette_tail=palette_tail,
        layers=layers,
        extras=extras,
        materials=MaterialTable(materials),
    )


# --- the graph ---------------------------------------------------------------


def _find_root(nodes: dict[int, tuple]) -> int | None:
    """Node 0 when it is an nTRN, else the lowest nTRN no node lists as a child."""
    if not nodes:
        return None
    if 0 in nodes and nodes[0][1] == "nTRN":
        return 0
    children: set[int] = set()
    for node in nodes.values():
        if node[1] == "nTRN":
            children.add(node[3])
        elif node[1] == "nGRP":
            children.update(node[3])
    for node_id in sorted(nodes):
        if nodes[node_id][1] == "nTRN" and node_id not in children:
            return node_id
    return None


def _walk(nodes: dict[int, tuple], root: int) -> list[tuple]:
    """Depth-first from `root`, one entry per nSHP: ([(model id, _f frame), ...],
    name, layer, R (3, 3), t (3,), above). R and t place the first frame;
    `above` is (R, t of the nTRN's parent, the nTRN's frame dicts), for the
    later frames."""
    out: list[tuple] = []
    identity = np.eye(3, dtype=np.int64)
    zero = np.zeros(3, dtype=np.int64)
    # An explicit stack keeps the depth-first order and takes any chain length;
    # a recursive walk would overflow on a few thousand chained nodes. `path`
    # holds the nodes above, so a cycle ends; the visit cap ends a graph that
    # lists one child many times at every level.
    stack: list[tuple] = [(root, identity, zero, "", 0, frozenset(), None)]
    visits = 0
    while stack:
        node_id, R_p, t_p, name, layer, path, above = stack.pop()
        node = nodes.get(node_id)
        if node is None or node_id in path:
            continue
        visits += 1
        if visits > MAX_VISITS:
            raise VoxError(f"scene graph too large: more than {MAX_VISITS} nodes to visit")
        path = path | {node_id}
        kind = node[1]
        if kind == "nTRN":
            _, _, attrs, child, frames, node_layer = node
            frame = frames[0] if frames else {}
            try:
                R_l = grid_ops.rotation_from_byte(int(frame.get(b"_r", b"4")))
            except ValueError as exc:
                raise VoxError(f"bad _r value in node {node_id}") from exc
            t_l = _parse_triple(frame.get(b"_t", b"0 0 0"))
            # The model takes the name and layer of the nTRN right above its nSHP.
            node_name = attrs.get(b"_name", b"").decode("utf-8", "replace")
            stack.append((child, R_p @ R_l, R_p @ t_l + t_p, node_name, node_layer, path, (R_p, t_p, frames)))
        elif kind == "nGRP":
            for child in reversed(node[3]):
                stack.append((child, R_p, t_p, "", 0, path, None))
        elif kind == "nSHP" and node[3]:
            out.append((node[3], name, layer, R_p, t_p, above))
            if len(out) > MAX_SHAPES:
                raise VoxError(f"scene graph too large: more than {MAX_SHAPES} shapes")
    return out


def _later_frames_present(above) -> bool:
    return above is not None and len(above[2]) >= 2


def _frame_number(frame: dict[bytes, bytes], default: int) -> int:
    try:
        return int(frame.get(b"_f", str(default).encode()))
    except ValueError as exc:
        raise VoxError(f"bad _f value {frame.get(b'_f')!r}") from exc


def _later_frames(above, R0, s, pivot, baked_shape, position) -> tuple[tuple[int, int, Cell], ...]:
    """The nTRN's frames after the first, as (frame number, rotation byte,
    move): relative to the baked voxels and the first frame, so writing them
    back next to the baked voxels places every frame where the file had it.

    A voxel at raw cell p shows at R (p - pivot) + t in frame k; with b the
    baked cell (b = R0 p + s) that is Q (b - pivot_b) + T for Q = R R0^-1 and
    T = -Q s - R pivot + t + Q pivot_b, the centre Rok writes. The first
    frame's T is position + pivot_b; the move is T minus that."""
    if above is None:
        return ()
    R_p, t_p, frames = above
    if len(frames) < 2:
        return ()
    pivot_b = np.asarray(baked_shape, dtype=np.int64) // 2
    T0 = np.asarray(position, dtype=np.int64) + pivot_b
    R0_inv = R0.T  # a signed permutation: its inverse is its transpose
    out = []
    for index, frame in enumerate(frames[1:], start=1):
        try:
            R_l = grid_ops.rotation_from_byte(int(frame.get(b"_r", b"4")))
        except ValueError as exc:
            raise VoxError(f"bad _r value {frame.get(b'_r')!r}") from exc
        R = R_p @ R_l
        t = R_p @ _parse_triple(frame.get(b"_t", b"0 0 0")) + t_p
        Q = R @ R0_inv
        T = -Q @ s - R @ pivot + t + Q @ pivot_b
        move = tuple(int(v) for v in T - T0)
        out.append((_frame_number(frame, index), grid_ops.rotation_to_byte(Q), move))
    return tuple(out)


def _parse_triple(text: bytes) -> np.ndarray:
    parts = text.split()
    if len(parts) != 3:
        raise VoxError(f"bad _t value {text!r}")
    try:
        return np.array([int(p) for p in parts], dtype=np.int64)
    except ValueError as exc:
        raise VoxError(f"bad _t value {text!r}") from exc


def _parse_string(content: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 4 > len(content):
        raise VoxError("truncated STRING")
    (length,) = struct.unpack_from("<i", content, offset)
    offset += 4
    if length < 0 or offset + length > len(content):
        raise VoxError("truncated STRING")
    return content[offset : offset + length], offset + length


def _parse_dict(content: bytes, offset: int) -> tuple[dict[bytes, bytes], int]:
    if offset + 4 > len(content):
        raise VoxError("truncated DICT")
    (count,) = struct.unpack_from("<i", content, offset)
    offset += 4
    if count < 0:
        raise VoxError("negative DICT count")
    out: dict[bytes, bytes] = {}
    for _ in range(count):
        key, offset = _parse_string(content, offset)
        value, offset = _parse_string(content, offset)
        out[key] = value
    return out, offset


def _parse_int(content: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(content):
        raise VoxError("truncated node")
    return struct.unpack_from("<i", content, offset)[0], offset + 4


def _parse_ntrn(content: bytes) -> tuple:
    """(id, "nTRN", attrs, child id, [frame dicts], layer id)."""
    node_id, offset = _parse_int(content, 0)
    attrs, offset = _parse_dict(content, offset)
    child, offset = _parse_int(content, offset)
    _, offset = _parse_int(content, offset)  # reserved
    layer, offset = _parse_int(content, offset)
    count, offset = _parse_int(content, offset)
    frames = []
    for _ in range(max(count, 0)):
        frame, offset = _parse_dict(content, offset)
        frames.append(frame)
    return node_id, "nTRN", attrs, child, frames, layer


def _parse_ngrp(content: bytes) -> tuple:
    """(id, "nGRP", attrs, [child ids])."""
    node_id, offset = _parse_int(content, 0)
    attrs, offset = _parse_dict(content, offset)
    count, offset = _parse_int(content, offset)
    children = []
    for _ in range(max(count, 0)):
        child, offset = _parse_int(content, offset)
        children.append(child)
    return node_id, "nGRP", attrs, children


def _parse_nshp(content: bytes) -> tuple:
    """(id, "nSHP", attrs, [(model id, _f frame number)]): an animated shape lists
    one model per frame it changes at."""
    node_id, offset = _parse_int(content, 0)
    attrs, offset = _parse_dict(content, offset)
    count, offset = _parse_int(content, offset)
    models = []
    for index in range(max(count, 0)):
        model_id, offset = _parse_int(content, offset)
        model_attrs, offset = _parse_dict(content, offset)
        models.append((model_id, _frame_number(model_attrs, index)))
    return node_id, "nSHP", attrs, models


# --- chunks ------------------------------------------------------------------


def _iter_chunks(data: bytes, offset: int, end: int):
    """Yield (chunk_id, content) for each top-level chunk in [offset, end)."""
    while offset < end:
        if offset + 12 > end:
            raise VoxError("truncated chunk header")
        cid = data[offset : offset + 4]
        content_size, children_size = struct.unpack_from("<ii", data, offset + 4)
        if content_size < 0 or children_size < 0:
            # A negative size would move the walk backwards, or not at all.
            raise VoxError(f"negative size in {cid!r} chunk")
        content_start = offset + 12
        content_end = content_start + content_size
        children_end = content_end + children_size
        if children_end > end:
            raise VoxError(f"truncated {cid!r} chunk")
        yield cid, data[content_start:content_end]
        offset = children_end


def _parse_size(content: bytes) -> Cell:
    if len(content) < 12:
        raise VoxError("SIZE chunk too short")
    sx, sy, sz = struct.unpack_from("<iii", content, 0)
    for dim, name in ((sx, "x"), (sy, "y"), (sz, "z")):
        if dim < 1 or dim > MAX_DIM:
            raise VoxError(f"SIZE {name}={dim} is out of range 1..{MAX_DIM}")
    return int(sx), int(sy), int(sz)


def _parse_xyzi(content: bytes, size: Cell) -> np.ndarray:
    if len(content) < 4:
        raise VoxError("XYZI chunk too short")
    (count,) = struct.unpack_from("<i", content, 0)
    if count < 0:
        raise VoxError("negative voxel count")
    expected = 4 + count * 4
    if len(content) < expected:
        raise VoxError("XYZI chunk truncated")
    voxels = np.zeros(size, dtype=np.uint8)
    if count == 0:
        return voxels
    raw = np.frombuffer(content, dtype=np.uint8, offset=4, count=count * 4)
    raw = raw.reshape(count, 4)
    xs = raw[:, 0].astype(np.intp)
    ys = raw[:, 1].astype(np.intp)
    zs = raw[:, 2].astype(np.intp)
    idx = raw[:, 3]
    sx, sy, sz = size
    valid = (xs < sx) & (ys < sy) & (zs < sz) & (idx != 0)
    voxels[xs[valid], ys[valid], zs[valid]] = idx[valid]
    return voxels


def _parse_rgba(content: bytes) -> np.ndarray:
    if len(content) < 1024:
        raise VoxError("RGBA chunk too short")
    raw = np.frombuffer(content[:1024], dtype=np.uint8).reshape(256, 4)
    palette = np.zeros((256, 4), dtype=np.uint8)
    # The .vox format maps file colours [0..254] onto palette indices [1..255].
    palette[1:256] = raw[0:255]
    return palette
