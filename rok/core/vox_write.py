"""Write .vox files in a flat layout.

Chunk order: SIZE/XYZI per model, then SIZE/XYZI per extra state (models in
order, their states in order), root nTRN, nGRP, nTRN/nSHP per model, LAYR x N,
RGBA, one MATL per material table entry, then the extras (rOBJ, rCAM, NOTE and
any MATL that did not parse) verbatim. The first frame has no _r; rotations
are baked into the arrays on read. The .vox animation goes back as it
came (ModelData.frames and .states): later frames with _r, _t and _f, and one
nSHP entry per state with its _f. A model without either writes exactly what
was written before animation was kept.
"""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from . import grid_ops
from .material import encode_matl
from .safe_write import write_bytes
from .vox_read import ModelData, SceneData, VoxModel

VOX_VERSION = 150


def write_vox(path: str | Path, model: VoxModel) -> None:
    """Write a VoxModel to disk as a one-model .vox file. The old file stays
    until the new one is complete."""
    write_bytes(path, dumps(model))


def dumps(model: VoxModel) -> bytes:
    """Serialise a VoxModel as a one-model scene: default layers, no extras."""
    size = tuple(int(d) for d in model.size)
    position = tuple(-(d // 2) for d in size)
    data = SceneData(
        models=[ModelData(size=size, voxels=model.voxels, name="", position=position)],
        palette=model.palette,
        palette_tail=b"\0\0\0\0",
        layers=list(DEFAULT_LAYERS),
        extras=[],
    )
    return dumps_scene(data)


def write_scene(path: str | Path, data: SceneData) -> None:
    """Write a SceneData to disk. The old file stays until the new one is complete."""
    write_bytes(path, dumps_scene(data))


def dumps_scene(data: SceneData) -> bytes:
    """Serialise a SceneData to .vox bytes."""
    chunks: list[bytes] = []
    # Linked copies share one model; each shows it with its own rotation.
    # Copies with an animation are written on their own, baked.
    model_ids: list[int] = []
    rotations: list[np.ndarray] = []
    centres: list[tuple[int, int, int]] = []
    shared: dict[int, tuple[int, np.ndarray]] = {}  # link -> (model id, shape)
    next_id = 0
    plain = [m.link for m in data.models if m.link and not m.states and not m.frames]
    for model in data.models:
        R = np.eye(3, dtype=np.int64)
        if model.link and not model.states and not model.frames and plain.count(model.link) > 1:
            R = np.asarray(model.orient, dtype=np.int64)
            if model.link not in shared:
                source = grid_ops.apply_rotation(model.voxels, R.T)[0]
                chunks.extend(_model_chunks(source))
                shared[model.link] = (next_id, source)
                next_id += 1
            model_id, source = shared[model.link]
        else:
            source = model.voxels
            chunks.extend(_model_chunks(source))
            model_id = next_id
            next_id += 1
        # The reader places the shape by position = t - s - R (size // 2).
        s = _shift(source.shape, R)
        pivot = np.asarray(source.shape, dtype=np.int64) // 2
        t = np.asarray(model.position, dtype=np.int64) + s + R @ pivot
        model_ids.append(model_id)
        rotations.append(R)
        centres.append(tuple(int(v) for v in t))
    count = len(data.models)
    # The extra states follow the models: state j of model i gets the next id.
    # An entry whose array is the model's own or an earlier entry's shows that
    # frame again: it names the same model id, so the voxels are written once.
    state_ids: list[list[tuple[int, int]]] = []
    for i, model in enumerate(data.models):
        ids = []
        written: list[tuple[np.ndarray, int]] = [(model.voxels, model_ids[i])]
        for frame, voxels in model.states:
            same = next((mid for array, mid in written if array is voxels), None)
            if same is None:
                chunks.extend(_model_chunks(voxels))
                same = next_id
                written.append((voxels, same))
                next_id += 1
            ids.append((same, int(frame)))
        state_ids.append(ids)
    layers = data.layers if data.layers else list(DEFAULT_LAYERS)
    layer_ids = {_layer_id(content) for content in layers}
    chunks.append(_chunk(b"nTRN", _ntrn(0, 1, -1, "", (0, 0, 0))))
    chunks.append(_chunk(b"nGRP", _ngrp(1, [2 + 2 * i for i in range(count)])))
    for i, model in enumerate(data.models):
        # A model keeps its layer; a .vox file needs a LAYR for every id an nTRN names.
        layer = int(model.layer) if int(model.layer) in layer_ids else 0
        rotation = grid_ops.rotation_to_byte(rotations[i])
        chunks.append(_chunk(b"nTRN", _ntrn(2 + 2 * i, 3 + 2 * i, layer, model.name, centres[i], model.frames, rotation)))
        chunks.append(_chunk(b"nSHP", _nshp(3 + 2 * i, model_ids[i], model.state_frame, state_ids[i])))
    for content in layers:
        chunks.append(_chunk(b"LAYR", content))
    chunks.append(_chunk(b"RGBA", _rgba_content(data.palette, data.palette_tail)))
    for material_id, material in data.materials.items():
        chunks.append(_chunk(b"MATL", encode_matl(material_id, material)))
    for cid, content in data.extras:
        chunks.append(_chunk(cid, content))
    main = _chunk(b"MAIN", b"", b"".join(chunks))
    return b"VOX " + struct.pack("<i", VOX_VERSION) + main


# --- encoders ----------------------------------------------------------------


def _string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<i", len(raw)) + raw


def _dict(pairs: list[tuple[str, str]]) -> bytes:
    out = struct.pack("<i", len(pairs))
    for key, value in pairs:
        out += _string(key) + _string(value)
    return out


def _model_chunks(voxels: np.ndarray) -> list[bytes]:
    sx, sy, sz = (int(d) for d in voxels.shape)
    return [_chunk(b"SIZE", struct.pack("<iii", sx, sy, sz)), _chunk(b"XYZI", _xyzi_content(voxels))]


def _shift(shape, R: np.ndarray) -> np.ndarray:
    """apply_rotation's shift s for an array of `shape` turned by R."""
    s = np.zeros(3, dtype=np.int64)
    for i in range(3):
        col = int(np.nonzero(R[i])[0][0])
        if R[i, col] < 0:
            s[i] = int(shape[col]) - 1
    return s


def _ntrn(node_id: int, child: int, layer: int, name: str, t, frames=(), rotation: int = 4) -> bytes:
    """An nTRN: its first frame at `t`, turned by the `rotation` byte (4: not
    turned), then the .vox format's later `frames` (frame number, rotation byte,
    move from `t`). A frame already at `_f 0` (a bake's first sample) replaces
    that first frame instead of following it, so every `_f` appears once."""
    attrs = [("_name", name)] if name else []
    tx, ty, tz = (int(v) for v in t)
    frames = list(frames)
    zero = frames.pop(0) if frames and int(frames[0][0]) == 0 else None
    first_rotation = int(zero[1]) if zero is not None else int(rotation)
    first_move = zero[2] if zero is not None else (0, 0, 0)
    first = [("_r", str(first_rotation))] if first_rotation != 4 else []
    x, y, z = (a + int(b) for a, b in zip((tx, ty, tz), first_move))
    first += [("_t", f"{x} {y} {z}")] if (x, y, z) != (0, 0, 0) else []
    if zero is not None:
        first += [("_f", "0")]
    dicts = [_dict(first)]
    for number, rotation, move in frames:
        pairs = [] if int(rotation) == 4 else [("_r", str(int(rotation)))]
        x, y, z = (a + int(b) for a, b in zip((tx, ty, tz), move))
        pairs += [("_t", f"{x} {y} {z}"), ("_f", str(int(number)))]
        dicts.append(_dict(pairs))
    return (
        struct.pack("<i", node_id)
        + _dict(attrs)
        + struct.pack("<iiii", child, -1, layer, len(dicts))
        + b"".join(dicts)
    )


def _ngrp(node_id: int, children: list[int]) -> bytes:
    return (
        struct.pack("<i", node_id)
        + _dict([])
        + struct.pack("<i", len(children))
        + b"".join(struct.pack("<i", c) for c in children)
    )


def _nshp(node_id: int, model_id: int, frame: int = 0, states=()) -> bytes:
    """An nSHP: the model, then one entry per state (model id, _f frame). The
    frame numbers only show when there are states, as a .vox file holds it."""
    entries = [(model_id, int(frame))] + list(states)
    out = struct.pack("<i", node_id) + _dict([]) + struct.pack("<i", len(entries))
    for mid, number in entries:
        out += struct.pack("<i", mid) + _dict([("_f", str(number))] if states else [])
    return out


def _layr(layer_id: int, name: str) -> bytes:
    return struct.pack("<i", layer_id) + _dict([("_name", name)]) + struct.pack("<i", -1)


def _layer_id(content: bytes) -> int:
    """The id a LAYR chunk content starts with; -1 for a content too short to hold one."""
    if len(content) < 4:
        return -1
    return struct.unpack_from("<i", content, 0)[0]


# The eight LAYR chunks of a fresh .vox file, 26 bytes each.
DEFAULT_LAYERS: tuple[bytes, ...] = tuple(_layr(i, str(i)) for i in range(8))


def _chunk(cid: bytes, content: bytes, children: bytes = b"") -> bytes:
    return cid + struct.pack("<ii", len(content), len(children)) + content + children


def _xyzi_content(voxels: np.ndarray) -> bytes:
    xs, ys, zs = np.nonzero(voxels)
    count = int(xs.size)
    header = struct.pack("<i", count)
    if count == 0:
        return header
    raw = np.empty((count, 4), dtype=np.uint8)
    raw[:, 0] = xs.astype(np.uint8, copy=False)
    raw[:, 1] = ys.astype(np.uint8, copy=False)
    raw[:, 2] = zs.astype(np.uint8, copy=False)
    raw[:, 3] = voxels[xs, ys, zs]
    return header + raw.tobytes()


def _rgba_content(palette: np.ndarray, tail: bytes = b"\0\0\0\0") -> bytes:
    """File colours [0..254] are palette indices [1..255]; the last entry is the tail."""
    raw = np.zeros((256, 4), dtype=np.uint8)
    raw[0:255] = palette[1:256]
    tail = bytes(tail)[:4].ljust(4, b"\0")
    raw[255] = np.frombuffer(tail, dtype=np.uint8)
    return raw.tobytes()
