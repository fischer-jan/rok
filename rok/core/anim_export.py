"""Animation exports to glTF.

Two kinds, both `.glb`:

- **Smooth**: each part is a node, with its parent as the parent node and
  its pivot as the node's origin, so the engine turns it about the pivot.
  Every clip is a glTF animation. A channel whose keys are all Hold writes
  STEP keys; the others are sampled once per clip frame and blend linearly.
  A part with frames, or with Show keys, holds its meshes in child nodes,
  one per frame, shown and hidden by a scale of 1 or 0. Markers, the frame
  rate, the loop mode and root motion go into the animation's extras.
  Simplifies: the turns are smooth, not voxels; colour motions are lost.
- **Baked**: every frame of every clip is one mesh of the whole scene, posed
  in Voxel look exactly as Rok shows it, and the clip shows one after the
  other (STEP scale keys). Frames that look the same share one mesh.
  Simplifies: big files; colour motions are lost.

Root motion off (a clip's setting, 6.4) plays the clip in place: a root
part's moves along X and Y (the ground) stay out of the export.

Both take `merge`: neighbouring faces of one colour become one rectangle
(mesh.build_mesh_merged), for fewer triangles. Off by default.

Rok is Z up, glTF Y up: (x, y, z) -> (x, z, -y), one voxel one unit.
numpy and stdlib only.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
import numpy as np

from . import anim, voxel_look
from .mesh import MeshData, build_mesh, build_mesh_merged
from .mesh_export import Glb, _y_up
from .safe_write import write_bytes

SMOOTH, BAKED = "SMOOTH", "BAKED"
KINDS = (SMOOTH, BAKED)
SIMPLIFIES = {
    SMOOTH: "glTF turns parts smoothly, not as voxels; colour motions stay out",
    BAKED: "every frame is its own mesh, so the file is big; colour motions stay out",
}

# Rok's axes in glTF's: a rotation R becomes _C @ R @ _C.T.
_C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_HIDDEN = (0.0, 0.0, 0.0)
_SHOWN = (1.0, 1.0, 1.0)


# --- shared pieces ---------------------------------------------------------------


def sample_frames(clip: anim.Clip) -> list[int]:
    """The clip frames an export samples, in play order. A loop ends on its
    first frame again, so the engine's loop closes; Ping-pong is played out
    there and back."""
    n = max(1, int(clip.length))
    if clip.loop_mode == anim.LOOP:
        return list(range(n + 1))
    if clip.loop_mode == anim.PING_PONG and n > 1:
        return list(range(2 * (n - 1) + 1))
    return list(range(n))


def roots_of(scene) -> dict[int, int | None]:
    """uid -> the parent uid each part keeps in the export. A parent that is
    gone, the part itself, or a loop back makes the part a root."""
    models = {m.uid: m for m in scene.models}
    out: dict[int, int | None] = {}
    for m in scene.models:
        parent, seen = m.parent, {m.uid}
        chain_ok = True
        p = parent
        while p is not None:
            if p not in models or p in seen:
                chain_ok = False
                break
            seen.add(p)
            p = models[p].parent
        out[m.uid] = parent if chain_ok and parent is not None else None
    return out


def in_place(clip: anim.Clip, scene) -> anim.Clip:
    """The clip as an export plays it. With root motion off, a root part's
    moves along X and Y are dropped (6.4: the clip plays in place)."""
    if clip.root_motion:
        return clip
    parents = roots_of(scene)
    out = copy.deepcopy(clip)
    for track in out.tracks:
        if track.target.kind == anim.PART and parents.get(track.target.uid, 0) is None:
            for name in (anim.MOVE_X, anim.MOVE_Y):
                track.properties.pop(name, None)
    return out


def _clip_extras(clip: anim.Clip) -> dict:
    return {
        "fps": int(clip.fps),
        "length": int(clip.length),
        "loop_mode": clip.loop_mode,
        "root_motion": bool(clip.root_motion),
        "markers": [{"name": m.name, "frame": int(m.time), "time": m.time / max(1, clip.fps)}
                    for m in sorted(clip.markers, key=lambda m: m.time)],
    }


def _quaternion(R: np.ndarray) -> np.ndarray:
    """A rotation matrix as glTF's quaternion (x, y, z, w)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _thin(times: list[float], values: np.ndarray, step: bool) -> tuple[list[float], np.ndarray]:
    """Drop the samples that change nothing: in a run of equal values (STEP)
    or on a straight line (LINEAR), only the ends stay."""
    if len(times) <= 2:
        return times, values
    keep = [0]
    for i in range(1, len(times) - 1):
        if step:
            if not np.allclose(values[i], values[keep[-1]], atol=1e-6):
                keep.append(i)
            continue
        a, b = keep[-1], i + 1
        f = (times[i] - times[a]) / (times[b] - times[a])
        if not np.allclose(values[i], values[a] + f * (values[b] - values[a]), atol=1e-5):
            keep.append(i)
    keep.append(len(times) - 1)
    return [times[i] for i in keep], values[keep]


@dataclass
class _Animation:
    """One glTF animation as it is built."""

    name: str
    extras: dict
    samplers: list = field(default_factory=list)
    channels: list = field(default_factory=list)

    def add(self, glb: Glb, node: int, path: str, times: list[float], values: np.ndarray, step: bool) -> None:
        times, values = _thin(times, np.asarray(values, dtype=np.float32), step)
        t = np.asarray(times, dtype=np.float32)
        kind = {"translation": "VEC3", "scale": "VEC3", "rotation": "VEC4"}[path]
        self.samplers.append({
            "input": glb.accessor(t, "SCALAR", bounds=True),
            "output": glb.accessor(values.astype(np.float32), kind),
            "interpolation": "STEP" if step else "LINEAR",
        })
        self.channels.append({"sampler": len(self.samplers) - 1, "target": {"node": node, "path": path}})

    def json(self) -> dict:
        return {"name": self.name, "samplers": self.samplers, "channels": self.channels, "extras": self.extras}


def _seconds(clip: anim.Clip, frames: list[int]) -> list[float]:
    return [k / max(1, clip.fps) for k in frames]


# --- smooth -------------------------------------------------------------------------


def _pivot(model, clip: anim.Clip | None) -> np.ndarray:
    return anim.pivot_of(clip.track_of(model.uid) if clip is not None else None, model)


def _all_hold(track: anim.Track | None, names) -> bool:
    return all(k.ease == anim.HOLD for n in names for k in (track.keys(n) if track else ()))


def _has_keys(track: anim.Track | None, names) -> bool:
    return track is not None and any(track.keys(n) for n in names)


def smooth_glb(scene, merge: bool = False) -> bytes:
    """The scene and its clips as a smooth glTF: see the module's text."""
    if not scene.models:
        raise ValueError("there are no parts to export")
    clips = [in_place(c, scene) for c in scene.clips]
    parents = roots_of(scene)
    models = {m.uid: m for m in scene.models}
    # A part's pivot is the same in every clip (it belongs to the part); an old
    # file's track anchor comes from the first clip that has one.
    pivots = {}
    for m in scene.models:
        track = next((c.track_of(m.uid) for c in clips if c.track_of(m.uid) is not None), None)
        pivots[m.uid] = anim.pivot_of(track, m)

    def origin(uid: int) -> np.ndarray:
        m = models[uid]
        return np.asarray(m.position, dtype=np.float64) + pivots[uid]

    def rest_translation(uid: int) -> np.ndarray:
        parent = parents[uid]
        return origin(uid) - (origin(parent) if parent is not None else 0.0)

    glb = Glb(scene.palette)
    nodes: list[dict] = []
    part_node: dict[int, int] = {}
    holders: dict[int, list[int]] = {}  # uid -> the nodes of frames 0, 1, ...
    for m in scene.models:
        part_node[m.uid] = len(nodes)
        nodes.append({"name": m.name or m.display_name(scene.models.index(m)),
                      "translation": [float(v) for v in _y_up(rest_translation(m.uid))]})
    for i, m in enumerate(scene.models):
        node = nodes[part_node[m.uid]]
        children = [part_node[c.uid] for c in scene.models if parents[c.uid] == m.uid]
        name = node["name"]
        grids = [m.grid.voxels] + list(m.states)
        shown = not scene.is_hidden(m) and not m.empty
        meshes = []
        build = build_mesh_merged if merge else build_mesh
        for k, voxels in enumerate(grids):
            data = build(np.asarray(voxels)) if shown else None
            ok = data is not None and data.vertices.shape[0] > 0
            meshes.append(glb.mesh(f"{name} {k + 1}" if len(grids) > 1 else name, data, -pivots[m.uid]) if ok else None)
        needs_holders = len(grids) > 1 or any(_has_keys(c.track_of(m.uid), (anim.VISIBLE,)) for c in clips)
        if needs_holders:
            holders[m.uid] = []
            for k, mesh in enumerate(meshes):
                h = {"name": f"{name} frame {k + 1}"}
                if mesh is not None:
                    h["mesh"] = mesh
                if k:
                    h["scale"] = list(_HIDDEN)
                holders[m.uid].append(len(nodes))
                children.append(len(nodes))
                nodes.append(h)
        elif meshes[0] is not None:
            node["mesh"] = meshes[0]
        if children:
            node["children"] = children

    moving = [m.uid for m in scene.models
              if any(_has_keys(c.track_of(m.uid), anim.MOVE + anim.TURN + (anim.SCALE,)) for c in clips)]
    animations = []
    for clip in clips:
        frames = sample_frames(clip)
        times = _seconds(clip, frames)
        a = _Animation(clip.name, _clip_extras(clip))
        for uid in moving:
            m, track, node = models[uid], clip.track_of(uid), part_node[uid]
            at = [clip.time_at(k) for k in frames]
            rest = rest_translation(uid)
            moves = np.array([_y_up(rest + np.array(anim.triple(track, anim.MOVE, t))) for t in at])
            a.add(glb, node, "translation", times, moves, _all_hold(track, anim.MOVE))
            quats = []
            for t in at:
                q = _quaternion(_C @ anim.matrix_from_euler(anim.triple(track, anim.TURN, t)) @ _C.T)
                if quats and float(np.dot(q, quats[-1])) < 0:
                    q = -q  # the short way from the sample before
                quats.append(q)
            a.add(glb, node, "rotation", times, np.array(quats), _all_hold(track, anim.TURN))
            sizes = np.array([np.array(anim.channel(track, anim.SCALE, t), dtype=np.float64)[[0, 2, 1]] / 100.0
                              for t in at])
            a.add(glb, node, "scale", times, sizes, _all_hold(track, (anim.SCALE,)))
        for uid, frame_nodes in holders.items():
            track = clip.track_of(uid)
            for k, node in enumerate(frame_nodes):
                values = []
                for f in frames:
                    t = clip.time_at(f)
                    on = bool(anim.channel(track, anim.VISIBLE, t))
                    state = int(anim.channel(track, anim.STATE, t)) if track is not None else 0
                    state = state if 0 <= state < len(frame_nodes) else 0
                    values.append(_SHOWN if on and state == k else _HIDDEN)
                a.add(glb, node, "scale", times, np.array(values), True)
        if a.channels:
            animations.append(a.json())
    roots = [part_node[m.uid] for m in scene.models if parents[m.uid] is None]
    return glb.dumps(nodes, roots, [a for a in animations], {"generator": "Rok", "kind": SMOOTH})


# --- baked --------------------------------------------------------------------------


def _quads(vertices: np.ndarray, uvs: np.ndarray) -> MeshData:
    """A MeshData of whole quads (four vertices each, in order)."""
    n = vertices.shape[0]
    return MeshData(vertices=vertices.astype(np.float32), loop_vertex_index=np.arange(n, dtype=np.int32),
                    loop_start=np.arange(0, n, 4, dtype=np.int32), loop_total=np.full(n // 4, 4, dtype=np.int32),
                    uvs=uvs.astype(np.float32))


def _merged(parts: list[tuple[MeshData, np.ndarray]]) -> MeshData | None:
    """One mesh of several, each moved by its offset (Z up, grid units)."""
    parts = [(d, o) for d, o in parts if d.vertices.shape[0]]
    if not parts:
        return None
    verts = np.concatenate([d.vertices + o.astype(np.float32) for d, o in parts]).astype(np.float32)
    uvs = np.concatenate([d.uvs for d, _o in parts]).astype(np.float32)
    return _quads(verts, uvs)


def _frame_parts(scene, clip: anim.Clip, t: float) -> list[tuple]:
    """What the scene shows at clip frame `t` in Voxel look: per shown part,
    (voxels, world cell of their (0, 0, 0)) or (voxels, None, matrix) for a
    part too big to turn in voxels (it turns smoothly, as the view shows it)."""
    fr = anim.evaluate(clip, scene, t)
    out = []
    for m in scene.models:
        if scene.is_hidden(m) or m.empty or not fr.visible.get(m.uid, True):
            continue
        source = fr.state.get(m.uid, 0)
        voxels = m.grid.voxels if not 0 < source <= len(m.states) else m.states[source - 1]
        pose = fr.poses.get(m.uid)
        position = np.asarray(m.position, dtype=np.int64)
        if pose is None or pose.is_rest():
            out.append((voxels, tuple(int(v) for v in position)))
            continue
        try:
            posed, at = voxel_look.posed(voxels, pose.linear, pose.pivot, pose.offset)
        except ValueError:
            out.append((voxels, None, pose.matrix(), tuple(int(v) for v in position)))
            continue
        out.append((posed, tuple(int(v) for v in position + np.asarray(at))))
    return out


def _key_of(parts: list[tuple]) -> str:
    h = hashlib.sha1()
    for part in parts:
        v = np.ascontiguousarray(part[0])
        h.update(repr((v.shape, part[1:] if part[1] is not None else (None, np.round(part[2], 6).tolist(),
                                                                         part[3]))).encode())
        h.update(v.tobytes())
    return h.hexdigest()


def _mesh_of(parts: list[tuple], merge: bool = False) -> MeshData | None:
    build = build_mesh_merged if merge else build_mesh
    pieces = []
    for part in parts:
        data = build(np.asarray(part[0]))
        if part[1] is not None:
            pieces.append((data, np.asarray(part[1], dtype=np.float32)))
            continue
        _v, _none, matrix, position = part
        if data.vertices.shape[0]:
            turned = data.vertices.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
            pieces.append((_quads(turned, data.uvs),
                           np.asarray(position, dtype=np.float32)))
    return _merged(pieces)


def baked_glb(scene, merge: bool = False) -> bytes:
    """Every frame of every clip as a mesh of the whole scene, shown in turn:
    see the module's text. The rest pose shows when no clip plays."""
    if not scene.models:
        raise ValueError("there are no parts to export")
    clips = [in_place(c, scene) for c in scene.clips]
    glb = Glb(scene.palette)
    nodes: list[dict] = []
    by_key: dict[str, int] = {}  # the look of a frame -> its node

    def node_for(parts: list[tuple], name: str, shown: bool) -> int:
        key = _key_of(parts)
        found = by_key.get(key)
        if found is not None:
            return found
        node = {"name": name}
        data = _mesh_of(parts, merge)
        if data is not None:
            node["mesh"] = glb.mesh(name, data)
        if not shown:
            node["scale"] = list(_HIDDEN)
        nodes.append(node)
        by_key[key] = len(nodes) - 1
        return len(nodes) - 1

    rest_parts = [(m.grid.voxels, tuple(int(v) for v in m.position)) for m in scene.models if not scene.is_hidden(m) and not m.empty]
    rest = node_for(rest_parts, "Rest", True)
    shows = []  # per clip: (clip, frames, the node each frame shows)
    for clip in clips:
        frames = sample_frames(clip)
        shown = [node_for(_frame_parts(scene, clip, f), f"{clip.name} {clip.time_at(f) + 1:.0f}", False)
                 for f in frames]
        shows.append((clip, frames, shown))
    if all(n == rest for _c, _f, shown in shows for n in shown) and not any(c.has_keys() for c in clips):
        raise ValueError("no clip has keys: export a mesh instead")
    animations = []
    for clip, frames, shown in shows:
        times = _seconds(clip, frames)
        a = _Animation(clip.name, _clip_extras(clip))
        for node in range(len(nodes)):
            values = np.array([_SHOWN if n == node else _HIDDEN for n in shown])
            a.add(glb, node, "scale", times, values, True)
        animations.append(a.json())
    return glb.dumps(nodes, range(len(nodes)), animations, {"generator": "Rok", "kind": BAKED})


def write(path, scene, kind: str = SMOOTH, merge: bool = False) -> None:
    """Write `scene` and its clips to `path` (a .glb) as `kind`."""
    if kind not in KINDS:
        raise ValueError(f"unknown glTF kind {kind!r}")
    data = smooth_glb(scene, merge) if kind == SMOOTH else baked_glb(scene, merge)
    write_bytes(path, data)


__all__ = ["BAKED", "KINDS", "SIMPLIFIES", "SMOOTH", "baked_glb", "in_place", "roots_of", "sample_frames",
           "smooth_glb", "write"]
