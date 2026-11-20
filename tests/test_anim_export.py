"""rok/core/anim_export.py: the glTF exports, played back by a small glTF
reader and compared vertex by vertex with the pose Rok shows."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from rok.core import anim, anim_export, rokfile, voxel_look
from rok.core.mesh import build_mesh
from rok.core.mesh_export import _y_up

ANIM = Path(__file__).parent / "fixtures" / "anim"
COMPONENTS = {5126: np.float32, 5125: np.uint32, 5121: np.uint8}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


class Gltf:
    """A .glb read back: nodes, meshes, and animations sampled at a time."""

    def __init__(self, data: bytes) -> None:
        magic, version, length = struct.unpack_from("<4sII", data, 0)
        assert magic == b"glTF" and version == 2 and length == len(data)
        jlen, _ = struct.unpack_from("<I4s", data, 12)
        self.doc = json.loads(data[20:20 + jlen])
        self.bin = data[20 + jlen + 8:]
        self.nodes = self.doc["nodes"]
        self.parent = {c: i for i, n in enumerate(self.nodes) for c in n.get("children", ())}

    def accessor(self, index: int) -> np.ndarray:
        acc = self.doc["accessors"][index]
        view = self.doc["bufferViews"][acc["bufferView"]]
        dtype = COMPONENTS[acc["componentType"]]
        n = acc["count"] * WIDTH[acc["type"]]
        a = np.frombuffer(self.bin, dtype=dtype, count=n, offset=view["byteOffset"])
        return a.reshape(acc["count"], -1) if WIDTH[acc["type"]] > 1 else a

    def animation(self, name: str) -> dict:
        return next(a for a in self.doc["animations"] if a["name"] == name)

    def trs(self, anim_json: dict | None, t: float) -> list[dict]:
        """Every node's translation, rotation and scale at time t."""
        out = [{"translation": np.array(n.get("translation", (0, 0, 0)), float),
                "rotation": np.array(n.get("rotation", (0, 0, 0, 1)), float),
                "scale": np.array(n.get("scale", (1, 1, 1)), float)} for n in self.nodes]
        for ch in (anim_json or {}).get("channels", ()):
            s = anim_json["samplers"][ch["sampler"]]
            times = self.accessor(s["input"]).astype(float)
            values = self.accessor(s["output"]).astype(float)
            i = int(np.searchsorted(times, t + 1e-5) - 1)  # the times are float32
            if i < 0:
                v = values[0]
            elif i >= len(times) - 1 or s["interpolation"] == "STEP":
                v = values[min(i, len(times) - 1)]
            else:
                f = (t - times[i]) / (times[i + 1] - times[i])
                v = values[i] + f * (values[i + 1] - values[i])
                if ch["target"]["path"] == "rotation":
                    v = v / np.linalg.norm(v)
            out[ch["target"]["node"]][ch["target"]["path"]] = v
        return out

    def world(self, trs: list[dict], node: int) -> np.ndarray:
        m = _matrix(**trs[node])
        while node in self.parent:
            node = self.parent[node]
            m = _matrix(**trs[node]) @ m
        return m

    def mesh_points(self, mesh: int) -> np.ndarray:
        prim = self.doc["meshes"][mesh]["primitives"][0]
        return self.accessor(prim["attributes"]["POSITION"]).astype(float)

    def shown_points(self, anim_name: str | None, t: float) -> dict[str, np.ndarray]:
        """Node name -> its vertices in the world at t, for every mesh node
        with a non-zero scale."""
        trs = self.trs(self.animation(anim_name) if anim_name else None, t)
        out = {}
        for i, n in enumerate(self.nodes):
            if "mesh" not in n:
                continue
            m = self.world(trs, i)
            if abs(np.linalg.det(m[:3, :3])) < 1e-9:
                continue
            p = self.mesh_points(n["mesh"])
            out[n["name"]] = p @ m[:3, :3].T + m[:3, 3]
        return out


def _matrix(translation, rotation, scale) -> np.ndarray:
    x, y, z, w = rotation
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r @ np.diag(scale)
    m[:3, 3] = translation
    return m


def _expected_smooth(scene, clip, frame: int, model, source: int = 0) -> np.ndarray:
    """Model's vertices in the world (glTF axes) as Rok poses them, turned smoothly."""
    voxels = model.grid.voxels if not source else model.states[source - 1]
    local = build_mesh(np.asarray(voxels)).vertices.astype(float)
    fr = anim.evaluate(anim_export.in_place(clip, scene), scene, frame)
    pose = fr.poses.get(model.uid)
    if pose is not None:
        m = pose.matrix()
        local = local @ m[:3, :3].T + m[:3, 3]
    return _y_up(local + np.asarray(model.position, float)).astype(float)


def _read(name: str):
    scene, _doc = rokfile.read(ANIM / f"{name}.rok")
    return scene


@pytest.mark.parametrize("name", ["walk", "coin", "propeller_spin"])
def test_smooth_parts_follow_the_pose_at_every_frame(name):
    scene = _read(name)
    g = Gltf(anim_export.smooth_glb(scene))
    for clip in scene.clips:
        for f in anim_export.sample_frames(clip):
            shown = g.shown_points(clip.name, f / clip.fps)
            for model in scene.models:
                want = _expected_smooth(scene, clip, f, model)
                got = shown[model.name]
                np.testing.assert_allclose(got, want, atol=1e-3, err_msg=f"{name} {model.name} frame {f}")


def test_smooth_rest_pose_is_the_scene_as_built():
    scene = _read("walk")
    shown = Gltf(anim_export.smooth_glb(scene)).shown_points(None, 0.0)
    for model in scene.models:
        local = build_mesh(model.grid.voxels).vertices.astype(float) + np.asarray(model.position, float)
        np.testing.assert_allclose(shown[model.name], _y_up(local), atol=1e-4)


def test_smooth_keeps_parents_and_pivots():
    scene = _read("walk")
    g = Gltf(anim_export.smooth_glb(scene))
    names = [n["name"] for n in g.nodes]
    body = names.index("Body")
    for child in ("Left leg", "Right leg", "Left arm", "Right arm", "Head"):
        assert g.parent[names.index(child)] == body
    roots = g.doc["scenes"][0]["nodes"]
    assert roots == [body]
    leg = scene.models[1]  # its pivot is the node's origin: the hip, not the box corner
    want = _y_up(np.asarray(leg.position, float) + np.array(leg.pivot2) / 2.0)
    np.testing.assert_allclose(g.world(g.trs(None, 0), names.index("Left leg"))[:3, 3], want, atol=1e-6)


def test_smooth_clip_extras_hold_markers_and_loop():
    scene = _read("walk")
    a = Gltf(anim_export.smooth_glb(scene)).animation("Walk")
    clip = scene.clips[0]
    assert a["extras"]["fps"] == clip.fps and a["extras"]["loop_mode"] == "LOOP"
    assert [m["frame"] for m in a["extras"]["markers"]] == sorted(m.time for m in clip.markers)
    assert len(a["extras"]["markers"]) == 2


def test_a_full_turn_stays_a_full_turn():
    """Quaternions take the short way: a spin from 0 to 360 degrees in two
    keys would play as no turn. Sampled every frame, each step is 15 degrees."""
    scene = _read("coin")
    g = Gltf(anim_export.smooth_glb(scene))
    a = g.animation(scene.clips[0].name)
    rot = next(ch for ch in a["channels"] if ch["target"]["path"] == "rotation")
    s = a["samplers"][rot["sampler"]]
    q = g.accessor(s["output"]).astype(float)
    assert s["interpolation"] == "LINEAR" and len(q) == 25
    steps = np.degrees(2 * np.arccos(np.clip(np.abs((q[1:] * q[:-1]).sum(axis=1)), 0, 1)))
    np.testing.assert_allclose(steps, 15.0, atol=1e-3)


def test_frames_are_child_meshes_shown_in_turn():
    scene = _read("face")
    g = Gltf(anim_export.smooth_glb(scene))
    clip = scene.clips[0]
    eyes = scene.models[1]
    a = g.animation(clip.name)
    channels = [ch for ch in a["channels"] if g.nodes[ch["target"]["node"]]["name"].startswith("Eyes frame")]
    assert {a["samplers"][ch["sampler"]]["interpolation"] for ch in channels} == {"STEP"}
    for f in (0, 71, 72, 75, 76):
        shown = g.shown_points(clip.name, f / clip.fps)
        source = 1 if 72 <= f < 76 else 0
        assert ("Eyes frame 2" in shown) == bool(source) and ("Eyes frame 1" in shown) != bool(source)
        key = f"Eyes frame {source + 1}"
        np.testing.assert_allclose(shown[key], _expected_smooth(scene, clip, f, eyes, source), atol=1e-4)


def _small_scene():
    """One bar with a Size, a Show and a Hold move, in a ping-pong clip."""
    from rok.core.grid import VoxelGrid
    from rok.core.palette import default_palette
    from rok.core.scene import VoxScene

    scene = VoxScene(palette=default_palette())
    voxels = np.zeros((4, 1, 1), np.uint8)
    voxels[:] = 7
    bar = scene.add(VoxelGrid((4, 1, 1), voxels, scene.palette), "Bar", (2, 0, 0), 0)
    clip = anim.Clip(name="Pulse", fps=10, length=5, loop_mode=anim.PING_PONG)
    track = anim.Track(anim.Target(anim.PART, bar.uid))
    anim.set_key(track, anim.SCALE, anim.Key(0, (100, 100, 100), anim.LINEAR))
    anim.set_key(track, anim.SCALE, anim.Key(4, (200, 100, 50), anim.LINEAR))
    anim.set_key(track, anim.MOVE_Z, anim.Key(0, 0, anim.HOLD))
    anim.set_key(track, anim.MOVE_Z, anim.Key(2, 3, anim.HOLD))
    anim.set_key(track, anim.VISIBLE, anim.Key(0, True))
    anim.set_key(track, anim.VISIBLE, anim.Key(3, False))
    clip.tracks.append(track)
    scene.clips = [clip]
    return scene


def test_size_show_hold_and_ping_pong():
    scene = _small_scene()
    clip = scene.clips[0]
    bar = scene.models[0]
    g = Gltf(anim_export.smooth_glb(scene))
    assert anim_export.sample_frames(clip) == list(range(9))  # there (0..4) and back (..8)
    a = g.animation("Pulse")
    move = next(ch for ch in a["channels"] if ch["target"]["path"] == "translation")
    assert a["samplers"][move["sampler"]]["interpolation"] == "STEP"
    for f in range(9):
        shown = g.shown_points("Pulse", f / clip.fps)
        t = clip.time_at(f)
        assert ("Bar frame 1" in shown) == (t < 3), f
        if t < 3:
            np.testing.assert_allclose(shown["Bar frame 1"], _expected_smooth(scene, clip, f, bar), atol=1e-4)


def test_root_motion_off_plays_in_place():
    scene = _small_scene()
    track = scene.clips[0].tracks[0]
    anim.set_key(track, anim.MOVE_X, anim.Key(4, 6, anim.LINEAR))
    anim.set_key(track, anim.MOVE_X, anim.Key(0, 0, anim.LINEAR))
    placed = anim_export.in_place(scene.clips[0], scene)
    assert not placed.tracks[0].keys(anim.MOVE_X) and placed.tracks[0].keys(anim.MOVE_Z)
    scene.clips[0].root_motion = True
    assert anim_export.in_place(scene.clips[0], scene).tracks[0].keys(anim.MOVE_X)


def _expected_voxels(scene, clip, frame: int) -> np.ndarray:
    """Every shown part as Voxel look builds it, as sorted world quads."""
    fr = anim.evaluate(anim_export.in_place(clip, scene), scene, frame)
    pts = []
    for m in scene.models:
        if not fr.visible.get(m.uid, True):
            continue
        source = fr.state.get(m.uid, 0)
        voxels = m.grid.voxels if not source else m.states[source - 1]
        pose = fr.poses.get(m.uid)
        at = np.zeros(3)
        if pose is not None and not pose.is_rest():
            voxels, at = voxel_look.posed(voxels, pose.linear, pose.pivot, pose.offset)
        v = build_mesh(np.asarray(voxels)).vertices.astype(float)
        pts.append(v + np.asarray(m.position, float) + np.asarray(at, float))
    return _sorted_quads(_y_up(np.concatenate(pts)).astype(float))


def _sorted_quads(points: np.ndarray) -> np.ndarray:
    q = np.round(points.reshape(-1, 12), 4)
    return q[np.lexsort(q.T[::-1])]


@pytest.mark.parametrize("name", ["walk", "face", "propeller_spin"])
def test_baked_frames_are_the_voxel_look(name):
    scene = _read(name)
    g = Gltf(anim_export.baked_glb(scene))
    for clip in scene.clips:
        for f in anim_export.sample_frames(clip):
            shown = g.shown_points(clip.name, f / clip.fps)
            assert len(shown) == 1, f"{name} frame {f}: {sorted(shown)}"
            got = _sorted_quads(next(iter(shown.values())))
            np.testing.assert_allclose(got, _expected_voxels(scene, clip, f), atol=1e-4)


def test_baked_rest_pose_shows_without_a_clip_and_frames_share_meshes():
    scene = _read("face")
    g = Gltf(anim_export.baked_glb(scene))
    assert list(g.shown_points(None, 0.0)) == ["Rest"]
    # 78 sampled frames show only two looks: eyes open (the rest) and closed.
    assert len(g.doc["meshes"]) == 2


def test_write_picks_the_kind(tmp_path):
    scene = _read("coin")
    for kind in anim_export.KINDS:
        path = tmp_path / f"{kind}.glb"
        anim_export.write(path, scene, kind)
        assert Gltf(path.read_bytes()).doc["scenes"][0]["extras"]["kind"] == kind
    with pytest.raises(ValueError):
        anim_export.write(tmp_path / "x.glb", scene, "FBX")


# --- .vox with frames -----------------------------------------------------------------


def _xyzi_count(data: bytes) -> int:
    return data.count(b"XYZI")


def test_vox_frames_play_at_ten_per_second_and_show_again():
    """The face blinks at frame 72 of 24 (3.0 s) and opens at 76: in a .vox
    that is frame 30 and 32, and the open eyes show again from the same model."""
    from rok.core.scene import VoxScene
    from rok.core.vox_read import loads_scene
    from rok.core.vox_write import dumps_scene

    scene = _read("face")
    data = dumps_scene(scene.to_data())
    assert _xyzi_count(data) == 3  # head, open eyes, closed eyes: nothing twice
    eyes = loads_scene(data).models[1]
    assert eyes.state_frame == 0 and [f for f, _v in eyes.states] == [30, 32]
    assert eyes.states[1][1] is eyes.voxels  # frame 32 is the model's own voxels again
    again = VoxScene.from_data(loads_scene(data))
    track = again.timeline.track_of(again.models[1].uid)
    assert len(again.models[1].states) == 1
    assert [(k.time, k.value) for k in track.keys(anim.STATE)] == [(0, 0), (30, 1), (32, 0)]


def test_a_copied_frame_stays_its_own_frame():
    """Frame: Add starts from a copy: same voxels, still a frame of its own."""
    from rok.core.scene import VoxScene
    from rok.core.vox_read import loads_scene
    from rok.core.vox_write import dumps_scene

    scene = _read("face")
    eyes = scene.models[1]
    eyes.states.append(eyes.grid.voxels.copy())
    again = VoxScene.from_data(loads_scene(dumps_scene(scene.to_data())))
    assert len(again.models[1].states) == 2


# --- File > Export Animation ---------------------------------------------------------


def test_the_export_animation_operator_writes_each_kind(tmp_path, monkeypatch):
    from rok.app import export_ops, opkit, recent
    from rok.app.app import RokApp
    from rok.app.object import document
    from rok.app.state import state
    from rok.core.vox_read import loads_scene

    class Host:
        def __getattr__(self, name):
            return lambda *a, **k: (10.0, 16.0) if name == "measure_text" else None

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    document.reset()
    state.__init__()
    try:
        app = RokApp(Host(), str(ANIM / "face.rok"))
        app.handle("start", {"width": 1200.0, "height": 800.0, "scale": 2.0})
        for kind, _label, suffix, _s in export_ops.ANIMATION_KINDS:
            out = tmp_path / kind.lower()  # no suffix: the kind's is added
            assert opkit.call("rok.export_animation", "EXEC", kind=kind, filepath=str(out)) == {"FINISHED"}
            data = out.with_suffix(suffix).read_bytes()
            if suffix == ".glb":
                assert Gltf(data).doc["animations"][0]["name"] == "Idle"
            else:
                assert [f for f, _v in loads_scene(data).models[1].states] == [30, 32]
        assert document.filepath.endswith("face.rok")  # an export leaves the document where it is
    finally:
        document.reset()
        state.__init__()


def _vertex_total(g: Gltf) -> int:
    return sum(g.doc["accessors"][p["attributes"]["POSITION"]]["count"]
               for m in g.doc["meshes"] for p in m["primitives"])


@pytest.mark.parametrize("name", ["walk", "face"])
def test_merged_faces_cut_the_vertices_of_both_glb_kinds(name):
    scene = _read(name)
    for build in (anim_export.smooth_glb, anim_export.baked_glb):
        plain, merged = Gltf(build(scene)), Gltf(build(scene, merge=True))
        assert 0 < _vertex_total(merged) < _vertex_total(plain)


def test_merged_smooth_parts_fill_the_same_boxes_at_every_frame():
    scene = _read("walk")
    plain, merged = Gltf(anim_export.smooth_glb(scene)), Gltf(anim_export.smooth_glb(scene, merge=True))
    for clip in scene.clips:
        for f in anim_export.sample_frames(clip):
            a, b = plain.shown_points(clip.name, f / clip.fps), merged.shown_points(clip.name, f / clip.fps)
            for model in scene.models:
                np.testing.assert_allclose(b[model.name].min(axis=0), a[model.name].min(axis=0), atol=1e-3)
                np.testing.assert_allclose(b[model.name].max(axis=0), a[model.name].max(axis=0), atol=1e-3)
