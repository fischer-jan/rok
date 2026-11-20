"""rok/core/mesh_export.py: GLB, OBJ and PLY files read back."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from rok.core import mesh_export, png
from rok.core.palette import default_palette
from rok.core.vox_read import read_scene

FIXTURES = Path(__file__).parent / "fixtures"


def one_voxel(colour: int = 5):
    v = np.zeros((1, 1, 1), dtype=np.uint8)
    v[0, 0, 0] = colour
    return v


def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2 and length == len(data)
    jlen, jtype = struct.unpack_from("<I4s", data, 12)
    assert jtype == b"JSON"
    doc = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack_from("<I4s", data, 20 + jlen)
    assert btype == b"BIN\0"
    blob = data[28 + jlen:28 + jlen + blen]
    return doc, blob


def accessor(doc, blob, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5121: np.uint8}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    raw = np.frombuffer(blob, dtype=dtype, count=acc["count"] * width, offset=view["byteOffset"])
    return raw.reshape(acc["count"], width) if width > 1 else raw


def test_one_voxel_is_six_faces_in_every_format(tmp_path):
    pal = default_palette()
    models = [("Cube", (0, 0, 0), one_voxel())]
    mesh_export.write(tmp_path / "c.glb", models, pal)
    doc, blob = read_glb(tmp_path / "c.glb")
    prim = doc["meshes"][0]["primitives"][0]
    assert doc["accessors"][prim["attributes"]["POSITION"]]["count"] == 24
    assert doc["accessors"][prim["indices"]]["count"] == 36
    mesh_export.write(tmp_path / "c.obj", models, pal)
    text = (tmp_path / "c.obj").read_text()
    assert text.count("\nv ") == 24 and text.count("\nf ") == 6
    mesh_export.write(tmp_path / "c.ply", models, pal)
    head = (tmp_path / "c.ply").read_bytes().split(b"end_header\n")[0].decode()
    assert "element vertex 24" in head and "element face 6" in head


def test_glb_is_y_up_and_places_each_model_at_its_position(tmp_path):
    pal = default_palette()
    tall = np.zeros((1, 1, 4), dtype=np.uint8)
    tall[..., :] = 7  # a column 4 voxels high along Rok's Z
    mesh_export.write(tmp_path / "t.glb", [("Pillar", (10, 20, 0), tall)], pal)
    doc, blob = read_glb(tmp_path / "t.glb")
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert acc["max"][1] - acc["min"][1] == pytest.approx(4.0)  # the height is glTF's Y
    assert doc["nodes"][0]["translation"] == [10.0, 0.0, -20.0]
    assert doc["nodes"][0]["name"] == "Pillar"


def test_glb_carries_the_palette_as_texture_and_vertex_colours(tmp_path):
    pal = default_palette()
    pal[5] = (200, 10, 30, 255)
    mesh_export.write(tmp_path / "p.glb", [("Cube", (0, 0, 0), one_voxel(5))], pal)
    doc, blob = read_glb(tmp_path / "p.glb")
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    colours = accessor(doc, blob, attrs["COLOR_0"])
    assert np.all(colours == [200, 10, 30, 255])
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    image = blob[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    _h, _w, rgba = png.decode(image, max_pixels=1 << 22)
    uvs = accessor(doc, blob, attrs["TEXCOORD_0"])
    u, v = uvs.mean(axis=0)
    size = rgba.shape[0]
    px = rgba[int(v * size), int(u * size)]  # glTF UVs have their origin at the top left
    assert tuple(px[:3]) == (200, 10, 30)
    assert doc["samplers"][0]["magFilter"] == 9728  # nearest: crisp voxels


def test_obj_writes_its_material_and_palette_beside_it(tmp_path):
    source = read_scene(FIXTURES / "robot.vox")
    mesh_export.write(tmp_path / "robot.obj", [("Robot", (0, 0, 0), source.models[0].voxels)], source.palette)
    assert "map_Kd robot_palette.png" in (tmp_path / "robot.mtl").read_text()
    _h, _w, rgba = png.decode((tmp_path / "robot_palette.png").read_bytes(), max_pixels=1 << 22)
    assert rgba.shape == (mesh_export.ATLAS_PX, mesh_export.ATLAS_PX, 4)


def test_ply_colours_match_the_voxels_colours(tmp_path):
    source = read_scene(FIXTURES / "robot.vox")
    vox = source.models[0].voxels
    mesh_export.write(tmp_path / "s.ply", [("Robot", (0, 0, 0), vox)], source.palette)
    data = (tmp_path / "s.ply").read_bytes()
    head, body = data.split(b"end_header\n", 1)
    n = int(head.decode().split("element vertex ")[1].split("\n")[0])
    verts = np.frombuffer(body[: n * 16], dtype=[("p", "<f4", 3), ("c", "u1", 4)])
    used = {tuple(source.palette[i][:3]) for i in np.unique(vox[vox != 0])}
    assert {tuple(c[:3]) for c in verts["c"]} <= used
    top = int(np.nonzero((vox != 0).any(axis=(0, 1)))[0].max()) + 1
    assert verts["p"].max(axis=0)[2] == pytest.approx(top)  # PLY stays Z up


def test_an_empty_scene_says_so(tmp_path):
    with pytest.raises(ValueError, match="no voxels"):
        mesh_export.write(tmp_path / "e.glb", [("E", (0, 0, 0), np.zeros((2, 2, 2), np.uint8))], default_palette())


def test_an_unknown_suffix_is_refused(tmp_path):
    with pytest.raises(ValueError, match="use .glb, .obj or .ply"):
        mesh_export.write(tmp_path / "x.fbx", [("C", (0, 0, 0), one_voxel())], default_palette())


def test_the_export_operator_writes_the_scene(tmp_path, monkeypatch):
    from rok.app import opkit, recent
    from rok.app.app import RokApp
    from rok.app.object import document
    from rok.app.state import state

    class Host:
        def __getattr__(self, name):
            return lambda *a, **k: (10.0, 16.0) if name == "measure_text" else None

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    document.reset()
    state.__init__()
    app = RokApp(Host(), str(FIXTURES / "phase3" / "two_models.vox"))
    app.handle("start", {"width": 1200.0, "height": 800.0, "scale": 2.0})
    assert opkit.call("rok.export_mesh", "EXEC", filepath=str(tmp_path / "two")) == {"FINISHED"}
    doc, _blob = read_glb(tmp_path / "two.glb")
    assert [n["name"] for n in doc["nodes"]] == ["Robot", "Car"]
    document.reset()
    state.__init__()


def quad_count_of(path: Path) -> int:
    doc, _blob = read_glb(path)
    return doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"] // 4


def test_a_glb_merges_faces_by_default_and_obj_and_ply_do_not(tmp_path):
    source = read_scene(FIXTURES / "car.vox")
    models = [("Car", (0, 0, 0), source.models[0].voxels)]
    mesh_export.write(tmp_path / "k.glb", models, source.palette)
    mesh_export.write(tmp_path / "k_off.glb", models, source.palette, merge=False)
    assert quad_count_of(tmp_path / "k.glb") * 2 < quad_count_of(tmp_path / "k_off.glb")
    mesh_export.write(tmp_path / "k.obj", models, source.palette)
    mesh_export.write(tmp_path / "k_on.obj", models, source.palette, merge=True)
    assert (tmp_path / "k.obj").read_text().count("\nf ") == quad_count_of(tmp_path / "k_off.glb")
    assert (tmp_path / "k_on.obj").read_text().count("\nf ") == quad_count_of(tmp_path / "k.glb")
    mesh_export.write(tmp_path / "k.ply", models, source.palette)
    head = (tmp_path / "k.ply").read_bytes().split(b"end_header\n")[0].decode()
    assert f"element face {quad_count_of(tmp_path / 'k_off.glb')}" in head


def test_merged_glb_keeps_the_size_and_colours_of_the_model(tmp_path):
    source = read_scene(FIXTURES / "robot.vox")
    models = [("Robot", (0, 0, 0), source.models[0].voxels)]
    for name, merge in (("on", True), ("off", False)):
        mesh_export.write(tmp_path / f"{name}.glb", models, source.palette, merge=merge)
    reads = {}
    for name in ("on", "off"):
        doc, blob = read_glb(tmp_path / f"{name}.glb")
        attrs = doc["meshes"][0]["primitives"][0]["attributes"]
        pos = doc["accessors"][attrs["POSITION"]]
        colours = accessor(doc, blob, attrs["COLOR_0"])
        reads[name] = (pos["min"], pos["max"], {tuple(c) for c in colours})
    assert reads["on"] == reads["off"]


def _scene_app(tmp_path, monkeypatch):
    from rok.app import recent
    from rok.app.app import RokApp
    from rok.app.object import document
    from rok.app.state import state

    class Host:
        def __getattr__(self, name):
            return lambda *a, **k: (10.0, 16.0) if name == "measure_text" else None

    monkeypatch.setattr(recent, "FOLDER", tmp_path / "config")
    monkeypatch.setattr(recent, "FILE", tmp_path / "config" / "recent.txt")
    document.reset()
    state.__init__()
    app = RokApp(Host(), str(FIXTURES / "car.vox"))
    app.handle("start", {"width": 1200.0, "height": 800.0, "scale": 2.0})
    return app, document, state


def test_the_export_operator_takes_merge_faces(tmp_path, monkeypatch):
    from rok.app import opkit

    app, document, state = _scene_app(tmp_path, monkeypatch)
    quads = {}
    for merge in ("AUTO", "ON", "OFF"):
        out = tmp_path / merge
        assert opkit.call("rok.export_mesh", "EXEC", filepath=str(out), merge=merge) == {"FINISHED"}
        quads[merge] = quad_count_of(out.with_suffix(".glb"))
    assert quads["AUTO"] == quads["ON"]  # a .glb merges by default
    assert quads["OFF"] > 2 * quads["ON"]
    document.reset()
    state.__init__()


def test_exporting_from_the_menu_asks_about_merging_before_it_opens_the_file_browser(tmp_path, monkeypatch):
    from rok.app import opkit

    app, document, state = _scene_app(tmp_path, monkeypatch)
    opened = []
    app.host.file_dialog = lambda token, save, title, filters, name, directory: opened.append((token, save))
    opkit.call("rok.export_mesh", "INVOKE")
    assert app.dialog is not None and not opened  # the options come first
    assert app.dialog.op.command_id == "rok.export_mesh"
    app.dialog._finish(True)
    assert len(opened) == 1 and opened[0][1] is True  # now the Save browser
    app.on_dialog(opened[0][0], path=str(tmp_path / "menu.glb"))
    assert quad_count_of(tmp_path / "menu.glb") > 0
    document.reset()
    state.__init__()
