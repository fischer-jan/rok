"""Mesh files for game engines: glTF binary (.glb), Wavefront (.obj) and
Stanford (.ply), written without any 3D package.

Each model becomes one mesh (face-culled quads, four vertices per quad), at
its place in the scene. One voxel is one unit. With `merge` the mesh comes
from `mesh.build_mesh_merged`: neighbouring faces of one colour are one
rectangle, which cuts the triangles by 3 to 10 times for a game engine. The
default merges a GLB and keeps one quad per voxel face in OBJ and PLY, which
people open in a modeller to edit.

- GLB: Y up (glTF's rule), one node per model, the palette atlas embedded as
  a PNG with nearest-pixel sampling so the voxels stay crisp, and each vertex
  also carries its colour (COLOR_0) for engines that prefer that.
- OBJ: Y up, one object per model, a .mtl and the palette atlas PNG beside it.
- PLY: Z up, quads with a colour on every vertex.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import png
from .mesh import ATLAS_PX, ATLAS_TILES, TILE_PX, build_mesh, build_mesh_merged

FORMATS = {".glb": "GLB", ".obj": "OBJ", ".ply": "PLY"}
MERGE_DEFAULT = {"GLB": True, "OBJ": False, "PLY": False}  # what `write` does when `merge` is None


# --- shared pieces -------------------------------------------------------------------


def palette_atlas(palette: np.ndarray) -> np.ndarray:
    """The palette as an (ATLAS_PX, ATLAS_PX, 4) uint8 image, top row first
    (as a PNG stores it). Tile i sits at column i % 16, row i // 16 counted
    from the bottom, which is where `mesh.tile_uvs` points."""
    pal = np.asarray(palette, dtype=np.uint8).reshape(256, 4).copy()
    pal[:, 3] = 255
    tiles = pal.reshape(ATLAS_TILES, ATLAS_TILES, 4)  # [row from bottom, column]
    img = np.repeat(np.repeat(tiles, TILE_PX, axis=0), TILE_PX, axis=1)
    return np.ascontiguousarray(img[::-1])  # top row first


def _colours_of(uvs: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """The palette colour of each vertex, read back from its UV's tile."""
    col = np.clip(np.floor(uvs[:, 0] * ATLAS_TILES), 0, ATLAS_TILES - 1).astype(np.int64)
    row = np.clip(np.floor(uvs[:, 1] * ATLAS_TILES), 0, ATLAS_TILES - 1).astype(np.int64)
    return np.asarray(palette, dtype=np.uint8).reshape(256, 4)[row * ATLAS_TILES + col]


def _normals(vertices: np.ndarray) -> np.ndarray:
    """One flat normal per quad, repeated on its four vertices."""
    q = vertices.reshape(-1, 4, 3)
    n = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.repeat(n, 4, axis=0).astype(np.float32)


def _triangles(n_verts: int) -> np.ndarray:
    base = np.arange(0, n_verts, 4, dtype=np.uint32)[:, None]
    return (base + np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)).reshape(-1)


def _y_up(points: np.ndarray) -> np.ndarray:
    """Z-up (Rok) to Y-up (glTF, OBJ): (x, y, z) -> (x, z, -y)."""
    p = np.asarray(points, dtype=np.float32)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1).astype(np.float32)


def model_meshes(models, merge: bool = False) -> list[tuple[str, tuple[float, float, float], object]]:
    """(name, position, MeshData) per model with voxels; `models` are
    (name, position, voxels) triples. `merge` joins faces of one colour."""
    build = build_mesh_merged if merge else build_mesh
    out = []
    for name, position, voxels in models:
        data = build(np.asarray(voxels))
        if data.vertices.shape[0]:
            out.append((name, tuple(float(p) for p in position), data))
    return out


# --- GLB ----------------------------------------------------------------------------


class Glb:
    """A glTF binary under construction: one buffer, the palette material and
    its atlas texture. `mesh` adds a mesh, `accessor` raw data (animations);
    `dumps` gives the file's bytes with the nodes, scenes and animations."""

    def __init__(self, palette) -> None:
        self.palette = palette
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []
        self.meshes: list[dict] = []

    def view(self, data: bytes, target: int | None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.blob.extend(data)
        self.views.append(view)
        return len(self.views) - 1

    def accessor(self, array: np.ndarray, kind: str, component: int = 5126, target: int | None = None,
                 normalized: bool = False, bounds: bool = False) -> int:
        view = self.view(np.ascontiguousarray(array).tobytes(), target)
        acc = {"bufferView": view, "componentType": component, "count": int(array.shape[0]), "type": kind}
        if normalized:
            acc["normalized"] = True
        if bounds:
            lo = array.min(axis=0) if array.ndim > 1 else np.array([array.min()])
            hi = array.max(axis=0) if array.ndim > 1 else np.array([array.max()])
            acc["min"] = [float(v) for v in lo]
            acc["max"] = [float(v) for v in hi]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def mesh(self, name: str, data, offset=(0.0, 0.0, 0.0)) -> int:
        """Add `data` (a MeshData, Z up, grid units) moved by `offset` as one
        mesh; returns its index."""
        verts = _y_up(data.vertices + np.asarray(offset, dtype=np.float32))
        normals = _y_up(_normals(data.vertices))
        uvs = data.uvs.copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]  # glTF's UV origin is the image's top left
        colours = _colours_of(data.uvs, self.palette)
        idx = _triangles(verts.shape[0])
        attrs = {
            "POSITION": self.accessor(verts, "VEC3", 5126, 34962, bounds=True),
            "NORMAL": self.accessor(normals, "VEC3", 5126, 34962),
            "TEXCOORD_0": self.accessor(uvs.astype(np.float32), "VEC2", 5126, 34962),
            "COLOR_0": self.accessor(colours, "VEC4", 5121, 34962, normalized=True),
        }
        indices = self.accessor(idx, "SCALAR", 5125, 34963)
        self.meshes.append({"name": name, "primitives": [{"attributes": attrs, "indices": indices, "material": 0}]})
        return len(self.meshes) - 1

    def dumps(self, nodes: list[dict], roots: list[int], animations=(), extras=None) -> bytes:
        image_view = self.view(png.encode(palette_atlas(self.palette)), None)
        while len(self.blob) % 4:
            self.blob.append(0)
        doc = {
            "asset": {"version": "2.0", "generator": "Rok"},
            "scene": 0,
            "scenes": [{"nodes": list(roots)}],
            "nodes": nodes,
            "meshes": self.meshes,
            "materials": [{
                "name": "Palette",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                         "roughnessFactor": 1.0},
            }],
            "textures": [{"sampler": 0, "source": 0}],
            "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
            "images": [{"bufferView": image_view, "mimeType": "image/png", "name": "palette"}],
            "buffers": [{"byteLength": len(self.blob)}],
            "bufferViews": self.views,
            "accessors": self.accessors,
        }
        if animations:
            doc["animations"] = list(animations)
        if extras:
            doc["scenes"][0]["extras"] = extras
        text = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        text += b" " * ((4 - len(text) % 4) % 4)
        body = (struct.pack("<I4s", len(text), b"JSON") + text
                + struct.pack("<I4s", len(self.blob), b"BIN\0") + bytes(self.blob))
        return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def write_glb(path, models, palette, merge: bool = False) -> None:
    meshes = model_meshes(models, merge)
    if not meshes:
        raise ValueError("there are no voxels to export")
    glb = Glb(palette)
    nodes = []
    for name, position, data in meshes:
        nodes.append({"name": name, "mesh": glb.mesh(name, data),
                      "translation": [float(v) for v in _y_up(np.asarray(position))]})
    Path(path).write_bytes(glb.dumps(nodes, range(len(nodes))))


# --- OBJ ----------------------------------------------------------------------------


def write_obj(path, models, palette, merge: bool = False) -> None:
    path = Path(path)
    meshes = model_meshes(models, merge)
    if not meshes:
        raise ValueError("there are no voxels to export")
    atlas_name = path.stem + "_palette.png"
    mtl_name = path.stem + ".mtl"
    lines = ["# Rok", f"mtllib {mtl_name}"]
    base = 0
    for name, position, data in meshes:
        verts = _y_up(data.vertices + np.asarray(position, dtype=np.float32))
        normals = _y_up(_normals(data.vertices))
        n = verts.shape[0]
        lines.append(f"o {name or 'Model'}")
        lines.append("usemtl palette")
        lines.extend(f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in verts)
        lines.extend(f"vt {u:.6f} {v:.6f}" for u, v in data.uvs)
        lines.extend(f"vn {x:.6g} {y:.6g} {z:.6g}" for x, y, z in normals[::4])
        quads = np.arange(n).reshape(-1, 4) + base + 1
        for q, (a, b, c, d) in enumerate(quads):
            k = base // 4 + q + 1
            lines.append(f"f {a}/{a}/{k} {b}/{b}/{k} {c}/{c}/{k} {d}/{d}/{k}")
        base += n
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path.parent / mtl_name).write_text(
        f"# Rok\nnewmtl palette\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {atlas_name}\n", encoding="utf-8")
    (path.parent / atlas_name).write_bytes(png.encode(palette_atlas(palette)))


# --- PLY ----------------------------------------------------------------------------


def write_ply(path, models, palette, merge: bool = False) -> None:
    meshes = model_meshes(models, merge)
    if not meshes:
        raise ValueError("there are no voxels to export")
    verts = np.concatenate([d.vertices + np.asarray(p, dtype=np.float32) for _n, p, d in meshes]).astype(np.float32)
    colours = np.concatenate([_colours_of(d.uvs, palette) for _n, _p, d in meshes])
    n = verts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment Rok\n"
        f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        f"element face {n // 4}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    vrec = np.zeros(n, dtype=[("p", "<f4", 3), ("c", "u1", 4)])
    vrec["p"], vrec["c"] = verts, colours
    frec = np.zeros(n // 4, dtype=[("k", "u1"), ("i", "<i4", 4)])
    frec["k"] = 4
    frec["i"] = np.arange(n, dtype=np.int32).reshape(-1, 4)
    Path(path).write_bytes(header + vrec.tobytes() + frec.tobytes())


def write(path, models, palette, merge: bool | None = None) -> str:
    """Write the file the suffix names; returns the format. `merge` None takes
    the format's default (MERGE_DEFAULT)."""
    fmt = FORMATS.get(Path(path).suffix.lower())
    if fmt is None:
        raise ValueError(f"unknown mesh format {Path(path).suffix!r}: use .glb, .obj or .ply")
    {"GLB": write_glb, "OBJ": write_obj, "PLY": write_ply}[fmt](
        path, models, palette, MERGE_DEFAULT[fmt] if merge is None else merge)
    return fmt


__all__ = ["FORMATS", "Glb", "MERGE_DEFAULT", "palette_atlas", "write", "write_glb", "write_obj", "write_ply", "ATLAS_PX"]
