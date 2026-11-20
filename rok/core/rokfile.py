"""The Rok file (`.rok`): a zip that keeps everything a scene has.

A `.vox` cannot keep everything Rok makes: live shaders, and animation a
`.vox` would bake or drop (anim.vox_losses). Such a scene saves as a Rok
file. The zip holds:

  - `scene.vox`: the parts at rest, with their voxel frames and no clip
    (a `.vox` reader turns the voxels by a frame's rotation, which would
    turn the rest pose itself). Unzip the file and any .vox editor opens it.
  - `rok.json`: what the `.vox` cannot hold, with "version":
      - "models": one [link, orient, pivot2, parent index, empty] per model index. `dumps_scene` writes
        an animated linked copy on its own, baked, without its link; this
        list gives the link back.
      - "clips": `anim.to_json` of every clip, each target's "model" a
        model index (`null` for a colour track), and "active_clip", the
        index of the clip the timeline shows.
      - "live": the live shaders (rok.core.live_shader), each with its
        model's index.
      - "colours": the colour store (rok.core.colour_store): every colour
        with its id, your order, unused colours, and the slots voxels hold.
      - "bricks": the brick tables (rok.core.brick), one
        [model index, table] per brick model.
      - "document": whatever the caller gave; handed back unread.
      - "cell": the length of one voxel, {"metres", "unit"} with unit "m" or
        "ft", or null when the labels stay in cells. A .vox has no place for it.

`pack` and `unpack` are the same document as one JSON text, the scene packed
inside as base64: for a test or a clipboard, not for disk.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
import zlib
from pathlib import Path

from . import anim, brick, link
from . import live_shader as live_core
from .colour_store import ColourStore
from .safe_write import write_bytes
from .scene import VoxScene
from .vox_read import VoxError, loads_scene
from .vox_write import dumps_scene

VERSION = 1


class RokFileError(ValueError):
    """`text` is not a Rok file's JSON: not JSON at all, or the wrong version."""


def _meta(scene: VoxScene, doc: dict) -> dict:
    """rok.json: what a `.vox` cannot hold."""
    models = [[int(model.link), [list(row) for row in model.orient],
               None if model.pivot2 is None else [int(v) for v in model.pivot2],
               None if model.parent is None or not scene.has_uid(model.parent) else scene.index_of(model.parent),
               bool(model.empty)]
              for model in scene.models]
    def clip_json(clip):
        out = anim.to_json(clip)
        for track, track_json in zip(clip.tracks, out["tracks"]):
            target = track.target
            track_json["target"]["model"] = None if target.kind == anim.COLOUR else scene.index_of(target.uid)
        return out

    clips = [clip_json(c) for c in scene.clips]
    live = [live_core.to_json(shader, scene.index_of(uid))
            for uid, shaders in scene.live.items() if scene.has_uid(uid) for shader in shaders]
    bricks = [[i, model.grid.bricks.to_json()] for i, model in enumerate(scene.models)
              if model.grid.bricks is not None]
    return {"version": VERSION, "models": models, "clips": clips,
            "active_clip": int(scene.active_clip), "live": live, "colours": scene.colours.to_json(),
            "cell": _cell_json(scene), "bricks": bricks, "document": doc}


def _check_version(data) -> None:
    if not isinstance(data, dict) or data.get("version") != VERSION:
        got = data.get("version") if isinstance(data, dict) else type(data).__name__
        raise RokFileError(f"Not a Rok file: version {got!r}, Rok reads version {VERSION}")


def _apply_meta(scene: VoxScene, data: dict) -> dict:
    """Give a scene read from `scene.vox` what rok.json holds. Returns the document."""
    _check_version(data)
    try:
        uids = [model.uid for model in scene.models]
        for model, entry in zip(scene.models, data.get("models", [])):
            if len(entry) != 5:
                raise ValueError("a model entry does not have five values")
            link_id, orient, pivot2, parent, empty = entry
            model.link = int(link_id)
            model.orient = link.as_orient(orient)
            if pivot2 is not None:
                model.pivot2 = tuple(int(v) for v in pivot2)
            if parent is not None:
                model.parent = uids[int(parent)]
            model.empty = bool(empty)
        if not data.get("clips"):
            raise ValueError("it holds no clips")
        scene.clips = [anim.from_json(c, uids=uids) for c in data["clips"]]
        scene.active_clip = min(max(int(data.get("active_clip", 0)), 0), len(scene.clips) - 1)
        if data.get("colours"):
            scene.colours = ColourStore.from_json(data["colours"], scene.palette, scene.materials)
        for item in data.get("live", []):
            index, shader = live_core.from_json(item)
            scene.live.setdefault(uids[index], []).append(shader)
        for index, table_json in data.get("bricks", []):
            grid = scene.models[int(index)].grid
            table = brick.BrickTable.from_json(table_json)
            fault = table.check(grid.voxels)
            if fault is not None:
                raise ValueError(f"brick table of model {index}: {fault}")
            grid.bricks = table
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise RokFileError(f"Not a Rok file: {exc}") from exc
    scene.next_link = max((model.link for model in scene.models), default=0) + 1
    _apply_cell(scene, data)
    return data.get("document", {})


def _cell_json(scene: VoxScene):
    """The voxel length for rok.json, or None when labels stay in cells."""
    metres = scene.cell_metres
    if isinstance(metres, (int, float)) and metres > 0:
        unit = scene.cell_unit if scene.cell_unit in ("m", "ft") else "m"
        return {"metres": float(metres), "unit": unit}
    return None


def _apply_cell(scene: VoxScene, data: dict) -> None:
    """Read "cell". A missing or broken entry leaves the labels in cells."""
    scene.cell_metres = None
    scene.cell_unit = "m"
    cell = data.get("cell")
    if not isinstance(cell, dict):
        return
    unit = cell.get("unit")
    if unit not in ("m", "ft"):
        unit = "m"
    try:
        metres = float(cell["metres"])
    except (KeyError, TypeError, ValueError):
        return
    if metres > 0 and metres < 1e12:
        scene.cell_metres = metres
        scene.cell_unit = unit


def pack(scene: VoxScene, doc: dict) -> str:
    """The scene and `doc` as one JSON text, the scene packed inside."""
    data = _meta(scene, doc)
    data["scene"] = base64.b64encode(zlib.compress(dumps_scene(scene.to_data(rest=True)))).decode("ascii")
    return json.dumps(data)


# A Rok file never unpacks to more than this; a bigger claim is a broken or hostile file.
MAX_SCENE_BYTES = 1 << 30


def unpack(text: str) -> tuple[VoxScene, dict]:
    """The inverse of pack. Raises RokFileError, starting "Not a Rok file",
    on a text this Rok cannot read: not JSON, not version 1, or a scene,
    models or clips entry that does not decode."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise RokFileError(f"Not a Rok file: {exc}") from exc
    _check_version(data)
    try:
        inflate = zlib.decompressobj()
        blob = inflate.decompress(base64.b64decode(data["scene"]), MAX_SCENE_BYTES)
        if inflate.unconsumed_tail:
            raise ValueError("scene too large")
        scene = VoxScene.from_data(loads_scene(blob))
    except (KeyError, ValueError, TypeError, VoxError, zlib.error) as exc:
        raise RokFileError(f"Not a Rok file: {exc}") from exc
    return scene, _apply_meta(scene, data)


SCENE_NAME = "scene.vox"
META_NAME = "rok.json"


def dumps(scene: VoxScene, doc: dict) -> bytes:
    """The Rok file's bytes: a zip of scene.vox and rok.json."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(SCENE_NAME, dumps_scene(scene.to_data(rest=True)))
        z.writestr(META_NAME, json.dumps(_meta(scene, doc)))
    return out.getvalue()


def loads(data: bytes) -> tuple[VoxScene, dict]:
    """The inverse of dumps. Raises RokFileError on bytes this Rok cannot read."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in (SCENE_NAME, META_NAME):
                if z.getinfo(name).file_size > MAX_SCENE_BYTES:
                    raise ValueError(f"{name} too large")
            blob = z.read(SCENE_NAME)
            meta = json.loads(z.read(META_NAME))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise RokFileError(f"Not a Rok file: {exc}") from exc
    try:
        scene = VoxScene.from_data(loads_scene(blob))
    except VoxError as exc:
        raise RokFileError(f"Not a Rok file: {exc}") from exc
    return scene, _apply_meta(scene, meta)


def write(path, scene: VoxScene, doc: dict) -> None:
    """Write a Rok file. The old file stays until the new one is complete."""
    write_bytes(path, dumps(scene, doc))


def read(path) -> tuple[VoxScene, dict]:
    return loads(Path(path).read_bytes())


def needs_rok(scene: VoxScene) -> tuple[str, ...]:
    """Why a `.vox` cannot keep this scene, as short phrases; () when it can."""
    reasons = []
    if any(scene.live.get(m.uid) for m in scene.models):
        reasons.append("live shaders")
    if anim.vox_losses(scene) or (scene.timeline.has_keys() and scene.timeline.fps != anim.VOX_FPS):
        reasons.append("animation a .vox would simplify")
    if any(m.pivot2 is not None or m.parent is not None for m in scene.models):
        reasons.append("pivots and parents")
    metres = scene.cell_metres
    if isinstance(metres, (int, float)) and metres > 0:
        reasons.append("a voxel length")
    if any(m.grid.bricks is not None for m in scene.models):
        reasons.append("bricks")
    return tuple(reasons)
