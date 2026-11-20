"""Onion skin: in Animate mode, faint copies of
each part's pose at the key before and the key after the play head. The copy
before is violet, the one after ember; a part with no key on either side
shows no copy. The copies are the posed grids Voxel look would show
(anim_mode._posed), as faces, in one draw layer that only rebuilds when the
play head, the clip or the voxels change.
"""

from __future__ import annotations

import numpy as np

from rok.core import anim, hud

from . import anim_mode
from . import object as voxel_object
from .gfx import pen
from .handlers import registry
from .object import document
from .state import state

BEFORE = (0.55, 0.42, 0.95, 0.18)
AFTER = (0.91, 0.36, 0.23, 0.18)
Z = 9  # under the view overlay (overlay.VIEW_Z = 10)

_cache: dict = {}  # (uid, time, stroke serial, pose key) -> (tris, place)
_MAX = 96


def neighbours(track: anim.Track | None, t: int) -> tuple[int | None, int | None]:
    """The key before and the key after frame t on the part's track."""
    if track is None:
        return None, None
    times = track.key_times()
    before = max((k for k in times if k < t), default=None)
    after = min((k for k in times if k > t), default=None)
    return before, after


def copies(scene, clip: anim.Clip, t: int) -> list[tuple[int, int, np.ndarray, tuple]]:
    """(uid, time, faces (N, 3) in world units, colour) for every part's copy before and after t."""
    out = []
    frames: dict[int, anim.Frame] = {}
    for model in scene.models:
        track = clip.track_of(model.uid)
        before, after = neighbours(track, t)
        for time, colour in ((before, BEFORE), (after, AFTER)):
            if time is None:
                continue
            fr = frames.get(time)
            if fr is None:
                fr = frames[time] = anim.evaluate(clip, scene, time)
            pose = fr.poses.get(model.uid)
            voxels = model.grid.visible
            source = fr.state.get(model.uid, 0)
            if 0 < source <= len(model.states):
                voxels = model.states[source - 1]
            if pose is None:
                grid, origin = voxels, (0, 0, 0)
                key = (model.uid, time, document.stroke_serial, "rest", source)
            else:
                pose_key, hit = anim_mode._posed(model, voxels, pose, source)
                if hit is None:
                    continue
                grid, origin = hit
                key = (model.uid, time, document.stroke_serial, pose_key)
            found = _cache.get(key)
            if found is None:
                mask = grid != 0
                if not mask.any():
                    continue
                tris, _lines = hud.mask_surface(mask, (0, 0, 0), tuple(int(d) - 1 for d in mask.shape), 0.0)
                found = _cache[key] = tris
                while len(_cache) > _MAX:
                    del _cache[next(iter(_cache))]
            place = np.asarray(voxel_object.model_origin(model), dtype=np.float64) + np.asarray(origin, dtype=np.float64)
            out.append((model.uid, time, found + place.astype(np.float32), colour))
    return out


def _draw(context) -> None:
    if not state.animating or not state.onion_skin or document.scene is None:
        return
    clip = anim_mode.clip()
    if clip is None or not clip.tracks:
        return
    for _uid, _time, tris, colour in copies(document.scene, clip, anim_mode.frame()):
        pen.tris(tris, colour)


registry.add_draw("anim_onion", Z, "VIEW", _draw)
