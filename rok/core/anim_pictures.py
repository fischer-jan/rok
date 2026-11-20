"""Pictures of a clip: the sprite sheet, and the frames of a GIF or a video.
The pictures come from the renderer
(rok/app/anim_pictures.py); this module only plans and assembles them.

- **Sprite sheet**: one row per direction, one column per frame, every cell
  the same size and the same framing, on a transparent background. The JSON
  beside it is in the array format Aseprite writes: one entry per cell, one
  frame tag per direction, the pivot as a slice, and Rok's markers.
- **Transparency**: the renderer draws each picture twice, on black and on
  white. A pixel the background shows through changes between the two, and
  by how much says how much shows through (`matte`).

numpy and stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import anim

DIRECTIONS = (1, 4, 8, 16)


def video_times(clip: anim.Clip) -> list[float]:
    """The clip frames a GIF or a video shows, once round: a loop without its
    closing frame (the player loops), Ping-pong there and back."""
    n = max(1, int(clip.length))
    if clip.loop_mode == anim.PING_PONG and n > 1:
        return [float(clip.time_at(k)) for k in range(2 * (n - 1))]
    return [float(k) for k in range(n)]


def sheet_times(clip: anim.Clip, fps: int = 0) -> list[float]:
    """The clip times of a sprite sheet's columns at `fps` (0: the clip's
    own). Each direction plays them one way; the tag says how to loop."""
    fps = int(fps) or int(clip.fps)
    count = max(1, math.ceil(clip.length * fps / max(1, clip.fps) - 1e-9))
    return [k * clip.fps / fps for k in range(count)]


def tag_direction(clip: anim.Clip) -> str:
    return "pingpong" if clip.loop_mode == anim.PING_PONG else "forward"


def direction_angles(count: int) -> list[float]:
    """The turns of the camera about the up axis, in degrees, one per direction."""
    if count not in DIRECTIONS:
        raise ValueError(f"a sprite sheet has 1, 4, 8 or 16 directions, not {count}")
    return [360.0 * k / count for k in range(count)]


def _to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


def matte(on_black: np.ndarray, on_white: np.ndarray) -> np.ndarray:
    """One RGBA picture from the same picture drawn on black and on white
    (H x W x 3 or 4 uint8, sRGB). The renderer blends in linear light, so
    the alpha is 1 minus the difference there, and the colour is what is
    left of the picture on black."""
    b = _to_linear(np.asarray(on_black, dtype=np.float64)[..., :3])
    w = _to_linear(np.asarray(on_white, dtype=np.float64)[..., :3])
    alpha = np.clip(1.0 - (w - b).mean(axis=-1), 0.0, 1.0)
    alpha[alpha < 1.5 / 255] = 0.0
    colour = b / np.maximum(alpha, 1e-6)[..., None]
    out = np.zeros(b.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.rint(_to_srgb(colour)).astype(np.uint8)
    out[..., 3] = np.rint(alpha * 255).astype(np.uint8)
    out[out[..., 3] == 0, :3] = 0
    return out


@dataclass(frozen=True)
class Sheet:
    """A sprite sheet: the picture and the JSON that describes it."""

    image: np.ndarray  # H x W x 4 uint8
    json: dict


def assemble(rows: list[list[np.ndarray]], clip: anim.Clip, fps: int, angles: list[float], image_name: str,
             pivot: tuple[float, float] | None = None) -> Sheet:
    """The sheet of `rows` (one per direction, one RGBA cell per frame) and
    its Aseprite JSON. `pivot` is the point (pixels in a cell) the sprite
    stands on; `fps` the sheet's frame rate."""
    if not rows or not rows[0]:
        raise ValueError("a sprite sheet needs at least one cell")
    ch, cw = rows[0][0].shape[:2]
    cols = len(rows[0])
    image = np.zeros((ch * len(rows), cw * cols, 4), dtype=np.uint8)
    frames = []
    tags = []
    fps = int(fps) or int(clip.fps)
    duration = round(1000 / max(1, fps))
    for r, (row, angle) in enumerate(zip(rows, angles)):
        name = clip.name if len(rows) == 1 else f"{clip.name} {angle:g}"
        tags.append({"name": name, "from": r * cols, "to": r * cols + cols - 1, "direction": tag_direction(clip)})
        for c, cell in enumerate(row):
            image[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
            frames.append({
                "filename": f"{name} {c}",
                "frame": {"x": c * cw, "y": r * ch, "w": cw, "h": ch},
                "rotated": False,
                "trimmed": False,
                "spriteSourceSize": {"x": 0, "y": 0, "w": cw, "h": ch},
                "sourceSize": {"w": cw, "h": ch},
                "duration": duration,
            })
    meta = {
        "app": "Rok",
        "version": "1",
        "image": image_name,
        "format": "RGBA8888",
        "size": {"w": int(image.shape[1]), "h": int(image.shape[0])},
        "scale": "1",
        "frameTags": tags,
        "layers": [],
        "slices": [],
        "fps": fps,
        "loop": clip.loop_mode != anim.ONCE,
        "directions": [float(a) for a in angles],
        # A marker's column: the first frame at or after its clip frame.
        "markers": [{"name": m.name, "frame": min(cols - 1, math.ceil(m.time * fps / max(1, clip.fps) - 1e-9))}
                    for m in sorted(clip.markers, key=lambda m: m.time)],
    }
    if pivot is not None:
        px, py = (int(round(v)) for v in pivot)
        meta["slices"].append({"name": "pivot", "color": "#0000ffff",
                               "keys": [{"frame": 0, "bounds": {"x": 0, "y": 0, "w": cw, "h": ch},
                                         "pivot": {"x": px, "y": py}}]})
    return Sheet(image, {"frames": frames, "meta": meta})


__all__ = ["DIRECTIONS", "Sheet", "assemble", "direction_angles", "matte", "sheet_times", "tag_direction",
           "video_times"]
