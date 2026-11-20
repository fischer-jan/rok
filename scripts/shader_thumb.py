"""Render a shader's gallery thumbnail: assets/icons/shaders/<group>__<name>.png.

    .venv-rok/bin/python scripts/shader_thumb.py hard_surface/greebles   (no names: every shader)

The shader runs on a demo model with the settings in SETUPS, as the dialog
runs it, and the result is drawn in the Iso look from the usual angle,
256 px square on a transparent background. The Iso look draws its own
background, so the transparency comes
from two Studio renders, on black and on white: what differs is background.
The floor grid is faint in both and drops out.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rok.app import xs_ops  # noqa: E402
from rok.app.looks import LOOKS  # noqa: E402
from rok.core import png, xs  # noqa: E402
from rok.core.palette import default_palette  # noqa: E402
from rok.core.vox_read import VoxModel  # noqa: E402
from rok.core.vox_write import write_vox  # noqa: E402

OUT = ROOT / "assets" / "icons" / "shaders"
SIZE = 256
SCALE = 4  # rendered this many times larger, then averaged down
GREY, RED, GREEN = 250, 216, 155  # The .vox default palette slots (155 is 33cc33)


def model(name: str) -> np.ndarray:
    """The demo models."""
    if name.startswith("empty:"):
        return np.zeros(tuple(int(n) for n in name[6:].split("x")), np.uint8)
    if name == "house":
        v = np.zeros((32, 32, 24), np.uint8)
        v[2:30, 2:30, 0:2] = GREY
        v[8:24, 8:24, 2:14] = GREY
        v[10:22, 10:22, 2:13] = 0
        v[14:18, 8, 2:8] = 0
        v[6:26, 6:26, 14:16] = RED
        v[9:23, 9:23, 16:18] = RED
        return v
    if name == "hill":
        v = np.zeros((32, 32, 20), np.uint8)
        x, y = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        h = np.clip(15 - np.hypot(x - 12, y - 16) * 0.8, 1, 18).astype(int)
        h = np.maximum(h, np.clip(12 - np.hypot(x - 26, y - 8) * 0.9, 1, 18).astype(int))
        z = np.arange(20)[None, None, :]
        v[z < h[:, :, None]] = GREEN
        return v
    if name == "scatter":
        rng = np.random.default_rng(4)
        v = np.zeros((20, 20, 24), np.uint8)
        v[2:18, 2:18, 0] = GREY
        cells = rng.integers(0, [20, 20, 24], size=(260, 3))
        v[cells[:, 0], cells[:, 1], cells[:, 2]] = RED
        return v
    if name == "hull":  # Rok's greebles demo: big enough for its panels
        v = np.zeros((60, 40, 26), np.uint8)
        v[2:58, 4:36, 2:14] = GREY
        v[14:46, 10:30, 14:19] = GREY
        return v
    if name == "station":  # Rok's hull_windows demo: tall walls for rows of windows
        v = np.zeros((48, 36, 32), np.uint8)
        v[4:44, 4:32, 1:22] = GREY
        v[12:36, 10:26, 22:30] = GREY
        return v
    if name == "meadow_slab":  # the stream's demo: grass on earth, cut through
        v = np.zeros((40, 40, 12), np.uint8)
        v[:, :, 0:5] = 95  # earth
        v[:, :, 5] = 125  # grass
        return v
    if name == "beach":  # the sea's demo: sand that rises to one side
        v = np.zeros((40, 40, 16), np.uint8)
        x, y = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
        shore = 22 + 3 * np.sin(y / 6.0)
        h = np.clip(2 + (x - shore + 8) * 0.5, 1, 10).astype(int)
        z = np.arange(16)[None, None, :]
        v[z < h[:, :, None]] = 45  # sand
        return v
    raise ValueError(name)


# name: (model, colour, {argument name: value}, colours count or None)
SETUPS = {
    "nature/terrain_tiles": ("empty:64x64x28", 40, {}, None),
    "nature/rocks": ("empty:32x32x12", 40, {"Size": 5, "Density": 90, "Moss": 30}, None),
    "nature/cliffs": ("empty:56x56x40", 40, {"Seed": 3}, None),
    "nature/woodland": ("empty:56x56x32", 40, {"Density": 75, "Size": 17}, None),
    "nature/pine_forest": ("empty:56x56x36", 40, {"Density": 80, "Size": 22, "Snow": 55}, None),
    "nature/autumn_grove": ("empty:56x56x32", 40, {"Density": 75, "Size": 17}, None),
    "nature/jungle": ("empty:64x64x40", 40, {"Density": 85, "Size": 22}, None),
    "nature/world_map": ("empty:96x96x40", 40, {}, None),
    "nature/hills": ("empty:64x64x30", 40, {"Scale": 50}, None),
    "nature/mountains": ("empty:64x64x40", 40, {"Scale": 60}, None),
    "nature/islands": ("empty:64x64x30", 40, {"Scale": 50}, None),
    "nature/desert": ("empty:64x64x30", 40, {"Scale": 50, "Style": 2}, None),
    "nature/craters": ("empty:64x64x30", 40, {"Crater Size": 22}, None),
    "nature/volcano": ("empty:64x64x40", 40, {}, None),
    "nature/foliage": ("empty:24x24x20", 40, {}, None),
    "nature/trees": ("empty:28x28x36", 40, {}, None),
    "nature/noise_volume": ("empty:32x32x32", 145, {}, None),
    "nature/mushrooms": ("empty:32x32x28", 40, {}, None),
    "nature/meadow": ("hill", 40, {}, None),
    "nature/clouds": ("empty:96x96x32", 40, {"Size": 40, "Density": 100}, None),
    "weathering/cover": ("house", 40, {}, None),
    "weathering/erode": ("house", 250, {"Amount": 60}, None),
    "weathering/patches": ("house", 40, {}, None),
    "weathering/stains": ("house", 40, {"Length": 12, "Amount": 70}, None),
    "weathering/vines": ("house", 40, {"Density": 45}, None),
    "weathering/water_fill": ("hill", 40, {"Level": 38, "Waves": 3}, None),
    "water/stream": ("meadow_slab", 40, {"Width": 9, "Bend Size": 30}, None),
    "water/sea": ("beach", 40, {"Level": 45, "Depth": 8, "Ripple Size": 5}, None),
    "weathering/gravity": ("house", 250, {}, None),  # the roof caves in
    "weathering/cracks": ("hull", 40, {"Amount": 80, "Size": 8}, None),
    "patterns/bricks": ("house", 40, {}, None),
    "patterns/tiles": ("house", 40, {}, None),
    "patterns/stripes": ("house", 245, {"Mode": 1, "Width": 2}, 2),  # black and white
    "patterns/gradient": ("house", 19, {}, 6),  # pink to orange
    "patterns/noise_dither": ("house", 250, {"Colours": 1, "Only Current": 0, "Scale": 2}, 4),
    "patterns/outline": ("house", 40, {"Shade": 2}, None),
    "patterns/grime": ("house", 40, {"Dirt": 90, "Radius": 3, "Wear": 50}, None),
    "patterns/planks": ("house", 40, {}, None),
    "hard_surface/greebles": ("hull", 40, {"Size": 12, "Density": 90, "Seed": 3}, None),
    "hard_surface/city_blocks": ("empty:72x72x40", 40, {}, None),
    "hard_surface/blocks": ("empty:48x48x24", 250, {}, None),
    "hard_surface/hull_plating": ("hull", 40, {"Details": 70, "Relief": 2}, None),
    "hard_surface/hull_windows": ("station", 40, {"Features": 50}, None),
    "hard_surface/battle_damage": ("hull", 40, {}, None),
}


def render(name: str) -> Path:
    from rok.headless import Session

    model_name, colour, named, count = SETUPS[name]
    shader = xs.load(xs_ops.bundled_folder() / f"{name}.txt", name)
    args = [float(named.get(a.name, a.value)) for a in shader.args]
    v = model(model_name)
    new, pal = xs_ops.run_coloured(shader, v, colour=colour, palette=default_palette(), args=args, colour_count=count)
    if xs.is_brush(shader):
        new = np.where((v > 0) & (new > 0), new, v).astype(np.uint8)  # a brush paints the voxels there only
    LOOKS["_BLACK"] = dataclasses.replace(LOOKS["STUDIO"], background=(0.0, 0.0, 0.0))
    LOOKS["_WHITE"] = dataclasses.replace(LOOKS["STUDIO"], background=(1.0, 1.0, 1.0))
    n = SIZE * SCALE
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.vox"
        write_vox(path, VoxModel(new.shape, new, pal))
        with Session(path) as s:
            black = s.render("iso", (n, n), look="_BLACK", fit="voxels").astype(np.float64) / 255
            white = s.render("iso", (n, n), look="_WHITE", fit="voxels").astype(np.float64) / 255
            look = s.render("iso", (n, n), look="ISO", fit="voxels").astype(np.float64) / 255
    alpha = (1.0 - np.clip((white[..., :3] - black[..., :3]).mean(axis=2), 0, 1) > 0.5).astype(np.float64)
    img = np.dstack([look[..., :3] * alpha[..., None], alpha])  # premultiplied for the box filter
    img = img.reshape(SIZE, SCALE, SIZE, SCALE, 4).mean(axis=(1, 3))
    a = img[..., 3:4]
    img[..., :3] = np.where(a > 1e-3, img[..., :3] / np.maximum(a, 1e-3), 0.0)
    dest = OUT / f"{name.replace('/', '__')}.png"
    dest.write_bytes(png.encode(np.ascontiguousarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8))))
    return dest


def main() -> None:
    for name in sys.argv[1:] or SETUPS:
        print("wrote", render(name))


if __name__ == "__main__":
    main()
