"""Write the three test models: tests/fixtures/robot.vox, bird.vox and car.vox.

Run once: `.venv-rok/bin/python scripts/make_models.py`, then
`scripts/make_golden.py` for the goldens that are computed from these models.

The geometry is built here from boxes, spheres and rows of cells, so the models
are Rok's own. Each file holds what a typical saved .vox file holds: one
model, 8 layers, a palette, 256 MATL chunks and 13 rOBJ render settings.

robot.vox, 24 x 30 x 20: a small robot on two legs. Glass visor, two lit eyes
and a lit antenna, metal legs and arms. The floor corner x 0-3, y 0-2 is empty.
bird.vox, 80 x 68 x 22: a fire bird with its wings along x, its head at +y and
its tail at -y. Glass eyes, glowing wing tips, tail and crest, metal beak and legs.
car.vox, 40 x 60 x 14: a car that points along +y. Glass windows, lit head and
tail lights, metal wheel hubs and paint.

All three are symmetric in x (a mirror on X changes nothing), as the tests of
the X mirror and of flips expect.

The tests use the palette slots and their materials (tests/test_material.py,
tests/test_palette.py): which slots each model uses, which slots carry which
material, and that no two models share an exact colour in a used slot. The
material text is written out key by key in the .vox order, not with
rok.core.material, so the material tests compare the code against text it did
not make.
"""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rok.core.material import Material, MaterialTable  # noqa: E402
from rok.core.palette import DEFAULT_PALETTE  # noqa: E402
from rok.core.vox_read import ModelData, SceneData  # noqa: E402
from rok.core.vox_write import DEFAULT_LAYERS, dumps_scene  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"

# ---------------------------------------------------------------- materials

# The MATL keys in the .vox order, with its Plain values.
MATL_KEYS = ("_type", "_weight", "_rough", "_spec", "_spec_p", "_ior", "_att", "_g0", "_g1", "_gw", "_flux", "_ldr")
MATL_PLAIN = ("_diffuse", "1", "0.1", "0.5", "0.5", "0.3", "0", "-0.5", "0.8", "0.7", "0", "0")


def matl(kind: str = "_diffuse", **values: str) -> Material:
    """One MATL slot as a .vox file holds it: every key, Plain values except `values`."""
    text = dict(zip(MATL_KEYS, MATL_PLAIN))
    text["_type"] = kind
    for key, value in values.items():
        assert "_" + key in text, key
        text["_" + key] = value
    return Material(tuple((k.encode(), text[k].encode()) for k in MATL_KEYS))


def table(slots: dict[int, Material]) -> MaterialTable:
    """All 256 MATL ids in order, Plain where `slots` has nothing."""
    return MaterialTable({i: slots.get(i, matl()) for i in range(1, 257)})


ROBOT_MATERIALS = {
    169: matl("_plastic", rough="0.8"),
    177: matl("_glass", weight="0.43", ior="0.1"),
    185: matl("_emit", weight="0.05", flux="4"),
    186: matl("_emit", weight="0.65", flux="1"),
    193: matl("_plastic", rough="0.73"),
    209: matl("_metal", weight="0.59", rough="0.67", spec="0.1"),
}
BIRD_MATERIALS = {
    81: matl("_plastic", rough="0.78", spec_p="0.04"),
    162: matl("_emit", flux="2"),
    169: matl("_glass", weight="0.52", rough="0.04", ior="0.23"),
    170: matl("_emit", weight="0.73", flux="1"),
    171: matl("_emit", flux="2"),
    177: matl("_plastic", rough="0.72", spec_p="0.05"),
    178: matl("_plastic", rough="0.66", spec_p="0.05"),
    180: matl("_metal", rough="0.52"),
}
CAR_MATERIALS = {
    79: matl("_plastic", weight="0.82", rough="0.58", spec_p="0.62"),
    87: matl("_metal", weight="0.78", rough="0.66"),
    95: matl("_plastic", weight="0.71"),
    103: matl("_metal", weight="0.76", rough="0.6"),
    161: matl("_plastic", rough="0.28", spec="0.83"),
    169: matl("_emit", flux="1"),
    170: matl("_emit", flux="1"),
    171: matl("_emit", flux="1"),
    172: matl("_emit", flux="2"),
    173: matl("_glass", weight="0.85", rough="0.81", ior="0.24", att="0.77"),
    174: matl("_emit", weight="0.84", flux="4"),
    209: matl("_metal", weight="0.9", rough="0.85", spec="0.58", spec_p="0.18"),
    211: matl("_metal", weight="0.83", rough="0.25", spec="0.73"),
}

# ------------------------------------------------------------------ palettes


def base_palette() -> np.ndarray:
    """A hue ramp in 15 rows of 17 shades. Slot 1 is white as in the .vox
    default palette; no other slot holds its default colour."""
    pal = np.zeros((256, 4), dtype=np.uint8)
    for i in range(1, 256):
        row, step = divmod(i - 1, 17)
        hue = row / 15.0
        value = 1.0 - step / 18.0
        k = (np.array([5.0, 3.0, 1.0]) + hue * 6.0) % 6.0  # HSV to RGB, saturation 0.6
        rgb = value * (1.0 - 0.6 * np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0))
        pal[i, :3] = np.round(255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        pal[i, 3] = 255
        if np.array_equal(pal[i, :3], DEFAULT_PALETTE[i, :3]):
            pal[i, 2] ^= 1
    pal[1] = (255, 255, 255, 255)
    return pal


def palette_with(colours: dict[int, tuple[int, int, int]]) -> np.ndarray:
    pal = base_palette()
    for i, rgb in colours.items():
        pal[i] = (*rgb, 255)
    return pal


ROBOT_COLOURS = {
    169: (214, 120, 40),   # head shell
    177: (60, 150, 190),   # visor glass
    185: (255, 60, 30),    # eyes
    186: (40, 255, 160),   # antenna
    193: (170, 172, 178),  # body
    201: (70, 72, 80),     # feet
    209: (110, 112, 118),  # legs and arms
}
BIRD_COLOURS = {
    81: (120, 30, 20),     # leading edges and tail
    162: (255, 250, 200),  # eye glints
    169: (20, 24, 40),     # eye glass
    170: (90, 200, 255),   # crest
    171: (255, 120, 20),   # glowing wing tips
    177: (200, 50, 30),    # body and head
    178: (230, 90, 40),    # wings
    179: (240, 200, 160),  # belly
    180: (150, 150, 160),  # beak and legs
}
CAR_COLOURS = {
    79: (40, 90, 160),     # body
    87: (30, 60, 120),     # cabin
    95: (30, 30, 34),      # tyres and bumpers
    103: (180, 182, 186),  # hubs
    172: (255, 245, 210),  # head lights
    173: (140, 200, 230),  # windows
    174: (255, 30, 40),    # tail lights
}

# ---------------------------------------------------------- render settings


def _dict(pairs: list[tuple[str, str]]) -> bytes:
    out = struct.pack("<i", len(pairs))
    for key, value in pairs:
        for text in (key.encode(), value.encode()):
            out += struct.pack("<i", len(text)) + text
    return out


# 13 rOBJ chunks: the render settings a .vox file carries.
RENDER = [
    [("_type", "_env"), ("_mode", "0")],
    [("_type", "_inf"), ("_i", "0.8"), ("_k", "255 250 240"), ("_angle", "40 30"), ("_area", "0.05"), ("_disk", "0")],
    [("_type", "_uni"), ("_i", "0.4"), ("_k", "200 210 220")],
    [("_type", "_atm"), ("_ray_d", "0.3"), ("_ray_k", "60 110 255"), ("_mie_d", "0.3"), ("_mie_k", "255 255 255"),
     ("_mie_g", "0.8"), ("_o3_d", "0"), ("_o3_k", "100 255 100")],
    [("_type", "_fog_uni"), ("_d", "0"), ("_k", "255 255 255"), ("_g", "0.5")],
    [("_type", "_lens"), ("_proj", "0"), ("_fov", "40"), ("_aperture", "0.2"), ("_blade_n", "0"), ("_blade_r", "0")],
    [("_type", "_film"), ("_expo", "1"), ("_vig", "0.1"), ("_aces", "1"), ("_gam", "2.2")],
    [("_type", "_bloom"), ("_mix", "0.2"), ("_scale", "0.1"), ("_aspect", "0.5"), ("_threshold", "1")],
    [("_type", "_ground"), ("_color", "100 100 100"), ("_hor", "0.1")],
    [("_type", "_bg"), ("_color", "0 0 0")],
    [("_type", "_edge"), ("_color", "0 0 0"), ("_width", "0.2")],
    [("_type", "_grid"), ("_color", "0 0 0"), ("_spacing", "1"), ("_width", "0.02"), ("_display", "0")],
    [("_type", "_setting"), ("_ground", "1"), ("_grid", "0"), ("_edge", "0"), ("_bg_c", "0"), ("_bg_a", "0"),
     ("_scale", "1 1 1"), ("_cell", "1")],
]

# ------------------------------------------------------------------ geometry


def box(v, x0, x1, y0, y1, z0, z1, colour) -> None:
    """Fill the cells x0..x1, y0..y1, z0..z1 (inclusive)."""
    v[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = colour


def ellipsoid(v, centre, radii, colour, where=None) -> None:
    x, y, z = np.indices(v.shape)
    d = sum(((c - m) / r) ** 2 for c, m, r in zip((x + 0.5, y + 0.5, z + 0.5), centre, radii))
    inside = d <= 1.0
    if where is not None:
        inside &= where(x, y, z)
    v[inside] = colour


def make_robot() -> np.ndarray:
    v = np.zeros((24, 30, 20), dtype=np.uint8)
    box(v, 6, 11, 11, 19, 0, 1, 201)  # feet
    box(v, 12, 17, 11, 19, 0, 1, 201)
    box(v, 7, 10, 12, 17, 2, 5, 209)  # legs
    box(v, 13, 16, 12, 17, 2, 5, 209)
    box(v, 5, 18, 9, 20, 6, 13, 193)  # body
    box(v, 6, 17, 10, 19, 7, 12, 0)  # the body is a shell
    box(v, 2, 4, 13, 16, 7, 12, 209)  # arms
    box(v, 19, 21, 13, 16, 7, 12, 209)
    box(v, 2, 4, 12, 17, 6, 6, 169)  # hands
    box(v, 19, 21, 12, 17, 6, 6, 169)
    box(v, 7, 16, 10, 19, 14, 18, 169)  # head
    box(v, 8, 15, 10, 10, 15, 17, 177)  # visor, on the -y face
    v[10, 10, 16] = v[13, 10, 16] = 185  # eyes behind the visor line
    v[9, 10, 16] = v[14, 10, 16] = 185
    box(v, 11, 12, 14, 15, 19, 19, 186)  # antenna
    return v


def make_bird() -> np.ndarray:
    v = np.zeros((80, 68, 22), dtype=np.uint8)
    cx = 40.0  # the bird is symmetric about the plane x = 39.5
    # Wings: one row of cells per x, swept back (-y) and rising toward the tips.
    for x in range(80):
        d = 31 - x if x < 32 else x - 48
        if d < 0:
            continue
        y0 = 22 - d // 10
        y1 = 40 - (d * 2) // 10
        z = 9 + d // 8
        tip = d >= 27
        v[x, y0 : y1 + 1, z : z + 2] = 171 if tip else 178
        v[x, y1, z : z + 2] = 171 if tip else 81  # the leading edge
    # Tail: a fan that widens toward -y.
    for y in range(0, 16):
        w = 3 + (15 - y) // 2
        v[int(cx) - w : int(cx) + w, y, 9:11] = 171 if y < 3 else 81
    ellipsoid(v, (cx, 32, 11), (9, 22, 6), 177)  # body
    ellipsoid(v, (cx, 32, 11), (9, 22, 6), 179, where=lambda x, y, z: z < 8)  # belly
    ellipsoid(v, (cx, 55, 14), (7, 7, 7), 177)  # head
    box(v, 38, 41, 61, 67, 12, 14, 180)  # beak
    box(v, 38, 41, 65, 67, 14, 14, 0)
    box(v, 39, 40, 53, 57, 20, 21, 170)  # crest
    box(v, 34, 34, 58, 60, 13, 15, 169)  # eyes
    box(v, 45, 45, 58, 60, 13, 15, 169)
    v[34, 60, 15] = v[45, 60, 15] = v[34, 59, 15] = v[45, 59, 15] = 162
    box(v, 36, 37, 28, 29, 0, 5, 180)  # legs
    box(v, 42, 43, 28, 29, 0, 5, 180)
    box(v, 35, 38, 28, 31, 0, 0, 180)  # toes
    box(v, 41, 44, 28, 31, 0, 0, 180)
    return v


def make_car() -> np.ndarray:
    v = np.zeros((40, 60, 14), dtype=np.uint8)
    box(v, 4, 35, 4, 55, 3, 7, 79)  # body
    box(v, 5, 34, 5, 54, 4, 6, 0)  # the body is a shell
    box(v, 7, 32, 18, 40, 8, 12, 87)  # cabin
    box(v, 7, 7, 20, 38, 9, 11, 173)  # side windows
    box(v, 32, 32, 20, 38, 9, 11, 173)
    box(v, 9, 30, 40, 40, 9, 11, 173)  # windscreen (front, +y)
    box(v, 9, 30, 18, 18, 9, 11, 173)  # rear window
    box(v, 9, 30, 20, 38, 13, 13, 79)  # roof
    _, y, z = np.indices(v.shape)
    for cy in (12, 47):  # wheels
        disc = (y + 0.5 - (cy + 0.5)) ** 2 + (z + 0.5 - 4.5) ** 2
        for x0, x1 in ((0, 3), (36, 39)):
            span = np.zeros(v.shape, dtype=bool)
            span[x0 : x1 + 1] = True
            v[span & (disc <= 16.5)] = 95
            v[span & (disc <= 3.0)] = 103
    box(v, 5, 34, 56, 57, 2, 3, 95)  # bumpers
    box(v, 5, 34, 2, 3, 2, 3, 95)
    box(v, 6, 10, 55, 55, 5, 6, 172)  # head lights
    box(v, 29, 33, 55, 55, 5, 6, 172)
    box(v, 6, 9, 4, 4, 6, 6, 174)  # tail lights
    box(v, 30, 33, 4, 4, 6, 6, 174)
    return v


MODELS = {
    "robot": (make_robot, ROBOT_COLOURS, ROBOT_MATERIALS),
    "bird": (make_bird, BIRD_COLOURS, BIRD_MATERIALS),
    "car": (make_car, CAR_COLOURS, CAR_MATERIALS),
}


def scene_of(name: str) -> SceneData:
    make, colours, materials = MODELS[name]
    voxels = make()
    used = set(np.unique(voxels).tolist()) - {0}
    assert used == set(colours), (name, sorted(used))
    size = voxels.shape
    position = (-(size[0] // 2), -(size[1] // 2), 0)
    return SceneData(
        models=[ModelData(size, voxels, "", position)],
        palette=palette_with(colours),
        palette_tail=b"\0\0\0\0",
        layers=list(DEFAULT_LAYERS),
        extras=[(b"rOBJ", _dict(pairs)) for pairs in RENDER],
        materials=table(materials),
    )


def main() -> None:
    for name in MODELS:
        path = FIXTURES / f"{name}.vox"
        data = scene_of(name)
        path.write_bytes(dumps_scene(data))
        count = int(np.count_nonzero(data.models[0].voxels))
        print(f"wrote {path} ({data.models[0].size}, {count} voxels)")


if __name__ == "__main__":
    main()
