"""Write the Phase 2 and Phase 3 golden files from the model fixtures.

Run once, after scripts/make_models.py: `uv run python scripts/make_golden.py`. The Phase 2 goldens use the
slow reference floods in tests/reference.py, never the fast code, so they are
independent of the code they check.

tests/fixtures/phase2/face_regions.json: per fixture, the first and the last
filled voxel (np.argwhere order) with each of their +Z, +X and -Y faces that is
exposed, plus one wall_region entry on the floor.
tests/fixtures/phase2/connected_colour.json: per fixture, the first, middle and
last filled voxel.
tests/fixtures/phase3/two_models.vox: robot.vox and car.vox in one scene,
written by the Phase 3 writer. tests/test_vox_scene.py pins its graph bytes
by hand, so the golden cannot drift with the writer.
tests/fixtures/anim/spin.vox: reference._spin_file() with a fresh LINEAR
ROTATION baked onto it (Task 2.3); tests/test_vox_scene.py pins its frame
dicts by hand too.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import reference  # noqa: E402
from rok.core import anim, palette  # noqa: E402
from rok.core.scene import VoxScene  # noqa: E402
from rok.core.vox_read import ModelData, SceneData, loads_scene, read_scene, read_vox  # noqa: E402
from rok.core.vox_write import DEFAULT_LAYERS, dumps_scene  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
OUT = FIXTURES / "phase2"
OUT3 = FIXTURES / "phase3"
OUT_ANIM = FIXTURES / "anim"
FACE_NORMALS = ((0, 0, 1), (1, 0, 0), (0, -1, 0))


def is_exposed(voxels, voxel, normal) -> bool:
    """True when the neighbour on the normal side is empty or outside the grid."""
    n = [int(voxel[i]) + int(normal[i]) for i in range(3)]
    for i in range(3):
        if n[i] < 0 or n[i] >= voxels.shape[i]:
            return True
    return int(voxels[n[0], n[1], n[2]]) == 0


def summary(cells) -> dict:
    return {
        "count": int(np.asarray(cells).reshape(-1, 3).shape[0]),
        "bbox": reference.cells_bbox(cells),
        "sha1": reference.cells_sha1(cells),
    }


def first_empty_floor_column(voxels):
    empty = np.argwhere(voxels[:, :, 0] == 0)
    if empty.shape[0] == 0:
        return None
    return [int(empty[0][0]), int(empty[0][1])]


def face_entries(stem, voxels) -> list[dict]:
    filled = np.argwhere(voxels != 0)
    entries = []
    for voxel in (filled[0], filled[-1]):
        voxel = [int(c) for c in voxel]
        for normal in FACE_NORMALS:
            if not is_exposed(voxels, voxel, normal):
                continue
            cells = reference.face_region_ref(voxels, voxel, normal)
            entries.append(
                {"fixture": stem, "voxel": voxel, "normal": list(normal), "wall": False}
                | summary(cells)
            )
    column = first_empty_floor_column(voxels)
    if column is not None:
        voxel = [column[0], column[1], -1]
        normal = (0, 0, 1)
        target = [voxel[i] + normal[i] for i in range(3)]
        cells = reference.wall_region_ref(voxels, target, normal)
        entries.append(
            {"fixture": stem, "voxel": voxel, "normal": list(normal), "wall": True}
            | summary(cells)
        )
    return entries


def colour_entries(stem, voxels) -> list[dict]:
    filled = np.argwhere(voxels != 0)
    entries = []
    for voxel in (filled[0], filled[len(filled) // 2], filled[-1]):
        voxel = [int(c) for c in voxel]
        mask = reference.connected_colour_ref(voxels, voxel)
        entries.append({"fixture": stem, "voxel": voxel} | summary(np.argwhere(mask)))
    return entries


def write_two_models() -> Path:
    """robot.vox at (-40, 0, 0) named Robot plus car.vox at (10, -5, 3) named Car."""
    robot = read_scene(FIXTURES / "robot.vox")
    car = read_scene(FIXTURES / "car.vox")
    robot_used = np.bincount(robot.models[0].voxels.ravel(), minlength=256)
    car_used = np.bincount(car.models[0].voxels.ravel(), minlength=256)
    merged, remap = palette.merge(robot.palette, robot_used, car.palette, car_used)
    car_voxels = remap[car.models[0].voxels]
    data = SceneData(
        models=[
            ModelData(robot.models[0].size, robot.models[0].voxels, "Robot", (-40, 0, 0)),
            ModelData(car.models[0].size, car_voxels, "Car", (10, -5, 3)),
        ],
        palette=merged,
        palette_tail=b"\0\0\0\0",
        layers=list(DEFAULT_LAYERS),
        extras=[],
    )
    OUT3.mkdir(parents=True, exist_ok=True)
    path = OUT3 / "two_models.vox"
    path.write_bytes(dumps_scene(data))
    return path


def write_anim_goldens() -> Path:
    """spin.vox: reference._spin_file()'s cube with a fresh LINEAR ROTATION
    key at 0 (0, 0, 0) and 4 (0, 0, 360), on a VOX_FPS timeline, baked and
    written back (Task 2.3, section 5.4)."""
    scene = VoxScene.from_data(loads_scene(reference._spin_file()))
    model = scene.models[0]
    track = anim.Track(anim.Target(anim.PART, model.uid))
    anim.set_key(track, anim.ROTATION, anim.Key(0, (0, 0, 0), anim.LINEAR))
    anim.set_key(track, anim.ROTATION, anim.Key(4, (0, 0, 360), anim.LINEAR))
    scene.timeline.tracks.append(track)
    scene.timeline.fps = anim.VOX_FPS
    OUT_ANIM.mkdir(parents=True, exist_ok=True)
    path = OUT_ANIM / "spin.vox"
    path.write_bytes(dumps_scene(scene.to_data()))
    return path


def main() -> None:
    faces = []
    colours = []
    for path in sorted(FIXTURES.glob("*.vox")):
        voxels = read_vox(path).voxels
        faces.extend(face_entries(path.stem, voxels))
        colours.extend(colour_entries(path.stem, voxels))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "face_regions.json").write_text(json.dumps(faces, indent=1) + "\n")
    (OUT / "connected_colour.json").write_text(json.dumps(colours, indent=1) + "\n")
    print(f"wrote {len(faces)} face entries and {len(colours)} colour entries to {OUT}")
    print(f"wrote {write_two_models()}")
    print(f"wrote {write_anim_goldens()}")


if __name__ == "__main__":
    main()
