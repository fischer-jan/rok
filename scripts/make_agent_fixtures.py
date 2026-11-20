"""Write the agent tool fixtures in tests/fixtures/agent/.

Run once: `.venv-rok/bin/python scripts/make_agent_fixtures.py`.

landscape.vox: a 64 x 64 x 24 piece of land: a ground slab 2 cells thick
(colour 10), a hill (11) in the back half, and a red post (12) at the front
edge near x = 8. A camera low at the front edge, looking along the ground in
+y, sees the post near and big, and the hill behind it.

many_colours.vox: a 24 x 4 x 4 bar of 24 columns, each its own palette number
from 1 to 24, with a different count each (column k has k + 1 cells). The
facts must list all 24 colours.

crate.vox: a 6 x 6 x 6 crate (colour 5, a lid of colour 7), to add to a scene
with rok_scene do=import.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rok.core.palette import default_palette  # noqa: E402
from rok.core.vox_read import VoxModel  # noqa: E402
from rok.core.vox_write import write_vox  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "agent"


def landscape() -> np.ndarray:
    v = np.zeros((64, 64, 24), dtype=np.uint8)
    v[:, :, 0:2] = 10
    x, y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    height = np.clip(14 - np.hypot(x - 36, y - 46) * 0.7, 0, None).astype(int)  # a round hill at (36, 46)
    for z in range(2, 16):
        v[:, :, z][height + 2 > z] = 11
    v[7:10, 5:8, 2:12] = 12  # the post at the front edge
    return v


def many_colours() -> np.ndarray:
    v = np.zeros((24, 4, 4), dtype=np.uint8)
    for k in range(24):
        cells = [(k, j % 4, j // 4) for j in range(k + 1) if j < 16]
        for c in cells:
            v[c] = k + 1
    return v


def crate() -> np.ndarray:
    v = np.zeros((6, 6, 6), dtype=np.uint8)
    v[:, :, :5] = 5
    v[:, :, 5] = 7
    return v


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    palette = default_palette()
    for index, rgb in ((5, (150, 105, 60)), (7, (90, 60, 35)), (10, (200, 175, 120)), (11, (80, 150, 60)),
                       (12, (220, 40, 40))):
        palette[index, :3] = rgb
    for name, voxels in (("landscape", landscape()), ("many_colours", many_colours()), ("crate", crate())):
        write_vox(OUT / f"{name}.vox", VoxModel(voxels.shape, voxels, palette))
        print(f"wrote {OUT / name}.vox: {int((voxels != 0).sum())} voxels")


if __name__ == "__main__":
    main()
