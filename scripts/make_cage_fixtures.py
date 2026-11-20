"""Write the cage's golden files.

Run once: `.venv-rok/bin/python scripts/make_cage_fixtures.py`.

tests/fixtures/cage/wing.vox: a small wing, 24 cells long (x), a chord that
narrows from 12 cells at the root to 4 at the tip (y), 3 cells thick (z), in
four colours: root, middle, tip and a leading-edge stripe.
tests/fixtures/cage/wing_sweep.json: a 3 x 3 x 3 cage on the wing's box with the
tip points moved back (+y) by 8 cells and the middle points by 3, the format of
the MCP tool.
tests/fixtures/cage/wing_sweep_expected.vox: the numpy path's result. Look at
it with scripts/drive.py before you keep it: it is the golden file.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rok.core import lattice as L  # noqa: E402
from rok.core.palette import default_palette  # noqa: E402
from rok.core.vox_read import VoxModel  # noqa: E402
from rok.core.vox_write import write_vox  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "cage"
GRID = (32, 32, 8)
LEADING, ROOT_C, MID_C, TIP_C = 5, 2, 3, 4


def make_wing() -> np.ndarray:
    v = np.zeros(GRID, dtype=np.uint8)
    for i in range(24):
        x = 4 + i
        chord = round(12 - 8 * i / 23)
        colour = ROOT_C if i < 8 else MID_C if i < 16 else TIP_C
        for y in range(6, 6 + chord):
            for z in range(2, 5):
                v[x, y, z] = LEADING if y == 6 else colour
    return v


def sweep_offsets() -> dict:
    out = {}
    for j in range(3):
        for k in range(3):
            out[f"2,{j},{k}"] = [0, 8, 0]
            out[f"1,{j},{k}"] = [0, 3, 0]
    return out


def write(path: Path, voxels: np.ndarray) -> None:
    write_vox(path, VoxModel(size=voxels.shape, voxels=voxels, palette=default_palette()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wing = make_wing()
    mask = wing != 0
    cells = np.argwhere(mask)
    bounds = (tuple(int(c) for c in cells.min(axis=0)), tuple(int(c) for c in cells.max(axis=0)))
    spec = {"bounds": [list(bounds[0]), list(bounds[1])], "counts": [3, 3, 3], "offsets": sweep_offsets()}
    (OUT / "wing_sweep.json").write_text(json.dumps(spec, indent=1) + "\n")
    lat = L.offset_lattice(L.rest_lattice(bounds, spec["counts"]), spec["offsets"])
    want, _mask = L.deform_grid(wing, mask, lat)
    write(OUT / "wing.vox", wing)
    write(OUT / "wing_sweep_expected.vox", want)
    print("wing", int(mask.sum()), "cells; swept", int((want != 0).sum()))


if __name__ == "__main__":
    main()
