"""Time one step of the cage's deformation.

`.venv-rok/bin/python scripts/time_cage.py` prints, for a dense cube of 32, 64
and 128 cells and for tests/fixtures/bird.vox (the whole model selected), the best of a few
steps of `deform_grid` (numpy; skipped above 64) and `live_deform_grid` (native, the exact kernel
and the fast pass, which is the one a drag runs),
with a 3 x 3 x 3 cage and one point dragged three cells. The plan's budget is 30 ms
a step at 128 cells.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rok.core import lattice as L  # noqa: E402
from rok.core.vox_read import read_vox  # noqa: E402


def best(fn, runs=3):
    out = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return min(out) * 1000.0


def case(name, voxels, mask, numpy_too):
    cells = np.argwhere(mask)
    lat = L.rest_lattice((cells.min(axis=0), cells.max(axis=0)), 3)
    lat.points[2, 1, 1] += (3.0, 0.0, 0.0)
    line = f"{name:>22}: {int(mask.sum()):>8} cells"
    if numpy_too:
        line += f"  numpy {best(lambda: L.deform_grid(voxels, mask, lat), 1):9.1f} ms"
    if L._native() is not None:
        L.live_deform_grid(voxels, mask, lat)
        line += f"  exact {best(lambda: L.live_deform_grid(voxels, mask, lat, exact=True)):8.1f} ms"
        line += f"  fast {best(lambda: L.live_deform_grid(voxels, mask, lat)):8.1f} ms"
    print(line, flush=True)


def main():
    print(f"{os.cpu_count()} cores; native {'yes' if L._native() is not None else 'no'}")
    for s in (32, 64, 128):
        v = np.zeros((s,) * 3, dtype=np.uint8)
        v[2 : s - 2, 2 : s - 2, 2 : s - 2] = 3
        case(f"dense cube {s}", v, v != 0, s <= 64)
    bird = read_vox(ROOT / "tests" / "fixtures" / "bird.vox").voxels
    case("bird, all", bird, bird != 0, True)


if __name__ == "__main__":
    main()
