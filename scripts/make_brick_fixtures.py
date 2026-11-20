"""Write the brick fixtures in tests/fixtures/brick: a small house built
by Place clicks from the floor, with and without Mirror X.

  house.vox          the house's voxels
  house.json         {"size", "clicks", "bricks"}: the clicks and the table they make
  house_mirror.json  the left half's clicks with Mirror X on: the same house

Each click is [x, y, z, w, d, turn, colour]: the first cell, the start
height, the footprint, the quarter turns and the palette index.

Run: .venv-rok/bin/python scripts/make_brick_fixtures.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rok.core import brick  # noqa: E402
from rok.core.grid import VoxelGrid  # noqa: E402
from rok.core.vox_write import write_vox  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "brick"
SIZE = (10, 10, 8)  # the house is symmetric about the middle plane x = 5
RED, WHITE, GREY = 37, 1, 250

# A floor slab, walls in running bond with a door at the front, a lintel and a roof row.
CLICKS = [
    [1, 1, 0, 8, 8, 0, GREY],
    [1, 1, 0, 4, 1, 0, RED], [5, 1, 0, 4, 1, 0, RED],       # back wall, row 1
    [1, 8, 0, 2, 1, 0, RED], [7, 8, 0, 2, 1, 0, RED],       # front wall, a door in the middle
    [1, 2, 0, 6, 1, 3, RED], [8, 2, 0, 6, 1, 3, RED],       # side walls, turned
    [1, 1, 0, 2, 1, 0, RED], [3, 1, 0, 4, 1, 0, RED], [7, 1, 0, 2, 1, 0, RED],  # row 2: the bond
    [1, 8, 0, 2, 1, 0, RED], [7, 8, 0, 2, 1, 0, RED],
    [1, 2, 0, 6, 1, 3, RED], [8, 2, 0, 6, 1, 3, RED],
    [1, 8, 0, 8, 1, 0, WHITE],                              # lintel over the door
    [1, 1, 0, 8, 8, 0, GREY],                               # roof
]

# The left half, Mirror X on; a brick across the middle is its own mirror image.
MIRROR_CLICKS = [
    [1, 1, 0, 8, 8, 0, GREY],
    [1, 1, 0, 4, 1, 0, RED], [1, 8, 0, 2, 1, 0, RED], [1, 2, 0, 6, 1, 3, RED],
    [1, 1, 0, 2, 1, 0, RED], [3, 1, 0, 4, 1, 0, RED], [1, 8, 0, 2, 1, 0, RED], [1, 2, 0, 6, 1, 3, RED],
    [1, 8, 0, 8, 1, 0, WHITE],
    [1, 1, 0, 8, 8, 0, GREY],
]


def build(clicks, axes) -> tuple[VoxelGrid, list]:
    grid = VoxelGrid(SIZE)
    for x, y, z, w, d, turn, colour in clicks:
        placed, reason = brick.click(grid, x, y, z, w, d, turn, colour, axes)
        if reason is not None:
            raise SystemExit(f"click {[x, y, z, w, d, turn, colour]} refused: {reason}")
    return grid, grid.bricks.to_json()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid, table = build(CLICKS, (False, False, False))
    write_vox(OUT / "house.vox", grid.to_model())
    (OUT / "house.json").write_text(json.dumps({"size": SIZE, "clicks": CLICKS, "bricks": table}) + "\n")
    _grid, mirrored = build(MIRROR_CLICKS, (True, False, False))
    (OUT / "house_mirror.json").write_text(json.dumps({"size": SIZE, "clicks": MIRROR_CLICKS,
                                                       "axes": [True, False, False],
                                                       "bricks": mirrored}) + "\n")
    print(f"house: {len(table)} bricks, {int((grid.voxels != 0).sum())} voxels; mirrored: {len(mirrored)} bricks")


if __name__ == "__main__":
    main()
