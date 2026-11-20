"""Write the golden files of the buckyball and Crease.

Run once: `.venv-rok/bin/python scripts/make_facet_fixtures.py`.

tests/fixtures/facets/framed_buckyball_24.vox: the Framed Buckyball filling a
24 x 24 x 24 box, the panes in colour 79 and the bars in 165.
tests/fixtures/facets/canopy.vox: a plain buckyball squashed into a
32 x 20 x 16 box, as a canopy is long, narrow and low.
tests/fixtures/facets/canopy_crease_expected.vox: Crease on the canopy with
its defaults (20 degrees, width 1, raised bars in colour 165).
Look at them with scripts/drive.py before you keep them: they are the golden files.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rok.core import facets  # noqa: E402
from rok.core.palette import default_palette  # noqa: E402
from rok.core.vox_read import VoxModel  # noqa: E402
from rok.core.vox_write import write_vox  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "facets"
GLASS, FRAME = 79, 165


def write(path: Path, voxels: np.ndarray) -> None:
    write_vox(path, VoxModel(size=voxels.shape, voxels=voxels, palette=default_palette()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    filled, frame = facets.bucky_grid((24, 24, 24), framed=True)
    ball = np.where(frame, FRAME, np.where(filled, GLASS, 0)).astype(np.uint8)
    write(OUT / "framed_buckyball_24.vox", ball)
    filled, _frame = facets.bucky_grid((32, 20, 16))
    canopy = np.where(filled, GLASS, 0).astype(np.uint8)
    write(OUT / "canopy.vox", canopy)
    creased = facets.crease(canopy, FRAME)
    write(OUT / "canopy_crease_expected.vox", creased)
    print("framed ball", int((ball != 0).sum()), "canopy", int(filled.sum()), "bars", int((creased == FRAME).sum()))


if __name__ == "__main__":
    main()
