"""Drive the real app through the cage and save screenshots.

  .venv-rok/bin/python scripts/drive_cage.py OUT_DIR [FILE.vox]

Selects the part of the model with x under 30 (the left wing of tests/fixtures/bird.vox),
runs Model > Cage, drags a point and shoots before, with the cage and after.
Needs a window: on a machine without a display, run it under xvfb-run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import drive
from drive import *
from rok.app import opkit, cage_ops
from rok.app.object import document
from rok.app.state import state
import rok_native

def select_wing(a):
    v = document.grid.voxels
    m = np.zeros(v.shape, bool); m[:30] = v[:30] != 0
    document.selection.replace(m)
    a.handle("resize", {"width": a.camera.width, "height": a.camera.height})
def show(a):
    opkit.call("rok.cage", "EXEC")
def tip_drag(a):
    a.redraw()
    ptr, ids, pts, depth = state.cage.layout
    i = ids.index("CAGE_0_2_1")
    H = a.camera.height
    s = {"x": float(pts[i][0]), "y": H - float(pts[i][1])}
    a.handle("mouse_move", s); a.handle("mouse_down", {"button": "left"})
    for k in range(1, 7):
        a.handle("mouse_move", {"x": s["x"] - 12 * k, "y": s["y"] + 4 * k})
    a.handle("mouse_up", {"button": "left"})

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
host = rok_native.Host("Rok (driven)", None)
app = RokApp(host, (sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "bird.vox")))
steps = [("shot","opened"), ("call", select_wing), ("shot","selected"), ("call", show), ("shot","cage"),
         ("call", tip_drag), ("shot","bent"), ("quit",)]
d = Driver(host, app, out, steps)
host.run(d.handle)
