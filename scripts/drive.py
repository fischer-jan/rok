"""Drive the real Rok app with scripted events and save screenshots.

  .venv-rok/bin/python scripts/drive.py OUT_DIR [FILE.vox]

The window opens, the steps below run one per tick, and every "shot" step
writes OUT_DIR/NN_name.png. The events go straight into the app's handler, so
real mouse input during the run can interfere: keep hands off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A scripted run keeps its recent-files list apart from yours.
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ.setdefault("ROK_CONFIG", tempfile.mkdtemp(prefix="rok-drive-"))

import rok_native  # noqa: E402

from rok.app.app import RokApp  # noqa: E402
from rok.core import png  # noqa: E402

STEP_S = 0.25


class Driver:
    def __init__(self, host, app: RokApp, out: Path, steps) -> None:
        self.host, self.app, self.out, self.steps = host, app, out, list(steps)
        self.n = 0

    def handle(self, kind: str, data: dict) -> None:
        if kind == "screenshot":
            rgba = np.frombuffer(data["rgba"], dtype=np.uint8).reshape(int(data["height"]), int(data["width"]), 4)
            path = self.out / f"{data['token']}.png"
            path.write_bytes(png.encode(rgba))
            print("wrote", path, flush=True)
            return
        if kind == "tick" and self.steps:
            self.run_step(self.steps.pop(0))
            self.host.tick(STEP_S)
        self.app.handle(kind, data)
        if kind == "start":
            self.host.tick(STEP_S * 2)

    def run_step(self, step) -> None:
        kind, *args = step
        if kind == "shot":
            self.n += 1
            self.host.screenshot(f"{self.n:02d}_{args[0]}")
            self.host.request_redraw()
        elif kind == "event":
            self.app.handle(args[0], args[1])
        elif kind == "call":
            args[0](self.app)
        elif kind == "quit":
            self.host.quit()


def centre(app: RokApp) -> dict:
    return {"x": app.camera.width / 2, "y": app.camera.height / 2}


def default_steps(out: Path):
    """Open, look, open the New dialog, cancel it, save, quit."""
    ev = lambda kind, **data: ("event", kind, data)
    from rok.app import opkit

    return [
        ("shot", "opened"),
        ("call", lambda a: a.handle("mouse_move", centre(a))),
        ("shot", "hover"),
        ("call", lambda a: a.handle("menu", {"id": "rok.new"})),
        ("shot", "new_dialog"),
        ev("key_down", key="Escape", code="Escape", text="", repeat=False),
        ("shot", "dialog_closed"),
        ("call", lambda a: opkit.call("rok.save_as", "EXEC", filepath=str(out / "saved.vox"))),
        ("shot", "saved"),
        ("quit",),
    ]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    path = sys.argv[2] if len(sys.argv) > 2 else None
    host = rok_native.Host("Rok (driven)", None)
    app = RokApp(host, path)
    driver = Driver(host, app, out, default_steps(out))
    host.run(driver.handle)


if __name__ == "__main__":
    main()
