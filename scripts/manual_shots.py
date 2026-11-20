"""Take the screenshots for the manual by driving the real app.

  .venv-rok/bin/python scripts/manual_shots.py OUT_DIR [SCENE ...]

Each scene opens a test model in a driven window, runs its steps one per
tick, and writes OUT_DIR/<scene>/NN_name.png. Without scene names every scene
runs. Keep hands off the mouse and keyboard while it runs: the events go
straight into the app, and real input mixes in.

`scripts/manual_images.py` turns the PNGs into the WebP files of `manual/img`.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import drive  # noqa: E402
import rok_native  # noqa: E402

from rok.app import chrome, opkit  # noqa: E402
from rok.app.app import RokApp  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
BODY = {"x": 735.0, "y": 331.0}  # a point on the bird's body, in a 1470 x 801 window
AWAY = {"x": 900.0, "y": 650.0}  # empty floor: no hover, no tooltip


# --- steps ----------------------------------------------------------------------------


class SafeDriver(drive.Driver):
    """A step that fails prints its error and the run goes on; a raise would
    stop the window's event loop and the run would hang."""

    def run_step(self, step) -> None:
        try:
            super().run_step(step)
        except Exception:
            print("STEP FAILED", step[0], flush=True)
            traceback.print_exc()


def wait(n: int = 1) -> list:
    return [("call", lambda a: None) for _ in range(n)]


def shot(name: str) -> tuple:
    return ("shot", name)


def call(fn) -> tuple:
    return ("call", fn)


def mods(app, **on) -> None:
    m = {"shift": False, "ctrl": False, "alt": False, "meta": False}
    m.update({("meta" if k == "cmd_key" else k): v for k, v in on.items()})
    app.handle("modifiers", m)


def _point(app, p) -> dict:
    return p(app) if callable(p) else p


def move(p) -> tuple:
    return call(lambda a: a.handle("mouse_move", _point(a, p)))


def click(p, button: str = "left", **on) -> tuple:
    def go(a):
        a.handle("mouse_move", _point(a, p))
        mods(a, **on)
        a.handle("mouse_down", {"button": button})
        a.handle("mouse_up", {"button": button})
        mods(a)
    return call(go)


def down(p, button: str = "left", **on) -> tuple:
    def go(a):
        a.handle("mouse_move", _point(a, p))
        mods(a, **on)
        a.handle("mouse_down", {"button": button})
    return call(go)


def up(button: str = "left") -> tuple:
    def go(a):
        a.handle("mouse_up", {"button": button})
        mods(a)
    return call(go)


def key(k: str, text: str | None = None, **on) -> tuple:
    text = k if text is None and len(k) == 1 else (text or "")

    def go(a):
        mods(a, **on)
        a.handle("key_down", {"key": k, "code": k, "text": text, "repeat": False})
        a.handle("key_up", {"key": k, "code": k, "text": "", "repeat": False})
        mods(a)
    return call(go)


def key_down(k: str, text: str = "") -> tuple:
    return call(lambda a: a.handle("key_down", {"key": k, "code": k, "text": text, "repeat": False}))


def key_up(k: str, text: str = "") -> tuple:
    return call(lambda a: a.handle("key_up", {"key": k, "code": k, "text": text, "repeat": False}))


def menu(ident: str) -> tuple:
    return call(lambda a: a.handle("menu", {"id": ident}))


def run_op(ident: str, **props) -> tuple:
    return call(lambda a: opkit.call(ident, "EXEC", **props))


# --- places in the window ---------------------------------------------------------------


def widget(ident: str):
    """The centre of a rail widget, or of a widget on the open flyout."""
    def p(a) -> dict:
        from rok.app.state import state
        from rok.core import chrome as kit

        lay = chrome.layout_for(a.region)
        for w in tuple(kit.flyout_widgets(lay, state.chrome_open)) + tuple(lay.widgets):
            if w.id == ident:
                x, y, ww, hh = w.rect
                return {"x": x + ww / 2, "y": a.region.height - (y + hh / 2)}
        raise KeyError(ident)
    return p


def nav(name: str):
    """The centre of a round button under the nav cube: CENTRE, QUAD, PAN, ZOOM, RENDER, STEADY."""
    def p(a) -> dict:
        from rok.app import nav_cube

        x, y = nav_cube.buttons(a.region)[name]
        return {"x": float(x), "y": float(a.region.height - y)}
    return p


def rim(name: str):
    """A rim button of the nav cube: SECTION, MIRROR, or an open row's MIRROR_X and so on."""
    def p(a) -> dict:
        from rok.app import nav_cube

        x, y, _r = nav_cube.rim_buttons(a.region)[name]
        return {"x": float(x), "y": float(a.region.height - y)}
    return p


def big_swatch(a) -> dict:
    """The palette card's current-colour swatch; a click opens the editor."""
    from rok.app import overlay

    x, y, w, h = overlay._dock.chip
    return {"x": x + w / 2, "y": a.region.height - (y + h / 2)}


def random_switch(a) -> dict:
    from rok.core import brick_bin

    lay = brick_bin.bin_layout(1.0, a.region.width, a.region.height)
    item = next(i for i in lay.items if i.kind == "RANDOM")
    x, y, w, h = item.rect
    return {"x": x + w / 2, "y": a.region.height - (y + h / 2)}


def house_top(a) -> dict:
    return {"x": a.region.width / 2, "y": a.region.height * 0.42}


# --- scenes ------------------------------------------------------------------------------


def scene_welcome() -> tuple[str | None, list]:
    return None, [*wait(10), shot("welcome"), *wait(10), shot("welcome2"),
                  key("Escape"), *wait(2), menu("rok.new"), *wait(2), shot("new_dialog"), key("Escape")]


def scene_bird() -> tuple[str, list]:
    steps = [
        move(AWAY), shot("window"),
        move(BODY), *wait(), shot("ghost_attach"),
        call(lambda a: mods(a, alt=True)), move({"x": 740.0, "y": 335.0}), *wait(), shot("ghost_erase"),
        call(lambda a: mods(a)), move(AWAY),
        move(widget("TOOL_ATTACH")), *wait(3), shot("attach_flyout"), move(AWAY), *wait(),
        move(widget("MODEL_TOOLS")), *wait(3), shot("model_flyout"), move(AWAY), *wait(),
        move(widget("LAYERS_BUTTON")), *wait(3), shot("layers_flyout"), move(AWAY), *wait(),
        click(big_swatch), *wait(2), shot("colour_popup"), key("Escape"), move(AWAY), *wait(),
        key_down("Space", " "), *wait(2), shot("star"), key_up("Space", " "), *wait(),
        key("k", cmd_key=True), *wait(), *[key(ch) for ch in "mirr"], *wait(), shot("command_palette"), key("Escape"), *wait(),
        *[key("PageDown") for _ in range(6)], *wait(), shot("section"), *[key("PageUp") for _ in range(6)],
        key("z"), key("m"), down({"x": 520.0, "y": 240.0}), move({"x": 700.0, "y": 350.0}), move({"x": 900.0, "y": 460.0}), up(),
        *wait(2), shot("selection"), run_op("rok.deselect"), key("t"), key("v"), move(AWAY), *wait(),
    ]
    for name in ("ISO", "SOFT", "NIGHT", "CERAMIC", "PLAIN", "STUDIO", "SUN", "PAPER", "CLAY", "FLAT", "BLUEPRINT", "TOON"):
        steps += [run_op("rok.look", name=name), *wait(2), shot(f"look_{name.lower()}")]
    steps += [
        run_op("rok.look", name="STUDIO"), *wait(),
        run_op("rok.theme", name="BRIGHT"), *wait(2), shot("bright"), run_op("rok.theme", name="DARK"), *wait(),
        menu("rok.view_settings"), *wait(2), shot("view_settings"), key("Escape"), *wait(),
        menu("rok.shader_gallery"), *wait(2), shot("shader_gallery"), key("Escape"), *wait(),
        click(nav("QUAD")), *wait(2), shot("quad"), click(nav("QUAD")), *wait(),
        key("Tab"), move(AWAY), *wait(2), shot("scene"),
        move(widget("NEW_BUTTON")), *wait(3), shot("new_flyout"), move(AWAY), *wait(),
        click(BODY), *wait(2), shot("gizmo"),
        move(widget("SCENE_TOOLS")), *wait(3), shot("scene_tools_flyout"), move(AWAY), *wait(),
        key("Enter"), *wait(),
        move(BODY), *[key("ArrowUp") for _ in range(3)], key("ArrowRight"), *wait(), shot("key_cursor"), key("Escape"),
    ]
    return "bird.vox", steps


def scene_bird_more() -> tuple[str, list]:
    return "bird.vox", [
        move(AWAY), *wait(2),
        move(widget("TOOL_ATTACH")), *wait(3), click(widget("TOOL_ATTACH_PRIMITIVES")), *wait(3), shot("primitives_flyout"),
        move(AWAY), *wait(2),
        key("f"), move(BODY), *wait(), shot("mode_face"),
        key("b"), down({"x": 600.0, "y": 300.0}), move({"x": 700.0, "y": 330.0}), move({"x": 820.0, "y": 380.0}), *wait(),
        shot("mode_box"), up(), *wait(), menu("rok.undo"), *wait(),
        key("l"), down({"x": 600.0, "y": 300.0}), move({"x": 820.0, "y": 380.0}), *wait(), shot("mode_line"), up(), *wait(),
        menu("rok.undo"), *wait(), key("v"), move(AWAY), *wait(),
        click(rim("MIRROR")), *wait(2), shot("mirror_open"), click(rim("MIRROR_X")), *wait(), move(BODY), *wait(2),
        shot("mirror_ghost"), run_op("rok.toggle_mirror", axis=0), move(AWAY), *wait(),
        click(rim("SECTION")), *wait(2), shot("section_open"), move(AWAY), *wait(),
        run_op("rok.select_all"), *wait(), menu("rok.cage"), move(AWAY), *wait(3), shot("cage"),
        run_op("rok.cage_done"), *wait(), run_op("rok.deselect"), *wait(),
        menu("rok.history_strip"), *wait(3), shot("history_strip"), menu("rok.history_strip"), *wait(),
        menu("rok.render_settings"), *wait(3), shot("render_settings"), key("Escape"), *wait(),
        click(nav("RENDER")), move(AWAY), *wait(14), shot("rendered"), click(nav("RENDER")), move(AWAY), *wait(2),
        menu("rok.cad_view"), move(AWAY), *wait(3), shot("cad"), menu("rok.cad_view"), *wait(),
        click(widget("STROKE_DESIGN")), *wait(3), shot("stroke_designer"), key("Escape"), *wait(),
    ]


def scene_brick() -> tuple[str, list]:
    return "brick/house.vox", [
        move(AWAY), shot("house"), menu("rok.brick_mode"), move(AWAY), *wait(3), shot("brick_mode"),
        run_op("rok.convert_to_bricks"), *wait(2), shot("brick_converted"),
        key("3"), move(house_top), *wait(2), shot("brick_ghost"), click(house_top), *wait(2), shot("brick_placed"),
        key("r"), *wait(), shot("brick_turned"),
        click(random_switch), *wait(2), move(house_top), *wait(2), shot("brick_random"),
        key("u"), move(house_top), *wait(2), shot("brick_pull"),
        key("Escape"), move(AWAY), *wait(), menu("rok.brick_mode"), *wait(2), shot("brick_off"),
    ]


def scene_anim() -> tuple[str, list]:
    return "anim/walk.rok", [
        move(AWAY), *wait(2), shot("walk_scene"),
        run_op("rok.animate", action="ENTER"), move(AWAY), *wait(4), shot("animate"),
        menu("rok.animation_menu"), *wait(3), shot("animate_menu"),
        key("Space", " ", shift=True), *wait(5), shot("playing"), key("Space", " ", shift=True), *wait(),
    ]


def scene_car() -> tuple[str, list]:
    return "car.vox", [move(AWAY), *wait(2), shot("model"), key("Tab"), *wait(2), shot("scene")]


def scene_robot() -> tuple[str, list]:
    return "robot.vox", [move(AWAY), *wait(2), shot("model"), key("Tab"), *wait(2), shot("scene")]


SCENES = {
    "welcome": scene_welcome, "bird": scene_bird, "bird_more": scene_bird_more,
    "brick": scene_brick, "anim": scene_anim, "car": scene_car, "robot": scene_robot,
}


def run_scene(name: str, out: Path) -> None:
    """One scene is one process: the app keeps its document in module state."""
    fixture, steps = SCENES[name]()
    out.mkdir(parents=True, exist_ok=True)
    host = rok_native.Host("Rok (manual shots)", None)
    app = RokApp(host, str(FIXTURES / fixture) if fixture else None)
    host.run(SafeDriver(host, app, out, list(steps) + [("quit",)]).handle)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    names = sys.argv[2:] or list(SCENES)
    if len(names) > 1:
        # One window per process: run each scene in a child, each with its own
        # settings folder, or the next start asks about the last one's recovery copy
        # in a native alert and waits for an answer.
        import os
        import subprocess
        import tempfile

        for name in names:
            print("==", name, flush=True)
            env = {**os.environ, "ROK_CONFIG": tempfile.mkdtemp(prefix="rok-manual-")}
            subprocess.run([sys.executable, __file__, str(out), name], check=False, env=env)
        return
    run_scene(names[0], out / names[0])


if __name__ == "__main__":
    main()
