"""The shader-limit tasks of scripts/mcp_eval.py can be done with Rok's tools.

Each task gets a short reference solution: the tool calls a model would make,
through Session.tool as the MCP server makes them. The result is saved and
opened again, as the eval does, and the task's own check must pass. A naive
solution that leaves out the shader limit must fail the check, so each task
really needs its feature. No `claude -p` runs here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("rok_native")

ROOT = Path(__file__).resolve().parents[1]
BIRD = ROOT / "tests" / "fixtures" / "bird.vox"


def _eval_module():
    spec = importlib.util.spec_from_file_location("mcp_eval", ROOT / "scripts" / "mcp_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _eval_module()


@pytest.fixture(scope="module")
def session():
    from rok.headless import Session

    s = Session()
    s.agent.feedback = "text"  # no picture per call: faster, and the checks read voxels
    yield s
    s.close()


def _shader(**args) -> dict:
    return {"tool": "rok_edit", "args": {"do": "shader", **args}}


def _new(*size, **colours) -> list[dict]:
    """A voxel art scene with its named palette (colours 1, 2, ... in the order given)."""
    return [{"tool": "rok_scene", "args": {"do": "new", "discard": True, "size": list(size)}},
            {"tool": "rok_edit", "args": {"do": "palette", "colours": colours}}]


LOOK = {"tool": "rok_look", "args": {}}  # ends the blockout: colours open


def _ground(*size, **colours) -> list[dict]:
    x, y = size[0] - 1, size[1] - 1
    return _new(*size, **colours) + [
        {"tool": "rok_shape", "args": {"kind": "box", "low": [0, 0, 0], "high": [x, y, 1], "colour": 1}}, LOOK]


def _box(low, high, colour=1, erase=False, mirror=None) -> dict:
    args = {"kind": "box", "low": low, "high": high, "colour": colour, "mode": "erase" if erase else "add"}
    if mirror:
        args["mirror"] = mirror
    return {"tool": "rok_shape", "args": args}


def _lantern() -> list[dict]:
    """Body, pointed roof and a hanging ring, in the middle step of the metal ramp (colour 2)."""
    return [_box([3, 3, 0], [8, 8, 9], 2),
            {"tool": "rok_shape", "args": {"kind": "pyramid", "low": [2, 2, 10], "high": [9, 9, 15], "colour": 2}},
            _box([5, 5, 16], [6, 6, 18], 2)]


# Each task: the reference solution, and a naive one without the shader limit.
SOLUTIONS = {
    "top_damage": (
        [_shader(shader="hard_surface/battle_damage", only_on=["top"])],
        [_shader(shader="hard_surface/battle_damage")],
    ),
    "edge_wear": (
        [_shader(shader="hard_surface/battle_damage", only_on=["edges"], feather=2)],
        [_shader(shader="hard_surface/battle_damage")],
    ),
    "dirt_fade": (
        [{"tool": "rok_select", "args": {"do": "box", "low": [0, 0, 0], "high": [29, 67, 21]}},
         _shader(shader="weathering/patches", args={"Kind": 1, "Amount": 100}, feather=6)],
        [{"tool": "rok_select", "args": {"do": "box", "low": [0, 0, 0], "high": [29, 67, 21]}},
         _shader(shader="weathering/patches", args={"Kind": 1, "Amount": 100})],
    ),
    "forest": (
        _ground(64, 32, 24, grass="#4a8a3a") + [_shader(shader="nature/woodland", volume=[0, 0, 2, 31, 31, 23])],
        _ground(64, 32, 24, grass="#4a8a3a") + [_shader(shader="nature/woodland")],
    ),
    "protect_eyes": (
        [{"tool": "rok_select", "args": {"do": "colour", "colour": EVAL.BIRD_GLASS}},
         {"tool": "rok_command", "args": {"do": "run", "id": "rok.mask_selection"}},
         _shader(shader="hard_surface/battle_damage", args={"Amount": 100})],
        [_shader(shader="hard_surface/battle_damage", args={"Amount": 100})],
    ),
    # One tree in a box at the corner: the volume is the tree's own grid.
    "corner_tree": (
        _ground(64, 64, 48, sand="#d8c08a") + [_shader(shader="nature/trees", volume=[4, 4, 2, 28, 28, 30])],
        _ground(64, 64, 48, sand="#d8c08a") + [_shader(shader="nature/trees")],  # one tree for the whole grid, in its middle
    ),
    # A line from its foot to its top; the naive one is a box's low to high corner, which cannot lean toward low x.
    "leaning_trunk": (
        _new(32, 32, 32, bark="#6a4a2a") + [{"tool": "rok_shape", "args": {
            "kind": "line", "start": [24, 16, 0], "end": [10, 16, 20], "colour": 1}}],
        _new(32, 32, 32, bark="#6a4a2a") + [{"tool": "rok_shape", "args": {
            "kind": "line", "low": [10, 16, 0], "high": [24, 16, 20], "colour": 1}}],
    ),
    # A face-joined trunk under a crown; the naive thin line touches its own cells only at edges.
    "joined_tree": (
        _new(32, 32, 32, bark="#6a4a2a", leaf="#4a8a3a") + [
            {"tool": "rok_shape", "args": {"kind": "line", "start": [20, 16, 0], "end": [12, 16, 16], "colour": 1,
                                             "solid": True}},
            {"tool": "rok_shape", "args": {"kind": "sphere", "low": [7, 11, 14], "high": [17, 21, 24], "colour": 1}},
            LOOK,
            {"tool": "rok_shape", "args": {"kind": "sphere", "low": [7, 11, 14], "high": [17, 21, 24], "colour": 2}}],
        _new(32, 32, 32, bark="#6a4a2a", leaf="#4a8a3a") + [
            {"tool": "rok_shape", "args": {"kind": "line", "start": [20, 16, 0], "end": [12, 16, 16], "colour": 1}},
            {"tool": "rok_shape", "args": {"kind": "sphere", "low": [7, 11, 14], "high": [17, 21, 24], "colour": 1}}],
    ),
    # A half cylinder run along its one-cell side; without along it runs along x: a plate.
    "canopy": (
        _new(40, 24, 12, glass="#9ad0f0", frame="#3a4250") + [
            {"tool": "rok_shape", "args": {"kind": "buckyball", "low": [2, 2, -11], "high": [37, 21, 11], "colour": 1}},
            LOOK,
            {"tool": "rok_shape", "args": {"kind": "framed_buckyball", "low": [2, 2, -11], "high": [37, 21, 11],
                                             "colour": 1, "frame_colour": 2}}],
        _new(40, 24, 12, glass="#9ad0f0") + [{"tool": "rok_shape", "args": {
            "kind": "buckyball", "low": [2, 2, -11], "high": [37, 21, 11], "colour": 1}}],
    ),
    "half_disc": (
        _new(32, 8, 16, stone="#8a8478") + [{"tool": "rok_shape", "args": {
            "kind": "half_cylinder", "low": [0, 4, 0], "high": [20, 4, 10], "along": "y", "colour": 1}}],
        _new(32, 8, 16, stone="#8a8478") + [{"tool": "rok_shape", "args": {
            "kind": "half_cylinder", "low": [0, 4, 0], "high": [20, 4, 10], "colour": 1}}],
    ),
    # Grass on the green tops only, in patches; the naive meadow covers both halves in single blades.
    "green_meadow": (
        [_shader(shader="nature/meadow", args={"Grass": 50, "Flowers": 0, "Clumps": 100}, only_colours=[12],
                 only_on=["top"], room=6)],
        [_shader(shader="nature/meadow")],
    ),
    # Erase what floats and the uneven slope, then step it evenly two columns at a time; the naive fix
    # only erases what floats, so the slope stays jagged.
    "fix_ramp": (
        [_box([18, 1, 9], [19, 2, 10], erase=True), _box([16, 1, 6], [16, 1, 11], erase=True),
         _box([0, 0, 1], [13, 3, 13], erase=True)]
        + [_box([2 * k, 0, 1], [2 * k + 1, 3, k], 1) for k in range(1, 6)] + [_box([12, 0, 1], [13, 3, 5], 1)],
        [_box([18, 1, 9], [19, 2, 10], erase=True), _box([16, 1, 6], [16, 1, 11], erase=True)],
    ),
    # A body, a roof and a ring in one colour, then shaded from the ramp; the naive one is not shaded.
    "lantern": (
        _new(12, 12, 20, metal="#8a7a5a") + [{"tool": "rok_edit", "args": {
            "do": "palette", "colours": {"metal": "#8a7a5a"}, "ramps": {"metal": 3}}}] + _lantern() + [
            LOOK, {"tool": "rok_edit", "args": {"do": "shade", "ramp": "metal", "colour": 2}}],
        _new(12, 12, 20, metal="#8a7a5a") + [{"tool": "rok_edit", "args": {
            "do": "palette", "colours": {"metal": "#8a7a5a"}, "ramps": {"metal": 3}}}] + _lantern(),
    ),
    # A beetle: body, head and legs mirrored across x, then eyes and shading; the naive one is a big ball.
    "icon_creature": (
        _new(24, 16, 16, shell="#3a6a4a", eye="#f0d040") + [{"tool": "rok_edit", "args": {
            "do": "palette", "colours": {"shell": "#3a6a4a"}, "ramps": {"shell": 3}}},
            _box([6, 4, 2], [17, 11, 7], 4), _box([9, 1, 3], [14, 3, 6], 4),
            _box([3, 5, 0], [5, 6, 3], 4, mirror="x"), _box([3, 9, 0], [5, 10, 3], 4, mirror="x"),
            LOOK, _box([10, 1, 5], [10, 1, 5], 2, mirror="x"),
            {"tool": "rok_edit", "args": {"do": "shade", "ramp": "shell", "colour": 4}}],
        _new(40, 40, 40, shell="#3a6a4a") + [
            {"tool": "rok_shape", "args": {"kind": "sphere", "low": [0, 0, 0], "high": [39, 39, 39], "colour": 1}}],
    ),
    # The cage (rok_deform): 5 x 2 x 2 points, the two wing-tip planes of points pulled back.
    "sweep_wings": (
        [{"tool": "rok_deform", "args": {"counts": [5, 2, 2], "offsets": {
            f"{i},{j},{k}": [0, -8, 0] for i in (0, 4) for j in range(2) for k in range(2)}}}],
        [{"tool": "rok_transform", "args": {"do": "move", "target": "model", "offset": [0, -8, 0]}}],  # all of it moves
    ),
}


def _animate(**args) -> dict:
    return {"tool": "rok_animate", "args": args}


# The animation tasks: the reference solution, and a naive one that misses what the check reads.
SCENE_SOLUTIONS = {
    "spin_propeller": (
        [_animate(do="parent", part="Propeller", parent="Hub"),
         _animate(do="motion", part="Propeller", name="spin", axis="z")],
        [_animate(do="motion", part="Propeller", name="spin", axis="z")],  # it does not follow the hub
    ),
    "swing_door": (
        [{"tool": "rok_select", "args": {"do": "box", "low": [2, 0, 0], "high": [11, 1, 15]}},
         _animate(do="make_part", name="Door"),
         _animate(do="pivot", part="Door", pivot=[0, 1.5, 0]),
         _animate(do="key", part="Door", frame=0, turn=[0, 0, 0]),
         _animate(do="key", part="Door", frame=12, turn=[0, 0, -90])],
        [_animate(do="key", part="Doorway", frame=12, turn=[0, 0, -90])],  # the whole doorway turns
    ),
    # import puts the crate beside the land; the move sets it on the ground at (40..45, 10..15).
    "crate_fit": (
        [{"tool": "rok_scene", "args": {"do": "import", "path": str(ROOT / "tests" / "fixtures" / "agent" / "crate.vox")}},
         "PLACE_CRATE"],
        [{"tool": "rok_scene", "args": {"do": "import", "path": str(ROOT / "tests" / "fixtures" / "agent" / "crate.vox")}}],
    ),
}


def _place_crate(session) -> dict:
    """The move that sets the imported (active) crate on the land at x 40..45, y 10..15, z 2."""
    scene = session.document.scene
    land, crate = scene.models[0], session.document.model
    want = [int(a) + b for a, b in zip(land.position, (40, 10, 2))]
    return {"tool": "rok_scene", "args": {"do": "move", "offset": [w - int(p) for w, p in zip(want, crate.position)]}}


def _solve_scene(session, task: str, steps: list[dict], tmp_path: Path):
    """Run the steps on the task's file, save a .rok, open it again: the scene."""
    kind, _prompt = EVAL.TASKS[task]
    session.open(EVAL.STARTS[kind])
    for step in steps:
        if step == "PLACE_CRATE":
            step = _place_crate(session)
        reply, _png = session.tool(step["tool"], step["args"])
        assert not reply.error, (task, step, reply.text)
    path = tmp_path / f"{task}{EVAL.result_suffix(task)}"
    reply, _png = session.tool("rok_scene", {"do": "save", "path": str(path)})
    assert not reply.error, reply.text
    session.open(path)
    return session.document.scene


def _solve(session, task: str, steps: list[dict], tmp_path: Path):
    """Run the steps on the task's start, save, open the saved file again: (before, after)."""
    kind, _prompt = EVAL.TASKS[task]
    before = None
    start_palette = session.document.scene.palette.copy() if session.document.scene is not None else None
    if kind in EVAL.STARTS:
        session.open(EVAL.STARTS[kind])
        before = session.document.grid.voxels.copy()
        start_palette = session.document.scene.palette.copy()
    for step in steps:
        reply, _png = session.tool(step["tool"], step["args"])
        assert not reply.error, (task, step, reply.text)
    path = tmp_path / f"{task}.vox"
    reply, _png = session.tool("rok_scene", {"do": "save", "path": str(path)})
    assert not reply.error, reply.text
    session.open(path)
    after = session.document.grid.voxels.copy()
    if start_palette is not None:
        after = EVAL.in_numbers_of(after, session.document.scene.palette, start_palette, before)
    return before, after


def test_every_new_task_has_a_check_and_a_solution():
    assert set(EVAL.CHECKS) == set(SOLUTIONS)
    assert set(EVAL.SCENE_CHECKS) == set(SCENE_SOLUTIONS)
    assert set(EVAL.CHECKS) | set(EVAL.SCENE_CHECKS) <= set(EVAL.TASKS)


@pytest.mark.parametrize("task", sorted(SCENE_SOLUTIONS))
def test_the_reference_animation_passes_the_check(session, task, tmp_path):
    ok, note = EVAL.SCENE_CHECKS[task](_solve_scene(session, task, SCENE_SOLUTIONS[task][0], tmp_path))
    assert ok, f"{task}: {note}"


@pytest.mark.parametrize("task", sorted(SCENE_SOLUTIONS))
def test_a_naive_animation_fails_the_check(session, task, tmp_path):
    ok, note = EVAL.SCENE_CHECKS[task](_solve_scene(session, task, SCENE_SOLUTIONS[task][1], tmp_path))
    assert not ok, f"{task}: {note}"


@pytest.mark.parametrize("task", sorted(SOLUTIONS))
def test_the_reference_solution_passes_the_check(session, task, tmp_path):
    before, after = _solve(session, task, SOLUTIONS[task][0], tmp_path)
    ok, note = EVAL.CHECKS[task](before, after)
    assert ok, f"{task}: {note}"


@pytest.mark.parametrize("task", sorted(SOLUTIONS))
def test_a_naive_solution_fails_the_check(session, task, tmp_path):
    before, after = _solve(session, task, SOLUTIONS[task][1], tmp_path)
    ok, note = EVAL.CHECKS[task](before, after)
    assert not ok, f"{task}: {note}"


def test_scope_not_selection_also_spares_the_eyes(session, tmp_path):
    """The other way to protect a part: run the shader on everything but the selection."""
    steps = [{"tool": "rok_select", "args": {"do": "colour", "colour": EVAL.BIRD_GLASS}},
             _shader(shader="hard_surface/battle_damage", args={"Amount": 100}, scope="not_selection")]
    before, after = _solve(session, "protect_eyes", steps, tmp_path)
    ok, note = EVAL.check_protect_eyes(before, after)
    assert ok, note


def test_a_cage_sweep_is_one_undo_step_and_keeps_the_colours(session, tmp_path):
    session.open(BIRD)
    before = session.document.grid.voxels.copy()
    reply, _png = session.tool("rok_deform", {"counts": [5, 2, 2], "offsets": {
        f"{i},{j},{k}": [0, -8, 0] for i in (0, 4) for j in range(2) for k in range(2)}})
    assert not reply.error, reply.text
    after = session.document.grid.voxels.copy()
    assert not (after == before).all() and set(after.flat) - {0} <= set(before.flat)
    reply, _png = session.tool("rok_scene", {"do": "undo"})
    assert not reply.error and (session.document.grid.voxels == before).all()
