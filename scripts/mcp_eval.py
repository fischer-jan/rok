"""The MCP test set: which feedback setting helps a model build best?

    .venv-rok/bin/python scripts/mcp_eval.py OUT_DIR [--model sonnet] [--only chair,sword] [--jobs 4]

Each task runs once per feedback setting (image, text) as `claude -p` with only
Rok's MCP server, headless, so it never touches an open Rok window. Every
run saves its model; this script renders each result as a quad picture and
writes OUT_DIR/report.md with turns, tokens, cost, time and Rok's checks. The
build quality is judged by looking at the pictures.

The tasks after "eyes" need the shader limits of rok_edit (only_on, scope,
feather, volume), the mask or rok_deform (sweep_wings). corner_tree needs a
one-tree shader in a volume, crate_fit rok_scene do=import. leaning_trunk needs a line
from start to end toward low x, half_disc a half_cylinder run along its thin side
(along), joined_tree a line with solid=true, green_meadow a meadow kept to the tops of one colour (only_colours,
only_on top, room) in solid patches (Clumps). Each of them has an automatic check in CHECKS that
reads the saved model; the report shows its result. The animation tasks (spin_propeller,
swing_door) need rok_animate; they save a .rok, and their checks in SCENE_CHECKS read
its parts and clip. fix_ramp, lantern and icon_creature test voxel art craft: an even
slope with no floating parts, shading from a ramp, and a small piece that reads as an
icon. tests/test_mcp_eval_tasks.py solves each of these tasks with the tools and proves
that its check passes.

"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-rok" / "bin" / "python"
FIXTURES = ROOT / "tests" / "fixtures"

TASKS = {
    "chair": ("new", "Build a small chair for a voxel game: four legs, a seat and a backrest, in a dark red wood "
                     "colour. Start a new 16 x 16 x 24 document."),
    "sword": ("new", "Build a pixel-art sword standing upright: a grey blade with a point, a gold crossguard, a brown "
                     "grip and a round gold pommel. Start a new 12 x 12 x 32 document."),
    "mushroom": ("new", "Build a toadstool: a white stem and a red domed cap with white spots on top. Start a new "
                        "24 x 24 x 24 document."),
    "house": ("new", "Build a small house: four walls of light stone, a door opening on the front (-y) side, a window "
                     "on each side wall, and a sloped red roof. Start a new 32 x 32 x 32 document."),
    "tree": ("new", "Build a low-poly tree for a game: a brown trunk and a green crown of a few overlapping blobs, "
                    "with a lighter green on top. Start a new 24 x 24 x 32 document."),
    "eyes": ("bird", "The open file is a voxel bird. Find its two glass eyes and paint them light blue (#8fd3ff). "
                     "Change nothing else."),
    "top_damage": ("bird", "The open file is a voxel bird that flew through a shower of falling embers. Give it burn "
                           "damage (dents and soot) only on the surfaces that face up: change only voxels that have an "
                           "empty cell directly above them. Every other voxel stays exactly as it is."),
    "edge_wear": ("bird", "The open file is a voxel bird. Give it wear and damage only along the outer edges and "
                          "corners of its body and wings, where two faces meet. Voxels on flat surfaces away from the "
                          "edges, and the inside, stay exactly as they are."),
    "dirt_fade": ("bird", "The open file is a voxel bird. Cover its -x end (its left wing), the cells with x from 0 to "
                          "29, densely in dirt patches. Toward x = 29 the dirt must fade out softly over about six "
                          "cells, not stop at a hard line. Nothing with x of 30 or more changes."),
    "forest": ("new", "Start a new 64 x 32 x 24 document. Make flat ground: fill the cells from [0, 0, 0] to "
                      "[63, 31, 1] with a grass green. Then grow a mixed woodland on the half with x from 0 to 31 only. "
                      "The other half stays open ground with nothing on it, and the ground itself stays whole."),
    "protect_eyes": ("bird", "The open file is a voxel bird that came out of a fire: cover the whole bird in heavy "
                             "damage. Its dark glass eyes must come through untouched: not one voxel of the glass may "
                             "change."),
    "sweep_wings": ("bird", "The open file is a voxel bird. Its wings reach out along x. Sweep both wing tips back: bend "
                            "the wings so their tips (the cells with x up to 6 and from 73 up) end about 8 cells further "
                            "toward -y, while the middle of the bird (x from 34 to 45) stays where it is. Keep the bird "
                            "whole: no new colours, about the same number of voxels."),
    "spin_propeller": ("propeller", "The open file is a propeller on a hub. Animate it: the propeller spins one full "
                                    "turn about the vertical axis every second, at 24 frames per second, and loops; "
                                    "the hub stays still, and the propeller follows the hub, so moving the hub would "
                                    "take the propeller along."),
    "swing_door": ("door", "The open file is a doorway: a grey frame and a brown door with a gold knob, all in one "
                           "model. Animate the door: it swings open by 90 degrees about its hinge, the vertical edge "
                           "on the -x side, within half a second, and stays open until the clip ends. The knob goes "
                           "with the door; the frame stays still."),
    "corner_tree": ("new", "Start a new 64 x 64 x 48 document. Make flat ground: fill the cells from [0, 0, 0] to "
                           "[63, 63, 1] with a sand colour. Then plant one leafy tree near a corner: every voxel of "
                           "the tree must lie in the cells with x from 4 to 28 and y from 4 to 28, and it must be at "
                           "least 12 cells tall. Nothing else stands on the ground, and the ground stays whole."),
    "crate_fit": ("landscape", "The open file is a piece of land with a hill and a red post. Add the crate from "
                               f"{FIXTURES / 'agent' / 'crate.vox'} to this scene as a model of its own, and set it "
                               "on the flat ground in front of the hill: its lowest voxels rest on top of the ground "
                               "(z = 2 in the land's cells), and its middle is near x = 42, y = 12 of the land. The "
                               "land itself stays exactly as it is."),
    "leaning_trunk": ("new", "Start a new 32 x 32 x 32 document. Build one thin, straight, leaning tree trunk in a "
                             "brown colour: its foot is the cell [24, 16, 0] and its top is the cell [10, 16, 20], so "
                             "it leans toward low x as it rises. It is one cell thick, with no steps or gaps."),
    "joined_tree": ("new", "Start a new 32 x 32 x 32 document. Build a leaning tree: a brown trunk from the cell "
                           "[20, 16, 0] up to about [12, 16, 16], leaning toward low x, and a round green crown on "
                           "its top. The whole tree must be one piece joined face to face: no part of it may touch "
                           "the rest only at an edge or a corner."),
    "half_disc": ("new", "Start a new 32 x 8 x 16 document. Build a half disc, like a fan or the face of an arch: "
                         "one cell thick, standing upright in the cells with y = 4, its flat side on the floor from "
                         "x = 0 to x = 20, and its round edge reaching up to z = 10 in the middle."),
    "green_meadow": ("grounds", "The open file is flat ground: brown earth (colour 30) on the half with x from 0 to "
                                "15, green grass (colour 12) on the half with x from 16 to 31. Grow a meadow of grass "
                                "on the green half only, in solid patches with bare ground between them, not "
                                "scattered single blades. The brown half stays bare, and the ground itself stays as "
                                "it is."),
    "canopy": ("new", "Build the cockpit canopy of a small spaceship: a faceted half dome of flat light-blue glass "
                      "panes held by a darker frame along every edge between the panes. It stands flat on the floor "
                      "(z = 0), is longer along x than across, and is mirror-symmetric across y. Start a new "
                      "40 x 24 x 12 document."),
    "fix_ramp": ("ramp", "The open file is a stone ramp. Its sloped top edge, seen from the front, steps unevenly, and "
                         "a thin antenna and a small block float beside it. Make the slope step evenly (every step "
                         "the same width, rising one cell at a time from x = 0) and remove what floats. Keep the "
                         "ramp in its place, with its flat top block at the high end."),
    "lantern": ("new", "Start a new 12 x 12 x 20 document. Build a lantern: a square body, a pointed roof and a "
                       "ring on top to hang it by, all one piece. Use at most 10 colours. Shade it as light from "
                       "above falls: faces that look up are lighter than faces that look sideways."),
    "icon_creature": ("new", "Build a small creature (a beetle, a cat or a robot) as voxel art: no more than 48 cells "
                             "on any side, at most 6000 voxels and 12 colours, one piece, mirror-symmetric from left "
                             "to right, and still readable as a 16-pixel icon seen from the front."),
}
# The file each kind of task opens; "new" starts an empty document.
STARTS = {"bird": FIXTURES / "bird.vox", "propeller": FIXTURES / "anim" / "propeller.vox",
          "door": FIXTURES / "anim" / "door.vox", "landscape": FIXTURES / "agent" / "landscape.vox",
          "grounds": FIXTURES / "agent" / "two_grounds.vox", "ramp": FIXTURES / "agent" / "jaggy_ramp.vox"}
SETTINGS = ("image", "text")

# The checks of the tasks above: (voxels before, voxels of the saved model) -> (passed, a short note).
# before holds the voxels of the opened file, or None for a task that starts a new document.
BIRD_GLASS = 169  # bird.vox: the dark glass of the eyes


def in_numbers_of(voxels, palette, start_palette, start_voxels=None):
    """`voxels` (coloured by `palette`) in the colour numbers of
    `start_palette`, matched by colour. A .vox save numbers the colours in the
    strip's order, so a check compares colours, not numbers. A colour the start
    does not hold gets a number the start's voxels do not use."""
    import numpy as np

    lut = np.zeros(256, dtype=np.uint8)
    taken = set() if start_voxels is None else set(int(k) for k in np.unique(start_voxels))
    found = {}
    for k in range(255, 0, -1):
        found[tuple(int(c) for c in start_palette[k])] = k  # the lowest number wins
    for k in sorted(set(int(k) for k in np.unique(voxels)) - {0}):
        colour = tuple(int(c) for c in palette[k])
        if colour in found:
            lut[k] = found[colour]
        else:
            lut[k] = next(s for s in range(255, 0, -1) if s not in taken and s not in set(lut.tolist()))
        taken.add(int(lut[k]))
    return lut[np.asarray(voxels, dtype=np.uint8)]


def _same_grid(before, after) -> str:
    return "" if before.shape == after.shape else f"the grid changed from {before.shape} to {after.shape}"


def _only_on(before, after, face: str) -> tuple[bool, str]:
    from rok.core.shader_scope import face_mask

    if bad := _same_grid(before, after):
        return False, bad
    changed = before != after
    outside = int((changed & ~face_mask(before, face)).sum())
    n = int(changed.sum())
    if n < 100:
        return False, f"only {n} cells changed; at least 100 expected"
    ok = outside <= n // 20
    return ok, f"{n} cells changed, {outside} of them off the {face.lower()} voxels (at most 5% allowed)"


def check_top_damage(before, after):
    return _only_on(before, after, "TOP")


def check_edge_wear(before, after):
    return _only_on(before, after, "EDGES")


def check_dirt_fade(before, after):
    if bad := _same_grid(before, after):
        return False, bad
    changed, filled = before != after, before != 0

    def rate(lo, hi):
        return changed[lo:hi + 1].sum() / max(1, filled[lo:hi + 1].sum())

    outside = int(changed[30:].sum())
    inner, edge = rate(18, 23), rate(28, 29)  # just inside the fade, and its last two cells
    ok = outside == 0 and inner >= 0.4 and edge <= 0.6 * inner
    return ok, (f"{outside} cells changed at x >= 30; {inner:.0%} of the voxels changed at x 18..23 (at least 40% "
                f"needed), {edge:.0%} at x 28..29 (at most 0.6 times the first)")


def check_forest(before, after):
    if after.shape != (64, 32, 24):
        return False, f"the grid is {after.shape}, not (64, 32, 24)"
    ground = bool((after[:, :, :2] != 0).all())
    left, right = int((after[:32, :, 2:] != 0).sum()), int((after[32:, :, 2:] != 0).sum())
    ok = ground and left >= 200 and right == 0
    return ok, (f"the ground is {'whole' if ground else 'broken'}; {left} voxels above it at x 0..31 "
                f"(at least 200 needed), {right} at x 32..63 (none allowed)")


def check_protect_eyes(before, after):
    if bad := _same_grid(before, after):
        return False, bad
    glass = before == BIRD_GLASS
    hit = int((glass & (after != before)).sum())
    rest = int(((after != before) & ~glass).sum())
    ok = hit == 0 and rest >= 1500
    return ok, f"{hit} glass voxels changed (none allowed); {rest} other cells changed (at least 1500 needed)"


def check_sweep_wings(before, after):
    import numpy as np

    if bad := _same_grid(before, after):
        return False, bad

    def mean_y(v, xs):
        cells = np.argwhere(v[xs, :, :] != 0)
        return float(cells[:, 1].mean()) if len(cells) else float("nan")

    tips = [slice(0, 7), slice(73, before.shape[0])]
    moved = [mean_y(before, t) - mean_y(after, t) for t in tips]
    mid = abs(mean_y(after, slice(34, 46)) - mean_y(before, slice(34, 46)))
    count = int((after != 0).sum()) / max(1, int((before != 0).sum()))
    new_colours = set(np.unique(after)) - set(np.unique(before))
    ok = min(moved) >= 4.0 and mid <= 1.5 and 0.7 <= count <= 1.3 and not new_colours
    return ok, (f"the tips moved back by {moved[0]:.1f} and {moved[1]:.1f} cells (at least 4 needed); the middle moved "
                f"{mid:.1f} (at most 1.5); {count:.0%} of the voxels remain (70 to 130%); new colours: {sorted(new_colours)}")


def check_corner_tree(before, after):
    import numpy as np

    if after.shape != (64, 64, 48):
        return False, f"the grid is {after.shape}, not (64, 64, 48)"
    ground = bool((after[:, :, :2] != 0).all())
    tree = np.argwhere(after[:, :, 2:] != 0)
    if not len(tree):
        return False, f"the ground is {'whole' if ground else 'broken'}; nothing stands on it"
    lo, hi = tree.min(axis=0), tree.max(axis=0)
    inside = bool((lo[:2] >= 4).all() and (hi[:2] <= 28).all())
    tall = int(hi[2] - lo[2] + 1)
    ok = ground and inside and tall >= 12 and len(tree) >= 300
    return ok, (f"the ground is {'whole' if ground else 'broken'}; {len(tree)} voxels above it in x {lo[0]}..{hi[0]}, "
                f"y {lo[1]}..{hi[1]} (4..28 allowed), {tall} cells tall (at least 12)")


def check_leaning_trunk(before, after):
    import numpy as np

    if after.shape != (32, 32, 32):
        return False, f"the grid is {after.shape}, not (32, 32, 32)"
    cells = np.argwhere(after != 0)
    if not len(cells):
        return False, "nothing was built"
    ends = {(24, 16, 0), (10, 16, 20)} <= {tuple(c) for c in cells.tolist()}
    one_per_layer = all(int((cells[:, 2] == z).sum()) == 1 for z in range(21))
    by_z = cells[np.argsort(cells[:, 2])]
    leans = bool(np.all(np.diff(by_z[:, 0]) <= 0) and np.all(np.abs(np.diff(by_z[:, 0])) <= 1))
    ok = ends and one_per_layer and leans and len(cells) == 21
    return ok, (f"{len(cells)} voxels (21 wanted); both ends there: {ends}; one cell per layer: {one_per_layer}; "
                f"x falls by at most one per layer: {leans}")


def check_joined_tree(before, after):
    import numpy as np

    from rok.core import region

    if after.shape != (32, 32, 32):
        return False, f"the grid is {after.shape}, not (32, 32, 32)"
    mask = after != 0
    parts = len(region.components(mask)) if mask.any() else 0
    foot = bool(mask[19:22, 15:18, 0].any())
    cells = np.argwhere(mask)
    crown = int((cells[:, 2] >= 14).sum()) if len(cells) else 0
    leans = bool(len(cells)) and float(cells[cells[:, 2] >= 14][:, 0].mean() if crown else 99) < 17
    ok = parts == 1 and foot and crown >= 60 and leans
    return ok, (f"{parts} part(s) face to face (1 wanted); foot at [20, 16, 0]: {foot}; {crown} voxels from z = 14 "
                f"up (at least 60); the top leans toward low x: {leans}")


def check_half_disc(before, after):
    import numpy as np

    if after.shape != (32, 8, 16):
        return False, f"the grid is {after.shape}, not (32, 8, 16)"
    filled = after != 0
    flat = bool(filled[:, 4].any()) and not filled[:, :4].any() and not filled[:, 5:].any()
    rows = filled[:, 4]
    floor = bool(rows[:21, 0].all()) and not rows[21:, 0].any()
    widths = [int(rows[:, z].sum()) for z in range(16)]
    tapers = all(a >= b for a, b in zip(widths, widths[1:])) and widths[10] >= 3 and widths[11] == 0
    narrow_top = widths[10] <= 9 and widths[5] >= 17
    ok = flat and floor and tapers and narrow_top
    return ok, (f"one cell thick at y = 4: {flat}; the floor row fills x 0..20: {floor}; row widths up the disc "
                f"{widths[:12]} (they must shrink to a short top row at z = 10)")


def check_green_meadow(before, after):
    import numpy as np

    if bad := _same_grid(before, after):
        return False, bad
    ground = bool(np.array_equal(after[:, :, :3], before[:, :, :3]))
    brown = int((after[:16, :, 3:] != 0).sum())
    grass = (after[16:, :, 3:] != 0).any(axis=2)
    cover = float(grass.mean())
    alike = float((grass[1:] == grass[:-1]).mean() + (grass[:, 1:] == grass[:, :-1]).mean()) / 2.0
    ok = ground and brown == 0 and 0.2 <= cover <= 0.9 and alike >= 0.8
    return ok, (f"the ground is {'as it was' if ground else 'changed'}; {brown} voxels over the brown half (none "
                f"allowed); grass on {cover:.0%} of the green half (20 to 90%); neighbours alike {alike:.0%} "
                "(at least 80%: patches, not single blades)")


def check_canopy(before, after):
    import numpy as np

    if after.shape != (40, 24, 12):
        return False, f"the grid is {after.shape}, not (40, 24, 12)"
    filled = after != 0
    total = int(filled.sum())
    if total == 0:
        return False, "the grid is empty"
    colours, counts = np.unique(after[filled], return_counts=True)
    top2 = sorted(counts, reverse=True)[:2]
    two = len(counts) >= 2 and top2[1] >= 0.05 * total
    mirror = float((filled == filled[:, ::-1]).mean())
    layers = [int(filled[:, :, z].sum()) for z in range(12)]
    dome = layers[0] >= 200 and all(a >= b for a, b in zip(layers, layers[1:])) and layers[-1] < layers[0] / 3
    long = int(filled.any(axis=(1, 2)).sum()) > int(filled.any(axis=(0, 2)).sum())
    ok = two and mirror >= 0.98 and dome and long
    return ok, (f"colours {dict(zip(colours.tolist(), counts.tolist()))} (two, the second at least 5%: {two}); "
                f"mirror-alike across y {mirror:.1%} (98%); cells per layer {layers} (a dome: shrinking, the floor "
                f"layer at least 200); longer along x: {long}")


def check_fix_ramp(before, after):
    from rok.core import craft

    if bad := _same_grid(before, after):
        return False, bad
    rep = craft.check(after)
    kept = int((after != 0).sum())
    ok = not rep.jags and len(rep.parts) == 1 and kept >= 150 and bool((after[:2, :, 0] != 0).any())
    return ok, (f"{len(rep.jags)} jagged slope(s) (none allowed); {len(rep.parts)} part(s) (one wanted); "
                f"{kept} voxels (at least 150 of the ramp stay)")


def _faces(after):
    """The voxels open to the top, and the voxels open only at a side."""
    import numpy as np

    v = after != 0
    pad = np.pad(v, 1)
    up = v & ~pad[1:-1, 1:-1, 2:]
    side = v & ~up & (~pad[2:, 1:-1, 1:-1] | ~pad[:-2, 1:-1, 1:-1] | ~pad[1:-1, 2:, 1:-1] | ~pad[1:-1, :-2, 1:-1])
    return up, side


def check_lantern(before, after):
    import numpy as np

    from rok.core import craft

    filled = after != 0
    if not filled.any():
        return False, "nothing was built"
    colours = len(np.unique(after[filled]))
    parts = len(craft.parts(after, limit=3))
    up, side = _faces(after)
    top = int(np.bincount(after[up]).argmax()) if up.any() else -1
    flank = int(np.bincount(after[side]).argmax()) if side.any() else -1
    shaded = top != flank
    ok = colours <= 10 and parts == 1 and shaded and max(after.shape) <= 20
    return ok, (f"{colours} colours (at most 10); {parts} part(s) (one); faces that look up mostly colour {top}, "
                f"faces that look sideways mostly colour {flank} (they must differ: shading)")


def check_icon_creature(before, after):
    import numpy as np

    from rok.agent_art import verdict
    from rok.core import craft, voxel_views

    filled = after != 0
    if not filled.any():
        return False, "nothing was built"
    box = np.argwhere(filled)
    size = (box.max(axis=0) - box.min(axis=0) + 1).tolist()
    voxels = int(filled.sum())
    colours = len(np.unique(after[filled]))
    parts = len(craft.parts(after, limit=3))
    sym = craft.symmetry(after).get("x", 0.0)
    crop, _ = voxel_views.cropped(after)
    sil = voxel_views.project(crop, "front")[0] != 0
    factor = max(1, -(-max(sil.shape) // 16))
    keep = craft.survival(sil, factor) if factor > 1 else 100.0
    kind, _why = verdict([after])
    ok = (max(size) <= 48 and voxels <= 6000 and colours <= 12 and parts == 1 and sym >= 90 and keep >= 60
          and kind == "voxel art")
    return ok, (f"{size} cells (at most 48 a side); {voxels} voxels (6000); {colours} colours (12); {parts} part(s); "
                f"symmetry x {sym}% (90); as a 16-pixel icon the front keeps {keep}% (60); {kind}")


CHECKS = {"top_damage": check_top_damage, "edge_wear": check_edge_wear, "dirt_fade": check_dirt_fade,
          "forest": check_forest, "protect_eyes": check_protect_eyes, "sweep_wings": check_sweep_wings,
          "corner_tree": check_corner_tree, "fix_ramp": check_fix_ramp, "lantern": check_lantern,
          "icon_creature": check_icon_creature, "leaning_trunk": check_leaning_trunk, "half_disc": check_half_disc, "joined_tree": check_joined_tree,
          "green_meadow": check_green_meadow, "canopy": check_canopy}


# The animation tasks' checks: the saved scene (parts, clips) -> (passed, a short note).


def _named(scene, name):
    return next((m for m in scene.models if m.name == name), None)


def _keys(clip, model) -> int:
    track = clip.track_of(model.uid)
    return 0 if track is None else sum(len(p.keys) for p in track.properties.values())


def check_spin_propeller(scene):
    from rok.core import anim

    hub, prop = _named(scene, "Hub"), _named(scene, "Propeller")
    if hub is None or prop is None:
        return False, "the Hub or the Propeller is gone"
    clip = scene.timeline
    track = clip.track_of(prop.uid)
    quarter = [abs(anim.channel(track, "TURN_Z", clip.length * k / 4.0)) for k in (1, 2, 3)]
    even = all(abs(q - 90 * k) <= 25 for q, k in zip(quarter, (1, 2, 3)))
    ok = (prop.parent == hub.uid and even and _keys(clip, hub) == 0 and clip.loop_mode == anim.LOOP
          and clip.fps == 24 and 0.9 <= clip.seconds <= 1.1)
    return ok, (f"the propeller {'follows' if prop.parent == hub.uid else 'does not follow'} the hub; its turn about z "
                f"at a quarter, half and three quarters of the clip: {', '.join(f'{q:.0f}' for q in quarter)} degrees "
                f"(90, 180, 270 wanted); the hub has {_keys(clip, hub)} keys (none allowed); the clip: "
                f"{clip.seconds:.2f} s at {clip.fps} fps, {clip.loop_mode.lower()}")


def check_swing_door(scene):
    import numpy as np

    from rok.core import anim

    doors = [m for m in scene.models if (m.grid.voxels == 3).any()]
    frame = _named(scene, "Doorway")
    if frame is None or len(doors) != 1 or doors[0] is frame:
        return False, "the door is not a part of its own (or the Doorway is gone)"
    door = doors[0]
    clip = scene.timeline
    track = clip.track_of(door.uid)
    # The hinge, in the doorway's own cells: the door's -x edge is at x = 2.
    pivot = np.asarray(door.position, dtype=float) + anim.pivot_of(track, door) - np.asarray(frame.position, dtype=float)
    turns = [abs(anim.channel(track, "TURN_Z", t)) for t in range(clip.length)]
    half = int(round(clip.fps * 0.5))
    knob = bool((door.grid.voxels == 4).any())
    ok = (door.parent == frame.uid and abs(pivot[0] - 2.0) <= 1.0 and turns[0] <= 5 and knob
          and all(abs(t - 90) <= 15 for t in turns[half:]) and _keys(clip, frame) == 0 and len(turns) > half)
    return ok, (f"the door {'follows' if door.parent == frame.uid else 'does not follow'} the frame; it turns about "
                f"x = {pivot[0]:.1f} (the hinge is at 2); its turn about z is {turns[0]:.0f} at frame 0 and "
                f"{min(turns[half:], default=0):.0f} to {max(turns[half:], default=0):.0f} degrees from frame {half} "
                f"to the end (90 wanted); the knob is {'on the door' if knob else 'left behind'}; the frame has "
                f"{_keys(clip, frame)} keys")


def check_crate_fit(scene):
    """The crate (216 voxels, from rok_scene do=import) stands on the land's ground near (42, 12)."""
    import numpy as np

    from rok.core.vox_read import read_scene

    land_file = read_scene(STARTS["landscape"]).models[0].voxels
    crates = [m for m in scene.models if int((m.grid.voxels != 0).sum()) == 216]
    lands = [m for m in scene.models if m.grid.voxels.shape == land_file.shape]
    if len(crates) != 1 or len(lands) != 1:
        return False, f"{len(crates)} crates and {len(lands)} lands in the scene (one of each wanted)"
    crate, land = crates[0], lands[0]
    same = bool(np.array_equal(land.grid.voxels, land_file))
    cells = np.argwhere(crate.grid.voxels != 0) + np.asarray(crate.position) - np.asarray(land.position)
    bottom = int(cells[:, 2].min())
    middle = (cells[:, :2].min(axis=0) + cells[:, :2].max(axis=0) + 1) / 2.0
    near = bool(np.all(np.abs(middle - (42.5, 12.5)) <= 3))
    ok = same and bottom == 2 and near
    return ok, (f"the land is {'unchanged' if same else 'changed'}; the crate's lowest voxels are at z = {bottom} of "
                f"the land (2 wanted), its middle at x {middle[0]:.1f}, y {middle[1]:.1f} (42.5, 12.5 wanted, 3 cells "
                "either way)")


SCENE_CHECKS = {"spin_propeller": check_spin_propeller, "swing_door": check_swing_door, "crate_fit": check_crate_fit}


def result_suffix(task: str) -> str:
    """A .rok keeps the parts and clips an animation task makes."""
    return ".rok" if task in SCENE_CHECKS else ".vox"


def run_one(out: Path, task: str, setting: str, model: str) -> dict:
    kind, prompt = TASKS[task]
    name = f"{task}-{setting}"
    work = Path("/tmp") / f"rok-eval-{name}"  # short: Rok's config holds a socket path
    work.mkdir(parents=True, exist_ok=True)
    env = {"ROK_LIVE": "never", "ROK_CONFIG": str(work), "ROK_FEEDBACK": setting, "ROK_IMAGE_SIZE": "1000"}
    if kind in STARTS:
        env["ROK_FILE"] = str(STARTS[kind])
    config = {"mcpServers": {"rok": {"command": str(PYTHON), "args": [str(ROOT / "rok" / "mcp_server.py")], "env": env}}}
    (work / "mcp.json").write_text(json.dumps(config))
    result_path = out / f"{name}{result_suffix(task)}"
    full = (f"Using only the rok tools: {prompt} When it is done, save it to {result_path} and say in two "
            "sentences what you built.")
    cmd = ["claude", "-p", full, "--mcp-config", str(work / "mcp.json"), "--strict-mcp-config",
           "--setting-sources", "project", "--allowedTools", "mcp__rok__*", "--model", model,
           "--output-format", "json"]
    done = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=1800)
    try:
        d = json.loads(done.stdout)
    except ValueError:
        d = {"is_error": True, "result": done.stdout[-500:] + done.stderr[-500:]}
    u = d.get("usage") or {}
    row = {
        "task": task, "setting": setting, "ok": not d.get("is_error") and result_path.exists(),
        "turns": d.get("num_turns"), "seconds": round((d.get("duration_ms") or 0) / 1000),
        "cost": d.get("total_cost_usd"),
        "input": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0),
        "output": u.get("output_tokens"), "summary": (d.get("result") or "").strip(),
    }
    (out / f"{name}.json").write_text(json.dumps(d, indent=1))
    return row


def render(out: Path, rows: list[dict]) -> None:
    """A quad picture and Rok's checks for every saved model."""
    script = out / "_render.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "import mcp_eval\n"
        f"out = Path({str(out)!r})\n"
        "session.open(mcp_eval.FIXTURES / 'bird.vox')\n"
        "bird = session.document.grid.voxels.copy()\n"
        "bird_palette = session.document.scene.palette.copy()\n"
        "facts = {}\n"
        "for vox in sorted([*out.glob('*.vox'), *out.glob('*.rok')]):\n"
        "    session.open(vox)\n"
        "    (out / (vox.stem + '.png')).write_bytes(session.render_png('quad', (800, 800)))\n"
        "    r, _ = session.tool('rok_check', {})\n"
        "    f = session.agent.facts()\n"
        "    facts[vox.stem] = {'voxels': f['voxels'], 'parts': len(r.data.get('parts', [])), 'colours': len(f['colours'])}\n"
        "    task = vox.stem.rsplit('-', 1)[0]\n"
        "    if task in mcp_eval.CHECKS:\n"
        "        before = bird if mcp_eval.TASKS[task][0] == 'bird' else None\n"
        "        after = mcp_eval.in_numbers_of(session.document.grid.voxels, session.document.scene.palette, bird_palette, bird)\n"
        "        ok, note = mcp_eval.CHECKS[task](before, after)\n"
        "        facts[vox.stem]['check'] = ('pass: ' if ok else 'FAIL: ') + note\n"
        "    if task in mcp_eval.SCENE_CHECKS:\n"
        "        ok, note = mcp_eval.SCENE_CHECKS[task](session.document.scene)\n"
        "        facts[vox.stem]['check'] = ('pass: ' if ok else 'FAIL: ') + note\n"
        "(out / '_facts.json').write_text(json.dumps(facts))\n")
    subprocess.run([str(PYTHON), "-m", "rok", "run", str(script)], cwd=ROOT, check=True,
                   env={**os.environ, "ROK_CONFIG": "/tmp/rok-eval-render"})
    facts = json.loads((out / "_facts.json").read_text())
    for row in rows:
        row.update(facts.get(f"{row['task']}-{row['setting']}", {}))


def report(out: Path, rows: list[dict]) -> None:
    lines = ["# Rok MCP test set", "",
             "| Task | Feedback | Saved | Turns | Seconds | Cost ($) | Input tokens | Voxels | Parts | Colours | Check |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        check = r.get("check", "-").split(":")[0]
        lines.append(f"| {r['task']} | {r['setting']} | {'yes' if r['ok'] else 'no'} | {r['turns']} | {r['seconds']} | "
                     f"{(r['cost'] or 0):.2f} | {r['input']} | {r.get('voxels', '-')} | {r.get('parts', '-')} | "
                     f"{r.get('colours', '-')} | {check} |")
    lines.append("")
    for r in rows:
        lines += [f"## {r['task']} ({r['setting']})", "", f"![{r['task']}]({r['task']}-{r['setting']}.png)", ""]
        if "check" in r:
            lines += [f"Check: {r['check']}", ""]
        lines += [r["summary"], ""]
    (out / "report.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", default="sonnet")
    p.add_argument("--only", default="")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in TASKS if not a.only or t in a.only.split(",")]
    jobs = [(t, s) for t in tasks for s in SETTINGS]
    with ThreadPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(lambda j: run_one(out, j[0], j[1], a.model), jobs))
    render(out, rows)
    report(out, rows)
    print(out / "report.md")
    for r in rows:
        print(r["task"], r["setting"], "saved" if r["ok"] else "FAILED", r["turns"], "turns", r["cost"])


if __name__ == "__main__":
    sys.exit(main())
