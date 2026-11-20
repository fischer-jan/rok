"""The star menu's contents and layout (rok/core/star.py)."""

import math
from pathlib import Path

import pytest

from rok.core import chrome as kit
from rok.core import star

ICONS = Path(__file__).resolve().parents[1] / "assets" / "icons" / "chrome" / "1x"


def _all_chip_ids(rays):
    return [c.id for r in rays for c in r.chips]


def test_every_chip_is_a_rail_button_or_one_of_the_stars_own():
    own = {star.STAR_MODEL, star.STAR_SHADERS, star.LAST_COMMAND}
    rail = (set(kit.TOOL_IDS) | set(kit.TOOL_MODE_OF_ID) | set(kit.SOLID_OF_ID) | set(kit.COMMAND_OF_MODEL_ID)
            | set(kit.CHOICE_OF_ID)
            | {kit.SCENE_PICK_ID, kit.SCENE_CUT_ID, kit.SCENE_DROP_ID, kit.SCENE_DROP_ONTO_ID, kit.SCENE_FIT_ID})
    for rays in (star.model_rays(), star.scene_rays()):
        ids = _all_chip_ids(rays)
        assert len(ids) == len(set(ids))
        for ident in ids:
            assert ident in rail or ident in own, ident


def test_the_tools_keep_the_wheels_directions_and_carry_their_modes():
    rays = {r.id: r for r in star.model_rays()}
    assert (rays["ATTACH"].angle, rays["ERASE"].angle, rays["SELECT"].angle, rays["PAINT"].angle) == (90, 0, 270, 180)
    assert [c.id for c in rays["ATTACH"].chips][:2] == ["TOOL_ATTACH", "TOOL_ATTACH_VOXEL"]
    by = [c.id for c in rays["SELECT_BY"].chips]
    assert by == [f"TOOL_SELECT_{m}" for m in kit.SELECT_ONLY_MODES]
    assert not any(c.id.startswith("PRIMITIVE_") for r in star.model_rays() for c in r.chips)


def test_nothing_on_the_star_wipes_the_model_without_asking():
    ids = set(_all_chip_ids(star.model_rays(kit.MODEL_COMMAND_IDS))) | set(_all_chip_ids(star.scene_rays()))
    for destructive in ("clear_all", "fill_all", "fill_volume", "inverse", "remove_slice", "smooth", "mirror_half"):
        assert f"MODEL_{destructive.upper()}" not in ids
    assert "SCALE_DOWN" not in ids and kit.SCENE_DELETE_ID not in ids
    assert not any(i.startswith("LATHE_") or i == "MODEL_LATHE" for i in ids)


def test_the_model_arms_show_what_the_rail_shows():
    """View > Customize Rail Menu and Keys picks the star's Model chips too."""
    def model_ids(items):
        return [c.id for r in star.model_rays(items) if r.family == star.MODEL for c in r.chips]

    default = model_ids(kit.DEFAULT_MODEL_ITEMS)
    assert "MODEL_LOOP" not in default and "MODEL_DIAGONAL" not in default and "MODEL_RESIZE" in default
    every = model_ids(kit.MODEL_COMMAND_IDS)
    assert {"MODEL_LOOP", "MODEL_REPEAT", "MODEL_TWIST", "MODEL_DIAGONAL", "SCALE_UP"} <= set(every)
    # Ticked or not, the star never gets a command that wipes the model without asking.
    assert not {"MODEL_CLEAR_ALL", "MODEL_FILL_ALL", "MODEL_INVERSE"} & set(every)
    # Up to six commands sit on one arm; more share two.
    arms = [r for r in star.model_rays(kit.DEFAULT_MODEL_ITEMS) if r.family == star.MODEL]
    assert len(arms) == 1 and len(arms[0].chips) - 1 <= star.MODEL_ARM
    six = ("model_split", "fit_grid", "drop_to_floor", "diagonal", "scale", "resize")
    assert len([r for r in star.model_rays(six) if r.family == star.MODEL]) == 1
    two = [r for r in star.model_rays(six + ("loop",)) if r.family == star.MODEL]
    assert [len(r.chips) for r in two] == [1 + 4, 3]  # the Model chip, then the seven split 4 and 3
    bare = star.model_rays(())
    assert [c.id for r in bare if r.family == star.MODEL for c in r.chips] == [star.STAR_MODEL]
    assert star.rays_for("MODEL", False, ("loop",)) == star.model_rays(("loop",))


def test_the_last_command_has_its_own_arm_where_the_primitives_were():
    grow = next(c for c in kit.COMMANDS if c.id == "grow")
    assert not any(r.family == star.LAST for r in star.model_rays())  # none before the first command
    rays = star.model_rays(kit.DEFAULT_MODEL_ITEMS, grow)
    last = [r for r in rays if r.family == star.LAST]
    assert len(last) == 1 and last[0].first and 90 < last[0].angle < 180
    chip = last[0].chips[0]
    assert (chip.id, chip.label, chip.glyph, chip.text) == (star.LAST_COMMAND, "Grow Voxels", "", "G")  # no icon: a letter
    hollow = next(c for c in kit.COMMANDS if c.id == "hollow")
    assert star.last_chip(hollow).glyph == "HOLLOW"
    assert any(r.family == star.LAST for r in star.scene_rays(grow))
    for glyph in set(star.COMMAND_GLYPHS.values()) | set(kit.MODEL_GLYPHS.values()):
        assert (ICONS / f"{glyph}.png").exists(), glyph


def test_the_last_three_colours_have_their_own_arm_beside_paint():
    assert not any(r.id == "COLOURS" for r in star.model_rays())  # none before the first stroke
    chips = [star.colour_chip(i, (10 * i, 20, 30), f"Red {i}") for i in (4, 9, 2, 7)]
    rays = star.model_rays(kit.DEFAULT_MODEL_ITEMS, None, chips)
    arm = [r for r in rays if r.id == "COLOURS"]
    assert len(arm) == 1 and arm[0].family == star.PAINT
    assert [c.id for c in arm[0].chips] == [f"{star.STAR_COLOUR}{i}" for i in (4, 9, 2)]  # three, newest first
    assert arm[0].chips[0].rgb == (40, 20, 30) and arm[0].chips[0].label == "Red 4"
    paint = next(r.angle for r in rays if r.first and r.family == star.PAINT)
    assert abs((arm[0].angle - paint + 180) % 360 - 180) <= 70
    assert star.colour_chip(5, (1, 2, 3)).label == "Colour 5"
    assert star.rays_for("MODEL", False, kit.DEFAULT_MODEL_ITEMS, None, chips) == rays
    assert not any(r.id == "COLOURS" for r in star.rays_for(kit.LEVEL_SCENE, False, colours=chips))


def test_with_every_arm_out_no_two_chips_touch():
    grow = next(c for c in kit.COMMANDS if c.id == "grow")
    chips = [star.colour_chip(i, (i, i, i)) for i in (1, 2, 3)]
    rays = star.model_rays(kit.DEFAULT_MODEL_ITEMS, grow, chips)
    lay = star.star_layout(rays, (2000, 2000), 4000, 4000, 1.0)
    pts = [c.centre for c in lay.chips if c.shown > 0]
    for i, (ax, ay) in enumerate(pts):
        for bx, by in pts[i + 1:]:
            assert math.hypot(ax - bx, ay - by) >= star.CHIP * 0.95


def test_the_angles_are_not_symmetric():
    angles = sorted(r.angle for r in star.model_rays())
    gaps = [(b - a) for a, b in zip(angles, angles[1:] + [angles[0] + 360])]
    assert len(set(round(g) for g in gaps)) > 3


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_closed_no_two_chips_touch_and_the_star_stays_small(scale):
    rays = star.model_rays()
    lay = star.star_layout(rays, (2000, 2000), 4000, 4000, scale)
    shown = [c for c in lay.chips if c.shown > 0]
    assert len(shown) == sum(min(star.SHOW, len(r.chips)) for r in rays)
    pts = [c.centre for c in shown]
    for i, (ax, ay) in enumerate(pts):
        for bx, by in pts[i + 1:]:
            assert math.hypot(ax - bx, ay - by) >= star.CHIP * scale * 0.95
    reach = max(math.hypot(x - 2000, y - 2000) for x, y in pts)
    assert reach < 200 * scale


def test_an_arm_reaches_out_when_the_pointer_comes_to_it():
    rays = star.model_rays()
    attach = next(i for i, r in enumerate(rays) if r.id == "ATTACH")
    lay = star.star_layout(rays, (600, 600), 1200, 1200, 1.0)
    second = next(c for c in lay.chips if c.ray == attach and c.index == 1)
    assert star.arm_under(lay, *second.centre, None) == attach
    extents = [0.0] * len(rays)
    extents[attach] = 1.0
    out = star.star_layout(rays, (600, 600), 1200, 1200, 1.0, extents=extents)
    arm = [c for c in out.chips if c.ray == attach]
    assert all(c.shown == 1.0 for c in arm) and len(arm) == len(rays[attach].chips)
    tip = arm[-1]
    assert star.arm_under(out, *tip.centre, attach) == attach
    assert star.chip_at(out, *tip.centre, attach) is tip
    assert star.arm_under(out, 600, 600, attach) is None  # back in the centre: none


def test_near_an_edge_the_arms_bend_along_it_and_stay_in_the_view():
    rays = star.model_rays()
    w, h = 1200, 800
    for origin in ((60, 400), (1150, 760), (600, 30)):
        lay = star.star_layout(rays, origin, w, h, 1.0, extents=[1.0] * len(rays))
        for c in lay.chips:
            x, y = c.centre
            assert 0 <= x - c.side / 2 and x + c.side / 2 <= w and 0 <= y - c.side / 2 and y + c.side / 2 <= h


def test_the_rays_unroll_and_sway_slowly():
    rays = star.model_rays()
    early = star.star_layout(rays, (600, 600), 1200, 1200, 1.0, opened=0.05)
    done = star.star_layout(rays, (600, 600), 1200, 1200, 1.0, opened=5.0)
    assert sum(c.shown for c in early.chips) < 0.5 * sum(c.shown for c in done.chips)
    assert all(c.shown == 1.0 for c in done.chips if c.index < star.SHOW)
    moved = star.star_layout(rays, (600, 600), 1200, 1200, 1.0, opened=5.0, seconds=2.5)
    shift = max(math.hypot(a.centre[0] - b.centre[0], a.centre[1] - b.centre[1]) for a, b in zip(done.chips, moved.chips))
    assert 0.5 < shift < 25  # it moves, a little


def test_a_chip_is_picked_under_the_pointer_and_not_in_the_centre():
    rays = star.model_rays()
    attach = next(i for i, r in enumerate(rays) if r.id == "ATTACH")
    extents = [0.0] * len(rays)
    extents[attach] = 1.0
    lay = star.star_layout(rays, (600, 600), 1200, 1200, 1.0, extents=extents)
    target = next(c for c in lay.chips if c.chip.id == "TOOL_ATTACH_BOX")
    assert star.chip_at(lay, *target.centre, attach) is target
    assert star.chip_at(lay, 600, 600) is None
    assert star.centre_label(lay, target) == "Attach · Box"
    tool = next(c for c in lay.chips if c.chip.id == "TOOL_ERASE")
    assert star.centre_label(lay, tool) == "Erase"


def test_every_arm_belongs_to_a_tool_family_and_sits_near_its_tool():
    rays = star.model_rays()
    tools = {r.family: r.angle for r in rays if r.first}
    assert set(tools) == {"ATTACH", "ERASE", "PAINT", "SELECT", star.MODEL}
    model = [r for r in rays if r.family == star.MODEL]
    assert {c.id for r in model for c in r.chips} >= {"MODEL_MODEL_SPLIT", "MODEL_RESIZE", "SCALE_UP"}
    for r in rays:
        assert r.family in tools
        gap = abs((r.angle - tools[r.family] + 180) % 360 - 180)
        assert gap <= 70, (r.id, gap)
    assert not any(c.id.startswith(("LEVEL_", "STAR_MIRROR")) for r in rays for c in r.chips)


def test_a_tools_own_chip_is_bigger_and_nothing_touches_it():
    lay = star.star_layout(star.model_rays(), (600, 600), 1200, 1200, 1.0)
    origins = [c for c in lay.chips if c.origin]
    assert sorted(c.chip.id for c in origins) == [star.STAR_MODEL, "TOOL_ATTACH", "TOOL_ERASE", "TOOL_PAINT", "TOOL_SELECT"]
    assert all(c.side == pytest.approx(star.CHIP * star.ORIGIN) for c in origins)
    shown = [c for c in lay.chips if c.shown > 0]
    for a in shown:
        for b in shown:
            if a is not b:
                d = math.hypot(a.centre[0] - b.centre[0], a.centre[1] - b.centre[1])
                assert d >= (a.side + b.side) / 2 - 0.5


def test_the_scene_and_animate_stars_are_small():
    assert {r.id for r in star.scene_rays()} == {"SCENE_TOOLS"}
    assert star.rays_for("MODEL", True) == ()  # no star in Animate
    ids = {c.id for r in star.model_rays() for c in r.chips}
    assert "STAR_LAYERS" not in ids and "STAR_SEARCH" not in ids
    assert star.rays_for(kit.LEVEL_SCENE, False) == star.scene_rays()


def test_the_rails_edge_counts_as_the_views_edge():
    rays = star.model_rays()
    lay = star.star_layout(rays, (90, 400), 1200, 800, 1.0, extents=[1.0] * len(rays), left=60)
    assert all(c.centre[0] - c.side / 2 >= 60 - 1e-6 for c in lay.chips)


def test_with_nothing_in_the_way_every_arm_is_straight():
    rays = star.model_rays()
    lay = star.star_layout(rays, (3000, 3000), 6000, 6000, 1.0, opened=5.0, extents=[1.0] * len(rays))
    for i, ray in enumerate(rays):
        pts = [c.centre for c in lay.chips if c.ray == i]
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        for x, y in pts[1:-1]:
            # Distance from the line through the first and the last chip.
            d = abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / max(1e-9, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
            assert d < 0.5, (ray.id, d)


def test_an_arm_that_is_out_rolls_up_a_little_to_its_side():
    rays = star.model_rays()
    attach = next(i for i, r in enumerate(rays) if r.id == "ATTACH")
    extents = [0.0] * len(rays)
    extents[attach] = 1.0
    straight = star.star_layout(rays, (3000, 3000), 6000, 6000, 1.0, extents=extents)
    for side in (1.0, -1.0):
        wiggles = [0.0] * len(rays)
        wiggles[attach] = side
        rolled = star.star_layout(rays, (3000, 3000), 6000, 6000, 1.0, extents=extents, wiggles=wiggles)
        a = [c.centre for c in straight.chips if c.ray == attach][-1]
        b = [c.centre for c in rolled.chips if c.ray == attach][-1]
        # Attach points up: rolling to one side moves the tip sideways.
        assert 60 < abs(b[0] - a[0]) < 220 and (b[0] - a[0]) * side < 0
        # The base stays where it was; the roll is at the tip.
        first = [c.centre for c in straight.chips if c.ray == attach][1]
        first_rolled = [c.centre for c in rolled.chips if c.ray == attach][1]
        assert abs(first_rolled[0] - first[0]) < 3
    closed = star.star_layout(rays, (3000, 3000), 6000, 6000, 1.0, wiggles=[1.0] * len(rays))
    assert [c.centre for c in closed.chips] == [c.centre for c in star.star_layout(rays, (3000, 3000), 6000, 6000, 1.0).chips]


def test_a_rolled_arm_keeps_its_chips_apart():
    rays = star.model_rays()
    for i, ray in enumerate(rays):
        extents = [0.0] * len(rays)
        extents[i] = 1.0
        wiggles = [0.0] * len(rays)
        wiggles[i] = 1.0
        lay = star.star_layout(rays, (3000, 3000), 6000, 6000, 1.0, extents=extents, wiggles=wiggles)
        arm = [c for c in lay.chips if c.ray == i]
        for x, a in enumerate(arm):
            for b in arm[x + 1:]:
                assert math.hypot(a.centre[0] - b.centre[0], a.centre[1] - b.centre[1]) >= (a.side + b.side) / 2 - 1, ray.id


def test_an_arm_sways_a_little_and_now_and_then_rolls_all_the_way_slowly():
    samples = [star.curl_amount(t / 10.0, key) for key in range(8) for t in range(0, 10 * 60)]
    gentle = sum(1 for v in samples if abs(v) <= star.SWAY_ROLL + 1e-9) / len(samples)
    assert 0.5 < gentle < 0.95  # most of the time: a gentle roll
    assert any(0.05 < abs(v) <= star.SWAY_ROLL for v in samples)  # which is not straight
    assert max(samples) > 0.95 and min(samples) < -0.95  # sometimes all the way, either side
    for key in range(2):
        values = [star.curl_amount(t / 60.0, key) for t in range(0, 60 * 24)]
        assert max(abs(a - b) for a, b in zip(values, values[1:])) < 0.04  # slow, no jumps
