"""The star's play (rok/core/star_play.py): the arms after a long stillness."""

import math

import pytest

from rok.core import star, star_play

W, H = 1400.0, 800.0


def _scene(seed: int = 0) -> star_play.Scene:
    rays = star.model_rays()
    starts = star.starts(star.closed(rays), 1.0)
    targets = {star_play.NAV_CUBE: (1310.0, 710.0), star_play.PALETTE: (250.0, 120.0),
               star_play.HINT: (700.0, 16.0), star_play.RAIL: [(10.0, 700.0 - 50 * k, 44.0, 44.0) for k in range(6)],
               star_play.SWATCHES: [((120.0 + 16 * k, 60.0), (200, 40 * k, 90)) for k in range(5)]}
    return star_play.Scene(rays, (700.0, 400.0), starts, W, H, 70.0, targets, seed, (9.0, 41.0, 12.0),
                           "Click or drag to place blocks")


def _base(scene):
    layout = star.star_layout(scene.rays, scene.origin, W, H, 1.0, opened=5.0, ray_starts=scene.starts, left=scene.left)
    return star_play.base_pose(scene, layout)


def _seed_with(thing: str, first: bool = False) -> int:
    """A seed whose hold plays `thing` (first, if asked)."""
    for seed in range(5000):
        acts = star_play.schedule(seed, 300.0)
        if (acts[0][0] == thing) if first else any(n == thing for n, _s in acts):
            return seed
    raise AssertionError(thing)


def _start_of(thing: str, seed: int) -> float:
    return next(start for name, start in star_play.schedule(seed, 400.0) if name == thing)


def test_nothing_plays_before_the_stillness_is_long_enough():
    scene = _scene()
    base = _base(scene)
    assert star_play.thing_at(star_play.START - 0.01) is None
    assert star_play.pose(scene, base, 3.0) is None


def test_single_things_come_at_random_with_rest_between_them():
    firsts = set()
    counts = {name: 0 for name, _d, _w in star_play.THINGS}
    for seed in range(400):
        acts = star_play.schedule(seed, 200.0)
        first_name, first_start = acts[0]
        firsts.add(first_name)
        assert star_play.START <= first_start <= star_play.START + star_play.START_JITTER
        for (a, sa), (b, sb) in zip(acts, acts[1:]):
            assert a != b  # never the same thing twice in a row
            assert sb - (sa + star_play.DURATION[a]) == pytest.approx(star_play.REST)  # the plain star between
        for name, _start in acts:
            counts[name] += 1
    assert len(firsts) >= 6  # random, not a fixed order
    assert counts["FLOWER"] > 5 * counts["DEEP"] > 0  # the deep is rare, but it comes


def test_between_two_things_the_star_rests():
    scene = _scene()
    base = _base(scene)
    (name, start), (_next, next_start) = star_play.schedule(scene.seed, 100.0)[:2]
    assert star_play.pose(scene, base, start + star_play.DURATION[name] / 2) is not None
    assert star_play.pose(scene, base, start + star_play.DURATION[name] + 1.0) is None
    assert star_play.pose(scene, base, next_start - 0.1) is None


@pytest.mark.parametrize("thing", [n for n, _d, _w in star_play.THINGS])
def test_every_thing_poses_every_chip_inside_the_view_and_grows_out_of_the_star(thing):
    seed = _seed_with(thing)
    scene = _scene(seed)
    base = _base(scene)
    start = _start_of(thing, seed)
    end = start + star_play.DURATION[thing]
    t = start
    while t < end:
        pose = star_play.pose(scene, base, t)
        assert pose is not None, t
        for i, ray in enumerate(scene.rays):
            assert len(pose.chips[i]) == len(ray.chips), (t, ray.id)
            for x, y in pose.chips[i]:
                assert math.isfinite(x) and math.isfinite(y), (t, ray.id)
                assert -5 <= x <= W + 5 and -5 <= y <= H + 5, (t, ray.id, x, y)
            assert len(pose.arms[i]) >= 2
        t += 0.31
    for edge in (start + 0.005, end - 0.005):  # no chip jumps where the thing starts or ends
        pose = star_play.pose(scene, base, edge)
        jump = max(math.hypot(p[0] - q[0], p[1] - q[1]) for i in range(len(base.chips))
                   for p, q in zip(base.chips[i], pose.chips[i]))
        assert jump < 12.0, (thing, edge, jump)


def test_an_arm_swallows_the_nav_cube_and_gives_it_back():
    seed = _seed_with("EAT")
    scene = _scene(seed)
    base = _base(scene)
    eat = _start_of("EAT", seed)
    assert star_play.pose(scene, base, eat + 3.0).ui[star_play.NAV_CUBE][2] is False  # shaken, still there
    eaten = star_play.pose(scene, base, eat + 5.5)
    assert eaten.ui[star_play.NAV_CUBE][2] is True and eaten.bulge is not None
    assert star_play.pose(scene, base, eat + 10.0).ui[star_play.NAV_CUBE][2] is False


def test_the_deep_darkens_and_opens_an_eye():
    seed = _seed_with("DEEP")
    scene = _scene(seed)
    deep = star_play.pose(scene, _base(scene), _start_of("DEEP", seed) + 8.0)
    assert deep.darken > 0.5 and len(deep.tentacles) == 8
    assert deep.eye is not None and deep.eye[0] > 0.0


def test_the_clock_shows_the_time_with_three_hands():
    seed = _seed_with("CLOCK")
    scene = _scene(seed)
    pose = star_play.pose(scene, _base(scene), _start_of("CLOCK", seed) + 3.0)
    ox, oy = scene.origin
    hour = scene.ray_index("ATTACH")
    x, y = pose.chips[hour][-1]
    angle = math.degrees(math.atan2(x - ox, y - oy)) % 360  # clockwise from twelve
    assert angle == pytest.approx((9 + 41 / 60) * 30, abs=3)  # the hour hand at 9:41


def test_drumming_lights_the_rail_and_the_ribbon_takes_the_hint():
    seed = _seed_with("DRUM")
    scene = _scene(seed)
    drum = star_play.pose(scene, _base(scene), _start_of("DRUM", seed) + 3.0)
    assert drum.flashes and all(0 <= a <= 1 for _r, a in drum.flashes)
    seed = _seed_with("RIBBON")
    scene = _scene(seed)
    ribbon = star_play.pose(scene, _base(scene), _start_of("RIBBON", seed) + 3.5)
    assert ribbon.ui[star_play.HINT][2] is True
    assert "".join(t[0] for t in ribbon.texts) == scene.hint


def test_a_dip_colours_the_arm_and_the_sleepers_wake_with_a_start():
    seed = _seed_with("DIP")
    scene = _scene(seed)
    dip = star_play.pose(scene, _base(scene), _start_of("DIP", seed) + 5.0)
    paint = scene.ray_index("PAINT")
    assert dip.tints[paint][1] > 0.5
    assert max(star_play.wake_weight("SLEEP", w / 50) for w in range(51)) > 1.05  # past the star, then back
    assert star_play.wake_weight("SLEEP", 1.0) == pytest.approx(1.0)
    assert star_play.wake_weight("FLOWER", 0.5) <= 1.0
