"""The star's play: what its arms do while Space is held and nothing moves. Pure.

The star stays a quick menu: the play starts only after about START seconds
with the pointer still and no arm reaching out, and a move ends it. Then
single things happen, one at a time, picked at random, some much rarer than
others (THINGS), with at least REST seconds of the plain star between them:

- a flower; a turning spiral; a beating heart; a jellyfish;
- an arm shakes the palette card; an arm eats the nav cube and spits it back;
- juggling, a stadium wave, a nap (a move wakes them with a start), a clock
  with the time of day;
- arms drum on the rail, dip into a palette colour, pull the hint line away
  like a ribbon;
- an arm tries to escape over the edge (rare), the star splits in two (rare),
  a dance party with disco lights (rarer);
- two arms wrestle, one wins, the other sulks (rare);
- the deep: the view darkens, the arms writhe, tentacles creep in from the
  edges and an eye opens in the middle (very rare).

`pose(scene, base, t)` says where every chip and arm is `t` seconds into the
stillness, or None while the star rests. A thing blends in from the star and
back out to it, so nothing jumps. Points are region pixels (y up).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

START = 6.0  # about this many seconds of stillness before the first thing
START_JITTER = 1.5  # the first thing comes up to this much later
REST = 10.0  # the plain star between two things, at least
# (name, seconds, weight): a weight 10 times smaller is 10 times rarer.
THINGS = (
    ("FLOWER", 4.0, 5.0),
    ("SPIRAL", 4.0, 5.0),
    ("HEART", 4.0, 4.0),
    ("JELLY", 5.0, 4.0),
    ("SHAKE", 6.5, 3.0),
    ("EAT", 11.5, 2.0),
    ("FIGHT", 15.0, 1.0),
    ("DEEP", 15.0, 0.4),
    ("JUGGLE", 6.0, 4.0),
    ("WAVE", 5.0, 4.0),
    ("SLEEP", 9.0, 3.0),
    ("CLOCK", 6.0, 3.0),
    ("DRUM", 6.0, 3.0),
    ("DIP", 9.5, 3.0),
    ("RIBBON", 7.5, 2.0),
    ("ESCAPE", 8.5, 1.5),
    ("MITOSIS", 8.0, 1.0),
    ("DANCE", 8.0, 0.8),
)
HINT, RAIL, SWATCHES = "HINT", "RAIL", "SWATCHES"
DURATION = {name: seconds for name, seconds, _w in THINGS}
BLEND_S = 1.0  # a phase blends in from the last one over this long
EXIT_S = 0.35  # a move: the arms go back to the star this fast
FIGHTERS = ("ATTACH", "ERASE")
EATER, SHAKER = "ERASE", "PAINT"  # the arm that eats the nav cube, the arm that shakes the palette
NAV_CUBE, PALETTE = "NAV_CUBE", "PALETTE"

Point = tuple[float, float]


@dataclass
class Pose:
    """Where everything is. Per ray: its chips' centres, a size factor and an
    alpha per chip, and the arm's path from the star's middle."""

    chips: list[list[Point]]
    sizes: list[list[float]]
    alphas: list[list[float]]
    arms: list[list[Point]]
    darken: float = 0.0  # 0..1 over the whole view
    tentacles: list[tuple[list[Point], float]] = field(default_factory=list)  # (path, width at the base)
    eye: tuple[float, float, float] | None = None  # (how open 0..1, look x, look y in -1..1)
    sparks: list[tuple[float, float, float, float]] = field(default_factory=list)  # x, y, radius, alpha
    bulge: tuple[float, float, float] | None = None  # x, y, radius: a swallowed thing in an arm
    ui: dict = field(default_factory=dict)  # NAV_CUBE / PALETTE / HINT -> (dx, dy, hidden)
    texts: list[tuple[str, float, float, float, float]] = field(default_factory=list)  # text, x, y, size, alpha
    flashes: list[tuple[tuple[float, float, float, float], float]] = field(default_factory=list)  # (rect, alpha)
    tints: dict = field(default_factory=dict)  # ray index -> ((r, g, b) 0..1, strength 0..1)
    spots: list[tuple[list[Point], tuple[float, float, float, float]]] = field(default_factory=list)  # light wedges
    cores: list[Point] = field(default_factory=list)  # the star's middles, when there are two


# --- small geometry ---------------------------------------------------------------------


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _lerp(a: Point, b: Point, w: float) -> Point:
    return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)


def length(path) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def resample(path, n: int) -> list[Point]:
    """`n` points spread evenly along a polyline (its ends included)."""
    path = [tuple(p) for p in path]
    if n <= 0:
        return []
    if len(path) == 1 or n == 1:
        return [path[-1] if n == 1 and len(path) > 1 else path[0]] * n
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    total = sum(seg) or 1e-9
    out = []
    k, walked = 0, 0.0
    for j in range(n):
        want = total * j / (n - 1)
        while k < len(seg) - 1 and walked + seg[k] < want:
            walked += seg[k]
            k += 1
        f = 0.0 if seg[k] <= 1e-9 else min(1.0, (want - walked) / seg[k])
        out.append(_lerp(path[k], path[k + 1], f))
    return out


def head(path, fraction: float) -> list[Point]:
    """The first `fraction` of a polyline by length."""
    fraction = min(max(fraction, 0.0), 1.0)
    if fraction >= 1.0 or len(path) < 2:
        return list(path)
    want = length(path) * fraction
    out = [path[0]]
    walked = 0.0
    for a, b in zip(path, path[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if walked + d >= want:
            out.append(_lerp(a, b, 0.0 if d <= 1e-9 else (want - walked) / d))
            return out
        walked += d
        out.append(b)
    return out


def point_at(path, fraction: float) -> Point:
    return head(path, fraction)[-1]


def curve(a: Point, b: Point, bend: float, steps: int = 24) -> list[Point]:
    """A soft arc from a to b, bowed sideways by `bend` pixels."""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy) or 1.0
    cx, cy = mx - dy / d * bend, my + dx / d * bend
    out = []
    for k in range(steps + 1):
        s = k / steps
        out.append(((1 - s) ** 2 * a[0] + 2 * (1 - s) * s * cx + s * s * b[0],
                    (1 - s) ** 2 * a[1] + 2 * (1 - s) * s * cy + s * s * b[1]))
    return out


def _dir(angle_deg: float) -> Point:
    a = math.radians(angle_deg)
    return (math.cos(a), math.sin(a))


# --- the phases ------------------------------------------------------------------------


def schedule(seed: int, until: float) -> list[tuple[str, float]]:
    """The things of one hold that start before `until`: (name, start). Picked
    at random by weight, never the same twice in a row, REST seconds apart."""
    rng = random.Random(seed)
    out: list[tuple[str, float]] = []
    t = START + rng.random() * START_JITTER
    while t <= until:
        choices = [(n, w) for n, _d, w in THINGS if not out or n != out[-1][0]]
        pick = rng.random() * sum(w for _n, w in choices)
        for name, w in choices:
            pick -= w
            if pick <= 0.0:
                break
        out.append((name, t))
        t += DURATION[name] + REST
    return out


def thing_at(t: float, seed: int = 0) -> tuple[str, float] | None:
    """(the thing, seconds into it) at `t`, or None while the star rests."""
    if t < START:
        return None
    for name, start in reversed(schedule(seed, t)):
        if t >= start:
            return (name, t - start) if t - start < DURATION[name] else None
    return None


@dataclass
class Scene:
    """What the poses need to know besides the time."""

    rays: tuple  # star.Ray
    origin: Point
    starts: tuple[float, ...]  # how far each ray starts from the middle
    width: float
    height: float
    left: float  # where the free view starts
    targets: dict  # NAV_CUBE / PALETTE / HINT -> their middles; RAIL -> button rects; SWATCHES -> (middle, rgb)
    seed: int = 0
    clock: tuple[float, float, float] = (10.0, 10.0, 30.0)  # the time of day: hours, minutes, seconds
    hint: str = ""  # the hint line's text

    def start_point(self, i: int) -> Point:
        dx, dy = _dir(self.rays[i].angle)
        r = self.starts[i]
        return self.clamp((self.origin[0] + dx * r, self.origin[1] + dy * r))

    def clamp(self, p: Point, margin: float = 16.0) -> Point:
        return (min(max(p[0], self.left + margin), self.width - margin),
                min(max(p[1], margin), self.height - margin))

    def ray_index(self, ident: str) -> int | None:
        return next((i for i, r in enumerate(self.rays) if r.id == ident), None)


def _arm_from(scene: Scene, i: int, path) -> list[Point]:
    """The arm: from near the star's middle to the start of `path`, then along it."""
    ox, oy = scene.origin
    p0 = path[0]
    d = math.hypot(p0[0] - ox, p0[1] - oy) or 1.0
    near = (ox + (p0[0] - ox) / d * 26.0, oy + (p0[1] - oy) / d * 26.0)
    return [near] + list(path)


def _place(scene: Scene, i: int, path, size: float, arm=None) -> tuple[list[Point], list[float], list[float], list[Point]]:
    n = len(scene.rays[i].chips)
    chips = resample(path, n)
    return chips, [size] * n, [1.0] * n, _arm_from(scene, i, arm if arm is not None else path)


def _curl(scene: Scene, i: int, t: float, lean: Point | None = None):
    """An arm with nothing to do rolls up near where it starts, slowly turning."""
    base = scene.start_point(i)
    a0 = math.radians(scene.rays[i].angle) + 0.4 * math.sin(t * 0.9 + i)
    if lean is not None:
        a0 = math.atan2(lean[1] - base[1], lean[0] - base[0])
    path = []
    for k in range(40):
        s = k / 39
        r = 26.0 * (1 - s) + 3.0
        a = a0 + 3.4 * math.pi * s
        path.append((base[0] + math.cos(a0) * 24 * s + math.cos(a) * r * s,
                     base[1] + math.sin(a0) * 24 * s + math.sin(a) * r * s))
    n = len(scene.rays[i].chips)
    chips = resample(path, n)
    sizes = [0.5 * (1 - 0.4 * k / max(1, n - 1)) for k in range(n)]
    return chips, sizes, [1.0] * n, _arm_from(scene, i, path)


def _empty(scene: Scene) -> Pose:
    n = len(scene.rays)
    return Pose([[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)])


def _set(pose: Pose, i: int, parts) -> None:
    pose.chips[i], pose.sizes[i], pose.alphas[i], pose.arms[i] = parts


# SHAPES ----------------------------------------------------------------------------------

def _shape(scene: Scene, name: str, u: float) -> Pose:
    pose = _empty(scene)
    ox, oy = scene.origin
    n = len(scene.rays)
    if name == "HEART":
        counts = [len(r.chips) for r in scene.rays]
        total = sum(counts)
        beat = 1.0 + 0.07 * max(0.0, math.sin(2 * math.pi * u * 1.1)) ** 8
        k = 7.2 * beat
        heart = []
        for j in range(181):
            a = 2 * math.pi * j / 180
            hx = 16 * math.sin(a) ** 3
            hy = 13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)
            heart.append(scene.clamp((ox + hx * k, oy + 30 + hy * k)))
        done = 0
        for i in range(n):
            lo, hi = done / total, (done + counts[i]) / total
            done += counts[i]
            seg = heart[int(lo * 180):max(int(hi * 180), int(lo * 180) + 2) + 1]
            _set(pose, i, _place(scene, i, seg, 0.5))
        return pose
    for i, ray in enumerate(scene.rays):
        count = len(ray.chips)
        a0 = 360.0 * i / n
        if name == "FLOWER":
            a = a0 + 18.0 * u
            path = []
            for j in range(31):
                s = j / 30
                r = 46 + 92 * math.sin(math.pi * s)
                path.append(scene.clamp((ox + math.cos(math.radians(a + 34 * (s - 0.5) * 2)) * r,
                                         oy + math.sin(math.radians(a + 34 * (s - 0.5) * 2)) * r)))
            _set(pose, i, _place(scene, i, path, 0.5))
        elif name == "SPIRAL":
            path = []
            for j in range(41):
                s = j / 40
                r = 30 + 26 * count * s
                a = math.radians(a0 + 200 * s + 70 * u)
                path.append(scene.clamp((ox + math.cos(a) * r, oy + math.sin(a) * r)))
            chips, _sz, alphas, arm = _place(scene, i, path, 0.5)
            sizes = [0.62 - 0.3 * k / max(1, count - 1) for k in range(count)]
            _set(pose, i, (chips, sizes, alphas, arm))
        else:  # JELLY: the star is the bell, the arms hang under it and sway
            bob = 14 * math.sin(2 * math.pi * u / 1.6)
            spread = (i + 0.5) / n - 0.5
            base = (ox + spread * 150, oy - 20 + bob - 12 * math.cos(spread * math.pi))
            path = []
            for j in range(31):
                s = j / 30
                drop = 26 * count * s
                sway = 16 * s * math.sin(2 * math.pi * (s * 1.2 - u * 0.7) + i)
                path.append(scene.clamp((base[0] + sway + spread * 40 * s, base[1] - drop)))
            _set(pose, i, _place(scene, i, path, 0.5))
    return pose


# GRAB ------------------------------------------------------------------------------------


def _reach(scene: Scene, i: int, target: Point, fraction: float, u: float, offset: Point = (0.0, 0.0)):
    start = scene.start_point(i)
    goal = (target[0] + offset[0], target[1] + offset[1])
    bend = 60 * math.sin(u * 1.7 + i)
    path = head(curve(start, goal, bend), max(fraction, 0.05))
    chips, sizes, alphas, arm = _place(scene, i, path, 0.55)
    return chips, sizes, alphas, arm, path


def _eat(scene: Scene, u: float) -> Pose:
    """An arm reaches for the nav cube, shakes it, swallows it, and spits it back."""
    pose = _empty(scene)
    eater, cube = scene.ray_index(EATER), scene.targets.get(NAV_CUBE)
    if eater is None or cube is None:
        return pose
    hidden = 4.6 <= u < 9.0
    shake = (9.0 * math.sin(2 * math.pi * u * 6.0), 3.0 * math.sin(2 * math.pi * u * 4.0)) if 2.0 <= u < 4.6 else (0.0, 0.0)
    if u < 2.0:
        reach = ease(u / 2.0)
    elif u < 4.6:
        reach = 1.0
    elif u < 7.0:
        reach = 1.0 - 0.65 * ease((u - 4.6) / 2.4)
    elif u < 8.5:
        reach = 0.35 + 0.65 * ease((u - 7.5) / 1.0)
    elif u < 9.5:
        reach = 1.0
    else:
        reach = 1.0 - ease((u - 9.5) / 2.0)
    bounce = (0.0, 16.0 * math.exp(-3.0 * (u - 9.0)) * math.sin(2 * math.pi * (u - 9.0) * 2.2)) if u >= 9.0 else (0.0, 0.0)
    chips, sizes, alphas, arm, path = _reach(scene, eater, cube, reach, u, shake)
    _set(pose, eater, (chips, sizes, alphas, arm))
    pose.ui[NAV_CUBE] = (shake[0] + bounce[0], shake[1] + bounce[1], hidden)
    if 4.6 <= u < 7.0:  # swallowed: a lump goes down the arm
        pose.bulge = (*point_at(path, 1.0 - ease((u - 4.6) / 2.4)), 12.0)
    elif 8.0 <= u < 9.0:  # and comes back up
        pose.bulge = (*point_at(path, ease((u - 8.0) / 1.0)), 12.0)
    return pose


def _shake(scene: Scene, u: float) -> Pose:
    """An arm reaches for the palette card, shakes it, and lets go."""
    pose = _empty(scene)
    shaker, card = scene.ray_index(SHAKER), scene.targets.get(PALETTE)
    if shaker is None or card is None:
        return pose
    if u < 2.0:
        reach = ease(u / 2.0)
    elif u < 5.0:
        reach = 1.0
    else:
        reach = 1.0 - ease((u - 5.0) / 1.5)
    holding = 2.0 <= u < 5.0
    shake = (7.0 * math.sin(2 * math.pi * u * 4.5), 5.0 * math.sin(2 * math.pi * u * 3.1)) if holding else (0.0, 0.0)
    chips, sizes, alphas, arm, _path = _reach(scene, shaker, card, reach, u, shake)
    _set(pose, shaker, (chips, sizes, alphas, arm))
    wobble = 6.0 * math.exp(-2.5 * (u - 5.0)) * math.sin(2 * math.pi * (u - 5.0) * 3.0) if u >= 5.0 else 0.0
    pose.ui[PALETTE] = (shake[0] + wobble, shake[1], False)
    return pose


# FIGHT -----------------------------------------------------------------------------------


def _fight(scene: Scene, u: float) -> Pose:
    pose = _empty(scene)
    ox, oy = scene.origin
    ring = scene.clamp((ox, oy + 120), 80)
    a_i, b_i = (scene.ray_index(r) for r in FIGHTERS)
    winner = (a_i, b_i)[scene.seed % 2] if a_i is not None and b_i is not None else None
    if winner is None:
        return pose
    loser = b_i if winner == a_i else a_i
    jitter = (6 * math.sin(u * 17.0), 6 * math.cos(u * 13.0))
    meet = (ring[0] + jitter[0], ring[1] + jitter[1])
    if 11.0 <= u < 12.0:  # the winner pushes
        base = scene.start_point(loser)
        meet = _lerp(meet, base, 0.35 * ease(u - 11.0))
    for i, sign in ((a_i, 1.0), (b_i, -1.0)):
        start = scene.start_point(i)
        if u >= 12.0 and i == loser:
            # Beaten: it pulls back and rolls up, drooping and trembling.
            droop = (start[0] + 2 * math.sin(u * 23.0), start[1] - 30 * ease((u - 12.0) / 1.5))
            path = []
            for k in range(36):
                s = k / 35
                r = 22 * (1 - s) + 3
                a = -math.pi / 2 + 3.6 * math.pi * s * sign
                path.append((droop[0] + math.cos(a) * r * s, droop[1] + math.sin(a) * r * s - 20 * s))
            n = len(scene.rays[i].chips)
            _set(pose, i, (resample(path, n), [0.42] * n, [0.8] * n, _arm_from(scene, i, path)))
            continue
        if u >= 12.0 and i == winner:
            # It won: up it goes, the tip twirling.
            top = scene.clamp((start[0], start[1] + 150), 30)
            path = curve(start, top, 20 * math.sin(u * 2.0))
            tip = path[-1]
            path += [(tip[0] + 18 * math.cos(u * 9 + k * 0.5) * k / 8, tip[1] + 18 * math.sin(u * 9 + k * 0.5) * k / 8)
                     for k in range(1, 9)]
            _set(pose, i, _place(scene, i, path, 0.6))
            continue
        reach = ease(u / 1.5)
        path = head(curve(start, meet, 40 * sign), reach)
        if u >= 1.5:  # wrestling: the tips wrap around each other
            spin = u * 4.0 * sign
            wrap = [(meet[0] + math.cos(spin + k * 0.45 * sign) * (24 - 1.6 * k),
                     meet[1] + math.sin(spin + k * 0.45 * sign) * (24 - 1.6 * k)) for k in range(12)]
            path = path + wrap
        _set(pose, i, _place(scene, i, path, 0.55))
    if 1.5 <= u < 12.0:
        rng = random.Random(scene.seed * 7919 + int(u * 12))
        for _k in range(7):
            a = rng.random() * 2 * math.pi
            d = 10 + 40 * rng.random()
            pose.sparks.append((meet[0] + math.cos(a) * d, meet[1] + math.sin(a) * d, 2.0 + 2.5 * rng.random(),
                                0.4 + 0.6 * rng.random()))
    return pose


# DEEP ------------------------------------------------------------------------------------


def _edge_anchors(scene: Scene, count: int) -> list[tuple[Point, Point]]:
    """Points spread around the view's edge, each with the direction inward."""
    w, h, left = scene.width, scene.height, scene.left
    per = 2 * (w - left) + 2 * h
    out = []
    for k in range(count):
        d = per * (k + 0.37) / count
        if d < w - left:
            out.append(((left + d, 0.0), (0.0, 1.0)))
        elif d < w - left + h:
            out.append(((w, d - (w - left)), (-1.0, 0.0)))
        elif d < 2 * (w - left) + h:
            out.append(((w - (d - (w - left) - h), h), (0.0, -1.0)))
        else:
            out.append(((left, h - (d - 2 * (w - left) - h)), (1.0, 0.0)))
    return out


def _deep(scene: Scene, u: float) -> Pose:
    pose = _empty(scene)
    ox, oy = scene.origin
    grow = 1.0 + 0.6 * ease(u / 6.0)
    for i, ray in enumerate(scene.rays):
        start = scene.start_point(i)
        count = len(ray.chips)
        a = math.radians(ray.angle)
        dx, dy = math.cos(a), math.sin(a)
        path = []
        for j in range(41):
            s = j / 40
            along = 30 * count * grow * s
            side = 22 * s * math.sin(2 * math.pi * (s * 1.6 - 0.35 * u) + i * 1.3)
            path.append(scene.clamp((start[0] + dx * along - dy * side, start[1] + dy * along + dx * side)))
        chips, _sz, alphas, arm = _place(scene, i, path, 0.5)
        sizes = [0.55 - 0.25 * k / max(1, count - 1) for k in range(count)]
        _set(pose, i, (chips, sizes, alphas, arm))
    ending = ease((DURATION["DEEP"] - u) / 3.0)  # the last seconds: the deep withdraws
    pose.darken = 0.62 * ease(u / 4.0) * ending
    for k, (anchor, inward) in enumerate(_edge_anchors(scene, 8)):
        reach = ease((u - 1.5 - 0.4 * k) / 5.0) * ending
        if reach <= 0.0:
            continue
        dist = math.hypot(ox - anchor[0], oy - anchor[1])
        to = ((ox - anchor[0]) / (dist or 1), (oy - anchor[1]) / (dist or 1))
        aim = (to[0] * 0.6 + inward[0] * 0.4, to[1] * 0.6 + inward[1] * 0.4)
        total = dist * 0.55 * reach
        path = []
        for j in range(33):
            s = j / 32
            side = 26 * s * math.sin(2 * math.pi * (s * 1.3 - 0.25 * u) + k * 1.7)
            path.append((anchor[0] + aim[0] * total * s - aim[1] * side, anchor[1] + aim[1] * total * s + aim[0] * side))
        pose.tentacles.append((path, 28.0))
    opened = ease((u - 3.0) / 1.5)
    beat = (u - 3.0) % 4.3
    blink = 1.0 - (1.0 - abs(beat - 0.09) / 0.09 if beat < 0.18 else 0.0)
    pose.eye = (opened * blink * ending, 0.55 * math.sin(0.7 * u), 0.35 * math.sin(1.1 * u + 1.0))
    if ending <= 0.0:
        pose.eye = None
    return pose


# MORE THINGS -------------------------------------------------------------------------------


def _from_base(base: Pose) -> Pose:
    """A copy of the star's own pose to change."""
    return Pose([list(c) for c in base.chips], [list(z) for z in base.sizes], [list(a) for a in base.alphas],
                [list(a) for a in base.arms])


def _juggle(scene: Scene, u: float) -> Pose:
    """One arm throws its outer chips up and catches them, a fountain."""
    pose = _empty(scene)
    i = scene.ray_index("ATTACH")
    if i is None:
        return pose
    start = scene.start_point(i)
    hand = scene.clamp((start[0], start[1] + 40))
    n = len(scene.rays[i].chips)
    arm = [start, hand]
    chips = [start, hand]
    period = 1.3
    for k in range(2, n):
        phase = ((u * 1.0 - k * period / max(1, n - 2)) % period) / period
        side = 46 if k % 2 else -46
        x = hand[0] + side * math.sin(math.pi * phase)
        y = hand[1] + 150 * 4 * phase * (1 - phase)
        chips.append(scene.clamp((x, y)))
    pose.chips[i] = chips
    pose.sizes[i] = [0.7] * n
    pose.alphas[i] = [1.0] * n
    pose.arms[i] = _arm_from(scene, i, arm)
    return pose


def _wave(scene: Scene, u: float, base: Pose) -> Pose:
    """A stadium wave: the chips stand up one after another around the star."""
    pose = _from_base(base)
    ox, oy = scene.origin
    front = (u / 2.4) * 2 * math.pi  # two laps
    for i in range(len(pose.chips)):
        for k, (x, y) in enumerate(pose.chips[i]):
            a = math.atan2(y - oy, x - ox)
            d = (a - front + math.pi) % (2 * math.pi) - math.pi
            lift = math.exp(-(d * d) / 0.18)
            r = math.hypot(x - ox, y - oy) or 1.0
            pose.chips[i][k] = (x + (x - ox) / r * 34 * lift, y + (y - oy) / r * 34 * lift)
            pose.sizes[i][k] = 1.0 + 0.45 * lift
    return pose


def _sleep(scene: Scene, u: float) -> Pose:
    """The arms droop and doze; z z z floats up. A move wakes them with a start."""
    pose = _empty(scene)
    breath = 1.0 + 0.06 * math.sin(2 * math.pi * u / 3.2)
    for i, ray in enumerate(scene.rays):
        start = scene.start_point(i)
        side = 1.0 if math.cos(math.radians(ray.angle)) >= 0 else -1.0
        path = []
        for j in range(25):
            s_ = j / 24
            path.append(scene.clamp((start[0] + side * 60 * s_ * breath, start[1] - 40 * s_ * s_ - 6 * math.sin(s_ * 3))))
        n = len(ray.chips)
        pose.chips[i], pose.sizes[i], pose.alphas[i] = resample(path, n), [0.5] * n, [0.8] * n
        pose.arms[i] = _arm_from(scene, i, path)
    ox, oy = scene.origin
    for k in range(3):
        age = (u - k * 0.9) % 2.7
        if u < k * 0.9:
            continue
        pose.texts.append(("z", ox + 14 + 12 * age + 6 * math.sin(age * 3), oy + 20 + 28 * age,
                           11 + 6 * age, max(0.0, 1.0 - age / 2.7)))
    return pose


def _clock(scene: Scene, u: float) -> Pose:
    """A clock face: the chips round the dial, three arms as the hands, the time of day."""
    pose = _empty(scene)
    ox, oy = scene.origin
    h, m, sec = scene.clock
    sec = sec + u  # the second hand runs on
    hands = {"ATTACH": ((h % 12 + m / 60) / 12, 0.5), "ERASE": ((m + sec / 60) / 60, 0.8),
             "SELECT": ((sec % 60) // 1 / 60, 0.92)}
    radius = 120.0
    ring = []
    for i, ray in enumerate(scene.rays):
        if ray.id in hands:
            turn, length_ = hands[ray.id]
            a = math.pi / 2 - 2 * math.pi * turn
            tip = scene.clamp((ox + math.cos(a) * radius * length_, oy + math.sin(a) * radius * length_))
            path = [(ox + math.cos(a) * 20, oy + math.sin(a) * 20), tip]
            n = len(ray.chips)
            pose.chips[i] = resample(path, n)
            pose.sizes[i] = [0.5] * n
            pose.alphas[i] = [1.0] * n
            pose.arms[i] = [scene.origin] + path
        else:
            ring += [(i, k) for k in range(len(ray.chips))]
    for slot, (i, k) in enumerate(ring):
        a = math.pi / 2 - 2 * math.pi * slot / max(1, len(ring))
        if not pose.chips[i]:
            n = len(scene.rays[i].chips)
            pose.chips[i], pose.sizes[i], pose.alphas[i] = [scene.origin] * n, [0.42] * n, [1.0] * n
        pose.chips[i][k] = scene.clamp((ox + math.cos(a) * radius, oy + math.sin(a) * radius))
    for i in range(len(scene.rays)):
        if not pose.arms[i] and pose.chips[i]:
            pose.arms[i] = [pose.chips[i][0], pose.chips[i][0]]  # no arm: the chips stand alone on the dial
    return pose


def _drum(scene: Scene, u: float) -> Pose:
    """Arms drum on the rail's buttons; each lights up as it is hit."""
    pose = _empty(scene)
    buttons = scene.targets.get(RAIL) or []
    drummers = [scene.ray_index(r) for r in ("PAINT", "SHADERS", "SELECT_BY")]
    drummers = [i for i in drummers if i is not None]
    if not buttons or not drummers:
        return pose
    beat = 0.24
    hits = int(u / beat)
    rng = random.Random(scene.seed * 31 + 7)
    plan = [rng.randrange(len(buttons)) for _ in range(int(DURATION["DRUM"] / beat) + 2)]
    reach = ease(u / 1.0) * ease((DURATION["DRUM"] - u) / 1.0)
    for d, i in enumerate(drummers):
        mine = [plan[h] for h in range(len(plan)) if h % len(drummers) == d]
        k = min(hits // len(drummers), len(mine) - 1)
        x, y, w, h = buttons[mine[k]]
        local = (u % (beat * len(drummers))) / (beat * len(drummers))
        lift = 18 * abs(math.sin(math.pi * local))
        target = (x + w / 2 + 8, y + h / 2 + lift)
        chips, sizes, alphas, arm, _path = _reach(scene, i, target, reach, u)
        _set(pose, i, (chips, sizes, alphas, arm))
    for h_ in range(max(0, hits - 4), hits + 1):
        if u >= 1.0 and h_ < len(plan):
            pose.flashes.append((buttons[plan[h_]], max(0.0, 1.0 - (u - h_ * beat) / 0.7) * 0.6))
    return pose


def _dip(scene: Scene, u: float) -> Pose:
    """An arm dips into a palette swatch; the colour climbs up the arm and fades."""
    pose = _empty(scene)
    swatches = scene.targets.get(SWATCHES) or []
    i = scene.ray_index("PAINT")
    if not swatches or i is None:
        return pose
    centre, rgb = swatches[random.Random(scene.seed).randrange(len(swatches))]
    if u < 2.0:
        reach = ease(u / 2.0)
    elif u < 3.6:
        reach = 1.0
    else:
        reach = 1.0 - 0.8 * ease((u - 3.6) / 1.4)
    dunk = (0.0, -6.0 * math.sin(math.pi * min(1.0, max(0.0, (u - 2.0) / 0.6))))
    chips, sizes, alphas, arm, _path = _reach(scene, i, centre, reach, u, dunk)
    _set(pose, i, (chips, sizes, alphas, arm))
    strength = ease((u - 2.4) / 1.2) * ease((DURATION["DIP"] - 0.6 - u) / 1.5)
    if strength > 0:
        pose.tints[i] = (tuple(c / 255.0 for c in rgb), strength)
    return pose


def _ribbon(scene: Scene, u: float) -> Pose:
    """An arm pulls the hint line's text off the bottom bar like a ribbon, then puts it back."""
    pose = _empty(scene)
    i = scene.ray_index("SELECT")
    hint = scene.targets.get(HINT)
    if i is None or hint is None or not scene.hint:
        return pose
    if u < 1.5:
        reach = ease(u / 1.5)
    elif u < 5.8:
        reach = 1.0
    else:
        reach = 1.0 - ease((u - 5.8) / 1.5)
    pulled = ease((u - 1.5) / 1.5) * ease((5.8 - u) / 1.2) if 1.5 <= u < 5.8 else 0.0
    tip = (hint[0] + 120 * pulled * math.sin(u * 1.3), hint[1] + 170 * pulled)
    chips, sizes, alphas, arm, path = _reach(scene, i, tip, reach, u)
    _set(pose, i, (chips, sizes, alphas, arm))
    if pulled > 0.02:
        pose.ui[HINT] = (0.0, 0.0, True)
        letters = scene.hint[:60]
        n = len(letters)
        # Before the pull the letters lie in a row on the bar; pulled, they hang
        # from the tip as a ribbon that flutters, one letter every 7.5 pixels.
        for k, ch in enumerate(letters):
            flat = (hint[0] + (k - n / 2) * 7.0, hint[1])
            s_ = k * 7.5
            hang = (tip[0] + 22 * math.sin(s_ / 40.0 - u * 3.0) * min(1.0, s_ / 60.0) + 0.3 * s_,
                    tip[1] - s_ * 0.95)
            x, y = _lerp(flat, hang, pulled)
            pose.texts.append((ch, x, y, 12.0, 1.0))
    return pose


def _escape(scene: Scene, u: float) -> Pose:
    """One arm sneaks to the edge of the view and tries to climb out; two others pull it back."""
    pose = _empty(scene)
    runner = scene.ray_index("SELECT_BY")
    helpers = [j for j in (scene.ray_index("SELECT"), scene.ray_index("ERASE")) if j is not None]
    if runner is None:
        return pose
    start = scene.start_point(runner)
    edge = (scene.width - 4, max(30.0, start[1] - 120)) if start[0] > scene.width / 2 else (scene.left + 4, start[1] - 120)
    if u < 3.0:
        out = ease(u / 3.0)
    elif u < 6.0:
        out = 1.0
    else:
        out = 1.0 - ease((u - 6.0) / 2.0) + 0.05 * math.sin(u * 20) * (u < 7.5)
    climb = (0.0, 30 * math.sin(u * 6) * (3.0 <= u < 6.0))
    goal = (edge[0] + climb[0], edge[1] + climb[1])
    path = head(curve(start, goal, 30 * math.sin(u * 5)), max(0.05, out))
    n = len(scene.rays[runner].chips)
    sizes = [0.45] * n
    pose.chips[runner], pose.sizes[runner], pose.alphas[runner] = resample(path, n), sizes, [1.0] * n
    pose.arms[runner] = _arm_from(scene, runner, path)
    grip = point_at(path, 0.6)
    for d, j in enumerate(helpers):
        reach = ease((u - 3.8 - 0.3 * d) / 1.2) * ease((8.3 - u) / 1.0) if u >= 3.8 else 0.0
        if reach <= 0.0:
            continue
        chips, szs, alphas, arm, _p = _reach(scene, j, grip, reach, u)
        _set(pose, j, (chips, szs, alphas, arm))
    return pose


def _mitosis(scene: Scene, u: float) -> Pose:
    """The star splits in two small stars that turn round each other, then join again."""
    pose = _empty(scene)
    ox, oy = scene.origin
    apart = 80 * ease(u / 2.0) * ease((DURATION["MITOSIS"] - u) / 2.0)
    turn = 2 * math.pi * ease((u - 1.5) / 5.0)
    cores = [scene.clamp((ox + math.cos(turn) * apart * sgn, oy + math.sin(turn) * apart * sgn), 60) for sgn in (1, -1)]
    pose.cores = cores
    for i, ray in enumerate(scene.rays):
        c = cores[i % 2]
        a = math.radians(ray.angle) + turn
        n = len(ray.chips)
        shown = min(n, 3)
        path = [(c[0] + math.cos(a) * (18 + 24 * k), c[1] + math.sin(a) * (18 + 24 * k)) for k in range(shown)]
        chips = path + [path[-1]] * (n - shown)
        pose.chips[i] = [scene.clamp(p) for p in chips]
        pose.sizes[i] = [0.55] * n
        pose.alphas[i] = [1.0] * shown + [0.0] * (n - shown)
        pose.arms[i] = [c] + path
    return pose


_DANCE_COLOURS = ((0.31, 0.78, 0.50), (0.93, 0.33, 0.30), (0.96, 0.70, 0.24), (0.38, 0.62, 0.98), (0.69, 0.47, 0.96))


def _dance(scene: Scene, u: float, base: Pose) -> Pose:
    """A dance party: the star sways and pulses on the beat, the chips flash, disco lights sweep."""
    pose = _from_base(base)
    ox, oy = scene.origin
    beat = 0.5
    k = int(u / beat)
    pulse = 1.0 + 0.18 * math.exp(-((u % beat) / 0.12))
    sway = math.radians(14 * math.sin(2 * math.pi * u / 2.0))
    c, s_ = math.cos(sway), math.sin(sway)
    for i in range(len(pose.chips)):
        for j, (x, y) in enumerate(pose.chips[i]):
            dx, dy = (x - ox) * pulse, (y - oy) * pulse
            pose.chips[i][j] = scene.clamp((ox + dx * c - dy * s_, oy + dx * s_ + dy * c))
        arm = pose.arms[i]
        pose.arms[i] = [(ox + ((x - ox) * c - (y - oy) * s_) * pulse, oy + ((x - ox) * s_ + (y - oy) * c) * pulse)
                        for x, y in arm]
        pose.tints[i] = (_DANCE_COLOURS[(i + k) % len(_DANCE_COLOURS)], 0.7)
    top = (ox, scene.height + 40)
    for d in range(3):
        a = -math.pi / 2 + 0.6 * math.sin(u * (0.9 + 0.3 * d) + d * 2.1)
        spread = 0.13
        far = scene.height * 1.4
        wedge = [top, (top[0] + math.cos(a - spread) * far, top[1] + math.sin(a - spread) * far),
                 (top[0] + math.cos(a + spread) * far, top[1] + math.sin(a + spread) * far)]
        rgb = _DANCE_COLOURS[(d + k) % len(_DANCE_COLOURS)]
        pose.spots.append((wedge, (rgb[0], rgb[1], rgb[2], 0.13 * ease(u / 1.0) * ease((DURATION["DANCE"] - u) / 1.0))))
    return pose


# --- the whole play ------------------------------------------------------------------------


def blend(a: Pose, b: Pose, w: float, per_ray=None) -> Pose:
    """`a` turning into `b`; `per_ray` (optional) weights each ray on its own."""
    out = _empty_like(b)
    for i in range(len(b.chips)):
        wi = w if per_ray is None else per_ray[i]
        n = len(b.chips[i])
        ac = a.chips[i] if len(a.chips[i]) == n else resample(a.chips[i] or b.chips[i], n)
        asz = a.sizes[i] if len(a.sizes[i]) == n else [1.0] * n
        aal = a.alphas[i] if len(a.alphas[i]) == n else [0.0] * n
        out.chips[i] = [_lerp(p, q, wi) for p, q in zip(ac, b.chips[i])]
        out.sizes[i] = [p + (q - p) * wi for p, q in zip(asz, b.sizes[i])]
        out.alphas[i] = [p + (q - p) * wi for p, q in zip(aal, b.alphas[i])]
        m = max(len(a.arms[i]), len(b.arms[i]), 2)
        out.arms[i] = [_lerp(p, q, wi) for p, q in zip(resample(a.arms[i] or b.arms[i], m),
                                                      resample(b.arms[i] or a.arms[i], m))]
    out.darken = a.darken + (b.darken - a.darken) * w
    out.tentacles = b.tentacles if w >= 0.5 else a.tentacles
    out.eye = b.eye if w >= 0.5 else a.eye
    out.sparks = b.sparks if w >= 0.5 else a.sparks
    out.bulge = b.bulge if w >= 0.5 else a.bulge
    for name in ("texts", "flashes", "spots", "cores"):
        setattr(out, name, getattr(b if w >= 0.5 else a, name))
    out.tints = {i: (rgb, k * w) for i, (rgb, k) in b.tints.items()} if w >= 0.5 else \
        {i: (rgb, k * (1 - w)) for i, (rgb, k) in a.tints.items()}
    keys = set(a.ui) | set(b.ui)
    for key in keys:
        ax, ay, ah = a.ui.get(key, (0.0, 0.0, False))
        bx, by, bh = b.ui.get(key, (0.0, 0.0, False))
        out.ui[key] = (ax + (bx - ax) * w, ay + (by - ay) * w, bh if w >= 0.5 else ah)
    return out


def _empty_like(p: Pose) -> Pose:
    n = len(p.chips)
    return Pose([[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)])


_POSES = {
    "FLOWER": lambda scene, u: _shape(scene, "FLOWER", u),
    "SPIRAL": lambda scene, u: _shape(scene, "SPIRAL", u),
    "HEART": lambda scene, u: _shape(scene, "HEART", u),
    "JELLY": lambda scene, u: _shape(scene, "JELLY", u),
    "SHAKE": _shake,
    "EAT": _eat,
    "FIGHT": _fight,
    "DEEP": _deep,
    "JUGGLE": _juggle,
    "SLEEP": _sleep,
    "CLOCK": _clock,
    "DRUM": _drum,
    "DIP": _dip,
    "RIBBON": _ribbon,
    "ESCAPE": _escape,
    "MITOSIS": _mitosis,
}
_ON_BASE = {"WAVE": _wave, "DANCE": _dance}  # things that move the star as it is


def base_pose(scene: Scene, layout) -> Pose:
    """The star as it is (a star.StarLayout): every chip of every ray, the
    ones the closed star does not show hidden at its last shown chip."""
    pose = _empty(scene)
    by_ray: dict[int, list] = {}
    for chip in layout.chips:
        by_ray.setdefault(chip.ray, []).append(chip)
    for i, ray in enumerate(scene.rays):
        shown = sorted(by_ray.get(i, []), key=lambda c: c.index)
        n = len(ray.chips)
        chips, sizes, alphas = [], [], []
        for k in range(n):
            c = shown[min(k, len(shown) - 1)] if shown else None
            chips.append(c.centre if c is not None else scene.start_point(i))
            sizes.append(1.0)
            alphas.append(min(1.0, c.shown) if c is not None and k < len(shown) else 0.0)
        pose.chips[i], pose.sizes[i], pose.alphas[i] = chips, sizes, alphas
        pose.arms[i] = [tuple(p) for p in layout.arms[i]] if i < len(layout.arms) else []
    return pose


def pose(scene: Scene, base: Pose, t: float) -> Pose | None:
    """The arms `t` seconds into the stillness, or None while the star rests
    (before the first thing and between two). A thing grows out of the star
    (`base`) and goes back to it; an arm it leaves alone stays as the star has it."""
    found = thing_at(t, scene.seed)
    if found is None:
        return None
    name, u = found
    now = _ON_BASE[name](scene, u, base) if name in _ON_BASE else _POSES[name](scene, u)
    for i in range(len(now.chips)):
        if not now.chips[i]:  # not in this thing: the star's own arm
            now.chips[i], now.sizes[i], now.alphas[i], now.arms[i] = (
                base.chips[i], base.sizes[i], base.alphas[i], base.arms[i])
    out = ease((DURATION[name] - u) / BLEND_S)
    return blend(base, now, ease(u / BLEND_S) * out)


def wake_weight(thing: str | None, w: float) -> float:
    """How far the arms are back at the star, `w` of the way through the
    going back. From sleep they overshoot: they wake with a start."""
    w = min(max(w, 0.0), 1.0)
    if thing == "SLEEP":
        c1 = 2.6  # the overshoot of an ease out with a step back
        return 1.0 + (c1 + 1.0) * (w - 1.0) ** 3 + c1 * (w - 1.0) ** 2
    return ease(w)
