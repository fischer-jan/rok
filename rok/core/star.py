"""The star: Rok's quick menu at the cursor (hold Space).

Rays go out from the cursor at uneven angles, one per category, each a row
of chips; a ray is as long as its category. The four tools keep the old
wheel's directions (Attach up, Erase right, Select down, Paint left) and
carry their modes, so a tool and a mode are one move. The other rays hold
Select's extra modes, the safe whole-model tools (nothing that clears or
overwrites the model without a dialog) and Shaders, and
the last command, which runs again. Every arm belongs to one of the four tools and
takes its colour; a tool's own chip is a little bigger and bold.

Rays start at different distances, so neighbours do not collide and the star
stays small. Closed, an arm shows its first two chips; when the pointer comes
to it, it reaches out to its full length and unrolls its tip, and the others
fade. When the star opens the rays unroll from a curl; while it is open they
sway very slowly. A ray that reaches the edge of the view bends
and runs along it; rays on the same edge lie side by side in lanes, like
octopus arms.

Chip ids are the rail's widget ids where the rail has the same button, so a
chip does exactly what that button does; the few chips of the star's own
start with STAR_.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import chrome as kit
from .edit import ATTACH, ERASE, PAINT, SELECT

CHIP = 38  # a chip's side, at scale 1
GAP = 6
STEP = CHIP + GAP  # from one chip centre to the next along a ray
R_MIN = 48  # the nearest a ray may start
DEAD = 26  # a release this near the centre picks nothing
MARGIN = 10  # from the view's edge to the outer lane
SUB = 3.0  # path step, pixels at scale 1
BEND = 34.0  # turning radius where a ray meets an edge, at scale 1
UNFURL_S = 0.32  # seconds the rays take to unroll
SWAY_DEG = 2.2  # the slow sway, in degrees
SWAY_S = 11.0  # its period, seconds
CURVE = 0.0  # per pixel: 0 keeps an arm straight where nothing is in its way
SHOW = 2  # chips an arm shows until the pointer comes to it

ORIGIN = 1.2  # a tool chip, where its arm starts, is this much bigger
SCENE = "SCENE"  # the family of the scene's tools; the others are the four tools
MODEL = "MODEL"  # the family of the whole-model tools
TIP_TURN = math.radians(160.0)  # how far the tip of an arm that is out turns as it rolls up
ROLL_POWER = 2.0  # the roll grows with the square of the way along: the base stays straight, the tip rolls hard
CURL_SLOT_S = 6.0  # an arm that is out may roll all the way once in each slot of this many seconds
CURL_CHANCE = 0.4  # the chance that it does
CURL_UP_S, CURL_HOLD_S, CURL_DOWN_S = 1.3, 1.0, 1.7  # a full roll: slow up, hold, slow down
SWAY_ROLL = 0.16  # the gentle roll left and right the rest of the time
STAR_MODEL = "STAR_MODEL"
STAR_SHADERS = "STAR_SHADERS"
LAST_COMMAND = "LAST_COMMAND"  # the chip that runs the last command again
STAR_COLOUR = "STAR_COLOUR:"  # + a palette index: make that colour current
LAST = "LAST"  # the family of its arm
# Icons for the commands outside the Model menu; a command without one gets a letter.
COMMAND_GLYPHS = {
    "hollow": "HOLLOW", "bridge": "BRIDGE", "smooth": "SMOOTH", "select_colour": "BY_COLOUR",
    "select_connected": "CONNECTED", "select_invert": "INVERSE", "flip": "FLIP", "rotate": "TURN",
    "selection_delete": "DELETE", "cut": "CUT", "scene_flip": "FLIP", "model_add": "CUBE_PLUS",
    "model_add_file": "FROM_FILE", "terrain_new": "TERRAIN", "heightmap_new": "HEIGHTMAP",
    "model_delete": "DELETE", "scale": "SCALE", "mirror_half": "MIRROR_HALF", "lathe": "LATHE",
}

LEFT, RIGHT, BOTTOM, TOP = "LEFT", "RIGHT", "BOTTOM", "TOP"


@dataclass(frozen=True)
class Chip:
    id: str
    label: str
    glyph: str = ""  # an icon name, or "" for a text chip
    text: str = ""  # the letter on a text chip
    rgb: tuple[int, int, int] | None = None  # a colour chip: drawn as a swatch of this colour


@dataclass(frozen=True)
class Ray:
    id: str
    label: str
    angle: float  # degrees, counter-clockwise from the right
    chips: tuple[Chip, ...]
    family: str  # ATTACH, ERASE, PAINT or SELECT (SCENE in the scene): the arm's colour
    first: bool = False  # a tool's own arm: placed first, its first chip the tool (bigger, bold)


@dataclass(frozen=True)
class PlacedChip:
    ray: int
    index: int
    chip: Chip
    centre: tuple[float, float]
    side: float
    shown: float  # 0 .. 1 while the ray unrolls
    origin: bool = False  # the tool chip at the start of its own arm


@dataclass(frozen=True)
class StarLayout:
    origin: tuple[float, float]
    scale: float
    rays: tuple[Ray, ...]
    chips: tuple[PlacedChip, ...]
    arms: tuple[tuple[tuple[float, float], ...], ...]  # one polyline per ray, centre to last chip
    dead: float


# --- what the rays hold ---------------------------------------------------------------


def _tool_ray(ident: str, label: str, angle: float, tool: str, modes) -> Ray:
    widget = f"TOOL_{tool}"
    chips = [Chip(widget, kit.TOOL_RING_TIPS[tool].split(" (")[0].split(" ")[0], kit.RING_TOOL_GLYPHS[tool])]
    chips += [Chip(f"{widget}_{m}", kit.MODE_NAMES[m], kit.RING_GLYPHS.get(m, m)) for m in modes]
    return Ray(ident, label, angle, tuple(chips), tool, first=True)


def last_chip(command: kit.Command) -> Chip:
    """The chip of the last command: its icon, or its first letter."""
    glyph = kit.MODEL_GLYPHS.get(command.id) or COMMAND_GLYPHS.get(command.id, "")
    name = command.name.rstrip(".")
    return Chip(LAST_COMMAND, name, glyph, "" if glyph else name[:1].upper())


def colour_chip(index: int, rgb, name: str = "") -> Chip:
    """A recent colour on the star: a swatch; a pick makes it current."""
    return Chip(f"{STAR_COLOUR}{int(index)}", name or f"Colour {int(index)}", rgb=tuple(int(c) for c in rgb))


def _colour_rays(colours) -> tuple[Ray, ...]:
    """The last colours you drew with, newest first: an arm of
    their own beside Paint, in Paint's colour family; none before the first."""
    chips = tuple(colours)[:3]
    return (Ray("COLOURS", "Colour", 157.0, chips, PAINT),) if chips else ()


def _last_rays(last: "kit.Command | Chip | None") -> tuple[Ray, ...]:
    """The arm of the last command, between Attach and Paint where the
    primitives were; none before the first command. A
    primitive comes as the rail's own chip."""
    if last is None:
        return ()
    chip = last if isinstance(last, Chip) else last_chip(last)
    return (Ray("LAST", "Last", 135.0, (chip,), LAST, first=True),)


def _command_chip(command_id: str) -> Chip:
    names: dict[str, str] = {}
    for c in kit.COMMANDS:
        names.setdefault(c.id, c.name.split(":")[0])
    return Chip(f"MODEL_{command_id.upper()}", names.get(command_id, command_id).rstrip("."), kit.MODEL_GLYPHS[command_id])


BUILD_MODES = ("VOXEL", "FACE", "BOX", "CENTER", "LINE", "PATTERN")
SAFE_COMMANDS = ("model_split", "fit_grid", "drop_to_floor", "diagonal")  # nothing is lost
DIALOG_COMMANDS = ("loop", "repeat", "twist", "resize")  # each asks first in a dialog
MODEL_ARM = 6  # the most Model commands one arm holds, not counting the Model chip


def model_rays(model_items: tuple[str, ...] = kit.DEFAULT_MODEL_ITEMS,
               last: kit.Command | None = None, colours=()) -> tuple[Ray, ...]:
    """The star inside a model. Every arm belongs to a family: one per tool, where Paint keeps the shaders and Select its
    other modes; the Model tools, whose first chip opens the rail's whole
    Model menu; and the last command, which runs again. Each family sits
    around its first arm.

    `model_items` are the commands the rail's Model menu shows (View >
    Customize Rail Menu and Keys); the Model arms show the same ones, but
    only those that lose nothing or ask first."""
    shown = set(model_items)
    safe = tuple(_command_chip(c) for c in SAFE_COMMANDS if c in shown)
    if "scale" in shown:
        safe += (Chip("SCALE_UP", "Scale ×2", "SCALE"),)
    chips = safe + tuple(_command_chip(c) for c in DIALOG_COMMANDS if c in shown)
    # One arm holds them all up to MODEL_ARM; more share two arms.
    first = chips if len(chips) <= MODEL_ARM else chips[:-(-len(chips) // 2)]
    model = (Ray("MODEL_SAFE", "Model", 40.0, (Chip(STAR_MODEL, "Model", "MODEL"),) + first, MODEL, first=True),)
    if len(first) < len(chips):
        model += (Ray("MODEL_ASK", "Model", 62.0, chips[len(first):], MODEL),)
    return (
        _tool_ray("ERASE", "Erase", 0.0, ERASE, BUILD_MODES + ("RECT",)),
        *model,
        _tool_ray("ATTACH", "Attach", 90.0, ATTACH, BUILD_MODES + ("CLONE",)),
        *_last_rays(last),
        *_colour_rays(colours),
        _tool_ray("PAINT", "Paint", 180.0, PAINT, BUILD_MODES + ("RECT", "GRADIENT")),
        Ray("SHADERS", "Paint", 207.0, (Chip(STAR_SHADERS, "Shaders", "SHADERS"),), PAINT),
        Ray("SELECT_BY", "Select", 243.0,
            tuple(Chip(f"TOOL_SELECT_{m}", kit.MODE_NAMES[m], m) for m in kit.SELECT_ONLY_MODES), SELECT),
        _tool_ray("SELECT", "Select", 270.0, SELECT, ("VOXEL", "FACE", "RECT", "LASSO", "BOX")),
    )


def scene_rays(last: kit.Command | None = None) -> tuple[Ray, ...]:
    """The star in the scene: its tools (no Delete), and the last command."""
    return (
        Ray("SCENE_TOOLS", "Scene", 0.0, (
            Chip(kit.SCENE_PICK_ID, "Pick", "PICK"),
            Chip(kit.SCENE_CUT_ID, "Cut", "CUT"),
            Chip(kit.SCENE_DROP_ID, "Drop to Floor", "DROP_FLOOR"),
            Chip(kit.SCENE_DROP_ONTO_ID, "Drop", "DROP_ONTO"),
            Chip(kit.SCENE_FIT_ID, "Fit", "FIT_GRID"),
        ), SCENE),
        *_last_rays(last),
    )


def rays_for(level: str, animating: bool, model_items: tuple[str, ...] = kit.DEFAULT_MODEL_ITEMS,
             last: kit.Command | None = None, colours=()) -> tuple[Ray, ...]:
    """The star's arms for where you are; none in Animate."""
    if animating:
        return ()
    return scene_rays(last) if level == kit.LEVEL_SCENE else model_rays(model_items, last, colours)


# --- where the chips go ------------------------------------------------------------------


def _free_path(angle: float, r0: float, count: int, s: float) -> list[tuple[float, float]]:
    """The chip centres of a ray in open space, relative to the centre."""
    heading = math.radians(angle)
    x, y = math.cos(heading) * r0, math.sin(heading) * r0
    out = [(x, y)]
    sub = SUB * s
    walked = 0.0
    while len(out) < count:
        heading += CURVE / s * sub
        x += math.cos(heading) * sub
        y += math.sin(heading) * sub
        walked += sub
        if walked >= len(out) * STEP * s - 1e-6:
            out.append((x, y))
    return out


def starts(rays, scale: float) -> tuple[float, ...]:
    """How far from the centre each ray starts. The tool rays start nearest;
    every other ray moves out until none of its chips touches a chip placed
    before it, so the star is as small as the angles allow."""
    s = float(scale)
    placed: list[tuple[float, float, float]] = []  # x, y, half the chip's side
    out = [0.0] * len(rays)
    # The tool rays first, then the longer rays: the short ones fill the gaps.
    order = sorted(range(len(rays)), key=lambda i: (not rays[i].first, -len(rays[i].chips), i))
    for i in order:
        ray = rays[i]
        halves = [CHIP * s * (ORIGIN if ray.first and k == 0 else 1.0) / 2 for k in range(len(ray.chips))]
        r = R_MIN * s * (ORIGIN if ray.first else 1.0)
        while True:
            pts = _free_path(ray.angle, r, len(ray.chips), s)
            if all(math.hypot(px - qx, py - qy) >= ha + hb + GAP * 0.5 * s
                   for (px, py), ha in zip(pts, halves) for qx, qy, hb in placed):
                break
            r += 2.0 * s
        out[i] = r
        placed.extend((px, py, ha) for (px, py), ha in zip(pts, halves))
    return tuple(out)


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 1.0 - (1.0 - u) ** 3


def sway(ray_index: int, seconds: float) -> float:
    """The slow sway of one ray, in degrees."""
    return SWAY_DEG * math.sin(2 * math.pi * seconds / SWAY_S) + 0.6 * math.sin(2 * math.pi * seconds / 7.0 + ray_index * 1.7)


def curl_amount(seconds: float, key: int) -> float:
    """How far an arm that has been out `seconds` rolls up, -1 (left) to 1
    (right). All the time it rolls a little, left and right, slowly; now and
    then (a chance in every slot, at a random moment) it rolls all the way to
    a random side, slowly, holds, and lets go again. `key`
    makes each arm's rolls its own."""
    import random

    phase = random.Random(key).random() * 2 * math.pi
    gentle = SWAY_ROLL * (0.7 * math.sin(2 * math.pi * seconds / 3.7 + phase)
                          + 0.3 * math.sin(2 * math.pi * seconds / 2.3 + 2.0 * phase))
    whole = CURL_UP_S + CURL_HOLD_S + CURL_DOWN_S
    k = math.floor(seconds / CURL_SLOT_S)
    for slot in (k - 1, k):  # a roll late in the slot before can still run
        rng = random.Random(key * 1_000_003 + slot)
        chance, when, side = rng.random(), rng.random(), rng.random()
        if slot < 0 or chance >= CURL_CHANCE:
            continue
        t = seconds - (slot * CURL_SLOT_S + when * max(CURL_SLOT_S - whole, 0.1))
        if 0.0 <= t <= whole:
            if t < CURL_UP_S:
                env = t / CURL_UP_S
            elif t < CURL_UP_S + CURL_HOLD_S:
                env = 1.0
            else:
                env = 1.0 - (t - CURL_UP_S - CURL_HOLD_S) / CURL_DOWN_S
            env = env * env * (3.0 - 2.0 * env)  # smooth at both ends
            target = 1.0 if side < 0.5 else -1.0
            return gentle * (1.0 - env) + target * env
    return gentle


def closed(rays) -> tuple[Ray, ...]:
    """The rays as the closed star shows them: their first SHOW chips."""
    return tuple(Ray(r.id, r.label, r.angle, r.chips[:SHOW], r.family, r.first) for r in rays)


def star_layout(rays, origin, region_width: float, region_height: float, scale: float,
                opened: float = 1.0, seconds: float = 0.0, ray_starts=None, extents=None,
                left: float = 0.0, wiggles=None) -> StarLayout:
    """The chips for a star at `origin`. `opened` is the seconds since the star
    opened (the unroll); `seconds` drives the sway (0: none). `extents` says
    for each ray how far it has reached out, 0 (closed: its first SHOW chips)
    to 1 (all of them); the part reaching out unrolls from a curl. `left` is
    where the free view starts: the arms treat the rail's edge as the view's.
    `wiggles` says for each ray how far it rolls up while it is out, -1 to 1
    (0: straight; curl_amount); a closed arm is always straight."""
    s = float(scale)
    rays = tuple(rays)
    ray_starts = ray_starts if ray_starts is not None else starts(closed(rays), s)
    extents = tuple(extents) if extents is not None else (0.0,) * len(rays)
    wiggles = tuple(wiggles) if wiggles is not None else (0.0,) * len(rays)
    ox, oy = float(origin[0]), float(origin[1])
    lanes: dict[str, int] = {}
    chips: list[PlacedChip] = []
    arms = []
    half = CHIP * ORIGIN * 0.5 * s  # room for the biggest chip at the view's edge
    step = STEP * s
    lo_x = float(left) + half
    for i, ray in enumerate(rays):
        # Rays unroll one after another, a little apart.
        u = _ease((opened - 0.018 * i) / UNFURL_S)
        extent = min(max(float(extents[i]), 0.0), 1.0)
        count = len(ray.chips)
        showing = min(SHOW, count) + extent * max(0, count - SHOW)  # chips out, fractional
        showing *= u
        heading = math.radians(ray.angle + (sway(i, seconds) if seconds else 0.0))
        r0 = ray_starts[i]
        x = min(max(ox + math.cos(heading) * r0, lo_x), region_width - half)
        y = min(max(oy + math.sin(heading) * r0, half), region_height - half)
        side = 1.0 if i % 2 else -1.0
        tip_curl = side * (1.0 - min(u, 0.35 + 0.65 * extent)) ** 2 / (14.0 * s)  # what unrolls
        # An arm that is out may roll up to a side like a tentacle: straight
        # at the base, hard at the tip (how far: wiggles, see curl_amount).
        arm_len = max((count - 1) * step, step)
        roll = float(wiggles[i]) * extent * TIP_TURN * (ROLL_POWER + 1.0) / arm_len
        # Never tighter at the tip than neighbouring chips allow (short arms turn a little less).
        tightest = 1.0 / (27.0 * s)
        roll = max(-tightest, min(tightest, roll))
        mine: dict[str, float] = {}  # this ray's lane inset on each edge it meets
        arm = [(ox + math.cos(heading) * DEAD * s, oy + math.sin(heading) * DEAD * s), (x, y)]
        walked = 0.0
        k = 0
        sub = SUB * s
        last = max(0.0, (math.ceil(showing) - 1)) * step
        while True:
            if walked >= k * step - 1e-6 and k < count:
                shown = min(1.0, max(0.0, showing - k))
                grow = 0.55 + 0.45 * shown
                origin = ray.first and k == 0
                chips.append(PlacedChip(i, k, ray.chips[k], (x, y), CHIP * s * grow * (ORIGIN if origin else 1.0),
                                        shown, origin))
                k += 1
            if walked >= last or k >= count:
                break
            heading = _steer(x - left, y, heading, sub, s, region_width - left, region_height, lanes, mine)
            curl = tip_curl if walked >= (SHOW - 1) * step else side * (1.0 - u) ** 2 / (14.0 * s)
            heading += (curl + CURVE / s + roll * (walked / arm_len) ** ROLL_POWER) * sub
            x = min(max(x + math.cos(heading) * sub, lo_x), region_width - half)
            y = min(max(y + math.sin(heading) * sub, half), region_height - half)
            walked += sub
            arm.append((x, y))
        arms.append(tuple(arm))
    return StarLayout((ox, oy), s, rays, tuple(chips), tuple(arms), DEAD * s)


def arm_under(layout: StarLayout, x: float, y: float, current: int | None) -> int | None:
    """Which arm reaches out for the pointer: the arm of the shown chip nearest
    to it (within a chip and a half); near nothing, the arm already out stays
    out; back in the centre, none."""
    ox, oy = layout.origin
    if math.hypot(x - ox, y - oy) < layout.dead:
        return None
    best, best_d = None, None
    for chip in layout.chips:
        if chip.shown < 0.5:
            continue
        cx, cy = chip.centre
        d = math.hypot(x - cx, y - cy)
        if current is not None and chip.ray == current:
            d *= 0.8  # the arm that is out keeps the pointer a little longer
        if d <= STEP * layout.scale * 1.5 and (best_d is None or d < best_d):
            best, best_d = chip.ray, d
    return best if best is not None else current


def _steer(x, y, heading, sub, s, w, h, lanes, mine) -> float:
    """Turn the heading away from an edge it is about to cross, onto the edge's
    direction nearest to where it points. Each ray gets its own lane on an
    edge, the first one out at MARGIN, the next one a chip further in."""
    dx, dy = math.cos(heading), math.sin(heading)
    bend = BEND * s
    for edge, dist, toward, tangent in (
        (LEFT, x, -dx, (0.0, 1.0)),
        (RIGHT, w - x, dx, (0.0, 1.0)),
        (BOTTOM, y, -dy, (1.0, 0.0)),
        (TOP, h - y, dy, (1.0, 0.0)),
    ):
        base = (MARGIN + CHIP * 0.5) * s
        if edge not in mine:
            if toward <= 0.05 or dist > base + bend + STEP * s * lanes.get(edge, 0):
                continue
            mine[edge] = base + STEP * s * lanes.get(edge, 0)
            lanes[edge] = lanes.get(edge, 0) + 1
        inset = mine[edge]
        if toward > 0.0 and dist < inset + bend:
            tx, ty = tangent
            if tx * dx + ty * dy < 0:
                tx, ty = -tx, -ty
            target = math.atan2(ty, tx)
            diff = (target - heading + math.pi) % (2 * math.pi) - math.pi
            # A steady turn near the edge; a sharp one once the ray is in its lane.
            rate = sub / (4.0 * s) if dist <= inset else sub / bend
            heading += max(-rate, min(rate, diff))
            dx, dy = math.cos(heading), math.sin(heading)
    return heading


def chip_at(layout: StarLayout, x: float, y: float, open_ray: int | None = None) -> PlacedChip | None:
    """The shown chip under a point: the nearest one within reach, the open
    arm's first; none in the centre."""
    ox, oy = layout.origin
    if math.hypot(x - ox, y - oy) < layout.dead:
        return None
    best, best_d = None, None
    for chip in layout.chips:
        if chip.shown < 0.99:
            continue
        cx, cy = chip.centre
        d = math.hypot(x - cx, y - cy)
        if chip.ray == open_ray:
            d *= 0.7  # the arm that is out wins where arms cross
        if d <= chip.side * 0.62 and (best_d is None or d < best_d):
            best, best_d = chip, d
    return best


def centre_label(layout: StarLayout, chip: PlacedChip | None, fallback: str = "") -> str:
    """What the centre says: the ray and the chip under the mouse."""
    if chip is None:
        return fallback
    ray = layout.rays[chip.ray]
    if chip.index == 0 and ray.first and ray.id in ("ATTACH", "ERASE", "PAINT", "SELECT"):
        return chip.chip.label
    return chip.chip.label if chip.chip.label.startswith(ray.label) else f"{ray.label} · {chip.chip.label}"
