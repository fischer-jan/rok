"""Animation: clips of keys, and the pose they give at one point in time.

A Clip has a length, a frame rate and a loop mode, and tracks. A Track
animates one target (a part, that is a model of the scene; or a colour)
through named channels, and each channel (a Property) holds keys: a time in
whole frames, a value, and an ease that says how the value goes to the next
key. HOLD keeps the value until the next key, as .vox frames
play; LINEAR, SMOOTH, EASE_IN and EASE_OUT blend.

A part's channels, all relative to the rest pose (the model as you build it):
  MOVE_X, MOVE_Y, MOVE_Z  cells from the rest place
  TURN_X, TURN_Y, TURN_Z  degrees about the part's pivot; 720 turns twice
  SCALE                   a triple of percent; (100, 100, 100) is the rest
  VISIBLE                 a bool; HOLD only
  STATE                   which voxel frame shows: 0 the model's own voxels,
                          1.. its extra ones (SceneModel.states); HOLD only
The turn is built X, then Y, then Z (Rz @ Ry @ Rx), so a turn about Z, the
up axis, is always one channel. Each axis is its own channel: the gizmo ring
you drag is the channel you key.

A colour's channels: COLOUR (an (r, g, b) 0..255), SLOT (an offset into
Target.slots; HOLD only) and GLOW (a percent of the HSV value).

`evaluate` turns a clip and a scene, at one point in time, into a Frame: the
pose the viewport, the exports and the voxel look read. numpy and stdlib
only.
"""

from __future__ import annotations

import base64
import colorsys
from dataclasses import dataclass, field

import numpy as np

from . import grid_ops
from .edit import Cell

HOLD, LINEAR, SMOOTH, EASE_IN, EASE_OUT = "HOLD", "LINEAR", "SMOOTH", "EASE_IN", "EASE_OUT"
EASES = (HOLD, LINEAR, SMOOTH, EASE_IN, EASE_OUT)
EASE_LABELS = {HOLD: "Hold", LINEAR: "Linear", SMOOTH: "Smooth", EASE_IN: "Ease in", EASE_OUT: "Ease out"}
PART, MASK, COLOUR = "PART", "MASK", "COLOUR"
MOVE_X, MOVE_Y, MOVE_Z = "MOVE_X", "MOVE_Y", "MOVE_Z"
TURN_X, TURN_Y, TURN_Z = "TURN_X", "TURN_Y", "TURN_Z"
MOVE = (MOVE_X, MOVE_Y, MOVE_Z)
TURN = (TURN_X, TURN_Y, TURN_Z)
SCALE, VISIBLE, STATE = "SCALE", "VISIBLE", "STATE"
COLOUR_RGB, SLOT, GLOW = "COLOUR", "SLOT", "GLOW"
# The groups a .vox frame holds together.
POSITION, ROTATION = "POSITION", "ROTATION"
GROUPS = {POSITION: MOVE, ROTATION: TURN}

PROPERTIES_OF: dict[str, tuple[str, ...]] = {
    PART: MOVE + TURN + (SCALE, VISIBLE, STATE),
    COLOUR: (COLOUR_RGB, SLOT, GLOW),
}
CHANNEL_LABELS = {
    MOVE_X: "Move X", MOVE_Y: "Move Y", MOVE_Z: "Move Z",
    TURN_X: "Turn X", TURN_Y: "Turn Y", TURN_Z: "Turn Z",
    SCALE: "Size", VISIBLE: "Show", STATE: "Frame",
    COLOUR_RGB: "Colour", SLOT: "Slot", GLOW: "Glow",
}
HOLD_ONLY = (VISIBLE, STATE, SLOT)

LOOP, ONCE, PING_PONG = "LOOP", "ONCE", "PING_PONG"
LOOP_MODES = (LOOP, ONCE, PING_PONG)

IDENTITY_ROTATION = 4  # the .vox rotation byte that turns nothing
VOX_FPS = 10  # a .vox file stores no frame rate; Rok plays its frames at this
DEFAULT_FPS = 24  # a new clip


@dataclass(frozen=True)
class Key:
    """One key: from `time` (a frame) the channel has `value`."""

    time: int
    value: object  # a number, a tuple (SCALE, COLOUR), or an int/bool (STATE, VISIBLE, SLOT)
    ease: str = HOLD


@dataclass
class Property:
    name: str
    keys: list[Key] = field(default_factory=list)  # sorted by time, one per frame


@dataclass(frozen=True)
class Target:
    kind: str
    uid: int = 0  # PART and MASK: the model; COLOUR: 0
    mask: bytes = b""  # MASK: np.packbits of a bool array of the grid shape
    slots: tuple[int, ...] = ()  # COLOUR: palette indices, in order


@dataclass
class Track:
    target: Target
    properties: dict[str, Property] = field(default_factory=dict)
    name: str = ""
    axis: int = 2  # the motions' axis
    settings: dict = field(default_factory=dict)  # a motion's fixed parameters (Shake's seed)

    def keys(self, name: str) -> list[Key]:
        prop = self.properties.get(name)
        return prop.keys if prop is not None else []

    def key_times(self) -> list[int]:
        """Every frame that holds a key on any channel, sorted."""
        return sorted({k.time for prop in self.properties.values() for k in prop.keys})


@dataclass(frozen=True)
class Marker:
    time: int
    name: str


@dataclass
class Clip:
    """One named animation. `length` is in frames and is set, not derived:
    keys after it show grey and do not play."""

    name: str = "Clip 1"
    fps: int = DEFAULT_FPS
    length: int = DEFAULT_FPS  # 1.0 s
    loop_mode: str = LOOP
    voxel_look: bool = True  # On by default
    root_motion: bool = False
    tracks: list[Track] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)

    def track_of(self, uid: int) -> Track | None:
        """The part track of model `uid`, or None."""
        for track in self.tracks:
            if track.target.kind == PART and track.target.uid == uid:
                return track
        return None

    def has_keys(self) -> bool:
        return any(prop.keys for track in self.tracks for prop in track.properties.values())

    @property
    def seconds(self) -> float:
        return self.length / max(1, self.fps)

    def time_at(self, t: float) -> float:
        """Clip time `t` (frames, any number) mapped into the clip by its loop mode."""
        n = max(1, int(self.length))
        if self.loop_mode == LOOP:
            return t % n
        if self.loop_mode == PING_PONG:
            if n == 1:
                return 0.0
            period = 2 * (n - 1)
            u = t % period
            return u if u <= n - 1 else period - u
        return min(max(t, 0.0), float(n - 1))


def set_key(track: Track, name: str, key: Key) -> None:
    """Put `key` on the channel, replacing a key at the same time."""
    prop = track.properties.setdefault(name, Property(name))
    prop.keys = sorted([k for k in prop.keys if k.time != key.time] + [key], key=lambda k: k.time)


def remove_keys(track: Track, name: str, times) -> int:
    """Remove the channel's keys at `times`. Returns how many went; an empty
    channel goes too."""
    prop = track.properties.get(name)
    if prop is None:
        return 0
    times = set(times)
    before = len(prop.keys)
    prop.keys = [k for k in prop.keys if k.time not in times]
    if not prop.keys:
        del track.properties[name]
    return before - len(prop.keys)


def move_keys(track: Track, name: str, times, offset: int) -> None:
    """Shift the channel's keys whose time is in `times` by `offset` frames.
    A moved key replaces a key it lands on."""
    prop = track.properties.get(name)
    if prop is None:
        return
    times = set(times)
    moved = [Key(k.time + offset, k.value, k.ease) for k in prop.keys if k.time in times]
    landing = {k.time for k in moved}
    kept = [k for k in prop.keys if k.time not in times and k.time not in landing]
    prop.keys = sorted(kept + moved, key=lambda k: k.time)


def set_ease(track: Track, name: str, times, ease: str) -> None:
    """Set the ease of the channel's keys whose time is in `times`. A Hold-only
    channel stays Hold."""
    prop = track.properties.get(name)
    if prop is None or name in HOLD_ONLY:
        return
    times = set(times)
    prop.keys = [Key(k.time, k.value, ease) if k.time in times else k for k in prop.keys]


def _ease_for_new_key(track: Track, name: str, frame: int, default: str) -> str:
    """The ease a key written at `frame` takes: the ease of the key it
    updates, else of the key before it (a key dropped into a Linear Spin
    keeps the spin linear), else `default`."""
    if name in HOLD_ONLY:
        return HOLD
    keys = track.keys(name)
    for key in keys:
        if key.time == frame:
            return key.ease
    before = [key for key in keys if key.time < frame]
    if before:
        return before[-1].ease
    return default


def key_change(track: Track, name: str, frame: int, value, rest, default_ease: str = SMOOTH) -> None:
    """Write a key for a change at `frame` (a gizmo release, K). The first
    key on a channel at a frame other than 0 also writes `rest` at frame 0,
    so the rest pose is never lost by accident (section 6.1)."""
    if not track.keys(name) and frame != 0:
        set_key(track, name, Key(0, rest, HOLD if name in HOLD_ONLY else default_ease))
    set_key(track, name, Key(frame, value, _ease_for_new_key(track, name, frame, default_ease)))


def _eased(f: float, ease: str) -> float:
    if ease == SMOOTH:
        return f * f * (3.0 - 2.0 * f)
    if ease == EASE_IN:
        return f * f
    if ease == EASE_OUT:
        return 1.0 - (1.0 - f) * (1.0 - f)
    return f


def value_at(prop: Property, t: float):
    """The channel's value at time t (frames). Before the first key it is the
    first key's value; a HOLD key keeps its value until the next key; the
    other eases blend numbers (a tuple or a scalar) toward the next key."""
    keys = prop.keys
    if not keys:
        return None
    if t <= keys[0].time:
        return keys[0].value
    for a, b in zip(keys, keys[1:]):
        if a.time <= t < b.time:
            if a.ease == HOLD:
                return a.value
            f = _eased((t - a.time) / (b.time - a.time), a.ease)
            if isinstance(a.value, tuple):
                return tuple(x + (y - x) * f for x, y in zip(a.value, b.value))
            return a.value + (b.value - a.value) * f
    return keys[-1].value


REST = {SCALE: (100, 100, 100), VISIBLE: True, STATE: 0, GLOW: 100}


def rest_of(name: str):
    """A part channel's value at rest."""
    return REST.get(name, 0)


def channel(track: Track | None, name: str, t: float):
    """The channel's value at t, or its rest value without keys."""
    prop = None if track is None else track.properties.get(name)
    if prop is None or not prop.keys:
        return rest_of(name)
    return value_at(prop, t)


def triple(track: Track | None, names, t: float) -> tuple[float, float, float]:
    return tuple(float(channel(track, n, t)) for n in names)


# --- rotation: a quarter-turn byte as a triple of degrees --------------------


def _euler_matrix(rx, ry, rz) -> np.ndarray:
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_from_euler(degrees) -> np.ndarray:
    """The rotation matrix for X, then Y, then Z degrees: Rz @ Ry @ Rx.
    The inverse of euler_from_matrix for a triple of multiples of 90."""
    rx, ry, rz = degrees
    return _euler_matrix(rx, ry, rz)


def euler_from_matrix(R: np.ndarray) -> tuple[int, int, int]:
    """One of the 24 rotation matrices (det +1) as a triple of degrees, each a
    multiple of 90 in 0..270. Tries the triples in order (rx, then ry, then
    rz), so the first match, the one with the most zero components, wins and
    the mapping is one function of R. ValueError on a reflection (det -1)."""
    R = np.rint(np.asarray(R, dtype=np.float64)).astype(np.int64)  # a float matrix may carry trig noise
    if int(round(np.linalg.det(R))) != 1:
        raise ValueError(f"R is a reflection, not a rotation: {R!r}")
    for rx in range(0, 360, 90):
        for ry in range(0, 360, 90):
            for rz in range(0, 360, 90):
                M = np.rint(_euler_matrix(rx, ry, rz)).astype(np.int64)
                if np.array_equal(M, R):
                    return (rx, ry, rz)
    raise ValueError(f"no whole-quarter Euler matches {R!r}")  # unreachable for a signed permutation


# --- evaluate: the scene's pose at one point in time -------------------------


@dataclass(frozen=True)
class Pose:
    """One part's pose: its turn and size (`linear`) about `pivot`, then a
    move by `offset`, all in grid units relative to the rest place."""

    linear: np.ndarray  # 3 x 3
    pivot: np.ndarray  # grid units, cell (i, j, k) filling [i, i + 1)
    offset: np.ndarray  # cells

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.linear
        m[:3, 3] = self.pivot - self.linear @ self.pivot + self.offset
        return m

    def is_rest(self) -> bool:
        return bool(np.allclose(self.linear, np.eye(3)) and not np.any(self.offset))

    @classmethod
    def from_matrix(cls, m: np.ndarray, pivot) -> Pose:
        """The pose whose matrix() is `m`, about `pivot`."""
        pivot = np.asarray(pivot, dtype=np.float64)
        linear = np.array(m[:3, :3], dtype=np.float64)
        offset = np.array(m[:3, 3], dtype=np.float64) - (pivot - linear @ pivot)
        return cls(linear, pivot, offset)


def _translation(v) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = v
    return m


def with_parents(poses: dict, scene, clip: Clip) -> dict:
    """The poses with every parent's pose applied to its children (6.3: a
    child's pose is its parent's pose, then its own). A part without keys
    under a posed parent gets a pose too. A parent chain that loops stops
    where it would repeat."""
    models = {model.uid: model for model in scene.models}
    if not any(m.parent is not None and m.parent in models and m.parent != m.uid for m in scene.models):
        return poses
    world: dict[int, np.ndarray] = {}

    def affine(uid: int, seen: tuple) -> np.ndarray:
        """World -> world: what the part's chain does to a world point."""
        found = world.get(uid)
        if found is not None:
            return found
        model = models[uid]
        pos = np.asarray(model.position, dtype=np.float64)
        own = poses[uid].matrix() if uid in poses else np.eye(4)
        a = _translation(pos) @ own @ _translation(-pos)
        parent = model.parent
        if parent is not None and parent in models and parent != uid and parent not in seen:
            a = affine(parent, seen + (uid,)) @ a
        world[uid] = a
        return a

    out: dict = {}
    for uid, model in models.items():
        a = affine(uid, (uid,))
        if np.allclose(a, np.eye(4)):
            continue
        pos = np.asarray(model.position, dtype=np.float64)
        total = _translation(-pos) @ a @ _translation(pos)
        out[uid] = Pose.from_matrix(total, pivot_of(clip.track_of(uid), model))
    return out


@dataclass(frozen=True)
class Frame:
    """The scene's pose at one point in time, as `evaluate` builds it. A part
    with no keys has no entry: it stands at rest."""

    poses: dict[int, Pose]  # uid -> pose
    visible: dict[int, bool] = field(default_factory=dict)
    state: dict[int, int] = field(default_factory=dict)  # uid -> which voxel frame shows
    palette: np.ndarray | None = None  # (256, 4) uint8; None = the rest palette
    emission: np.ndarray | None = None  # (256,) float32 scale of Light power; None = 1

    @property
    def transforms(self) -> dict[int, np.ndarray]:
        return {uid: pose.matrix() for uid, pose in self.poses.items()}


def pivot_of(track: Track | None, model) -> np.ndarray:
    """The point a part turns and sizes about, in grid units: the part's own
    pivot (SceneModel.pivot2, doubled), or the middle of its box."""
    cell = getattr(model, "pivot2", None)
    if cell is None:
        cell = tuple(model.grid.size)
    return np.array(cell, dtype=np.float64) / 2.0


def pose_of(track: Track | None, model, t: float) -> Pose:
    """A part's pose at clip time t."""
    degrees = triple(track, TURN, t)
    percent = channel(track, SCALE, t)
    linear = matrix_from_euler(degrees) @ np.diag([p / 100.0 for p in percent])
    offset = np.array(triple(track, MOVE, t), dtype=np.float64)
    return Pose(linear, pivot_of(track, model), offset)


def _scale_hsv_value(rgb, factor: float) -> np.ndarray:
    """`rgb` (3 uint8) with its HSV value scaled by `factor` and clipped to
    255; hue and saturation stand."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
    r, g, b = colorsys.hsv_to_rgb(h, s, min(v * factor, 1.0))
    return np.array([round(r * 255), round(g * 255), round(b * 255)], dtype=np.uint8)


def evaluate(clip: Clip, scene, t: float) -> Frame:
    """The scene's pose at frame `t`, mapped into the clip by its loop mode:
    one pose per part with keys, which frame and whether each such part
    shows, and the palette a colour track changes. A track whose model is
    gone is skipped."""
    t = clip.time_at(t)
    uids = {model.uid for model in scene.models}
    poses: dict[int, Pose] = {}
    visible: dict[int, bool] = {}
    state: dict[int, int] = {}
    palette = None
    emission = None
    for track in clip.tracks:
        target = track.target
        if target.kind == PART:
            if target.uid not in uids:
                continue
            model = scene.by_uid(target.uid)
            if any(track.keys(n) for n in MOVE + TURN + (SCALE,)):
                poses[target.uid] = pose_of(track, model, t)
            if track.keys(VISIBLE):
                visible[target.uid] = bool(value_at(track.properties[VISIBLE], t))
            if track.keys(STATE):
                state[target.uid] = int(value_at(track.properties[STATE], t))
        elif target.kind == COLOUR:
            colour_prop = track.properties.get(COLOUR_RGB)
            slot_prop = track.properties.get(SLOT)
            glow = track.properties.get(GLOW)
            has_colour = colour_prop is not None and colour_prop.keys
            has_slot = slot_prop is not None and slot_prop.keys and target.slots
            has_glow = glow is not None and glow.keys
            if (has_colour or has_slot or has_glow) and palette is None:
                palette = scene.palette.copy()
            if has_colour:
                r, g, b = value_at(colour_prop, t)
                for slot in target.slots:
                    palette[slot, :3] = [int(round(r)), int(round(g)), int(round(b))]
            if has_slot:
                offset = int(round(value_at(slot_prop, t)))
                rest = np.array([palette[s, :3] for s in target.slots])
                rotated = np.roll(rest, offset, axis=0)
                for i, slot in enumerate(target.slots):
                    palette[slot, :3] = rotated[i]
            if has_glow:
                factor = value_at(glow, t) / 100.0
                for slot in target.slots:
                    palette[slot, :3] = _scale_hsv_value(palette[slot, :3], factor)
                    if scene.materials.get(slot).type == "EMIT":
                        if emission is None:
                            emission = np.ones(256, dtype=np.float32)
                        emission[slot] = factor
    poses = with_parents(poses, scene, clip)
    return Frame(poses=poses, visible=visible, state=state, palette=palette, emission=emission)


# --- motions: ready-made keys -------------------------------------------------

PART_PRESETS = ("SPIN", "SWING", "BOB", "SHAKE")
COLOUR_PRESETS = ("GLOW", "BLINK", "FLOW")
PRESETS = PART_PRESETS + COLOUR_PRESETS
PRESET_LABELS = {"SPIN": "Spin", "SWING": "Swing", "BOB": "Bob", "SHAKE": "Shake",
                 "GLOW": "Glow", "BLINK": "Blink", "FLOW": "Flow"}


def default_anchor2(part, others) -> tuple[Cell, int]:
    """The default pivot for `part` (a SceneModel): the face of its box that
    has a filled voxel of another model of `others` right outside it is the
    touching face; the pivot is that face's centre, snapped to half cells
    (doubled world cells), and the axis is the face's normal. With no
    touching face: the box centre and axis 2."""
    lo, hi = part.box_world()
    for axis in range(3):
        for sign in (-1, 1):
            if _face_touches(lo, hi, axis, sign, others):
                centre2 = [lo[i] + hi[i] + 1 for i in range(3)]
                centre2[axis] = 2 * (lo[axis] if sign < 0 else hi[axis] + 1)
                return tuple(centre2), axis
    return tuple(lo[i] + hi[i] + 1 for i in range(3)), 2


def _face_touches(lo: Cell, hi: Cell, axis: int, sign: int, others) -> bool:
    """Whether a filled voxel of any model of `others` sits right outside the
    box `lo`..`hi` (inclusive, world cells) on the face named by `axis` and
    `sign` (-1 the box's low side, +1 its high side)."""
    coord = lo[axis] - 1 if sign < 0 else hi[axis] + 1
    region_lo, region_hi = list(lo), list(hi)
    region_lo[axis] = region_hi[axis] = coord
    for other in others:
        other_lo, other_hi = other.box_world()
        cross_lo = tuple(max(region_lo[i], other_lo[i]) for i in range(3))
        cross_hi = tuple(min(region_hi[i], other_hi[i]) for i in range(3))
        if any(cross_lo[i] > cross_hi[i] for i in range(3)):
            continue
        local = tuple(
            slice(cross_lo[i] - other_lo[i], cross_hi[i] - other_lo[i] + 1) for i in range(3)
        )
        if np.any(other.grid.voxels[local] != 0):
            return True
    return False


_SIX_NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def neighbour_colour(scene, slots) -> tuple[int, int, int]:
    """The most common palette colour among the six-neighbours of every
    voxel that uses one of `slots`, over every model of `scene`, not
    counting the slots themselves (the skin around the eyes for Blink).
    Black `(0, 0, 0)` when there is none."""
    slot_set = frozenset(slots)
    counts = np.zeros(256, dtype=np.int64)
    for model in scene.models:
        voxels = model.grid.voxels
        is_slot = np.isin(voxels, list(slot_set))
        if not is_slot.any():
            continue
        sx, sy, sz = voxels.shape
        for dx, dy, dz in _SIX_NEIGHBOURS:
            src = (
                slice(max(0, -dx), sx - max(0, dx)),
                slice(max(0, -dy), sy - max(0, dy)),
                slice(max(0, -dz), sz - max(0, dz)),
            )
            dst = (
                slice(max(0, dx), sx + min(0, dx)),
                slice(max(0, dy), sy + min(0, dy)),
                slice(max(0, dz), sz + min(0, dz)),
            )
            neighbours = voxels[dst][is_slot[src]]
            counts += np.bincount(neighbours, minlength=256)
    for slot in slot_set:
        counts[slot] = 0
    counts[0] = 0
    if not counts.any():
        return (0, 0, 0)
    winner = int(np.argmax(counts))
    return tuple(int(c) for c in scene.palette[winner, :3])


def preset_keys(name: str, track: Track, clip: Clip, scene=None) -> tuple[tuple[str, tuple[Key, ...]], ...]:
    """The (channel, keys) pairs of motion `name` for `track`, spread over the
    clip's length: a Spin turns once from the first frame to the last, a
    Swing goes out to both sides and back, a Bob goes up and down, a Shake
    jitters. The part motions work on `track.axis`. The caller writes the
    keys with `set_key`. `scene`, when given, supplies a Blink's colour."""
    n = max(1, int(clip.length))
    fps = clip.fps

    def at(fraction: float) -> int:
        return int(round(fraction * n))

    if name == "SPIN":
        turn = TURN[track.axis]
        return ((turn, (Key(0, 0.0, LINEAR), Key(n, 360.0, LINEAR))),)
    if name == "SWING":
        turn = TURN[track.axis]
        values = (0.0, 30.0, 0.0, -30.0, 0.0)
        return ((turn, tuple(Key(at(k / 4), v, SMOOTH) for k, v in enumerate(values))),)
    if name == "BOB":
        move = MOVE[track.axis]
        values = (0.0, 2.0, 0.0, -2.0, 0.0)
        return ((move, tuple(Key(at(k / 4), v, SMOOTH) for k, v in enumerate(values))),)
    if name == "SHAKE":
        seed = track.settings.get("seed")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
            track.settings["seed"] = seed
        rng = np.random.default_rng(seed)
        step = max(1, round(fps / 12))  # a jolt every twelfth of a second
        times = list(range(0, n, step))
        out = []
        for axis in range(2):  # sideways, not up and down
            keys = [Key(t, float(rng.integers(-1, 2)), HOLD) for t in times]
            keys.append(Key(n, 0.0, HOLD))
            out.append((MOVE[axis], tuple(keys)))
        return tuple(out)
    if name == "GLOW":
        return ((GLOW, (Key(0, 100, SMOOTH), Key(at(0.5), 200, SMOOTH), Key(n, 100, SMOOTH))),)
    if name == "BLINK":
        slot = track.target.slots[0] if track.target.slots else 0
        rest_colour = tuple(int(c) for c in scene.palette[slot, :3]) if scene is not None else (0, 0, 0)
        skin = track.settings.get("skin_colour")
        middle = tuple(int(c) for c in skin) if skin is not None else (0, 0, 0)
        shut = max(1, round(0.13 * fps))
        times = (0, max(0, n - shut - 1), max(0, n - 1))
        return ((COLOUR_RGB, tuple(Key(t, v, HOLD) for t, v in zip(times, (rest_colour, middle, rest_colour)))),)
    if name == "FLOW":
        times = [at(k / 6) for k in range(6)]
        return ((SLOT, tuple(Key(t, k, HOLD) for k, t in enumerate(times))),)
    raise ValueError(f"no motion named {name!r}")


def preset_kind(name: str) -> str:
    """The target kind (PART or COLOUR) motion `name` animates."""
    return PART if name in PART_PRESETS else COLOUR


# --- the .vox file's own frames --------------------------------------------------


def track_from_vox(uid: int, frames, state_frame: int = 0, states=()) -> Track | None:
    """The part track for a model's .vox animation, or None without any.

    `frames` are ModelData.frames: (frame number, rotation byte, move) after the
    first frame, which is the rest pose at frame 0. `state_frame` is the frame of
    the model's own voxels, `states` (frame, which frame shows) for the nSHP's
    other entries: 1.. a state, 0 the model's own voxels again.
    Every key is HOLD, as .vox frames play. A later frame whose byte
    mirrors relative to the rest reads as no turn, the frame's move kept."""
    if not frames and not states:
        return None
    track = Track(Target(PART, uid))
    if frames:
        for name in MOVE + TURN:
            set_key(track, name, Key(0, 0))
        for number, rotation, move in frames:
            try:
                degrees = euler_from_matrix(grid_ops.rotation_from_byte(int(rotation)))
            except ValueError:
                degrees = (0, 0, 0)
            for name, v in zip(MOVE, move):
                set_key(track, name, Key(int(number), int(v)))
            for name, v in zip(TURN, degrees):
                set_key(track, name, Key(int(number), int(v)))
        for name in MOVE + TURN:  # a channel that never changes is no channel
            if all(k.value == 0 for k in track.keys(name)):
                del track.properties[name]
    if states:
        set_key(track, STATE, Key(int(state_frame), 0))
        for number, index in states:
            set_key(track, STATE, Key(int(number), int(index)))
    return track


def vox_length(track: Track) -> int:
    """The loop length the .vox frames imply: one past the last key."""
    times = track.key_times()
    return times[-1] + 1 if times else 0


def _group_props(track: Track, group: str) -> list[Property]:
    return [track.properties[n] for n in GROUPS[group] if n in track.properties and track.properties[n].keys]


def _group_keys_exactly(track: Track, group: str) -> bool:
    """HOLD keys on whole cells (POSITION) or whole quarter turns (ROTATION)
    round-trip a .vox exactly; anything else does not."""
    step = 1 if group == POSITION else 90
    for prop in _group_props(track, group):
        for key in prop.keys:
            if key.ease != HOLD or float(key.value) % step != 0:
                return False
    return True


def _rounded_position(value) -> tuple[int, int, int]:
    """`value` (dx, dy, dz) rounded to whole cells."""
    return tuple(int(round(v)) for v in value)


def _rounded_rotation(value) -> tuple[int, int, int]:
    """`value` (rx, ry, rz) rounded to the nearest quarter turn (-44..44 is 0,
    45..134 is 90, and so on). It was rounded down before, which turned a
    leg keyed at -30 degrees a full quarter back."""
    return tuple(int(np.floor((v + 45.0 + 1e-9) / 90.0)) * 90 for v in value)


def bake_times(track: Track, group: str, length: int, fps: int) -> list[int]:
    """The `.vox` frame numbers a bake of a channel group writes: `.vox`
    frame `k` is sampled at `t = k * fps / VOX_FPS` while `t` is below the
    clip's length, rounded (whole cells, or the quarter turn at or below),
    and a frame stays only where the rounded value differs from the last
    kept frame's."""
    if not _group_props(track, group) or length <= 0:
        return []
    names = GROUPS[group]
    round_value = _rounded_rotation if group == ROTATION else _rounded_position
    times: list[int] = []
    previous = None
    k = 0
    while (t := k * fps / VOX_FPS) < length:
        rounded = round_value(triple(track, names, t))
        if rounded != previous:
            times.append(k)
            previous = rounded
        k += 1
    return times


def vox_from_track(track: Track | None, state_count: int, fps: int = VOX_FPS, length: int = 0):
    """(frames, state frame, states) for writing a model's track back as
    .vox frames; `vox_states` explains the last two. HOLD keys on whole
    cells and quarter turns write one frame per key time; anything else bakes
    at the `.vox` frame rate over `length` frames of the clip."""
    if track is None:
        return (), *vox_states(None, state_count)
    exact = _group_keys_exactly(track, POSITION) and _group_keys_exactly(track, ROTATION)
    frames = []
    if exact:
        times = sorted({k.time for g in GROUPS for p in _group_props(track, g) for k in p.keys} - {0})
        by_frame: dict[int, tuple] = {}
        for t in times:
            whole = _rounded_position(triple(track, TURN, t))
            M = np.rint(matrix_from_euler(whole)).astype(np.int64)
            f = round(t * VOX_FPS / fps)
            by_frame[f] = (f, int(grid_ops.rotation_to_byte(M)), _rounded_position(triple(track, MOVE, t)))
        frames = [by_frame[f] for f in sorted(by_frame)]
    else:
        ks = sorted(set(bake_times(track, POSITION, length, fps)) | set(bake_times(track, ROTATION, length, fps)))
        for k in ks:
            t = k * fps / VOX_FPS
            move = _rounded_position(triple(track, MOVE, t))
            degrees = _rounded_rotation(triple(track, TURN, t))
            M = np.rint(matrix_from_euler(degrees)).astype(np.int64)
            frames.append((k, int(grid_ops.rotation_to_byte(M)), move))
    state_frame, states = vox_states(track, state_count, fps, length)
    return tuple(frames), state_frame, states


def vox_states(track: Track | None, state_count: int, fps: int = VOX_FPS, length: int = 0):
    """(the frame of the model's own voxels, [(frame, which frame shows)]):
    a part's Frames row as .vox frames, at 10 per second. A frame can
    show more than once (a blink opens the eyes again), so an entry can name
    any frame, the model's own (0) too; the writer then writes those voxels
    once more. Without Frame keys, frame k shows at .vox frame k, as the
    .vox file had it. A painted frame that no key shows follows after the clip."""
    keys = [k for k in (track.keys(STATE) if track is not None else ()) if length <= 0 or k.time < length]
    if not keys:
        return 0, [(k, k) for k in range(1, state_count + 1)]
    shows: dict[int, int] = {}
    for key in keys:  # sorted by time; two keys on one .vox frame: the later wins
        index = int(key.value)
        if 0 <= index <= state_count:
            shows[round(key.time * VOX_FPS / max(1, fps))] = index
    ordered = []
    for f in sorted(shows):
        if not ordered or ordered[-1][1] != shows[f]:
            ordered.append((f, shows[f]))
    end = max(round(length * VOX_FPS / max(1, fps)), ordered[-1][0] + 1 if ordered else 0)
    own = next((i for i, (_f, index) in enumerate(ordered) if index == 0), None)
    if own is None:
        state_frame, end = end, end + 1
    else:
        state_frame = ordered.pop(own)[0]
    shown = {index for _f, index in ordered}
    for index in range(1, state_count + 1):
        if index not in shown:
            ordered.append((end, index))
            end += 1
    return state_frame, ordered


# --- losses: what a .vox cannot keep exactly ---------------------------------

BAKED, DROPPED = "BAKED", "DROPPED"

LOSS_TEXT: dict[str, dict[str, tuple[str, str]]] = {
    ROTATION: {BAKED: ("a .vox holds quarter turns that do not blend", "it gets {n} frames at 10 per second")},
    POSITION: {BAKED: ("a .vox holds whole-cell moves that jump", "it gets {n} frames at 10 per second")},
    SCALE: {DROPPED: ("a .vox cannot scale a model", "it stays at its rest size")},
    VISIBLE: {DROPPED: ("a .vox cannot hide a model over time", "it stays visible")},
    COLOUR_RGB: {DROPPED: ("a .vox has no colour animation", "it gets the normal colour")},
    SLOT: {DROPPED: ("a .vox has no colour animation", "it gets the normal colour")},
    GLOW: {DROPPED: ("a .vox has no colour animation", "it gets the normal colour")},
}

# Channels dropped whole when they carry any keys: a .vox has no way to keep them.
_DROPPED_IF_KEYED = (SCALE, VISIBLE, COLOUR_RGB, SLOT, GLOW)


@dataclass(frozen=True)
class Loss:
    """One channel group a .vox cannot keep exactly: `name` is the Track.name."""

    track_index: int
    property: str
    name: str
    level: str  # BAKED or DROPPED
    reason: str
    instead: str


def vox_losses(scene) -> tuple[Loss, ...]:
    """One Loss per track's moves or turns a .vox cannot keep exactly, plus
    one per track for each of Size, Show, Colour, Slot and Glow that has any
    keys. Frames (STATE) never lose."""
    losses = []
    clip = scene.timeline
    for index, track in enumerate(clip.tracks):
        for group in (POSITION, ROTATION):
            if not _group_props(track, group) or _group_keys_exactly(track, group):
                continue
            reason, instead = LOSS_TEXT[group][BAKED]
            n = len(bake_times(track, group, clip.length, clip.fps))
            losses.append(Loss(index, group, track.name, BAKED, reason, instead.format(n=n)))
        for name in _DROPPED_IF_KEYED:
            if track.keys(name):
                reason, instead = LOSS_TEXT[name][DROPPED]
                losses.append(Loss(index, name, track.name, DROPPED, reason, instead))
    return tuple(losses)


# --- JSON (the Rok file) -------------------------------------------------------


def _encode_value(value):
    return list(value) if isinstance(value, tuple) else value


def _decode_value(value):
    return tuple(value) if isinstance(value, list) else value


def _target_to_json(target: Target) -> dict:
    return {
        "kind": target.kind,
        "model": target.uid,
        "mask": base64.b64encode(target.mask).decode("ascii") if target.mask else None,
        "slots": list(target.slots),
    }


def _target_from_json(d: dict, uids: list[int]) -> Target:
    model = d.get("model")
    mask = d.get("mask")
    return Target(
        kind=d["kind"],
        uid=uids[model] if model is not None else 0,
        mask=base64.b64decode(mask) if mask else b"",
        slots=tuple(d.get("slots", ())),
    )


def _track_to_json(track: Track) -> dict:
    return {
        "target": _target_to_json(track.target),
        "name": track.name,
        "axis": track.axis,
        "settings": dict(track.settings),
        "properties": {
            name: [[k.time, _encode_value(k.value), k.ease] for k in prop.keys]
            for name, prop in track.properties.items()
        },
    }


def _track_from_json(d: dict, uids: list[int]) -> Track:
    track = Track(
        target=_target_from_json(d["target"], uids),
        name=d.get("name", ""),
        axis=d.get("axis", 2),
        settings=dict(d.get("settings", {})),
    )
    for name, keys in d.get("properties", {}).items():
        track.properties[name] = Property(name, [Key(int(t), _decode_value(v), e) for t, v, e in keys])
    return track


def to_json(clip: Clip) -> dict:
    """`clip` as a JSON-ready dict. A target's `uid` sits under "model" as is;
    rok/core/rokfile.py replaces it with the model index."""
    return {
        "name": clip.name,
        "fps": clip.fps,
        "length": clip.length,
        "loop_mode": clip.loop_mode,
        "voxel_look": clip.voxel_look,
        "root_motion": clip.root_motion,
        "markers": [[m.time, m.name] for m in clip.markers],
        "tracks": [_track_to_json(t) for t in clip.tracks],
    }


def from_json(d: dict, uids: list[int]) -> Clip:
    """The inverse of to_json for a dict whose targets' "model" is an index
    into `uids`."""
    tracks = [_track_from_json(t, uids) for t in d.get("tracks", [])]
    return Clip(
        name=d.get("name", "Clip 1"),
        fps=int(d.get("fps", DEFAULT_FPS)),
        length=int(d.get("length", DEFAULT_FPS)),
        loop_mode=d.get("loop_mode", LOOP),
        voxel_look=bool(d.get("voxel_look", True)),
        root_motion=bool(d.get("root_motion", False)),
        tracks=tracks,
        markers=[Marker(int(t), str(n)) for t, n in d.get("markers", [])],
    )
