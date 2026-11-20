"""The light of a picture: its lamps, sky, ground and finish, as data.

A file keeps its own `LightSetup` (`VoxScene.light`); a file without one
shows the default preset's. Every render preset turns into a setup
(`from_preset`), and `trace_settings` turns a setup and the scene's box into
the arguments of Host.set_render (native/src/trace.rs).

Two kinds of light:

- a sun: parallel light from far away. Its strength is the same at every
  scene size; its softness is its width in degrees.
- a lamp: a glowing ball near the model, `distance` scene radii from the
  scene's middle; its softness is its radius in scene radii. Its power grows
  with its distance, so moving it out changes its shadows, not how bright the
  model is.

Colours are linear RGB, as the tracer takes them. Angles are degrees:
azimuth around +Z (0 = +X, counter-clockwise seen from above), elevation
above the XY plane.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace

from . import render_presets as rp

SUN, LAMP = "SUN", "LAMP"
KINDS = (SUN, LAMP)
MAX_LIGHTS = 4  # the tracer's limit
CHUNK = b"MLGT"  # the .vox chunk that holds a file's light (JSON text)
VERSION = 1

SUN_W = rp.SUN_W_PER_M2  # a sun's irradiance per unit of power
LAMP_W = rp.AREA_W_PER_UNIT2  # a lamp's energy = power * LAMP_W * (distance * r)^2
LAMP_DISTANCE = rp.LIGHT_DISTANCE
LOWEST = -10.0  # a light's elevation may go this far below the horizon
HIGHEST = 90.0
TONES = (rp.CLIP, rp.FILM, rp.FILM_PUNCHY, rp.FILM_HIGH)
TONE_LABELS = {rp.CLIP: "Clipped", rp.FILM: "Film", rp.FILM_PUNCHY: "Punchy", rp.FILM_HIGH: "High"}


@dataclass(frozen=True)
class Light:
    kind: str = SUN
    azimuth: float = -45.0
    elevation: float = 50.0
    colour: tuple[float, float, float] = (1.0, 1.0, 1.0)
    power: float = 1.0
    softness: float = 2.0  # a sun: its width in degrees; a lamp: its radius in scene radii
    distance: float = LAMP_DISTANCE  # a lamp only: scene radii from the middle
    on: bool = True


@dataclass(frozen=True)
class LightSetup:
    lights: tuple[Light, ...] = (Light(),)
    sky_top: tuple[float, float, float] = (0.18, 0.18, 0.18)
    sky_horizon: tuple[float, float, float] = (0.18, 0.18, 0.18)
    sky_strength: float = 1.0
    backdrop: tuple[float, float, float] | None = None  # what the camera sees instead of the sky
    ground: tuple[float, float, float] | None = (0.5, 0.5, 0.5)  # None: no ground
    exposure: float = 0.0  # stops
    turn: float = 0.0  # every light turned this many degrees around the model (Turn All)
    tone: str = rp.FILM_PUNCHY
    bloom: float = 0.0
    outline: float = 0.0
    voxel_grid: float = 0.0
    transparent: bool = False  # pictures only: no background
    preset: str = rp.DEFAULT  # the preset it started from: a picture's camera and size
    extra: dict = field(default_factory=dict, compare=False, hash=False)  # fields a newer Rok wrote


# --- presets -------------------------------------------------------------------------


def _from_spec(spec: rp.LightSpec) -> Light:
    """One preset light as a Light that gives the tracer the same lamp."""
    if spec.kind == 'SUN':
        return Light(SUN, spec.azimuth, spec.elevation, spec.colour, spec.power, spec.angle)
    if spec.kind == 'AREA':  # rp.lamp: radius size * r / 2, the energy of LAMP_W
        return Light(LAMP, spec.azimuth, spec.elevation, spec.colour, spec.power, spec.size / 2.0)
    if spec.kind == 'POINT':  # rp.lamp: radius size * r, energy POINT_W r^2 over 4 pi (not pi)
        d = LAMP_DISTANCE
        power = spec.power * rp.POINT_W_PER_UNIT2 / (4.0 * LAMP_W * d * d)
        return Light(LAMP, spec.azimuth, spec.elevation, spec.colour, power, spec.size)
    raise ValueError(f"unknown light kind {spec.kind!r}")


def from_preset(preset: rp.Preset | str) -> LightSetup:
    """A preset's light as a setup. A sky becomes its sun and its two colours."""
    if isinstance(preset, str):
        preset = rp.preset_by_id(preset)
    world = preset.world
    lights = [_from_spec(s) for s in preset.lights]
    if world.kind == 'SKY':
        top, horizon, sun = rp._sky(replace(world, strength=1.0))
        lights.insert(0, _from_spec(sun))
    else:
        top = horizon = tuple(world.colour)
    return LightSetup(
        lights=tuple(lights[:MAX_LIGHTS]),
        sky_top=tuple(float(c) for c in top),
        sky_horizon=tuple(float(c) for c in horizon),
        sky_strength=float(world.strength),
        backdrop=world.backdrop,
        ground=tuple(preset.ground_colour) if preset.ground else None,
        exposure=float(preset.exposure),
        tone=preset.tone,
        bloom=float(preset.bloom),
        outline=float(preset.outline),
        voxel_grid=float(preset.voxel_grid),
        transparent=bool(preset.film_transparent),
        preset=preset.id,
    )


def default() -> LightSetup:
    """The light of a file that has none of its own."""
    return from_preset(rp.DEFAULT)


def same_as_preset(setup: LightSetup) -> bool:
    """True when the setup is its preset's light, unchanged."""
    try:
        return setup == from_preset(setup.preset)
    except KeyError:
        return False


# --- the tracer ----------------------------------------------------------------------


def lamp(light: Light, r: float, turn: float = 0.0) -> tuple:
    """One lamp for Host.set_render: (kind, x, y, z, r, g, b, size), the light
    turned `turn` degrees more around the model.

    A sun: (x, y, z) points toward it, the colour is its irradiance, size is
    its width in degrees. A lamp is a glowing sphere: (x, y, z) is its offset
    from the scene's middle, the colour its radiant intensity (W/sr), size its
    radius in cells."""
    d = rp.direction(light.azimuth + turn, light.elevation)
    if light.kind == SUN:
        e = light.power * SUN_W
        return (rp.SUN, *(float(v) for v in d), *(float(c) * e for c in light.colour), float(light.softness))
    dist = light.distance * r
    intensity = light.power * LAMP_W * dist * dist / math.pi
    pos = d * dist
    return (rp.SPHERE, *(float(v) for v in pos), *(float(c) * intensity for c in light.colour),
            float(max(light.softness * r, 0.25)))


def trace_settings(setup: LightSetup, lo, hi) -> dict:
    """Host.set_render's arguments (but `on` and `max_samples`) for the scene
    box [lo, hi] in cells."""
    r = rp.scene_radius(lo, hi)
    s = setup.sky_strength
    return {
        "lamps": [lamp(light, r, setup.turn) for light in setup.lights if light.on][:MAX_LIGHTS],
        "sky_top": tuple(c * s for c in setup.sky_top),
        "sky_horizon": tuple(c * s for c in setup.sky_horizon),
        "backdrop": setup.backdrop,
        "ground": setup.ground,
        "exposure": float(setup.exposure),
        "tone": rp.TONE_CODES.get(setup.tone, rp.TONE_CODES[rp.CLIP]),
        "bloom": float(setup.bloom),
        "outline": float(setup.outline),
        "voxel_grid": float(setup.voxel_grid),
        "transparent": bool(setup.transparent),
    }


# --- changes -------------------------------------------------------------------------


def clamp_elevation(elevation: float) -> float:
    return min(HIGHEST, max(LOWEST, float(elevation)))


def wrap(azimuth: float) -> float:
    """An azimuth in (-180, 180]."""
    a = (float(azimuth) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def with_ground(setup: LightSetup, on: bool, preset: rp.Preset | None = None) -> LightSetup:
    """The setup with its ground on or off. A ground turned on takes the
    preset's ground colour, or a mid grey."""
    if not on:
        return replace(setup, ground=None)
    if setup.ground is not None:
        return setup
    colour = tuple(preset.ground_colour) if preset is not None and preset.ground else (0.3, 0.3, 0.3)
    return replace(setup, ground=colour)


def with_light(setup: LightSetup, index: int, **changes) -> LightSetup:
    lights = list(setup.lights)
    lights[index] = replace(lights[index], **changes)
    return replace(setup, lights=tuple(lights))


def added(setup: LightSetup) -> LightSetup:
    """One more light (up to MAX_LIGHTS): a copy of the first, a third of the way
    around and at half its power, so it shows and does not hide behind it."""
    if len(setup.lights) >= MAX_LIGHTS:
        return setup
    base = setup.lights[0] if setup.lights else Light()
    new = replace(base, azimuth=wrap(base.azimuth + 120.0), power=base.power * 0.5, on=True)
    return replace(setup, lights=setup.lights + (new,))


def removed(setup: LightSetup, index: int) -> LightSetup:
    lights = list(setup.lights)
    if 0 <= index < len(lights):
        del lights[index]
    return replace(setup, lights=tuple(lights))


def srgb255(linear) -> tuple[int, int, int]:
    """A linear colour as a swatch shows it: sRGB 0..255 (clipped to 0..1 first)."""
    out = []
    for c in linear:
        c = min(1.0, max(0.0, float(c)))
        c = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        out.append(int(round(c * 255)))
    return tuple(out)


def linear_of(srgb) -> tuple[float, float, float]:
    """The inverse of srgb255: sRGB 0..255 to linear 0..1."""
    out = []
    for c in srgb:
        c = float(c) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


# --- files ---------------------------------------------------------------------------


def _rgb(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    r, g, b = (float(v) for v in value)
    return (r, g, b)


def to_json(setup: LightSetup) -> dict:
    data = asdict(setup)
    data.pop("extra", None)
    data["lights"] = [asdict(light) for light in setup.lights]
    out = dict(setup.extra)
    out.update(data)
    out["version"] = VERSION
    return out


def from_json(data: dict) -> LightSetup:
    """The inverse of to_json. Missing fields take their defaults; fields this
    Rok does not know are kept (`extra`) and written back. Raises ValueError
    on data that is not a light."""
    if not isinstance(data, dict):
        raise ValueError("a light is a JSON object")
    try:
        base = LightSetup()
        known = set(LightSetup.__dataclass_fields__) - {"extra", "lights"}
        lights = []
        for item in data.get("lights", [])[:MAX_LIGHTS]:
            light = Light()
            values = {}
            for key in Light.__dataclass_fields__:
                if key in item:
                    values[key] = item[key]
            light = replace(light, **values)
            if light.kind not in KINDS:
                raise ValueError(f"unknown light kind {light.kind!r}")
            lights.append(replace(light, azimuth=float(light.azimuth), elevation=clamp_elevation(light.elevation),
                                  colour=_rgb(light.colour), power=max(0.0, float(light.power)),
                                  softness=max(0.0, float(light.softness)),
                                  distance=max(0.5, float(light.distance)), on=bool(light.on)))
        values = {k: data[k] for k in known if k in data}
        setup = replace(base, lights=tuple(lights), **values)
        setup = replace(setup, sky_top=_rgb(setup.sky_top), sky_horizon=_rgb(setup.sky_horizon),
                        backdrop=_rgb(setup.backdrop), ground=_rgb(setup.ground),
                        sky_strength=float(setup.sky_strength), exposure=float(setup.exposure),
                        turn=wrap(setup.turn),
                        tone=setup.tone if setup.tone in TONES else rp.CLIP, bloom=float(setup.bloom),
                        outline=float(setup.outline), voxel_grid=float(setup.voxel_grid),
                        transparent=bool(setup.transparent), preset=str(setup.preset),
                        extra={k: v for k, v in data.items() if k not in known | {"lights", "version"}})
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"not a light: {exc}") from exc
    return setup


def to_chunk(setup: LightSetup) -> bytes:
    return json.dumps(to_json(setup), separators=(",", ":")).encode("utf-8")


def from_chunk(content: bytes) -> LightSetup:
    """A `.vox` chunk's light. Raises ValueError when it is not one."""
    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not a light: {exc}") from exc
    return from_json(data)
