"""Render presets: the light, world, ground and colour look of a picture, as
data, plus the camera and light maths. Numpy only.

rok.core.light turns a preset into a LightSetup, and a setup and the
scene's box into the arguments of Host.set_render (native/src/trace.rs). The
camera fields and the framing maths are for finished pictures (an isometric
camera, a size, a margin).

Light power keeps the same look at every scene size: lamps sit at
LIGHT_DISTANCE scene radii from the scene's middle, and their power grows with
the radius, so a 10-voxel and a 200-voxel model come out equally bright.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LightSpec:
    kind: str  # 'SUN' | 'AREA' | 'POINT'
    azimuth: float  # degrees around +Z, 0 = +X, counter-clockwise seen from above
    elevation: float  # degrees above the XY plane
    colour: tuple[float, float, float]
    power: float  # relative; light_energy turns it into watts
    size: float = 0.0  # AREA: side / r; POINT: radius / r
    angle: float = 1.0  # SUN: the sun's width in degrees (soft shadows)


@dataclass(frozen=True)
class WorldSpec:
    kind: str  # 'COLOR' | 'SKY'
    colour: tuple[float, float, float] = (0.0, 0.0, 0.0)
    strength: float = 1.0
    sun_elevation: float = 0.0  # SKY, degrees
    sun_rotation: float = 0.0  # SKY, degrees
    backdrop: tuple[float, float, float] | None = None  # COLOR: what the camera sees instead; None: colour


# The colour looks (Host.set_render tone): clipped, or the film curve.
CLIP, FILM, FILM_PUNCHY, FILM_HIGH = "CLIP", "FILM", "FILM_PUNCHY", "FILM_HIGH"
TONE_CODES = {CLIP: 1, FILM: 2, FILM_PUNCHY: 3, FILM_HIGH: 4}


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    camera_azimuth: float
    camera_elevation: float
    ortho: bool
    lens_mm: float  # perspective only; sensor 36 mm HORIZONTAL
    resolution: tuple[int, int]
    margin: float  # empty border as a fraction of the frame
    tone: str  # CLIP, FILM, FILM_PUNCHY or FILM_HIGH
    exposure: float  # stops
    film_transparent: bool
    ground: bool
    ground_colour: tuple[float, float, float]
    world: WorldSpec
    lights: tuple[LightSpec, ...]
    samples: int
    bloom: float = 0.0  # strength of the glow around bright lights; 0: none
    outline: float = 0.0  # how much darker the ink lines on shape edges are; 0: none
    voxel_grid: float = 0.0  # how much darker the border of every voxel face is (0.1 = 10 %); 0: none


AREA_W_PER_UNIT2 = 6.0  # AREA energy in W = power * AREA_W_PER_UNIT2 * d^2 with d = LIGHT_DISTANCE * r
SUN_W_PER_M2 = 3.0  # SUN irradiance = power * SUN_W_PER_M2
POINT_W_PER_UNIT2 = 20.0  # POINT energy = power * POINT_W_PER_UNIT2 * r^2
LIGHT_DISTANCE = 2.5  # lamps sit at LIGHT_DISTANCE * r from the scene's middle
MIN_RADIUS = 8.0

PRESETS = (
    # A plain studio look: a light grey sky, a grey ground, one sun from the
    # front right, colours clipped to white.
    Preset('STANDARD', 'Standard', 35, 25, False, 35.0, (1920, 1080), 0.08, CLIP, 0.0, False,
           True, (0.09, 0.09, 0.09), WorldSpec('COLOR', (0.58, 0.58, 0.58), 1.0),
           (LightSpec('SUN', -45, 50, (1.0, 1.0, 1.0), 0.8, angle=2.0),),
           128, bloom=0.3),
    # A product shot: a warm key, a cool fill and a rim, all soft.
    Preset('STUDIO', 'Studio', 35, 25, False, 35.0, (1920, 1080), 0.08, FILM_PUNCHY, 0.0, False,
           True, (0.5, 0.5, 0.5), WorldSpec('COLOR', (0.18, 0.18, 0.18), 1.0),
           (LightSpec('AREA', 45, 45, (1.0, 0.95, 0.88), 1.0, size=0.8),
            LightSpec('AREA', -60, 20, (0.85, 0.9, 1.0), 0.3, size=1.2),
            LightSpec('AREA', 180, 35, (1.0, 1.0, 1.0), 0.6, size=0.5)),
           128, bloom=0.2),
    # A low sun in a sky: warm light, long shadows.
    Preset('SUNSET', 'Sunset', -30, 15, False, 35.0, (1920, 1080), 0.08, FILM_PUNCHY, 0.0, False,
           True, (0.35, 0.3, 0.28), WorldSpec('SKY', strength=1.0, sun_elevation=8, sun_rotation=90),
           (), 160, bloom=0.25),
    # A dark blue night: a weak moon and a warm lantern.
    Preset('NIGHT', 'Night', 45, 30, False, 35.0, (1920, 1080), 0.08, FILM_HIGH, 0.0, False,
           True, (0.12, 0.12, 0.14), WorldSpec('COLOR', (0.01, 0.015, 0.03), 1.0),
           (LightSpec('SUN', 200, 50, (0.7, 0.8, 1.0), 0.25, angle=1.0),
            LightSpec('POINT', -40, 25, (1.0, 0.75, 0.45), 1.5, size=0.15)),
           160, bloom=0.4),
    # A game sprite: a true isometric camera, a square picture, no ground and a
    # transparent background.
    Preset('ISO', 'Isometric', 45, 35.264, True, 35.0, (2048, 2048), 0.06, CLIP, 0.0, True,
           False, (0.0, 0.0, 0.0), WorldSpec('COLOR', (0.35, 0.35, 0.35), 1.0),
           (LightSpec('SUN', 60, 55, (1.0, 0.98, 0.95), 0.67, angle=1.5),),
           96),
    # A toy diorama: isometric, a near-white backdrop, a dimmer sky and one
    # soft sun for the light, and dark ink lines along every shape edge.
    Preset('DIORAMA', 'Diorama', 45, 35.264, True, 35.0, (2048, 1536), 0.06, CLIP, 0.0, False,
           False, (0.0, 0.0, 0.0), WorldSpec('COLOR', (0.55, 0.57, 0.6), 1.0, backdrop=(0.8, 0.8, 0.8)),
           (LightSpec('SUN', -30, 55, (1.0, 0.97, 0.92), 1.1, angle=12.0),),
           128, outline=0.65, voxel_grid=0.1),
    # Diorama moods: the diorama camera on a near-black backdrop, so a tile
    # floats in the dark, and lighter ink lines. Day: a warm sun high on the
    # front side, a light cool sky fill and one stop more exposure, so muted
    # and dark colours still read; the backdrop is halved to stay near-black.
    Preset('DIORAMA_DAY', 'Diorama Day', 45, 35.264, True, 35.0, (2048, 2048), 0.06, FILM_PUNCHY, 1.0, False,
           False, (0.0, 0.0, 0.0), WorldSpec('COLOR', (0.55, 0.6, 0.68), 1.0, backdrop=(0.0125, 0.0125, 0.015)),
           (LightSpec('SUN', -30, 65, (1.0, 0.95, 0.85), 1.6, angle=4.0),),
           128, bloom=0.15, outline=0.25, voxel_grid=0.05),
    # Golden hour: a low orange sun from the front left, a violet sky fill and
    # a warm dark backdrop; long shadows, the sides of the tile glow.
    Preset('DIORAMA_GOLDEN', 'Diorama Golden Hour', 45, 35.264, True, 35.0, (2048, 2048), 0.06, FILM_PUNCHY, 0.0,
           False, False, (0.0, 0.0, 0.0), WorldSpec('COLOR', (0.3, 0.26, 0.4), 1.0, backdrop=(0.035, 0.025, 0.03)),
           (LightSpec('SUN', -15, 24, (1.0, 0.68, 0.4), 1.5, angle=3.0),),
           128, bloom=0.25, outline=0.25, voxel_grid=0.05),
    # Night: a weak blue moon and a dark blue sky; the light comes from the
    # scene's own glowing colours (windows, lamps), with a strong bloom.
    Preset('DIORAMA_NIGHT', 'Diorama Night', 45, 35.264, True, 35.0, (2048, 2048), 0.06, FILM_HIGH, 0.0, False,
           False, (0.0, 0.0, 0.0), WorldSpec('COLOR', (0.03, 0.045, 0.09), 1.0, backdrop=(0.01, 0.014, 0.03)),
           (LightSpec('SUN', -60, 40, (0.55, 0.65, 1.0), 0.4, angle=2.0),),
           160, bloom=0.5, outline=0.2, voxel_grid=0.05),
)

DEFAULT = "STUDIO"


def preset_by_id(pid: str) -> Preset:
    for preset in PRESETS:
        if preset.id == pid:
            return preset
    raise KeyError(f"no render preset {pid!r}")


# --- geometry ----------------------------------------------------------------------


def direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from the target toward a camera or light at these angles."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float64)


def view_azimuth(view_vec) -> float | None:
    """Azimuth in degrees of a vector from the target toward the eye, or None
    when it points (almost) straight up or down and has no side."""
    x, y = float(view_vec[0]), float(view_vec[1])
    if math.hypot(x, y) < 1e-3 * max(1e-12, float(np.linalg.norm(view_vec))):
        return None
    return math.degrees(math.atan2(y, x))


def iso_turn(preset_azimuth: float, azimuth: float | None) -> float:
    """The quarter turn (0, 90, 180 or 270 degrees) that brings a fixed-angle
    camera at `preset_azimuth` to the side nearest `azimuth`; 0 without one."""
    if azimuth is None:
        return 0.0
    return float((90 * round((azimuth - preset_azimuth) / 90.0)) % 360)


def scene_radius(lo, hi) -> float:
    """Half the diagonal of the box [lo, hi + 1], at least MIN_RADIUS."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64) + 1.0
    return max(MIN_RADIUS, 0.5 * float(np.linalg.norm(hi - lo)))


def box_corners(lo, hi) -> np.ndarray:
    """(8, 3) corners of the box [lo, hi + 1]."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64) + 1.0
    bits = np.array([[(i >> a) & 1 for a in range(3)] for i in range(8)], dtype=np.float64)
    return lo + bits * (hi - lo)


def camera_basis(view_vec) -> np.ndarray:
    """(3, 3) with columns right, up, back for a camera that looks along -view_vec.
    Up leans on world +Z; a camera on the Z axis uses +Y."""
    back = np.asarray(view_vec, dtype=np.float64)
    back = back / np.linalg.norm(back)
    right = np.cross([0.0, 0.0, 1.0], back)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross([0.0, 1.0, 0.0], back)
    right = right / np.linalg.norm(right)
    up = np.cross(back, right)
    return np.stack([right, up, back], axis=1)


def _camera_coords(corners, view_vec):
    """Centre and per-corner (x, y, z) in the camera basis; z is depth toward the scene."""
    corners = np.asarray(corners, dtype=np.float64)
    centre = corners.mean(axis=0)
    basis = camera_basis(view_vec)
    local = (corners - centre) @ basis  # columns: right, up, back
    return centre, local[:, 0], local[:, 1], -local[:, 2]


def frame_perspective(corners, view_vec, hfov: float, vfov: float, margin: float) -> tuple[np.ndarray, float]:
    """Camera location and distance so that every corner fits inside the margin.

    Per corner: x = (p - C) . right, y = (p - C) . up, z = (p - C) . forward.
    d = max over corners of |x| / (tan(hfov / 2) (1 - margin)) - z and the same for y.
    """
    centre, x, y, z = _camera_coords(corners, view_vec)
    kx = math.tan(hfov / 2) * (1.0 - margin)
    ky = math.tan(vfov / 2) * (1.0 - margin)
    d = float(max(np.max(np.abs(x) / kx - z), np.max(np.abs(y) / ky - z)))
    view = np.asarray(view_vec, dtype=np.float64)
    view = view / np.linalg.norm(view)
    return centre + view * d, d


def frame_ortho(corners, view_vec, aspect: float, margin: float) -> tuple[np.ndarray, float]:
    """Camera location and frame width for an orthographic camera.

    `aspect` is res_y / res_x: the frame is `width` wide and width * aspect
    high. The camera sits at 2 * radius from the centre.
    """
    centre, x, y, _z = _camera_coords(corners, view_vec)
    extent_x = float(x.max() - x.min())
    extent_y = float(y.max() - y.min())
    width = max(extent_x, extent_y / aspect) / (1.0 - margin)
    corners = np.asarray(corners, dtype=np.float64)
    radius = max(MIN_RADIUS, 0.5 * float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))))
    view = np.asarray(view_vec, dtype=np.float64)
    view = view / np.linalg.norm(view)
    return centre + view * 2.0 * radius, width


def fov(lens_mm: float, sensor_mm: float = 36.0) -> float:
    """Full field of view in radians: 2 atan(sensor / (2 lens))."""
    return 2.0 * math.atan(sensor_mm / (2.0 * lens_mm))


# --- lights ------------------------------------------------------------------------


SUN, SPHERE = 0, 1  # the lamp kinds of Host.set_render


def _sky(world: WorldSpec) -> tuple[tuple, tuple, LightSpec]:
    """A sky as a gradient and a sun: warm and low near the horizon, white and
    bright high up. (top colour, horizon colour, the sun)."""
    el = max(world.sun_elevation, 0.0)
    low = math.exp(-el / 12.0)  # 1 at the horizon, near 0 high up
    sun_col = (1.0, 0.95 - 0.55 * low, 0.88 - 0.8 * low)
    horizon = (0.9 + 0.1 * low, 0.78 - 0.33 * low, 0.68 - 0.5 * low)
    top = (0.3 - 0.12 * low, 0.42 - 0.14 * low, 0.65 - 0.1 * low)
    sun = LightSpec('SUN', world.sun_rotation, max(el, 1.0), sun_col, 1.4 - 0.4 * low, angle=1.5)
    s = world.strength
    return tuple(c * s for c in top), tuple(c * s for c in horizon), sun
