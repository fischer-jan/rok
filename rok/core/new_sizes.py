"""The standard grid sizes that File > New offers."""

from __future__ import annotations

from .grid_ops import MAX_SIZE, MIN_SIZE

DEFAULT = "SMALL"

# (id, name, size). The size is (x, y, z); z is the height.
STANDARDS = (
    ("ICON", "Icon", (16, 16, 16)),
    ("SMALL", "Small", (32, 32, 32)),
    ("MEDIUM", "Medium", (64, 64, 64)),
    ("LARGE", "Large", (128, 128, 128)),
    ("CHARACTER", "Character", (32, 32, 64)),
    ("VEHICLE", "Vehicle", (96, 48, 48)),
    ("TILE", "Tile", (64, 64, 16)),
    ("ROOM", "Room", (128, 128, 64)),
    ("SCENE", "Scene", (256, 256, 128)),
    ("HUGE", "Huge", (256, 256, 256)),
)


def size_of(preset_id: str) -> tuple[int, int, int]:
    """The size of the standard with this id."""
    return next(size for pid, _name, size in STANDARDS if pid == preset_id)


def label(name: str, size) -> str:
    """'Small  32 x 32 x 32': a standard's name and its size."""
    return f"{name}   {size[0]} × {size[1]} × {size[2]}"


def preset_for(size) -> str | None:
    """The id of the standard that has exactly this size, else None."""
    size = tuple(int(v) for v in size)
    return next((pid for pid, _name, s in STANDARDS if s == size), None)


def valid(size) -> bool:
    return len(size) == 3 and all(MIN_SIZE <= int(v) <= MAX_SIZE for v in size)
