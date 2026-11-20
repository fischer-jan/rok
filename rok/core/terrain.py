"""Seamless terrain tiles (New > Terrain in the scene).

The bundled shader rok/shaders/nature/terrain_tiles.txt computes its heights in
world columns: tile (tx, ty) of size S covers columns (tx * S, ty * S) to
((tx + 1) * S - 1, (ty + 1) * S - 1). Tiles placed at matching positions,
edge to edge, therefore meet without a seam. This module lays them out.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_TILES = 5  # tiles across and deep
MAX_TILE_SIZE = 256  # the largest model a .vox holds


@dataclass(frozen=True)
class Tile:
    tile_x: int
    tile_y: int
    position: tuple[int, int, int]  # the world cell of its (0, 0, 0)
    name: str


def tile_layout(across: int, deep: int, size: int, origin: tuple[int, int, int]) -> list[Tile]:
    """The tiles of an `across` x `deep` terrain of `size` x `size` models,
    the first at `origin`, row by row. Counts are clamped to 1..MAX_TILES."""
    across = max(1, min(MAX_TILES, int(across)))
    deep = max(1, min(MAX_TILES, int(deep)))
    size = int(size)
    ox, oy, oz = (int(v) for v in origin)
    return [
        Tile(tx, ty, (ox + tx * size, oy + ty * size, oz), f"Terrain {tx + 1}-{ty + 1}")
        for ty in range(deep)
        for tx in range(across)
    ]


def shader_args(args, tile_x: int, tile_y: int, seed: int, **named) -> list[float]:
    """The shader's argument values for one tile. `args` are the header's
    arguments (xs.ShaderArg): Tile X, Tile Y and Seed get the tile's and the
    terrain's values, an argument named in `named` (style=1 for "Style") its
    value, every other one its default."""
    wanted = {"tile x": tile_x, "tile y": tile_y, "seed": seed}
    wanted.update({key.replace("_", " "): value for key, value in named.items()})
    return [float(wanted.get(a.name.lower(), a.value)) for a in args]
