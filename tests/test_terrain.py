"""Terrain tiles: the layout (rok/core/terrain.py) and the bundled shader's header."""

from pathlib import Path

from rok.core import xs
from rok.core.terrain import MAX_TILES, shader_args, tile_layout

SHADER = Path(__file__).resolve().parent.parent / "rok" / "shaders" / "nature" / "terrain_tiles.txt"


def test_tiles_sit_edge_to_edge_in_the_order_of_their_numbers():
    tiles = tile_layout(3, 2, 64, (10, -20, 0))
    assert [(t.tile_x, t.tile_y) for t in tiles] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    for t in tiles:
        assert t.position == (10 + 64 * t.tile_x, -20 + 64 * t.tile_y, 0)
    assert len({t.name for t in tiles}) == len(tiles)


def test_the_layout_holds_at_most_five_by_five():
    tiles = tile_layout(9, 0, 32, (0, 0, 0))
    assert {t.tile_x for t in tiles} == set(range(MAX_TILES)) and {t.tile_y for t in tiles} == {0}


def test_the_bundled_shader_takes_tile_numbers_and_brings_its_colours():
    shader = xs.load(SHADER)
    names = [a.name for a in shader.args]
    assert names[:2] == ["Tile X", "Tile Y"] and names[-1] == "Seed"
    assert shader.colours is not None and shader.colours.start == 0
    assert len(shader.colours.rgb) == 6  # water, sand, grass, forest, rock, snow
    assert xs.prepare(shader.source)  # Rok accepts it for the GPU
    values = shader_args(shader.args, 2, 3, 77, style=1)
    assert values[:2] == [2.0, 3.0] and values[-1] == 77.0
    style = names.index("Style")
    assert values[style] == 1.0
    assert [v for k, v in enumerate(values[2:-1], 2) if k != style] == [
        a.value for k, a in enumerate(shader.args[2:-1], 2) if k != style]
