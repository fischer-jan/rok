"""VoxScene: models over one shared palette."""

import numpy as np
import pytest

from rok.core.grid import VoxelGrid
from rok.core.material import DEFAULT, MaterialTable
from rok.core.scene import DEFAULT_LAYERS, SceneModel, VoxScene, merged
from rok.core.select import Selection
from rok.core.vox_read import read_scene
from rok.core.vox_write import dumps_scene

from helpers import FIXTURES

GOLDEN = FIXTURES / "phase3" / "two_models.vox"


def _grid(size=(4, 4, 4), fill=0) -> VoxelGrid:
    grid = VoxelGrid(size)
    if fill:
        grid.voxels[:] = fill
    return grid


def _model_scene(stem: str) -> VoxScene:
    return VoxScene.from_data(read_scene(FIXTURES / f"{stem}.vox"))


# --- construction -----------------------------------------------------------------


def test_single_has_one_model_default_layers_no_extras():
    grid = _grid((6, 4, 2))
    scene = VoxScene.single(grid)
    assert len(scene.models) == 1
    model = scene.models[0]
    assert model.position == (-3, -2, -1)
    assert model.name == ""
    assert model.uid == 1
    assert scene.next_uid == 2
    assert scene.layers == list(DEFAULT_LAYERS)
    assert scene.extras == []
    assert isinstance(scene.materials, MaterialTable)
    assert not scene.materials
    assert scene.palette_tail == b"\0\0\0\0"
    assert model.grid is grid
    assert isinstance(model.selection, Selection)
    assert model.selection.size == (6, 4, 2)


def test_default_layers_are_eight_26_byte_entries():
    assert len(DEFAULT_LAYERS) == 8
    for i, layer in enumerate(DEFAULT_LAYERS):
        assert len(layer) == 26
        assert layer[:4] == i.to_bytes(4, "little")
        assert str(i).encode() in layer
        assert layer.endswith(b"\xff\xff\xff\xff")


def test_add_uses_minus_half_size_and_hands_out_uids():
    scene = VoxScene.single(_grid())
    a = scene.add(_grid((5, 3, 7)), "a")
    b = scene.add(_grid((2, 2, 2)), "b", (10, 20, 30))
    assert a.position == (-2, -1, -3)
    assert b.position == (10, 20, 30)
    assert [m.uid for m in scene.models] == [1, 2, 3]
    assert scene.index_of(3) == 2
    assert scene.by_uid(2) is a
    with pytest.raises(ValueError):
        scene.index_of(99)


def test_palette_aliasing():
    """Every grid holds the same palette array; one write shows in every model."""
    scene = _model_scene("robot")
    added = scene.add(VoxelGrid((3, 3, 3)))
    assert added.grid.palette is scene.palette
    assert scene.models[0].grid.palette is scene.palette
    scene.palette[7] = (1, 2, 3, 255)
    assert added.grid.palette[7].tolist() == [1, 2, 3, 255]
    assert scene.models[0].grid.palette[7].tolist() == [1, 2, 3, 255]
    # to_model still copies.
    model = added.grid.to_model()
    assert model.palette is not scene.palette
    model.palette[7] = 0
    assert scene.palette[7].tolist() == [1, 2, 3, 255]


def test_insert_and_remove():
    scene = VoxScene.single(_grid())
    b = scene.add(_grid(), "b")
    c = SceneModel(uid=scene.new_uid(), name="c", position=(0, 0, 0), grid=_grid(), selection=Selection((4, 4, 4)))
    scene.insert(1, c)
    assert [m.name for m in scene.models] == ["", "c", "b"]
    assert c.grid.palette is scene.palette
    removed = scene.remove(1)
    assert removed is c
    assert [m.name for m in scene.models] == ["", "b"]
    assert scene.remove(0).uid == 1
    assert scene.models == [b]
    with pytest.raises(ValueError):
        scene.remove(0)
    assert scene.models == [b]


def test_uids_are_never_reused():
    scene = VoxScene.single(_grid())
    scene.add(_grid())
    scene.remove(1)
    again = scene.add(_grid())
    assert again.uid == 3


# --- queries --------------------------------------------------------------------------


def test_box_world_and_world_bounds():
    scene = VoxScene.single(_grid((4, 6, 8)), (1, 2, 3))
    assert scene.models[0].box_world() == ((1, 2, 3), (4, 7, 10))
    assert scene.world_bounds() == ((1, 2, 3), (4, 7, 10))
    scene.add(_grid((2, 2, 2)), position=(-5, 10, 0))
    assert scene.world_bounds() == ((-5, 2, 0), (4, 11, 10))


def test_world_bounds_of_the_golden():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    assert scene.world_bounds() == ((-40, -5, 0), (49, 54, 19))


def test_merged_needs_at_least_one_model():
    with pytest.raises(ValueError):
        merged([])


def test_merged_single_model_is_its_own_grid():
    scene = VoxScene.single(_grid((3, 2, 1), fill=5), (4, -1, 2))
    voxels, position = merged(scene.models)
    assert position == (4, -1, 2)
    assert np.array_equal(voxels, scene.models[0].grid.voxels)


def test_merged_is_the_union_over_the_world_bounds():
    scene = VoxScene.single(_grid((2, 2, 2), fill=1), (0, 0, 0))
    scene.add(_grid((2, 2, 2), fill=2), position=(5, 5, 5))
    voxels, position = merged(scene.models)
    assert position == (0, 0, 0)
    assert voxels.shape == (7, 7, 7)
    assert (voxels[0:2, 0:2, 0:2] == 1).all()
    assert (voxels[5:7, 5:7, 5:7] == 2).all()
    # every other cell is empty: the two boxes do not touch.
    assert np.count_nonzero(voxels) == 8 + 8


def test_merged_overlap_puts_later_models_over_earlier():
    scene = VoxScene.single(_grid((2, 2, 2), fill=1), (0, 0, 0))
    scene.add(_grid((2, 2, 2), fill=2), position=(1, 0, 0))
    voxels, position = merged(scene.models)
    assert position == (0, 0, 0)
    assert voxels.shape == (3, 2, 2)
    assert (voxels[0] == 1).all()  # only the first model reaches here
    assert (voxels[1] == 2).all()  # shared column: the later model wins
    assert (voxels[2] == 2).all()  # only the second model reaches here


def test_merged_empty_cells_of_a_later_model_erase_nothing():
    scene = VoxScene.single(_grid((3, 1, 1)), (0, 0, 0))
    scene.models[0].grid.voxels[1, 0, 0] = 4
    scene.add(_grid((3, 1, 1)), position=(0, 0, 0))
    scene.models[1].grid.voxels[2, 0, 0] = 9
    voxels, _ = merged(scene.models)
    assert list(voxels[:, 0, 0]) == [0, 4, 9]


def test_merged_refuses_a_grid_past_the_size_limit():
    from rok.core.grid_ops import MAX_SIZE
    from rok.core.scene import TooBigToMerge, merge_size

    scene = VoxScene.single(_grid((2, 1, 1), fill=1), (0, 0, 0))
    scene.add(_grid((2, 1, 1), fill=2), position=(MAX_SIZE, 0, 0))
    assert merge_size(scene.models)[0] == MAX_SIZE + 2
    with pytest.raises(TooBigToMerge):
        merged(scene.models)


def test_merged_of_the_two_model_golden_holds_every_voxel_of_both():
    scene = VoxScene.from_data(read_scene(GOLDEN))
    lo, hi = scene.world_bounds()
    voxels, position = merged(scene.models)
    assert position == lo
    assert voxels.shape == tuple(h - l + 1 for l, h in zip(lo, hi))
    total = 0
    for model in scene.models:
        x0, y0, z0 = (p - l for p, l in zip(model.position, lo))
        sx, sy, sz = model.grid.size
        sub = voxels[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz]
        assert np.array_equal(sub, model.grid.voxels)
        total += int(np.count_nonzero(model.grid.voxels))
    # the two models do not overlap, so every voxel of both survives.
    assert int(np.count_nonzero(voxels)) == total


def test_used_counts_sums_every_grid():
    scene = _model_scene("robot")
    single = np.bincount(scene.models[0].grid.voxels.ravel(), minlength=256)
    assert scene.used_counts().shape == (256,)
    assert scene.used_counts().dtype == np.int64
    assert np.array_equal(scene.used_counts(), single)
    scene.add(_grid((2, 2, 2), fill=9))
    counts = scene.used_counts()
    assert counts[9] == single[9] + 8
    assert counts[0] == single[0]


def test_display_name_and_unique_name():
    scene = VoxScene.single(_grid())
    assert scene.models[0].display_name(0) == "Model 1"
    scene.models[0].name = "Ship"
    assert scene.models[0].display_name(0) == "Ship"
    assert scene.unique_name("Other") == "Other"
    assert scene.unique_name("Ship") == "Ship 2"
    scene.add(_grid(), "Ship 2")
    assert scene.unique_name("Ship") == "Ship 3"
    assert scene.add(_grid(), "").display_name(2) == "Model 3"
    assert scene.unique_name("Model 3") == "Model 3 2"  # a shown name counts as taken


def test_a_part_is_named_after_its_model_with_the_next_number():
    scene = VoxScene.single(_grid())  # shows "Model 1"
    assert scene.part_name("Model 1") == "Model 1.1"
    scene.add(_grid(), "Model 1.1")
    assert scene.part_name("Model 1") == "Model 1.2"
    assert scene.part_name("Model 1.1") == "Model 1.1.1"  # a part of a part
    assert scene.part_name("Robot") == "Robot.1"


def test_a_new_model_is_named_like_the_others():
    scene = VoxScene.single(_grid())  # shows "Model 1"
    assert scene.new_model_name() == "Model 2"
    scene.add(_grid(), scene.new_model_name())
    scene.add(_grid(), "Model 3")
    assert scene.new_model_name() == "Model 4"
    scene.models[0].name = "Model 4"  # a name that is taken is skipped
    assert scene.new_model_name() == "Model 5"


# --- file data ----------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["robot", "bird", "car"])
def test_from_data_to_data_round_trip(stem):
    data = read_scene(FIXTURES / f"{stem}.vox")
    scene = VoxScene.from_data(data)
    assert len(scene.models) == 1
    model = scene.models[0]
    assert model.position == data.models[0].position
    assert model.size() == data.models[0].size
    assert np.array_equal(model.grid.voxels, data.models[0].voxels)
    assert scene.layers == data.layers
    assert scene.extras == data.extras
    assert scene.palette_tail == data.palette_tail
    assert scene.materials == data.materials
    assert len(scene.materials) == 256
    again = scene.to_data()
    assert dumps_scene(again) == dumps_scene(data)
    assert again.materials is scene.materials


def test_from_data_copies_the_palette_and_aliases_it():
    data = read_scene(GOLDEN)
    scene = VoxScene.from_data(data)
    assert scene.palette is not data.palette
    assert np.array_equal(scene.palette, data.palette)
    for model in scene.models:
        assert model.grid.palette is scene.palette
    assert [m.name for m in scene.models] == ["Robot", "Car"]
    assert [m.uid for m in scene.models] == [1, 2]
    assert not scene.materials
    assert dumps_scene(scene.to_data()) == GOLDEN.read_bytes()


def test_from_data_copies_the_material_table():
    data = read_scene(FIXTURES / "robot.vox")
    scene = VoxScene.from_data(data)
    assert scene.materials is not data.materials
    assert scene.materials == data.materials
    metal = DEFAULT.with_type("METAL")
    scene.materials.set(3, metal)
    assert data.materials.get(3) == DEFAULT
    assert scene.to_data().materials.get(3) == metal


def test_from_data_without_layers_uses_defaults():
    data = read_scene(GOLDEN)
    data.layers = []
    scene = VoxScene.from_data(data)
    assert scene.layers == list(DEFAULT_LAYERS)


def test_layer_passes_from_data_to_data():
    data = read_scene(GOLDEN)
    data.models[1].layer = 3
    scene = VoxScene.from_data(data)
    assert [m.layer for m in scene.models] == [0, 3]
    assert [m.layer for m in scene.to_data().models] == [0, 3]


def test_add_takes_a_layer_and_defaults_to_zero():
    scene = VoxScene.single(_grid())
    assert scene.models[0].layer == 0
    assert scene.add(_grid(), "b", layer=2).layer == 2
