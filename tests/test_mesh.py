"""Face culling on small hand-built grids, and the merged mesh for exports."""

from pathlib import Path

import numpy as np
import pytest

from rok.core.grid import VoxelGrid
from rok.core.mesh import ATLAS_TILES, ATLAS_PX, build_chunk_mesh, build_mesh, build_mesh_merged, tile_uvs
from rok.core.vox_read import read_scene

FIXTURES = Path(__file__).parent / "fixtures"


def test_single_voxel_six_quads():
    voxels = np.zeros((1, 1, 1), dtype=np.uint8)
    voxels[0, 0, 0] = 1
    mesh = build_mesh(voxels)
    assert mesh.quad_count == 6
    assert mesh.vertices.shape == (24, 3)


def test_two_adjacent_voxels_ten_quads():
    voxels = np.zeros((2, 1, 1), dtype=np.uint8)
    voxels[0, 0, 0] = 1
    voxels[1, 0, 0] = 1
    assert build_mesh(voxels).quad_count == 10


def test_hidden_interior_voxel_adds_no_faces():
    solid = np.ones((3, 3, 3), dtype=np.uint8)
    # 6 faces * 9 cells. The centre voxel is fully hidden.
    assert build_mesh(solid).quad_count == 54
    shell = solid.copy()
    shell[1, 1, 1] = 0
    # Opening the centre exposes 6 inner faces, so the interior voxel
    # itself contributed 0 quads.
    assert build_mesh(shell).quad_count == 60


def test_empty_grid_has_no_faces():
    mesh = build_mesh(np.zeros((4, 4, 4), dtype=np.uint8))
    assert mesh.quad_count == 0
    assert mesh.vertices.shape == (0, 3)


def test_uvs_span_the_palette_tile_of_the_index():
    voxels = np.zeros((1, 1, 1), dtype=np.uint8)
    voxels[0, 0, 0] = 7 + 3 * 16  # column 7, row 3 of the atlas
    mesh = build_mesh(voxels)
    assert mesh.uvs.shape == (6 * 4, 2)
    tile = 1.0 / ATLAS_TILES
    inset = 0.5 / ATLAS_PX
    lo = np.array([7 * tile + inset, 3 * tile + inset])
    hi = np.array([8 * tile - inset, 4 * tile - inset])
    assert np.allclose(mesh.uvs.min(axis=0), lo)
    assert np.allclose(mesh.uvs.max(axis=0), hi)
    # Every face uses all four tile corners, in the fixed corner order.
    expected = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    for face in mesh.uvs.reshape(6, 4, 2):
        assert np.allclose(face, expected)


def test_tile_uvs_cover_all_256_tiles_without_overlap():
    uvs = tile_uvs(np.arange(256))
    assert uvs.shape == (256, 4, 2)
    assert (uvs >= 0).all() and (uvs <= 1).all()
    origins = np.floor(uvs[:, 0, :] * ATLAS_TILES).astype(int)
    assert len({tuple(o) for o in origins.tolist()}) == 256


def test_chunk_mesh_matches_full_mesh_on_one_chunk():
    grid = VoxelGrid((32, 32, 32))
    grid.set_voxel(0, 0, 0, 1)
    grid.set_voxel(31, 31, 31, 2)
    full = build_mesh(grid.voxels)
    chunk = build_chunk_mesh(grid.voxels, (0, 0, 0))
    assert chunk.quad_count == full.quad_count
    assert chunk.vertices.shape == full.vertices.shape


def test_chunk_mesh_culls_against_neighbour_chunk():
    voxels = np.zeros((64, 32, 32), dtype=np.uint8)
    voxels[31, 0, 0] = 1
    voxels[32, 0, 0] = 1
    chunk0 = build_chunk_mesh(voxels, (0, 0, 0))
    chunk1 = build_chunk_mesh(voxels, (1, 0, 0))
    # Shared face is culled on both sides. Faces outside this chunk are not emitted.
    assert chunk0.quad_count == 5
    assert chunk1.quad_count == 5
    assert build_mesh(voxels).quad_count == 10


# --- the merged mesh ------------------------------------------------------------------


def unit_faces(mesh) -> dict:
    """What a mesh draws, as unit faces: (axis, outward sign, plane, a, b) -> palette
    index. A quad of any size counts as the unit faces it covers, so a merged mesh
    and a culled one compare equal when they show the same surface."""
    faces = {}
    for q, uv in zip(mesh.vertices.reshape(-1, 4, 3), mesh.uvs.reshape(-1, 4, 2)):
        lo, hi = q.min(axis=0), q.max(axis=0)
        axis = int(np.nonzero(lo == hi)[0][0])
        normal = np.cross(q[1] - q[0], q[2] - q[0])
        sign = 1 if normal[axis] > 0 else -1
        tile = np.floor(uv.mean(axis=0) * ATLAS_TILES).astype(int)
        colour = int(tile[1] * ATLAS_TILES + tile[0])
        a, b = (k for k in range(3) if k != axis)
        for i in range(int(lo[a]), int(hi[a])):
            for j in range(int(lo[b]), int(hi[b])):
                key = (axis, sign, int(lo[axis]), i, j)
                assert key not in faces, "two quads cover one face"
                faces[key] = colour
    return faces


def test_a_solid_block_of_one_colour_is_six_rectangles():
    solid = np.ones((3, 4, 5), dtype=np.uint8)
    mesh = build_mesh_merged(solid)
    assert mesh.quad_count == 6
    assert unit_faces(mesh) == unit_faces(build_mesh(solid))


def test_faces_of_different_colours_do_not_join():
    row = np.zeros((4, 1, 1), dtype=np.uint8)
    row[:, 0, 0] = [1, 1, 2, 2]
    mesh = build_mesh_merged(row)
    # the two end caps, then +Y, -Y, +Z and -Z in two pieces each
    assert mesh.quad_count == 2 + 4 * 2
    assert unit_faces(mesh) == unit_faces(build_mesh(row))


def test_merged_quads_keep_the_winding_of_the_culled_ones():
    rng = np.random.default_rng(3)
    voxels = (rng.random((6, 5, 4)) < 0.6) * rng.integers(1, 4, (6, 5, 4))
    mesh = build_mesh_merged(voxels.astype(np.uint8))
    for q in mesh.vertices.reshape(-1, 4, 3):
        normal = np.cross(q[1] - q[0], q[2] - q[0])
        axis = int(np.argmax(np.abs(normal)))
        centre = q.mean(axis=0)
        probe = centre + np.eye(3)[axis] * np.sign(normal[axis]) * 0.5  # a cell just in front of the quad
        cell = np.floor(probe).astype(int)
        inside = np.all(cell >= 0) and np.all(cell < voxels.shape)
        assert not inside or voxels[tuple(cell)] == 0  # the quad faces away from its solid


@pytest.mark.parametrize("seed", range(5))
def test_merged_mesh_shows_the_same_surface_as_the_culled_one(seed):
    rng = np.random.default_rng(seed)
    voxels = ((rng.random((7, 6, 5)) < 0.55) * rng.integers(1, 5, (7, 6, 5))).astype(np.uint8)
    merged, culled = build_mesh_merged(voxels), build_mesh(voxels)
    assert unit_faces(merged) == unit_faces(culled)
    assert merged.quad_count <= culled.quad_count


@pytest.mark.parametrize("name", ["robot.vox", "bird.vox", "car.vox"])
def test_merged_mesh_of_a_real_model_is_the_same_surface_and_much_smaller(name):
    voxels = read_scene(FIXTURES / name).models[0].voxels
    merged, culled = build_mesh_merged(voxels), build_mesh(voxels)
    assert unit_faces(merged) == unit_faces(culled)
    assert merged.quad_count * 1.5 < culled.quad_count


def test_merged_mesh_of_nothing_is_empty():
    assert build_mesh_merged(np.zeros((4, 4, 4), dtype=np.uint8)).quad_count == 0
