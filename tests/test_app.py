"""rok/app: the camera, the GPU sync and the app's event handling, against a
host that only records what it is told (no window, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.app.camera import Camera
from rok.app.sync import GpuSync
from rok.core.grid import VoxelGrid
from rok.core.scene import VoxScene
from rok.core.vox_read import read_scene

FIXTURES = Path(__file__).parent / "fixtures"
BIRD = FIXTURES / "bird.vox"


def open_scene(path) -> VoxScene:
    return VoxScene.from_data(read_scene(path))


class RecordingHost:
    """Stands in for rok_native.Host: keeps every call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "measure_text":
                return (len(args[0]) * 7.0, 16.0)
            return None

        return record

    def named(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]

    def clear(self) -> None:
        self.calls.clear()


# --- camera -------------------------------------------------------------------------


def test_the_ray_through_the_middle_of_the_view_meets_the_target():
    cam = Camera(target=np.array([3.0, -2.0, 5.0]), distance=80.0)
    o, d = cam.ray(cam.width / 2, cam.height / 2)
    closest = o + d * np.dot(cam.target - o, d)
    assert np.allclose(closest, cam.target, atol=1e-6)


@pytest.mark.parametrize("x, y", [(0.0, 0.0), (1440.0, 900.0), (300.0, 700.0), (1200.0, 50.0)])
def test_a_point_on_a_ray_projects_back_to_the_pixel_of_the_ray(x, y):
    cam = Camera(target=np.zeros(3), yaw=0.4, pitch=0.3, distance=120.0)
    o, d = cam.ray(x, y)
    px = cam.project([o + d * 57.0])[0]
    assert np.allclose(px, (x, y), atol=1e-6)


def test_frame_keeps_every_corner_of_the_box_on_screen():
    cam = Camera.default()
    lo, hi = np.array([-40.0, -34.0, 0.0]), np.array([40.0, 34.0, 22.0])
    cam.frame(lo, hi)
    corners = [[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]
    px = cam.project(corners)
    assert np.all(px[:, 0] >= 0) and np.all(px[:, 0] <= cam.width)
    assert np.all(px[:, 1] >= 0) and np.all(px[:, 1] <= cam.height)


def test_orbit_never_tips_over_the_top():
    cam = Camera.default()
    cam.orbit(0.0, 100000.0)
    assert cam.pitch < np.pi / 2
    cam.orbit(0.0, -100000.0)
    assert cam.pitch > -np.pi / 2


# --- sync ---------------------------------------------------------------------------


def test_the_first_flush_sends_the_palette_and_every_model_whole():
    host = RecordingHost()
    scene = open_scene(BIRD)
    GpuSync(host).flush(scene)
    assert len(host.named("set_palette")) == 1
    (_name, (idx, voxels, position), _kw), = host.named("set_model")
    assert idx == 0 and voxels.shape == (80, 68, 22)
    assert position == tuple(float(p) for p in scene.models[0].position)


def test_one_changed_voxel_sends_only_its_chunk():
    host = RecordingHost()
    scene = open_scene(BIRD)
    sync = GpuSync(host)
    sync.flush(scene)
    host.clear()
    scene.models[0].grid.set_voxel(70, 5, 3, 9)
    sync.flush(scene)
    assert [c[0] for c in host.calls] == ["update_region"]
    _name, (idx, origin, box), _kw = host.calls[0]
    assert idx == 0 and origin == (64, 0, 0)
    assert box[70 - 64, 5, 3] == 9
    assert np.array_equal(box, scene.models[0].grid.voxels[64:80, 0:32, 0:22])


def test_a_flush_without_changes_sends_nothing():
    host = RecordingHost()
    scene = open_scene(BIRD)
    sync = GpuSync(host)
    sync.flush(scene)
    host.clear()
    assert sync.flush(scene) is False
    assert host.calls == []


def test_a_new_grid_size_sends_the_model_whole_again():
    host = RecordingHost()
    scene = VoxScene.single(VoxelGrid.empty((16, 16, 16)))
    sync = GpuSync(host)
    sync.flush(scene)
    host.clear()
    scene.models[0].grid.replace_voxels(np.ones((8, 9, 10), dtype=np.uint8))
    sync.flush(scene)
    (_name, (idx, voxels, _pos), _kw), = host.named("set_model")
    assert idx == 0 and voxels.shape == (8, 9, 10)


