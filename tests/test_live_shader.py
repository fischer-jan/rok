"""Live shaders' rules (rok/core/live_shader.py): take back, rerun, and
your edits win. numpy only, with plain functions as the shaders."""

from __future__ import annotations

import numpy as np

from rok.core import live_shader as core


def snow(_live, voxels):
    """Every voxel with air above it turns 9."""
    out = voxels.copy()
    top = (voxels > 0) & np.concatenate([voxels[:, :, 1:] == 0, np.ones(voxels.shape[:2] + (1,), bool)], axis=2)
    out[top] = 9
    return out


def red(_live, voxels):
    """Every 9 turns 4."""
    return np.where(voxels == 9, 4, voxels).astype(np.uint8)


def column(height: int) -> np.ndarray:
    v = np.zeros((1, 1, 6), np.uint8)
    v[0, 0, :height] = 2
    return v


def runner(fns):
    return lambda live, voxels: fns[live.shader](live, voxels)


def test_a_rerun_follows_the_model_under_it():
    shaders = [core.LiveShader(1, "snow")]
    run = runner({"snow": snow})
    out, shaders = core.restack(column(3), shaders, run)
    assert list(out[0, 0]) == [2, 2, 9, 0, 0, 0]
    out[0, 0, 3] = 2  # you add a voxel on top
    out, shaders = core.restack(out, shaders, run)
    assert list(out[0, 0]) == [2, 2, 2, 9, 0, 0]  # the old top is yours again


def test_a_cell_you_changed_keeps_your_value():
    out, shaders = core.restack(column(3), [core.LiveShader(1, "snow")], runner({"snow": snow}))
    out[0, 0, 2] = 5  # you paint the snow
    back = core.take_back(out, shaders)
    assert list(back[0, 0]) == [2, 2, 5, 0, 0, 0]


def test_stacked_shaders_come_off_in_reverse_order():
    run = runner({"snow": snow, "red": red})
    out, shaders = core.restack(column(3), [core.LiveShader(1, "snow"), core.LiveShader(2, "red")], run)
    assert list(out[0, 0]) == [2, 2, 4, 0, 0, 0]
    assert list(core.take_back(out, shaders)[0, 0]) == [2, 2, 2, 0, 0, 0]
    # Off: its voxels go, and the next shader works without them.
    off = [shaders[0].__class__(**{**shaders[0].__dict__, "on": False}), shaders[1]]
    out, shaders = core.restack(out, off, run)
    assert list(out[0, 0]) == [2, 2, 2, 0, 0, 0] and shaders[0].diff is None


def test_a_failed_run_leaves_the_voxels_and_says_why():
    out, shaders = core.restack(column(2), [core.LiveShader(1, "bad")], lambda live, v: "It broke")
    assert list(out[0, 0]) == [2, 2, 0, 0, 0, 0] and shaders[0].error == "It broke"


def test_a_new_size_does_not_fit():
    _out, shaders = core.restack(column(3), [core.LiveShader(1, "snow")], runner({"snow": snow}))
    assert core.fits(shaders[0], (1, 1, 6)) and not core.fits(shaders[0], (2, 1, 6))
