"""Live shaders: a shader that stays on a model and runs again after each edit.

A live shader keeps its settings and what its last run changed (a Diff: the
cells, their values before and after). After an edit of the model, Rok
takes every live shader of the model back, the last one first, and runs the
ones that are on again, in order, on the result. So the shaders always work
on your voxels, and your edits stay under them.

Taking a shader back writes each cell's value from before the run, but only
where the cell still holds what the shader wrote. A cell you changed since
keeps your value. A shader you bake keeps its voxels and forgets the rest; a
shader you remove is taken back first.

In Rok's UI "layer" means a layer of the .vox file (rok.core.layers), so these
are "live shaders". numpy only: the app runs the shaders (rok/app/live_shaders.py).
"""

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Diff:
    """What one run changed: flat cell indices, the values before and after."""

    shape: tuple[int, int, int]
    idx: np.ndarray
    before: np.ndarray
    after: np.ndarray

    @classmethod
    def of(cls, old: np.ndarray, new: np.ndarray) -> Diff:
        flat_old, flat_new = old.ravel(), new.ravel()
        idx = np.flatnonzero(flat_old != flat_new)
        return cls(tuple(int(v) for v in old.shape), idx, flat_old[idx].copy(), flat_new[idx].copy())

    def revert(self, voxels: np.ndarray) -> np.ndarray:
        """`voxels` with this run taken back: a new array. A cell that no
        longer holds the run's value keeps its value."""
        out = np.array(voxels, dtype=np.uint8, copy=True)
        if tuple(out.shape) != self.shape or not len(self.idx):
            return out
        flat = out.reshape(-1)
        ours = flat[self.idx] == self.after
        flat[self.idx[ours]] = self.before[ours]
        return out

    def __len__(self) -> int:
        return int(len(self.idx))

    @property
    def nbytes(self) -> int:
        return int(self.idx.nbytes + self.before.nbytes + self.after.nbytes)


@dataclass(frozen=True)
class LiveShader:
    """One live shader. `settings` are rok.xs_run's keyword arguments
    (a0.., colour, colour_count, iterations, colours_rgb, scope, only_on,
    only_colours, room, volume, volume_shape, feather). `selection` is the
    selection it was made with (a mask of the model's shape), `mirror` the
    mirror axes then. `diff` is what its last run changed (None: off, or it
    has not run)."""

    id: int
    shader: str
    settings: dict = field(default_factory=dict)
    selection: np.ndarray | None = None
    mirror: tuple[bool, bool, bool] = (False, False, False)
    on: bool = True
    diff: Diff | None = None
    error: str = ""  # why the last run failed, or ""

    @property
    def nbytes(self) -> int:
        sel = 0 if self.selection is None else int(self.selection.nbytes)
        return sel + (0 if self.diff is None else self.diff.nbytes)


# A run: the shader and the voxels without it -> the voxels with it, or an
# error text (the voxels stay as they are).
Run = Callable[[LiveShader, np.ndarray], "np.ndarray | str"]


def take_back(voxels: np.ndarray, shaders: list[LiveShader]) -> np.ndarray:
    """`voxels` with every shader's last run taken back, the last one first."""
    out = np.asarray(voxels, dtype=np.uint8)
    for live in reversed(shaders):
        if live.diff is not None:
            out = live.diff.revert(out)
    return np.array(out, dtype=np.uint8, copy=True)


def restack(voxels: np.ndarray, shaders: list[LiveShader], run: Run) -> tuple[np.ndarray, list[LiveShader]]:
    """Take every shader back and run the ones that are on again, in order.
    Returns the new voxels and the shaders with their new diffs."""
    out = take_back(voxels, shaders)
    done: list[LiveShader] = []
    for live in shaders:
        if not live.on:
            done.append(dataclasses.replace(live, diff=None, error=""))
            continue
        result = run(live, out)
        if isinstance(result, str):
            done.append(dataclasses.replace(live, diff=None, error=result))
            continue
        result = np.asarray(result, dtype=np.uint8)
        done.append(dataclasses.replace(live, diff=Diff.of(out, result), error=""))
        out = result
    return out, done


def fits(live: LiveShader, shape) -> bool:
    """True when the shader still fits a model of `shape` (a resize bakes it)."""
    shape = tuple(int(v) for v in shape)
    if live.selection is not None and tuple(live.selection.shape) != shape:
        return False
    return live.diff is None or live.diff.shape == shape


@dataclass
class LiveRecord:
    """The live shaders of one model before and after a change: undo puts
    the list back (scene.live)."""

    uid: int
    before: tuple[LiveShader, ...]
    after: tuple[LiveShader, ...]

    def undo(self, scene) -> None:
        _put(scene, self.uid, self.before)

    def redo(self, scene) -> None:
        _put(scene, self.uid, self.after)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.before) + sum(s.nbytes for s in self.after)


def _put(scene, uid: int, shaders) -> None:
    if shaders:
        scene.live[uid] = list(shaders)
    else:
        scene.live.pop(uid, None)


# --- the Rok file (rok/core/rokfile.py) -------------------------------------------


def _array_json(a: np.ndarray) -> dict:
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode("ascii")}


def _array_from(d: dict) -> np.ndarray:
    raw = base64.b64decode(d["data"])
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def to_json(live: LiveShader, model_index: int) -> dict:
    """A live shader for rok.json, with its model's index."""
    out = {"model": int(model_index), "id": int(live.id), "shader": live.shader, "settings": dict(live.settings),
           "mirror": [bool(m) for m in live.mirror], "on": bool(live.on)}
    if live.selection is not None:
        out["selection"] = _array_json(np.packbits(live.selection.ravel()))
        out["selection_shape"] = list(live.selection.shape)
    if live.diff is not None:
        d = live.diff
        out["diff"] = {"shape": list(d.shape), "idx": _array_json(d.idx.astype(np.int64)),
                       "before": _array_json(d.before), "after": _array_json(d.after)}
    return out


def from_json(d: dict) -> tuple[int, LiveShader]:
    """(model index, live shader) from rok.json."""
    selection = None
    if "selection" in d:
        shape = tuple(int(v) for v in d["selection_shape"])
        bits = np.unpackbits(_array_from(d["selection"]))[: int(np.prod(shape))]
        selection = bits.astype(bool).reshape(shape)
    diff = None
    if "diff" in d:
        dd = d["diff"]
        diff = Diff(tuple(int(v) for v in dd["shape"]), _array_from(dd["idx"]), _array_from(dd["before"]),
                    _array_from(dd["after"]))
    live = LiveShader(int(d["id"]), str(d["shader"]), dict(d.get("settings", {})), selection,
                      tuple(bool(m) for m in d.get("mirror", (False, False, False))), bool(d.get("on", True)), diff)
    return int(d["model"]), live
