"""Linked copies: models that share one shape.

A Shift-drag in the scene leaves a linked copy. Every model of a link group
shows the same shape, each with its own place and its own turn and flip;
an edit to the voxels of one (paint, stretch, fit) shows in all of them.
Make unique takes a model out of its group.

Each model keeps its own voxels, turned as it is shown, so every tool works
on it as on any other model. `SceneModel.orient` says how: the signed
permutation that turns the group's shared shape (the "source") into this
model's voxels. After a change to one member, `sync` rebuilds the others
from it. A .vox file stores the same thing natively: one model, several
nSHP nodes, one _r rotation each.
"""

from __future__ import annotations

import numpy as np

from . import brick, grid_ops

Cell = tuple[int, int, int]
Orient = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

IDENTITY: Orient = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def matrix(orient) -> np.ndarray:
    return np.asarray(orient, dtype=np.int64).reshape(3, 3)


def as_orient(R) -> Orient:
    m = matrix(R)
    return tuple(tuple(int(v) for v in row) for row in m)  # type: ignore[return-value]


def turned(orient, T) -> Orient:
    """The orient after the model turned or flipped by T in the world."""
    return as_orient(matrix(T) @ matrix(orient))


def turn_matrix(axis: int, turns: int) -> np.ndarray:
    """k quarter turns about `axis`, as grid_ops.rotate_model turns."""
    return np.linalg.matrix_power(grid_ops._QUARTER[int(axis)], int(turns) % 4)


def transform_matrix(kind: str, axis: int, turns: int = 1) -> np.ndarray:
    """The world matrix of a gizmo turn ("ROTATE") or flip ("FLIP")."""
    return turn_matrix(axis, turns) if kind == "ROTATE" else flip_matrix(axis)


def flip_matrix(axis: int) -> np.ndarray:
    m = np.eye(3, dtype=np.int64)
    m[axis, axis] = -1
    return m


def to_source(voxels: np.ndarray, orient) -> np.ndarray:
    """The group's shared shape from one member's voxels."""
    return grid_ops.apply_rotation(voxels, matrix(orient).T)[0]


def from_source(source: np.ndarray, orient) -> np.ndarray:
    """A member's voxels from the shared shape."""
    return grid_ops.apply_rotation(source, matrix(orient))[0]


def source_size(size, orient) -> Cell:
    """The shared shape's size from a member's size."""
    out = np.abs(matrix(orient).T) @ np.asarray(size, dtype=np.int64)
    return tuple(int(v) for v in out)  # type: ignore[return-value]


def _axes(orient):
    """For each world axis: (source axis, sign)."""
    m = matrix(orient)
    return [(int(np.nonzero(m[i])[0][0]), int(m[i, np.nonzero(m[i])[0][0]])) for i in range(3)]


def source_shift(d, orient, old_size, new_size) -> Cell:
    """How far the shared shape's low corner moved, in its own cells, when a
    member's box moved by `d` (world cells) and the shape went from
    `old_size` to `new_size` (source cells). A member's low corner on a world
    axis is the source's low end when the axis runs the same way, else the
    source's high end: that is where the sign comes in."""
    s = [0, 0, 0]
    for i, (a, sign) in enumerate(_axes(orient)):
        s[a] = int(d[i]) if sign > 0 else int(old_size[a]) - int(new_size[a]) - int(d[i])
    return (s[0], s[1], s[2])


def placed_shift(s, orient, old_size, new_size) -> Cell:
    """How far a member's box moves in the world when the shared shape's low
    corner moved by `s` and its size changed (the inverse of source_shift)."""
    d = [0, 0, 0]
    for i, (a, sign) in enumerate(_axes(orient)):
        d[i] = int(s[a]) if sign > 0 else int(old_size[a]) - int(new_size[a]) - int(s[a])
    return (d[0], d[1], d[2])


def members(scene, link: int) -> list:
    """The models of one link group, in scene order."""
    if not link:
        return []
    return [m for m in scene.models if m.link == link]


def is_linked(scene, model) -> bool:
    return len(members(scene, model.link)) > 1


def sync(scene, uid: int, d=(0, 0, 0)) -> list[int]:
    """Rebuild the other members of model `uid`'s group from it, after its
    voxels changed and its box moved by `d`. Returns the uids that changed.

    Only the cells that differ are written, so an edit rebuilds only the
    chunks it touched in each copy."""
    from .select import Selection

    source_model = scene.by_uid(uid)
    others = [m for m in members(scene, source_model.link) if m.uid != uid]
    if not others:
        return []
    source = to_source(source_model.grid.voxels, source_model.orient)
    new_size = tuple(int(v) for v in source.shape)
    old_size = source_size(others[0].grid.size, others[0].orient)
    s = source_shift(d, source_model.orient, old_size, new_size)
    changed = []
    for other in others:
        voxels = from_source(source, other.orient)
        grid = other.grid
        # A brick table follows the voxels; the copy mirrors an allowed change.
        retabled = _follow_table(source_model, other)
        with brick.unlocked():
            if voxels.shape == tuple(grid.size):
                cells = np.argwhere(grid.voxels != voxels)
                if len(cells) == 0:
                    if retabled:
                        changed.append(other.uid)
                    continue
                grid.set_voxels(cells, voxels[cells[:, 0], cells[:, 1], cells[:, 2]],
                                ignore_cut=True, ignore_lock=True)
            else:
                grid.replace_voxels(voxels)
                other.selection = Selection(grid.size)
        move = placed_shift(s, other.orient, old_size, new_size)
        other.position = tuple(int(p) + m for p, m in zip(other.position, move))
        changed.append(other.uid)
    return changed


def _follow_table(source_model, other) -> bool:
    """Give `other` the brick table of `source_model`, turned by their
    orients (or none when the source has none). True when it changed."""
    table = source_model.grid.bricks
    if table is not None:
        try:
            table = brick.member_table(table, source_model.grid.size, source_model.orient, other.orient)
        except ValueError:
            table = None  # a copy turned on its side cannot hold bricks; Convert refuses that group
    old = other.grid.bricks
    if old == table or (old is None and table is None):
        return False
    other.grid.bricks = table
    other.grid.mark_all_dirty()
    return True


def voxel_changes(record, undone: bool) -> list[tuple[int, Cell]]:
    """The models whose voxels a record changed, with how far each one's box
    moved: [(uid, d)], in the order the record applied them."""
    from .undo import BrickRecord, CompoundRecord, ModelRecord, SceneRecord

    out: list[tuple[int, Cell]] = []

    def walk(rec) -> None:
        if rec is None:
            return
        if isinstance(rec, CompoundRecord):
            for part in (reversed(rec.parts) if undone else rec.parts):
                walk(part)
        elif isinstance(rec, SceneRecord):
            walk(rec.inner)
        elif isinstance(rec, BrickRecord):
            out.append((rec.uid, (0, 0, 0)))
        elif isinstance(rec, ModelRecord) and rec.inner is not None and rec.orient_after is None:
            # (A turn or a flip in the scene changes this copy alone: its
            # orient changed with its voxels, the shared shape did not.)
            d = (0, 0, 0)
            if rec.position_before is not None and rec.position_after is not None:
                a, b = (rec.position_after, rec.position_before) if undone else (rec.position_before, rec.position_after)
                d = tuple(int(y) - int(x) for x, y in zip(a, b))
            out.append((rec.uid, d))

    walk(record)
    return out


def sync_after(scene, record, undone: bool = False) -> list[int]:
    """Bring every link group a record touched in line, from the first member
    it changed (two members of one group changed at once: the first wins).
    Returns the uids of the copies that changed."""
    done: set[int] = set()
    changed: list[int] = []
    for uid, d in voxel_changes(record, undone):
        try:
            model = scene.by_uid(uid)
        except ValueError:
            continue  # the record's model left the scene
        if not model.link or model.link in done:
            continue
        done.add(model.link)
        changed += sync(scene, uid, d)
    return changed
