"""The one path from the grids to the GPU.

`GpuSync.flush` compares the scene with what the host already has and sends
only the difference: whole models when the model list or a grid size changed,
otherwise the dirty chunks of each grid, and the positions and the palette.
A brick model's seams field (rok.core.brick) goes with its dirty chunks,
so the seams follow every change.
Tools, undo, redo and file loads all end with a flush; nothing else writes to
the host's models.
"""

from __future__ import annotations

import numpy as np

from rok.core.material import render_rows
from rok.core.scene import VoxScene


class GpuSync:
    def __init__(self, host) -> None:
        self.host = host
        self.uids: list[int] = []
        self.sizes: list[tuple[int, int, int]] = []
        self.positions: list[tuple[int, int, int]] = []
        self.palette: bytes = b""
        self.materials: bytes = b""
        self.styles: list = []
        # Per model: the pose key of the stand-in grid the host shows instead
        # of the model's own (Animate mode's Voxel look), or None.
        self.shown: list = []
        # Indices whose GPU model was replaced this flush. The CAD field was
        # fitted to the old one, so the view uploads it again.
        self.replaced: set[int] = set()
        self.seams: list[bool] = []  # per model: the host shows a seams field

    def reset(self) -> None:
        """Forget what the host has: the next flush sends everything."""
        for i in reversed(range(len(self.uids))):
            self.host.remove_model(i)
        self.uids, self.sizes, self.positions, self.styles, self.shown = [], [], [], [], []
        self.seams = []
        self.palette = b""
        self.materials = b""
        self.replaced = set()

    def flush(self, scene: VoxScene, style_of=None, turns=None, poses=None, palette=None) -> bool:
        """Send what changed. Returns True when anything was sent. `style_of(model)`
        gives (fade, hidden); `turns` maps uid -> 4 x 4 turn for a rotate preview.
        `poses` maps uid -> (key, voxels, position): a grid the host shows in
        place of the model's own, until the uid leaves `poses` (Animate mode).
        `voxels` None keeps the model's own voxels at `position`. `palette`,
        when given, shows instead of the scene's (a colour motion at the play
        head); None shows the scene's own again."""
        poses = poses or {}
        self.replaced = set()
        sent = False
        palette = np.ascontiguousarray(scene.palette if palette is None else palette, dtype=np.uint8)
        if palette.tobytes() != self.palette:
            self.host.set_palette(palette)
            self.palette = palette.tobytes()
            sent = True
        # The materials, for the rendered view (glass, metal, light).
        rows = render_rows(scene.materials)
        if rows.tobytes() != self.materials and hasattr(self.host, "set_materials"):
            self.host.set_materials(rows)
            self.materials = rows.tobytes()
            sent = True
        uids = [m.uid for m in scene.models]
        if uids != self.uids:
            for i in reversed(range(len(uids), len(self.uids))):
                self.host.remove_model(i)
            self.uids, self.sizes, self.positions, self.styles, self.shown = [], [], [], [], []
            self.seams = []
            for i, model in enumerate(scene.models):
                self._send_whole(i, model)
            self._send_poses(scene, poses)
            self._send_styles(scene, style_of, turns)
            return True
        for i, model in enumerate(scene.models):
            grid = model.grid
            if self._send_pose(i, model, poses.get(model.uid)):
                sent = True
                continue
            if tuple(grid.size) != self.sizes[i]:
                self._send_whole(i, model)
                sent = True
                continue
            dirty = grid.dirty_chunks()
            if self.seams[i] and grid.bricks is None:
                self.host.clear_seams(i)  # a voxel model again
                self.seams[i] = False
                sent = True
            if dirty:
                visible = grid.visible
                field = grid.bricks.field(grid.voxels) if grid.bricks is not None else None
                for chunk in sorted(dirty):
                    xs, ys, zs = grid.chunk_bounds(chunk)
                    box = np.ascontiguousarray(visible[xs, ys, zs])
                    self.host.update_region(i, (xs.start, ys.start, zs.start), box)
                    if field is not None:
                        self.host.set_seams(i, (xs.start, ys.start, zs.start), np.ascontiguousarray(field[xs, ys, zs]))
                if field is not None:
                    self.seams[i] = True
                grid.clear_dirty()
                sent = True
            pose = poses.get(model.uid)
            position = tuple(pose[2]) if pose is not None else tuple(model.position)  # a move-only pose
            if position != self.positions[i]:
                self.host.set_model_position(i, tuple(float(p) for p in position))
                self.positions[i] = position
                sent = True
        return self._send_styles(scene, style_of, turns) or sent

    def _send_poses(self, scene, poses) -> None:
        for i, model in enumerate(scene.models):
            self._send_pose(i, model, poses.get(model.uid))

    def _send_pose(self, i: int, model, pose) -> bool:
        """Show a stand-in grid for model i, or go back to its own. True when
        the model is posed (its own grid's changes wait until it is not)."""
        if pose is None:
            if self.shown[i] is not None:
                self.shown[i] = None
                self._send_whole(i, model)
            return False
        key, voxels, position = pose
        position = tuple(position)
        if voxels is None:
            if self.shown[i] is not None:  # from a stand-in back to the own voxels
                self._send_whole(i, model)
            self.shown[i] = None
            if position != self.positions[i]:
                self.host.set_model_position(i, tuple(float(p) for p in position))
                self.positions[i] = position
            return False
        if key != self.shown[i]:
            self.replaced.add(i)
            # A posed stand-in shows no seams: its cells are turned, the field is not.
            self.host.set_model(i, np.ascontiguousarray(voxels, dtype=np.uint8), tuple(float(p) for p in position))
            self.seams[i] = False
            self.sizes[i] = tuple(voxels.shape)
            self.positions[i] = position
            self.shown[i] = key
        # Posed, new or not: the model's own grid waits. (Returning False for an
        # unchanged pose let the size check below send the rest grid right after
        # the pose: the view snapped back to frame 0 on every step.)
        return True

    def _send_styles(self, scene, style_of, turns) -> bool:
        sent = False
        turns = turns or {}
        for i, model in enumerate(scene.models):
            style = (style_of(model) if style_of else (0.0, False))
            turn = turns.get(model.uid)
            key = (style, None if turn is None else np.asarray(turn, dtype=np.float64).tobytes())
            if self.styles[i] != key:
                self.host.set_model_style(i, float(style[0]), bool(style[1]))
                if self.styles[i] is None or self.styles[i][1] != key[1]:
                    self.host.set_model_turn(i, None if turn is None else np.asarray(turn, dtype=np.float64))
                self.styles[i] = key
                sent = True
        return sent

    def _send_whole(self, i: int, model) -> None:
        grid = model.grid
        self.replaced.add(i)
        self.host.set_model(i, np.ascontiguousarray(grid.visible), tuple(float(p) for p in model.position))
        grid.clear_dirty()
        has_seams = grid.bricks is not None
        if has_seams:
            self.host.set_seams(i, (0, 0, 0), np.ascontiguousarray(grid.bricks.field(grid.voxels)))
        if i < len(self.uids):
            self.uids[i], self.sizes[i], self.positions[i] = model.uid, tuple(grid.size), tuple(model.position)
            self.styles[i] = None
            self.seams[i] = has_seams
        else:
            self.uids.append(model.uid)
            self.sizes.append(tuple(grid.size))
            self.positions.append(tuple(model.position))
            self.styles.append(None)
            self.shown.append(None)
            self.seams.append(has_seams)
