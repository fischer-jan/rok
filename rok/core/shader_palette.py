"""The palette a shader sees, built new for each run, and the way back.

xs shaders were written for a palette of 255 numbered colours: the default
palette, a run of selected colours next to each other (`color_sel`), and the
colour you paint with (`i_color_index`). Rok keeps your colours in its own
store, in any order, so each run gets a palette of that kind:

- every colour the model's voxels use, on the slot it has in your palette
  (moved only when the run needs that place);
- the run of colours: the shader's own colours (its rok_colours line, after
  the marker when it has one), or, for a shader that asks for a colour list
  without own colours, a ramp of shades of the current colour, dark to light;
- your unused colours, as many as there are places left;
- every other slot: the default palette. A shader that looks for a darker or
  lighter shade of a colour finds it there.

After the run, `back` translates the result: a slot that holds one of your
colours goes back to it; any other colour the result uses (a shade from the
default palette, an own colour, a ramp shade) is a new colour for your
palette. numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colour_store import ramp, rgba
from .palette import DEFAULT_PALETTE

SLOTS = 256


class NoRoom(ValueError):
    """The model holds so many colours that the run's colour list has no place."""


@dataclass
class ShaderPalette:
    palette: np.ndarray  # (256, 4): what the shader sees
    to_shader: np.ndarray  # (256,) uint8: your slot -> the shader's
    colour: int  # the shader's slot of the colour you paint with (i_color_index)
    run: list[int]  # the shader's slots of the colour list, in order (color_sel)
    from_shader: dict[int, int] = field(default_factory=dict)  # the shader's slot -> your slot
    materials: dict = field(default_factory=dict)  # the shader's slot -> the material of its own colour there
    new_materials: dict = field(default_factory=dict)  # after `back`: your slot -> a new colour's material

    def to_run(self, voxels: np.ndarray) -> np.ndarray:
        """The model as the shader reads it."""
        return self.to_shader[np.asarray(voxels, dtype=np.uint8)]

    def back(self, result: np.ndarray, free_slot) -> tuple[np.ndarray, dict[int, tuple]]:
        """The result in your slots, and the new colours it needs: {slot: rgba}.
        `free_slot(colour, hint)` gives the slot for a colour that is not
        one of yours as it stands (an existing slot of exactly that colour, or
        a free one, slot `hint` first: a shader that returns a number keeps
        it where it can); it raises when there is no room."""
        lut = np.zeros(SLOTS, dtype=np.uint8)
        writes: dict[int, tuple] = {}
        present = np.bincount(np.asarray(result, dtype=np.uint8).ravel(), minlength=SLOTS) > 0
        for k in np.flatnonzero(present):
            k = int(k)
            if k == 0:
                continue
            if k in self.from_shader:
                lut[k] = self.from_shader[k]
                continue
            colour = rgba(self.palette[k])
            material = self.materials.get(k)
            slot = free_slot(colour, k) if material is None else free_slot(colour, k, material)
            lut[k] = slot
            if slot not in writes and slot not in self.from_shader.values():
                writes[slot] = colour
                if material is not None:
                    self.new_materials[slot] = material
        return lut[np.asarray(result, dtype=np.uint8)], writes


def build(palette: np.ndarray, voxels: np.ndarray, colour: int, *, own=None, marker: bool = False,
          count: int = 1, unused=(), own_materials=()) -> ShaderPalette:
    """The palette for one run on `voxels` (your slots) with the colour you
    paint with on slot `colour`. `own`: the shader's own colours, or None;
    `marker`: they come after the current colour (a marker shader). Without
    own colours, `count` > 1 makes the colour list a ramp of the current
    colour with that many shades. `unused`: your unused colours, as
    (colour, your slot or None) pairs; each takes its own slot when that is
    free, else any free one. `own_materials[k]`: own colour k's material
    (glass), None for plain."""
    palette = np.asarray(palette)
    colour = int(colour)
    used = np.bincount(np.asarray(voxels, dtype=np.uint8).ravel(), minlength=SLOTS) > 0
    used[0] = False
    current = rgba(palette[colour])
    if own is not None:
        listed = ([current] if marker else []) + [rgba(c) for c in own]
    elif count > 1:
        listed = list(ramp(current, count))
    else:
        listed = [current]
    out = np.array(DEFAULT_PALETTE, copy=True)
    # The run: the first places in a row that no voxel of the model holds, so
    # the model's colours keep their own slots.
    length = len(listed)
    start = next((s for s in range(1, SLOTS - length + 1) if not used[s:s + length].any()), None)
    if start is None:
        start = 1
    run = list(range(start, start + length))
    to_shader = np.arange(SLOTS, dtype=np.uint8)
    from_shader: dict[int, int] = {}
    taken = set(run)
    spare = [s for s in range(SLOTS - 1, 0, -1) if s not in taken and not used[s]]
    if int(used.sum()) + length > SLOTS - 1:
        raise NoRoom(f"the model holds {int(used.sum())} colours, and the shader needs {length} more places")
    for slot in np.flatnonzero(used):
        slot = int(slot)
        place = slot
        if place in taken:
            place = spare.pop(0)  # only when the model holds every place of a long run
        to_shader[slot] = place
        out[place] = palette[slot]
        from_shader[place] = slot
        taken.add(place)
    for k, value in enumerate(listed):
        out[run[k]] = value
    places = [s for s in range(1, SLOTS) if s not in taken]
    for value, slot in unused:
        place = slot if slot in places else (places[0] if places else None)
        if place is None:
            break
        out[place] = rgba(value)
        places.remove(place)
    # The current colour is the shader's i_color_index and, in a ramp or after
    # a marker, one of the run: its voxels read as that slot.
    if own is None or marker:
        at = listed.index(current)
        if used[colour]:
            from_shader.pop(int(to_shader[colour]), None)
        to_shader[colour] = run[at]
        from_shader[run[at]] = colour
        shader_colour = run[at]
    else:
        shader_colour = run[0]
    to_shader[0] = 0
    materials = {}
    if own is not None:
        first = 1 if marker else 0
        for k, material in enumerate(own_materials):
            if material is not None and first + k < len(run):
                materials[run[first + k]] = material
    return ShaderPalette(out, to_shader, shader_colour, run, from_shader, materials)
