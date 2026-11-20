"""One rule for free palette slots (rok/core/colours.py) and the commands that use it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rok.core import colours, rokfile, recolour, xs
from rok.core.material import DEFAULT, MaterialTable, parse_matl
from rok.core.palette import DEFAULT_PALETTE, default_palette, merge_claimed

FRAME_ONLY = Path(__file__).parent / "fixtures" / "palette" / "frame_only.rok"  # slot 200 only in Eyes' frame 2


def _used(*slots) -> np.ndarray:
    used = np.zeros(256, dtype=np.int64)
    for s in slots:
        used[s] = 1
    return used


def _glass() -> object:
    import struct

    def chunk(k: bytes, v: bytes) -> bytes:
        return struct.pack("<i", len(k)) + k + struct.pack("<i", len(v)) + v

    content = struct.pack("<ii", 1, 1) + chunk(b"_type", b"_glass")
    return parse_matl(content)[1]


def test_a_colour_only_a_frame_uses_counts_as_used():
    scene, _doc = rokfile.read(FRAME_ONLY)
    assert int(scene.used_counts()[200]) > 0
    free = colours.free_slots(scene.palette, scene.used_counts())
    assert 200 not in free


def test_two_colours_of_one_command_never_share_a_slot():
    palette = default_palette()
    old = tuple(int(c) for c in palette[255])  # an unused slot already holds it
    placed = colours.fit(palette, _used(1), [(old, None), ((1, 2, 3), None)])
    assert placed.slots[0] == 255
    assert placed.slots[1] not in (0, 255)
    assert placed.writes == {placed.slots[1]: ((1, 2, 3, 255), DEFAULT)}


def test_a_colour_given_twice_takes_one_slot():
    placed = colours.fit(default_palette(), _used(1), [((9, 9, 9), None), ((9, 9, 9), None)])
    assert placed.slots[0] == placed.slots[1]
    assert len(placed.writes) == 1


def test_pinned_and_used_slots_are_never_given_out():
    used = np.ones(256, dtype=np.int64)
    used[[10, 11, 12]] = 0
    placed = colours.fit(default_palette(), used, [((1, 2, 3), None)], pinned=(10, 11))
    assert placed.slots == [12]


def test_a_full_palette_raises_or_merges_to_the_nearest_colour():
    palette = default_palette()
    used = np.ones(256, dtype=np.int64)
    with pytest.raises(colours.Full):
        colours.fit(palette, used, [((250, 0, 0), None)])
    placed = colours.fit(palette, used, [((250, 0, 0), None)], merge=True)
    assert placed.merged == [0] and not placed.writes
    assert tuple(palette[placed.slots[0]][:3]) == (255, 0, 0)


def test_a_colour_of_another_alpha_or_material_is_not_reused():
    palette = default_palette()
    palette[20] = (5, 5, 5, 0)
    palette[21] = (6, 6, 6, 255)
    materials = MaterialTable()
    materials.set(21, _glass())
    placed = colours.fit(palette, _used(20, 21), [((5, 5, 5), None), ((6, 6, 6), None)], materials=materials)
    assert 20 not in placed.slots and 21 not in placed.slots


def test_a_new_colour_drops_the_glass_of_its_slot():
    palette = default_palette()
    materials = MaterialTable()
    materials.set(30, _glass())
    assert colours.write(palette, materials, {30: (1, 2, 3)})
    assert materials.get(30) == DEFAULT and tuple(palette[30]) == (1, 2, 3, 255)


def test_free_slots_at_their_default_colour_come_first():
    palette = default_palette()
    palette[1] = (1, 1, 1, 255)
    free = colours.free_slots(palette, _used())
    assert free[0] == 2 and free[-1] == 1
    assert np.array_equal(palette[2], DEFAULT_PALETTE[2])


def test_a_recolour_gives_each_colour_its_own_slot():
    palette = default_palette()
    palette[255] = (7, 7, 7, 255)
    palette[2] = (50, 0, 0, 255)
    palette[3] = (60, 0, 0, 255)
    before = np.zeros((2, 1, 1), dtype=np.uint8)
    after = np.array([2, 3], dtype=np.uint8).reshape(2, 1, 1)
    new, writes = recolour.apply(before, after, palette, {(50, 0, 0): (40, 40, 40), (60, 0, 0): (7, 7, 7)},
                                 _used(1))
    assert new[0, 0, 0] != new[1, 0, 0]
    assert int(new[1, 0, 0]) == 255 and 255 not in writes


def test_a_merge_keeps_the_metal_of_an_incoming_colour():
    palette = default_palette()
    palette[5] = (200, 0, 0, 255)
    incoming = default_palette()
    incoming[5] = (200, 0, 0, 255)
    theirs = MaterialTable()
    theirs.set(5, _glass())
    _merged, remap, claimed_by = merge_claimed(palette, _used(5), incoming, _used(5), materials=MaterialTable(),
                                               incoming_materials=theirs)
    assert remap[5] != 5
    assert claimed_by[remap[5]] == 5


def test_a_merge_never_takes_a_colour_you_made():
    incoming = default_palette()
    incoming[40] = (1, 2, 3, 255)
    _merged, remap, _ = merge_claimed(default_palette(), _used(), incoming, _used(40), keep=(40,))
    assert remap[40] != 40
