"""The colour store: entries with stable ids, leases on slots, one writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rok.core import colour_store as cs
from rok.core.material import DEFAULT, Material, MaterialTable
from rok.core.palette import DEFAULT_PALETTE, default_palette
from rok.core.vox_read import read_scene

ROBOT = Path(__file__).parent / "fixtures" / "robot.vox"
GLASS = Material(((b"_type", b"_glass"),))


def _robot():
    data = read_scene(ROBOT)
    palette = np.array(data.palette, dtype=np.uint8, copy=True)
    used = np.zeros(256, dtype=np.int64)
    for model in data.models:
        used += np.bincount(model.voxels.ravel(), minlength=256)
    used[0] = 0
    return palette, data.materials.copy(), used


def _fresh(*used_slots):
    used = np.zeros(256, dtype=np.int64)
    for s in used_slots:
        used[s] = 1
    store = cs.ColourStore.from_palette(default_palette(), MaterialTable(), used)
    return store, used


def test_a_file_becomes_used_entries_on_their_slots_then_its_own_unused_colours():
    palette, materials, used = _robot()
    store = cs.ColourStore.from_palette(palette, materials, used)
    used_slots = [s for s in range(1, 256) if used[s]]
    assert [store.slot_of(i) for i in store.used_ids(used)] == used_slots
    for slot in used_slots:
        assert store.entries[store.id_of(slot)].rgba == tuple(int(c) for c in palette[slot])
    for ident in store.unused_ids(used):
        entry = store.entries[ident]
        assert store.slot_of(ident) is None
        assert not any(np.array_equal(entry.rgba, DEFAULT_PALETTE[s]) and not used[s] and palette[s].tolist() ==
                       DEFAULT_PALETTE[s].tolist() for s in range(1, 256) if palette[s].tolist() == list(entry.rgba))


def test_a_default_palette_brings_no_unused_colours():
    store, used = _fresh(1)
    assert len(store.order) == 1


def test_new_colours_get_their_own_free_slots_and_drop_old_materials():
    store, used = _fresh(1, 2)
    store.materials.set(3, GLASS)
    plan = store.plan([(10, 20, 30), (40, 50, 60), (10, 20, 30)], used, pinned=(4,))
    ids = store.commit(plan)
    assert ids[0] == ids[2] != ids[1]
    slots = [store.slot_of(i) for i in ids[:2]]
    assert len(set(slots)) == 2 and not set(slots) & {1, 2, 4}
    for ident, slot in zip(ids[:2], slots):
        assert tuple(store.palette[slot]) == store.entries[ident].rgba
        assert store.materials.get(slot) == DEFAULT


def test_a_colour_already_in_the_store_is_reused():
    store, used = _fresh(5)
    red = store.entries[store.id_of(5)].rgba
    plan = store.plan([red], used)
    assert plan.ids == [store.id_of(5)] and not plan.new and not plan.leases


def test_another_material_or_alpha_is_another_colour():
    store, used = _fresh(5)
    red = store.entries[store.id_of(5)].rgba
    plan = store.plan([(red, GLASS), (*red[:3], 0)], used)
    assert len(plan.new) == 2


def test_an_unused_colour_keeps_its_entry_when_its_slot_is_taken_back():
    store, used = _fresh()
    first = store.commit(store.plan([(1, 2, 3)], used))[0]
    slot = store.slot_of(first)
    used_now = np.ones(256, dtype=np.int64)
    used_now[slot] = 0  # every other slot holds voxels; `first` has none
    second = store.commit(store.plan([(9, 9, 9)], used_now))[0]
    assert store.slot_of(second) == slot
    assert store.slot_of(first) is None
    assert store.entries[first].rgba == (1, 2, 3, 255)


def test_the_colour_you_paint_with_keeps_its_slot():
    store, used = _fresh()
    current = store.commit(store.plan([(1, 2, 3)], used))[0]
    slot = store.slot_of(current)
    used_now = np.ones(256, dtype=np.int64)
    used_now[slot] = 0
    plan = store.plan([(9, 9, 9)], used_now, pinned=(slot,))
    assert plan.short == 1


def test_a_full_store_merges_to_the_nearest_colour_of_the_same_kind():
    store, used = _fresh(*range(1, 256))
    plan = store.plan([(250, 1, 1)], used)
    assert plan.short == 1
    with pytest.raises(ValueError):
        store.commit(plan)
    store.merge_short(plan, used)
    (ident,) = store.commit(plan)
    assert store.entries[ident].rgba[:3] == (255, 0, 0)


def test_a_lease_for_the_colour_you_paint_with():
    store, used = _fresh(1)
    ident = store._add(cs.Entry((7, 7, 7, 255)))
    slot = store.lease(ident, used)
    assert slot is not None and slot != 1 and tuple(store.palette[slot]) == (7, 7, 7, 255)
    assert store.lease(ident, used) == slot


def test_editing_a_colour_recolours_its_slot_and_moving_it_changes_nothing_else():
    store, used = _fresh(5, 6)
    a = store.id_of(5)
    store.edit(a, (1, 1, 1), GLASS)
    assert tuple(store.palette[5]) == (1, 1, 1, 255) and store.materials.get(5) == GLASS
    before = store.palette.copy()
    store.move(a, 1)
    assert store.order[1] == a and np.array_equal(store.palette, before)


def test_only_unused_colours_can_be_removed():
    store, used = _fresh(5)
    spare = store.commit(store.plan([(1, 2, 3)], used))[0]
    assert store.remove([store.id_of(5), spare], used) == [spare]
    assert spare not in store.entries


def test_a_snapshot_brings_everything_back():
    store, used = _fresh(5)
    snap = store.snapshot()
    before = store.palette.copy()
    store.edit(store.id_of(5), (1, 1, 1))
    store.commit(store.plan([(2, 2, 2)], used))
    store.restore(snap)
    assert np.array_equal(store.palette[5], before[5])
    assert len(store.order) == 1


def test_the_store_survives_json():
    store, used = _fresh(5)
    store.edit(store.id_of(5), material=GLASS, name="Window")
    store.commit(store.plan([(1, 2, 3)], used))
    text = json.dumps(store.to_json())
    palette, materials = default_palette(), MaterialTable()
    back = cs.ColourStore.from_json(json.loads(text), palette, materials)
    assert back.entries == store.entries and back.order == store.order and back.id_at == store.id_at
    assert materials.get(5) == GLASS and np.array_equal(palette, store.palette)


def test_a_ramp_holds_the_colour_and_runs_dark_to_light():
    base = (180, 90, 40)
    shades = cs.ramp(base, 5)
    assert len(shades) == 5 and (*base, 255) in shades
    lightness = [cs.oklab(s)[0] for s in shades]
    assert lightness == sorted(lightness)
    assert cs.ramp(base, 1) == [(*base, 255)]


def test_oklab_round_trips():
    for colour in ((0, 0, 0), (255, 255, 255), (200, 30, 90), (12, 140, 230)):
        assert cs.from_oklab(cs.oklab(colour))[:3] == pytest.approx(colour, abs=1)


def test_a_vox_save_holds_only_the_used_colours_in_your_order_and_the_scene_stays():
    from rok.core import rokfile

    scene, _doc = rokfile.read(Path(__file__).parent / "fixtures" / "palette" / "frame_only.rok")
    store = scene.colours
    used_ids = store.used_ids(scene.used_counts())
    store.move(used_ids[-1], 0)  # the last used colour first
    before = scene.palette.copy()
    data = scene.to_vox()
    assert np.array_equal(scene.palette, before)  # the scene does not change
    order = store.used_ids(scene.used_counts())
    assert [cs.rgba(data.palette[k]) for k in range(1, len(order) + 1)] == [store.entries[i].rgba for i in order]
    for model, saved in zip(scene.models, data.models):
        assert np.array_equal(data.palette[saved.voxels], scene.palette[model.grid.voxels])
        for state, (_frame, voxels) in zip(model.states, saved.states):
            assert np.array_equal(data.palette[voxels], scene.palette[state])  # frames keep their colours
    assert np.array_equal(data.palette[len(order) + 1:], DEFAULT_PALETTE[len(order) + 1:])


def test_a_slot_painted_before_the_store_knew_it_becomes_a_colour():
    store, used = _fresh()
    used[9] = 3
    assert store.adopt_used(used)
    assert store.entries[store.id_of(9)].rgba == cs.rgba(DEFAULT_PALETTE[9])


def test_a_colour_motion_keeps_its_colour_on_its_slot():
    from rok.app.object import document
    from rok.core import anim
    pytest.importorskip("rok_native")  # Session needs the native module
    from rok.headless import Session

    s = Session()
    try:
        s.run("rok.new", size=tuple([8, 8, 8]), discard=True)
        store = document.scene.colours
        glow = store.add((250, 200, 40))
        slot = store.lease(glow, document.used_counts())
        document.scene.timeline.tracks.append(anim.Track(anim.Target(anim.COLOUR, slots=(slot,))))
        assert slot in document.pinned()
        assert slot not in store.free_slots(document.used_counts(), document.pinned())
    finally:
        s.close()
