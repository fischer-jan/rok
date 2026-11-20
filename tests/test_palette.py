"""Palette helpers: first_unused, new_colour_slot, nearest, merge, merge_claimed, swap and gradient."""

import numpy as np
import pytest

from rok.core.palette import (
    default_palette,
    first_unused,
    gradient,
    merge,
    merge_claimed,
    nearest,
    new_colour_slot,
    swap_remap,
    swap_slots,
)

from helpers import model_palette_counts


# --- first_unused ------------------------------------------------------------


def test_first_unused_skips_zero_and_exclude():
    counts = np.zeros(256, dtype=np.int64)
    assert first_unused(counts) == 1
    counts[1] = 4
    assert first_unused(counts) == 2
    assert first_unused(counts, exclude=2) == 3
    counts[1:] = 1
    assert first_unused(counts) is None
    counts[200] = 0
    assert first_unused(counts) == 200
    assert first_unused(counts, exclude=200) is None


def test_first_unused_on_robot():
    _, counts = model_palette_counts("robot")
    assert first_unused(counts) == 1
    counts[1:169] = 1
    assert first_unused(counts) == 170


# --- new_colour_slot ---------------------------------------------------------

WHITE = (255, 255, 255)


def test_new_colour_slot_keeps_a_colour_you_set_up():
    pal = default_palette()
    counts = np.zeros(256, dtype=np.int64)
    plain = np.ones(256, dtype=bool)
    pal[2, :3] = (10, 20, 30)  # set up, no voxels yet
    plain[3] = False  # a Glass slot
    # Slot 1 is white already, 2 and 3 were set up: 4 is the first one at its default.
    assert new_colour_slot(pal, counts, plain, WHITE) == 4
    assert new_colour_slot(pal, counts, plain, WHITE, exclude=(4,)) == 5


def test_new_colour_slot_takes_any_unused_slot_when_none_is_at_its_default():
    pal, counts = model_palette_counts("robot")
    plain = np.ones(256, dtype=bool)
    # Only slot 1 is at its default, and it is white already.
    assert new_colour_slot(pal, counts, plain, WHITE) == 2
    counts[1:] = 1
    assert new_colour_slot(pal, counts, plain, WHITE) is None


# --- nearest -----------------------------------------------------------------


def test_nearest_picks_the_smallest_rgb_distance():
    palette = default_palette()
    candidates = np.ones(256, dtype=bool)
    candidates[0] = False
    assert nearest(palette, (255, 255, 255), candidates) == 1
    assert nearest(palette, (250, 250, 250), candidates) == 1
    got = nearest(palette, (40, 90, 200), candidates)
    dist = ((palette[1:, :3].astype(int) - (40, 90, 200)) ** 2).sum(axis=1)
    assert dist[got - 1] == dist.min()


def test_nearest_honours_the_candidates():
    palette = default_palette()
    candidates = np.zeros(256, dtype=bool)
    candidates[[10, 20]] = True
    got = nearest(palette, (255, 255, 255), candidates)
    assert got in (10, 20)
    d10 = ((palette[10, :3].astype(int) - 255) ** 2).sum()
    d20 = ((palette[20, :3].astype(int) - 255) ** 2).sum()
    assert got == (10 if d10 <= d20 else 20)
    with pytest.raises(ValueError):
        nearest(palette, (0, 0, 0), np.zeros(256, dtype=bool))


# --- merge -------------------------------------------------------------------


def test_merge_bird_into_robot():
    base, base_used = model_palette_counts("robot")
    incoming, incoming_used = model_palette_counts("bird")
    merged, remap = merge(base, base_used, incoming, incoming_used)
    assert merged.shape == (256, 4) and merged.dtype == np.uint8
    assert remap.shape == (256,) and remap.dtype == np.uint8
    assert merged is not base
    assert remap[0] == 0
    # Free indices keep their numbers.
    for i in (81, 162, 170, 171, 178, 179, 180):
        assert remap[i] == i
        assert merged[i].tolist() == incoming[i].tolist()
    # The two shared indices have different colours, so they move to the lowest free slots.
    assert remap[169] == 1
    assert remap[177] == 2
    assert merged[1].tolist() == incoming[169].tolist()
    assert merged[2].tolist() == incoming[177].tolist()
    # Nothing the robot uses changes.
    for i in np.nonzero(base_used[1:])[0] + 1:
        assert merged[i].tolist() == base[i].tolist()
    # Every remapped colour equals the incoming colour.
    for i in np.nonzero(incoming_used[1:])[0] + 1:
        assert merged[remap[i]].tolist() == incoming[i].tolist()


def test_merge_car_into_robot_keeps_every_index():
    base, base_used = model_palette_counts("robot")
    incoming, incoming_used = model_palette_counts("car")
    merged, remap = merge(base, base_used, incoming, incoming_used)
    assert np.array_equal(remap, np.arange(256))
    for i in np.nonzero(incoming_used[1:])[0] + 1:
        assert merged[i].tolist() == incoming[i].tolist()
    for i in np.nonzero(base_used[1:])[0] + 1:
        assert merged[i].tolist() == base[i].tolist()


def test_merge_exact_colour_reuses_the_used_slot():
    base, base_used = model_palette_counts("robot")
    incoming = default_palette()
    incoming[5] = base[185]  # a colour the robot uses
    incoming[6] = base[185]  # twice
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[5] = incoming_used[6] = 1
    merged, remap = merge(base, base_used, incoming, incoming_used)
    assert remap[5] == 185
    assert remap[6] == 185
    assert np.array_equal(merged, base)


def test_merge_two_incoming_colours_share_one_claimed_slot():
    base = default_palette()
    base_used = np.zeros(256, dtype=np.int64)
    base_used[1:5] = 1
    incoming = default_palette()
    incoming[1] = (9, 9, 9, 255)
    incoming[2] = (9, 9, 9, 255)
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[1] = incoming_used[2] = 1
    merged, remap = merge(base, base_used, incoming, incoming_used)
    assert remap[1] == 5
    assert remap[2] == 5
    assert merged[5].tolist() == [9, 9, 9, 255]


def test_merge_full_palette_maps_to_the_nearest_colour():
    base = default_palette()
    base_used = np.ones(256, dtype=np.int64)
    incoming = default_palette()
    incoming[3] = (250, 250, 250, 255)  # near white, slot 1
    incoming[4] = (2, 2, 2, 255)  # near the darkest grey
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[3] = incoming_used[4] = 1
    merged, remap = merge(base, base_used, incoming, incoming_used)
    assert np.array_equal(merged, base)
    assert remap[3] == 1
    dist = ((base[1:, :3].astype(int) - (2, 2, 2)) ** 2).sum(axis=1)
    assert dist[remap[4] - 1] == dist.min()


def test_merge_ignores_unused_incoming_indices():
    base = default_palette()
    base_used = np.zeros(256, dtype=np.int64)
    incoming = np.zeros((256, 4), dtype=np.uint8)
    incoming_used = np.zeros(256, dtype=np.int64)
    merged, remap = merge(base, base_used, incoming, incoming_used)
    assert np.array_equal(merged, base)
    assert np.array_equal(remap, np.arange(256))


def test_merge_protect_keeps_the_current_colour():
    # Slot 200 holds the colour the user just made with "+": no voxel uses it yet,
    # so without `protect` an APPEND import would write over it.
    base = default_palette()
    base[200] = (7, 11, 200, 255)
    base_used = np.zeros(256, dtype=np.int64)
    base_used[1:200] = 1
    incoming = default_palette()
    incoming[200] = (1, 2, 3, 255)
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[200] = 1
    merged, remap = merge(base, base_used, incoming, incoming_used, protect=200)
    assert merged[200].tolist() == [7, 11, 200, 255]
    assert remap[200] == 201
    assert merged[201].tolist() == [1, 2, 3, 255]
    # Without the guard the colour is lost.
    plain, plain_remap = merge(base, base_used, incoming, incoming_used)
    assert plain[200].tolist() == [1, 2, 3, 255]
    assert plain_remap[200] == 200


def test_merge_protect_still_matches_an_equal_colour():
    base = default_palette()
    base[200] = (7, 11, 200, 255)
    base_used = np.zeros(256, dtype=np.int64)
    incoming = default_palette()
    incoming[9] = (7, 11, 200, 255)
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[9] = 1
    merged, remap = merge(base, base_used, incoming, incoming_used, protect=200)
    assert remap[9] == 200
    assert np.array_equal(merged, base)


def test_merge_protect_outside_the_range_does_nothing():
    base = default_palette()
    base_used = np.zeros(256, dtype=np.int64)
    incoming = default_palette()
    incoming[3] = (5, 5, 5, 255)
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[3] = 1
    for protect in (0, 256, -1):
        merged, remap = merge(base, base_used, incoming, incoming_used, protect=protect)
        assert remap[3] == 3
        assert merged[3].tolist() == [5, 5, 5, 255]


# --- merge_claimed -----------------------------------------------------------


def test_merge_claimed_bird_into_robot():
    base, base_used = model_palette_counts("robot")
    incoming, incoming_used = model_palette_counts("bird")
    merged, remap, claimed_by = merge_claimed(base, base_used, incoming, incoming_used)
    plain, plain_remap = merge(base, base_used, incoming, incoming_used)
    assert np.array_equal(merged, plain) and np.array_equal(remap, plain_remap)
    assert claimed_by.shape == (256,) and claimed_by.dtype == np.uint8
    assert claimed_by[0] == 0
    # Every slot that took a colour records the incoming index that wrote it.
    written = {}
    for i in np.nonzero(incoming_used[1:])[0] + 1:
        k = int(remap[i])
        if base_used[k] == 0 and k not in written:
            written[k] = int(i)
    assert {int(k): int(v) for k, v in enumerate(claimed_by) if v} == written
    assert claimed_by[1] == 169 and claimed_by[2] == 177 and claimed_by[81] == 81
    # Slots the robot uses are never claimed.
    for k in np.nonzero(base_used[1:])[0] + 1:
        assert claimed_by[k] == 0


def test_merge_claimed_records_the_first_index_only():
    base = default_palette()
    base_used = np.zeros(256, dtype=np.int64)
    base_used[1:5] = 1
    incoming = default_palette()
    incoming[1] = (9, 9, 9, 255)
    incoming[2] = (9, 9, 9, 255)
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[1] = incoming_used[2] = 1
    merged, remap, claimed_by = merge_claimed(base, base_used, incoming, incoming_used)
    assert remap[1] == 5 and remap[2] == 5
    assert claimed_by[5] == 1
    assert claimed_by.sum() == 1


def test_merge_claimed_exact_match_and_protect_claim_nothing():
    base, base_used = model_palette_counts("robot")
    incoming = default_palette()
    incoming[5] = base[185]
    incoming_used = np.zeros(256, dtype=np.int64)
    incoming_used[5] = 1
    _, remap, claimed_by = merge_claimed(base, base_used, incoming, incoming_used, protect=200)
    assert remap[5] == 185
    assert not claimed_by.any()


def test_merge_keeps_returning_two_values():
    base, base_used = model_palette_counts("robot")
    incoming, incoming_used = model_palette_counts("car")
    result = merge(base, base_used, incoming, incoming_used)
    assert len(result) == 2
    _, _, claimed_by = merge_claimed(base, base_used, incoming, incoming_used)
    assert np.array_equal(np.nonzero(claimed_by)[0], np.nonzero(incoming_used[1:])[0] + 1)
    for i in np.nonzero(claimed_by)[0]:
        assert claimed_by[i] == i


# --- swap_slots and swap_remap ------------------------------------------------


def test_swap_slots_exchanges_two_colours():
    base = default_palette()
    base[5] = (10, 20, 30, 255)
    base[9] = (40, 50, 60, 255)
    swapped = swap_slots(base, 5, 9)
    assert swapped is not base
    assert swapped.shape == (256, 4) and swapped.dtype == np.uint8
    assert swapped[5].tolist() == [40, 50, 60, 255]
    assert swapped[9].tolist() == [10, 20, 30, 255]
    others = [i for i in range(256) if i not in (5, 9)]
    assert np.array_equal(swapped[others], base[others])


def test_swap_slots_same_index_is_an_unchanged_copy():
    base = default_palette()
    swapped = swap_slots(base, 5, 5)
    assert swapped is not base
    assert np.array_equal(swapped, base)


def test_swap_remap_trades_the_two_indices():
    remap = swap_remap(5, 9)
    assert remap.shape == (256,) and remap.dtype == np.uint8
    assert remap[5] == 9 and remap[9] == 5
    others = np.ones(256, dtype=bool)
    others[[5, 9]] = False
    assert np.array_equal(remap[others], np.arange(256)[others])


def test_swap_remap_same_index_is_identity():
    assert np.array_equal(swap_remap(7, 7), np.arange(256))


def test_swap_remap_applies_to_a_voxel_array():
    # A model painted with 5 and 9 looks the same after swap_slots + this remap.
    voxels = np.array([0, 5, 9, 5, 9, 3], dtype=np.uint8)
    remapped = swap_remap(5, 9)[voxels]
    assert remapped.tolist() == [0, 9, 5, 9, 5, 3]


# --- gradient ------------------------------------------------------------------


def test_gradient_interpolates_the_slots_between():
    base = default_palette()
    base[10] = (0, 0, 0, 255)
    base[20] = (100, 200, 40, 255)
    out = gradient(base, 10, 20)
    assert out.shape == (256, 4) and out.dtype == np.uint8
    assert out[10].tolist() == [0, 0, 0, 255]  # the endpoints keep their colour
    assert out[20].tolist() == [100, 200, 40, 255]
    assert out[15].tolist() == [50, 100, 20, 255]  # halfway
    outside = [i for i in range(256) if i < 10 or i > 20]
    assert np.array_equal(out[outside], base[outside])


def test_gradient_is_the_same_from_either_end():
    base = default_palette()
    base[3] = (10, 20, 30, 255)
    base[8] = (90, 80, 70, 255)
    assert np.array_equal(gradient(base, 3, 8), gradient(base, 8, 3))


def test_gradient_equal_or_adjacent_slots_change_nothing():
    base = default_palette()
    assert np.array_equal(gradient(base, 4, 4), base)
    assert np.array_equal(gradient(base, 4, 5), base)
