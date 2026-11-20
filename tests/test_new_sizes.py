"""The standard sizes of File > New."""

from rok.core import new_sizes
from rok.core.grid_ops import MAX_SIZE, MIN_SIZE


def test_every_standard_fits_the_grid_limits():
    assert new_sizes.STANDARDS
    for _pid, _name, size in new_sizes.STANDARDS:
        assert new_sizes.valid(size), size
        assert all(MIN_SIZE <= v <= MAX_SIZE for v in size)


def test_ids_and_names_are_unique():
    ids = [s[0] for s in new_sizes.STANDARDS]
    assert len(set(ids)) == len(ids)
    sizes = [s[2] for s in new_sizes.STANDARDS]
    assert len(set(sizes)) == len(sizes)


def test_default_is_a_standard_and_lookup_round_trips():
    assert new_sizes.DEFAULT in [s[0] for s in new_sizes.STANDARDS]
    for pid, _name, size in new_sizes.STANDARDS:
        assert new_sizes.preset_for(new_sizes.size_of(pid)) == pid
        assert new_sizes.preset_for(size) == pid


def test_a_typed_size_that_is_no_standard_has_no_preset():
    assert new_sizes.preset_for((33, 32, 32)) is None
