"""Paint's Gradient mode: the ramp and the bands (rok/core/gradient.py)."""

import numpy as np

from rok.core import gradient


def test_a_ramp_has_its_ends_exactly_and_a_clean_middle():
    colours = gradient.ramp((255, 0, 0), (0, 0, 255), 5)
    assert len(colours) == 5 and colours[0] == (255, 0, 0) and colours[-1] == (0, 0, 255)
    middle = colours[2]
    assert middle[0] > 60 and middle[2] > 60  # through purple, not a muddy grey
    assert gradient.ramp((10, 20, 30), (10, 20, 30), 3) == [(10, 20, 30)] * 3
    assert len(gradient.ramp((0, 0, 0), (255, 255, 255), 99)) == gradient.MAX_STEPS


def test_bands_run_from_the_start_to_the_end_in_equal_steps():
    cells = np.array([[x, 0, 0] for x in range(10)])
    assert gradient.bands(cells, (0, 0, 0), (10, 0, 0), 5).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert gradient.bands(cells, (10, 0, 0), (0, 0, 0), 5).tolist() == [4, 4, 3, 3, 2, 2, 1, 1, 0, 0]
    assert gradient.bands(cells, (2, 0, 0), (2, 0, 0), 5).tolist() == [0] * 10  # no drag yet


def test_dither_mixes_only_neighbouring_bands():
    cells = np.argwhere(np.ones((16, 8, 8), bool))
    hard = gradient.bands(cells, (0, 4, 4), (16, 4, 4), 4)
    mixed = gradient.bands(cells, (0, 4, 4), (16, 4, 4), 4, dither=True)
    assert np.abs(hard - mixed).max() <= 1 and (hard != mixed).any()
    assert mixed.min() == 0 and mixed.max() == 3
