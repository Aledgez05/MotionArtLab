import numpy as np

from motion_art_lab.module1_processing.filters import mean_filter
from motion_art_lab.module1_processing.morphology import (
    binary_close,
    binary_dilate,
    binary_erode,
    binary_open,
)


def test_mean_filter_preserves_a_constant_and_spreads_an_impulse():
    constant = np.full((5, 5), 7, dtype=np.float32)
    np.testing.assert_allclose(mean_filter(constant, 3), 7)

    impulse = np.zeros((5, 5), dtype=np.float32)
    impulse[2, 2] = 9
    filtered = mean_filter(impulse, 3)
    assert filtered[2, 2] == np.float32(1)
    assert filtered[0, 0] == 0


def test_erosion_shrinks_a_block_and_removes_an_isolated_pixel():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    mask[4, 5] = True
    expected = np.zeros((6, 6), dtype=bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(binary_erode(mask), expected)


def test_dilation_expands_a_single_pixel_to_its_3x3_neighborhood():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(binary_dilate(mask), expected)


def test_opening_removes_specks_and_keeps_a_solid_interior():
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:9, 2:9] = True
    mask[0, 11] = True
    opened = binary_open(mask)
    assert not bool(opened[0, 11])
    assert bool(opened[5, 5])
    np.testing.assert_array_equal(opened, binary_dilate(binary_erode(mask)))


def test_closing_fills_a_one_pixel_hole():
    mask = np.ones((7, 7), dtype=bool)
    mask[3, 3] = False
    closed = binary_close(mask)
    assert bool(closed[3, 3])
    np.testing.assert_array_equal(closed, binary_erode(binary_dilate(mask)))
