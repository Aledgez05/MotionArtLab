import numpy as np
import pytest

from motion_art_lab.module1_processing.components import (
    component_stats,
    label_components,
    remove_small_components,
)


def test_two_separate_squares_have_expected_area_and_centroid():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3:5, 3:5] = True
    labels = label_components(mask, connectivity=8)
    regions = component_stats(labels)
    assert len(regions) == 2
    centroids = sorted(region.centroid_xy for region in regions)
    assert centroids[0] == pytest.approx((0.5, 0.5))
    assert centroids[1] == pytest.approx((3.5, 3.5))
    assert sorted(region.area_px for region in regions) == [4, 4]
    assert sorted(region.bbox_xywh for region in regions) == [(0, 0, 2, 2), (3, 3, 2, 2)]


def test_diagonal_pixels_depend_on_connectivity():
    mask = np.array([[True, False], [False, True]])
    assert len(component_stats(label_components(mask, 4))) == 2
    assert len(component_stats(label_components(mask, 8))) == 1


def test_small_components_are_removed():
    mask = np.zeros((20, 20), dtype=bool)
    mask[1, 1] = True
    mask[5:15, 5:15] = True
    clean = remove_small_components(mask, min_area=50, connectivity=8)
    assert not clean[1, 1]
    assert bool(clean[8, 8])
    assert int(clean.sum()) == 100
