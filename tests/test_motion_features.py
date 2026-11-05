import numpy as np
import pytest

from motion_art_lab.contracts import MotionRegion
from motion_art_lab.module1_processing.motion_features import (
    compute_motion_centroid,
    compute_motion_energy,
    compute_motion_fraction,
    pair_points,
    sample_emission_points,
    track_region_centroids,
    transform_coordinates,
)
from motion_art_lab.module2_experience.effects.trails import trail_decay


def test_motion_fraction_centroid_and_energy():
    mask = np.zeros((5, 4), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    assert compute_motion_fraction(mask) == pytest.approx(2 / 20)
    assert compute_motion_centroid(mask) == pytest.approx((0.5, 0.0))
    assert compute_motion_centroid(np.zeros((2, 2), dtype=bool)) is None

    diff = np.zeros((5, 4), dtype=np.float32)
    diff[0, 0] = 255
    diff[0, 1] = 255
    assert compute_motion_energy(diff, mask) == pytest.approx(1.0)
    assert compute_motion_energy(diff, np.zeros_like(mask)) == 0.0


def test_pair_points_skips_jumps_and_reports_displacement():
    previous = [(0.0, 0.0), (10.0, 10.0)]
    current = [(2.0, 1.0), (80.0, 80.0)]
    pairs = pair_points(previous, current, max_distance=5)
    assert len(pairs) == 1
    prev_i, curr_i, dx, dy = pairs[0]
    assert (prev_i, curr_i) == (0, 0)
    assert (dx, dy) == pytest.approx((2.0, 1.0))


def test_track_region_centroids_uses_the_same_limit():
    previous = [MotionRegion(1, 4, (0.0, 0.0), (0, 0, 2, 2))]
    current = [MotionRegion(2, 4, (1.0, 0.0), (1, 0, 2, 2))]
    pairs = track_region_centroids(previous, current, max_distance=2)
    assert len(pairs) == 1
    assert pairs[0][2] == pytest.approx(1.0)


def test_sample_emission_points_stay_inside_the_mask():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 2:5] = True
    region = MotionRegion(1, int(mask.sum()), (3.0, 2.0), (2, 1, 3, 3))
    points = sample_emission_points(mask, [region], max_points=4, rng=np.random.default_rng(0))
    assert 1 <= len(points) <= 4
    for x, y in points:
        assert mask[int(y), int(x)]


def test_transform_coordinates_mirror_and_scale():
    plain = transform_coordinates((0, 0), (4, 2), (0, 0, 40, 20), mirror=False)
    mirrored = transform_coordinates((0, 0), (4, 2), (0, 0, 40, 20), mirror=True)
    assert plain == pytest.approx((5.0, 5.0))
    assert mirrored == pytest.approx((35.0, 5.0))


def test_trail_decay_depends_on_time_and_persistence():
    assert trail_decay(0.8, 0.0) == 1.0
    assert trail_decay(0.9, 0.05) > trail_decay(0.2, 0.05)
    assert trail_decay(0.5, 0.2) < trail_decay(0.5, 0.05)
