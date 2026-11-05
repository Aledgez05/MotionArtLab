import numpy as np
import pytest

from motion_art_lab.module1_processing.temporal import temporal_smooth


def test_temporal_smooth_matches_the_recurrence():
    previous = np.zeros((2, 2), dtype=np.float32)
    current = np.ones((2, 2), dtype=np.float32)
    step = temporal_smooth(current, previous, 0.4)
    np.testing.assert_allclose(step, 0.4)
    nxt = temporal_smooth(np.zeros((2, 2)), step, 0.4)
    np.testing.assert_allclose(nxt, 0.24)


def test_temporal_smooth_rejects_beta_outside_unit_interval():
    mask = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        temporal_smooth(mask, None, 1.2)
