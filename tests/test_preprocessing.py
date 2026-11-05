import numpy as np
import pytest

from motion_art_lab.module1_processing.preprocessing import (
    bgr_to_rgb,
    normalize_frame,
    resize_nearest,
    rgb_to_grayscale,
)


def test_bgr_to_rgb_reorders_channels():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 10
    bgr[..., 1] = 20
    bgr[..., 2] = 30
    rgb = bgr_to_rgb(bgr)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_grayscale_uses_rec601_weights():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    gray = rgb_to_grayscale(rgb)
    assert gray.dtype == np.float32
    assert gray[0, 0] == pytest.approx(0.299 * 255)
    assert gray[0, 1] == 0


def test_resize_nearest_replicates_blocks():
    src = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = resize_nearest(src, 4, 4)
    expected = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(out, expected)


def test_normalize_rejects_bad_frames():
    with pytest.raises(ValueError):
        normalize_frame(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        normalize_frame(np.zeros((4, 4, 3), dtype=np.float32))
