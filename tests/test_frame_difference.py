import numpy as np

from motion_art_lab.module1_processing.frame_difference import FrameDiffDetector, binary_threshold


def _ready(detector: FrameDiffDetector, frame: np.ndarray, **kwargs):
    options = {"threshold": 25, "apply_mean_filter": False, "mean_filter_size": 3}
    options.update(kwargs)
    return detector.process(frame, **options)


def test_identical_frames_have_zero_difference_and_empty_mask():
    detector = FrameDiffDetector()
    image = np.full((4, 5), 40, dtype=np.uint8)
    warmup = _ready(detector, image)
    assert warmup.status == "warmup"
    assert np.all(warmup.difference == 0)
    ready = _ready(detector, image)
    assert ready.status == "ready"
    assert np.all(ready.difference == 0)
    assert not ready.raw_mask.any()


def test_single_changed_pixel_is_detected_at_that_coordinate():
    detector = FrameDiffDetector()
    base = np.full((5, 6), 10, dtype=np.uint8)
    _ready(detector, base)
    current = base.copy()
    current[2, 3] = 200
    out = _ready(detector, current)
    assert out.difference[2, 3] == 190
    assert out.raw_mask[2, 3]
    assert int(out.raw_mask.sum()) == 1


def test_uint8_subtraction_does_not_wrap():
    wrapped = np.array([0], dtype=np.uint8) - np.array([255], dtype=np.uint8)
    assert int(wrapped[0]) == 1

    detector = FrameDiffDetector()
    _ready(detector, np.zeros((1, 1), dtype=np.uint8))
    out = _ready(detector, np.full((1, 1), 255, dtype=np.uint8))
    assert float(out.difference[0, 0]) == 255.0
    assert bool(out.raw_mask[0, 0])


def test_threshold_is_strictly_greater_than_t():
    difference = np.array([[24, 25, 26]], dtype=np.float32)
    mask = binary_threshold(difference, 25)
    assert mask.tolist() == [[False, False, True]]


def test_reset_and_shape_change_return_to_warmup():
    detector = FrameDiffDetector()
    image = np.zeros((3, 3), dtype=np.uint8)
    _ready(detector, image)
    _ready(detector, image + 80)
    detector.reset()
    assert _ready(detector, image).status == "warmup"
    assert _ready(detector, np.zeros((4, 2), dtype=np.uint8)).status == "warmup"
