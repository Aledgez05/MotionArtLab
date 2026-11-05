from dataclasses import replace

import numpy as np
import pytest

from motion_art_lab.config import ProcessingSettings
from motion_art_lab.module1_processing.pipeline import MotionPipeline
from motion_art_lab.module1_processing.sources import SyntheticFrameSource


def _rgb_frame(
    width: int, height: int, block: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if block is not None:
        x, y, w, h = block
        frame[y : y + h, x : x + w] = 255
    return frame


def test_pipeline_warmup_then_detects_a_synthetic_rectangle():
    settings = replace(
        ProcessingSettings(),
        width=80,
        height=60,
        threshold=25,
        apply_mean_filter=False,
        opening_iterations=0,
        closing_iterations=0,
        min_component_area=50,
    )
    pipeline = MotionPipeline(settings)
    first = pipeline.process_rgb(_rgb_frame(80, 60), 0.0)
    assert first.status == "warmup"
    assert first.regions == []
    assert first.frame_rgb.shape == (60, 80, 3)
    assert first.raw_mask.shape == first.clean_mask.shape == (60, 80)
    assert first.flow_vectors is None

    second = pipeline.process_rgb(_rgb_frame(80, 60, (10, 10, 20, 20)), 0.03)
    assert second.status == "ready"
    assert second.motion_fraction > 0
    assert len(second.regions) == 1
    assert second.regions[0].area_px == 400
    assert second.regions[0].centroid_xy == pytest.approx((19.5, 19.5))
    assert second.processing_ms >= 0
    assert second.detector_name == "frame_diff"

    third = pipeline.process_rgb(_rgb_frame(80, 60, (10, 10, 20, 20)), 0.06)
    assert third.status == "ready"
    assert third.regions == []
    assert third.motion_fraction == 0


def test_opening_still_keeps_a_solid_rectangle():
    settings = replace(
        ProcessingSettings(),
        width=80,
        height=60,
        apply_mean_filter=False,
        opening_iterations=1,
        closing_iterations=0,
        min_component_area=50,
    )
    pipeline = MotionPipeline(settings)
    pipeline.process_rgb(_rgb_frame(80, 60), 0.0)
    result = pipeline.process_rgb(_rgb_frame(80, 60, (10, 10, 20, 20)), 0.03)
    assert len(result.regions) == 1
    assert result.regions[0].area_px == 400


def test_synthetic_source_is_labeled_and_changes_between_frames():
    source = SyntheticFrameSource(160, 120)
    assert source.is_synthetic
    assert source.label == "Demo sintética"
    ok_a, frame_a, _t1 = source.read()
    ok_b, frame_b, _t2 = source.read()
    assert ok_a and ok_b
    assert frame_a is not None and frame_b is not None
    assert frame_a.shape == (120, 160, 3)
    assert not np.array_equal(frame_a, frame_b)

    pipeline = MotionPipeline(
        replace(ProcessingSettings(), width=80, height=60, min_component_area=20)
    )
    pipeline.process_bgr(frame_a, 0.0)
    saw_motion = False
    latest = frame_b
    for index in range(8):
        ok, latest, timestamp = source.read()
        assert ok and latest is not None
        result = pipeline.process_bgr(latest, timestamp)
        if result.status == "ready" and result.motion_fraction > 0 and result.regions:
            saw_motion = True
            break
        assert index >= 0
    assert saw_motion
