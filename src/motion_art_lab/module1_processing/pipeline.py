"""Pipeline de un fotograma: preparar, detectar, limpiar y medir.

Orden:
1. Validar, pasar a RGB y redimensionar a la resolución de procesamiento.
2. Gris manual y diferencia absoluta contra el fotograma anterior.
3. Filtro de promedio opcional sobre la diferencia, después umbral.
4. Apertura y cierre, solo si sus iteraciones son mayores que cero.
5. Suavizado temporal opcional, vuelto a umbralizar en 0.5.
6. Componentes conectados y descarte por área.
"""

from __future__ import annotations

import time

import numpy as np

from motion_art_lab.config import ProcessingSettings
from motion_art_lab.contracts import MotionResult
from motion_art_lab.module1_processing.components import analyze_components
from motion_art_lab.module1_processing.frame_difference import FrameDiffDetector
from motion_art_lab.module1_processing.morphology import binary_close, binary_open, square_structure
from motion_art_lab.module1_processing.motion_features import (
    compute_motion_energy,
    compute_motion_fraction,
)
from motion_art_lab.module1_processing.preprocessing import (
    bgr_to_rgb,
    normalize_frame,
    resize_nearest,
    rgb_to_grayscale,
)
from motion_art_lab.module1_processing.temporal import temporal_smooth


class MotionPipeline:
    def __init__(self, settings: ProcessingSettings | None = None) -> None:
        self.settings = settings or ProcessingSettings()
        self.detector = FrameDiffDetector()
        self._temporal: np.ndarray | None = None

    def update_settings(self, settings: ProcessingSettings) -> None:
        if (settings.width, settings.height) != (self.settings.width, self.settings.height):
            self.reset()
        if settings.detector != self.settings.detector:
            self.reset()
        self.settings = settings

    def reset(self) -> None:
        self.detector.reset()
        self._temporal = None

    def process_bgr(self, frame_bgr: np.ndarray, timestamp_s: float) -> MotionResult:
        return self.process_rgb(bgr_to_rgb(frame_bgr), timestamp_s)

    def process_rgb(self, frame_rgb: np.ndarray, timestamp_s: float) -> MotionResult:
        started = time.perf_counter()
        settings = self.settings
        rgb = resize_nearest(normalize_frame(frame_rgb), settings.width, settings.height)
        gray = rgb_to_grayscale(rgb)
        detected = self.detector.process(
            gray,
            threshold=settings.threshold,
            apply_mean_filter=settings.apply_mean_filter,
            mean_filter_size=settings.mean_filter_size,
        )
        clean = detected.raw_mask
        regions = []
        if detected.status == "ready":
            structure = square_structure(settings.morph_kernel_size)
            if settings.opening_iterations > 0:
                clean = binary_open(clean, structure, settings.opening_iterations)
            if settings.closing_iterations > 0:
                clean = binary_close(clean, structure, settings.closing_iterations)
            if settings.temporal_smoothing_enabled:
                smoothed = temporal_smooth(clean, self._temporal, settings.temporal_beta)
                self._temporal = smoothed
                clean = smoothed >= np.float32(0.5)
            else:
                self._temporal = None
            clean, regions = analyze_components(
                clean,
                settings.min_component_area,
                settings.component_connectivity,
            )
        else:
            self._temporal = None
            clean = np.zeros(detected.raw_mask.shape, dtype=bool)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return MotionResult(
            timestamp_s=float(timestamp_s),
            frame_rgb=rgb,
            difference=detected.difference,
            raw_mask=detected.raw_mask,
            clean_mask=clean,
            regions=regions,
            motion_fraction=compute_motion_fraction(clean),
            mean_motion_energy=compute_motion_energy(detected.difference, clean),
            flow_vectors=None,
            detector_name=settings.detector,
            status=detected.status,
            processing_ms=elapsed_ms,
        )
