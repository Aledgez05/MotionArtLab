"""Contratos de datos entre el motor de visión y la interfaz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionRegion:
    """Región de movimiento en coordenadas de procesamiento: x horizontal, y vertical."""

    id: int
    area_px: int
    centroid_xy: tuple[float, float]
    bbox_xywh: tuple[int, int, int, int]


@dataclass
class MotionResult:
    """Resultado de un fotograma. Las máscaras comparten el tamaño del fotograma procesado.

    `frame_rgb` es uint8 (H, W, 3). `difference` es float32 (H, W).
    `raw_mask` y `clean_mask` son booleanas. `flow_vectors` queda en None en el MVP:
    Lucas–Kanade no forma parte de esta fase.
    """

    timestamp_s: float
    frame_rgb: np.ndarray
    difference: np.ndarray
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    regions: list[MotionRegion]
    motion_fraction: float
    mean_motion_energy: float
    flow_vectors: np.ndarray | None
    detector_name: str
    status: str
    processing_ms: float
