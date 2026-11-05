"""Diferencia absoluta entre fotogramas consecutivos.

D_t(x, y) = |I_t(x, y) - I_(t-1)(x, y)|
M_t(x, y) = D_t(x, y) > T

La resta se hace en float32. Restar uint8 directamente desborda de forma modular
(0 - 255 = 1) y no representa la magnitud del cambio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motion_art_lab.module1_processing.filters import mean_filter


@dataclass
class DetectorOutput:
    difference: np.ndarray
    raw_mask: np.ndarray
    status: str


def binary_threshold(difference: np.ndarray, threshold: float) -> np.ndarray:
    """Máscara booleana. La igualdad con T no activa el píxel: la condición es estricta."""

    values = np.asarray(difference, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("la diferencia debe ser 2D")
    return values > float(threshold)


class FrameDiffDetector:
    """Detector con un fotograma de calentamiento. `reset` olvida la referencia."""

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def process(
        self,
        gray: np.ndarray,
        *,
        threshold: float,
        apply_mean_filter: bool,
        mean_filter_size: int,
    ) -> DetectorOutput:
        current = np.asarray(gray, dtype=np.float32)
        if current.ndim != 2:
            raise ValueError("el fotograma en gris debe ser 2D")
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current.copy()
            return DetectorOutput(
                difference=np.zeros(current.shape, dtype=np.float32),
                raw_mask=np.zeros(current.shape, dtype=bool),
                status="warmup",
            )

        difference = np.abs(current - self._previous)
        self._previous = current.copy()
        compared = difference
        if apply_mean_filter and mean_filter_size >= 3:
            compared = mean_filter(difference, mean_filter_size)
        return DetectorOutput(
            difference=difference,
            raw_mask=binary_threshold(compared, threshold),
            status="ready",
        )
