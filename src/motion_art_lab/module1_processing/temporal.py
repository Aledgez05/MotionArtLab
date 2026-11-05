"""Suavizado temporal opcional: S_t = beta * M_t + (1 - beta) * S_(t-1)."""

from __future__ import annotations

import numpy as np


def temporal_smooth(
    mask: np.ndarray,
    previous: np.ndarray | None,
    beta: float,
) -> np.ndarray:
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta debe estar entre 0 y 1")
    current = np.asarray(mask, dtype=np.float32)
    if current.ndim != 2:
        raise ValueError("la máscara temporal debe ser 2D")
    if previous is None:
        return current.copy()
    prev = np.asarray(previous, dtype=np.float32)
    if prev.shape != current.shape:
        raise ValueError("la máscara previa no coincide en tamaño")
    weight = np.float32(beta)
    return weight * current + (np.float32(1.0) - weight) * prev
