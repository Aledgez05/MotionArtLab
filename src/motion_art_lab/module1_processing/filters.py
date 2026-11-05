"""Filtro de promedio espacial, implementado con desplazamientos de NumPy."""

from __future__ import annotations

import numpy as np


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Promedio de una ventana impar. El borde se replica para no oscurecer las esquinas.

    El resultado es float32. Conviene aplicarlo al mapa de diferencias, antes del umbral:
    promediar una máscara binaria la convertiría en grises.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("el tamaño del kernel debe ser impar y positivo")
    src = np.asarray(image, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError("el filtro de promedio espera una imagen 2D")
    pad = kernel_size // 2
    padded = np.pad(src, pad, mode="edge")
    height, width = src.shape
    acc = np.zeros((height, width), dtype=np.float32)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            acc += padded[dy : dy + height, dx : dx + width]
    return acc / np.float32(kernel_size * kernel_size)
