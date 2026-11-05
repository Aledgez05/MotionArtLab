"""Preparación de fotogramas con NumPy. OpenCV no interviene en estas conversiones."""

from __future__ import annotations

import numpy as np

# Luminancia aproximada Rec. 601.
_LUMA_R = np.float32(0.299)
_LUMA_G = np.float32(0.587)
_LUMA_B = np.float32(0.114)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Comprueba que el fotograma sea uint8, con forma (H, W, 3)."""

    if frame is None:
        raise ValueError("el fotograma está vacío")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("se esperaba una imagen de tres canales (H, W, 3)")
    if array.dtype != np.uint8:
        raise ValueError(f"se esperaba dtype uint8, llegó {array.dtype}")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("la imagen es demasiado pequeña")
    if not np.isfinite(array).all():
        raise ValueError("el fotograma contiene valores no finitos")
    return array


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Reordena canales BGR → RGB con un corte de NumPy."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("se esperaba una imagen BGR de tres canales")
    return np.ascontiguousarray(array[:, :, ::-1])


def rgb_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """0.299 R + 0.587 G + 0.114 B, en float32."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("se esperaba una imagen RGB de tres canales")
    rgb = array.astype(np.float32, copy=False)
    return _LUMA_R * rgb[..., 0] + _LUMA_G * rgb[..., 1] + _LUMA_B * rgb[..., 2]


def resize_nearest(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Redimensiona por vecino más cercano, con índices enteros."""

    if width < 1 or height < 1:
        raise ValueError("el tamaño de destino debe ser positivo")
    array = np.asarray(frame)
    if array.ndim not in (2, 3):
        raise ValueError("se esperaba una imagen 2D o 3D")
    src_h, src_w = array.shape[:2]
    if src_h < 1 or src_w < 1:
        raise ValueError("la imagen de origen está vacía")
    if src_w == width and src_h == height:
        return np.ascontiguousarray(array)
    ys = np.minimum((np.arange(height) * src_h) // height, src_h - 1)
    xs = np.minimum((np.arange(width) * src_w) // width, src_w - 1)
    return np.ascontiguousarray(array[ys][:, xs])
