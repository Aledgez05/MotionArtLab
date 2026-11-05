"""Conversión de imágenes NumPy a superficies y encaje con relación de aspecto."""

from __future__ import annotations

import numpy as np
import pygame


def fit_rect(
    src_w: int,
    src_h: int,
    dst_rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Letterbox: escala uniforme y centra. No estira la imagen."""

    dx, dy, dw, dh = dst_rect
    if src_w < 1 or src_h < 1 or dw < 1 or dh < 1:
        return dx, dy, 0, 0
    scale = min(dw / src_w, dh / src_h)
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))
    x = dx + (dw - width) // 2
    y = dy + (dh - height) // 2
    return x, y, width, height


def rgb_to_surface(rgb: np.ndarray, mirror: bool = False) -> pygame.Surface:
    """(H, W, 3) → superficie. surfarray usa (W, H, 3), por eso se transponen los ejes."""

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("se esperaba RGB uint8 (H, W, 3)")
    if mirror:
        image = image[:, ::-1]
    swapped = np.ascontiguousarray(np.transpose(image, (1, 0, 2)))
    return pygame.surfarray.make_surface(swapped)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    vis = np.asarray(mask, dtype=np.uint8) * np.uint8(255)
    return np.dstack((vis, vis, vis))


def difference_to_rgb(difference: np.ndarray) -> np.ndarray:
    """Escala directa 0–255, sin estirar el histograma."""

    vis = np.clip(np.asarray(difference), 0, 255).astype(np.uint8)
    return np.dstack((vis, vis, vis))


def blit_rgb(
    dest: pygame.Surface,
    rgb: np.ndarray,
    dest_rect: tuple[int, int, int, int],
    mirror: bool = False,
) -> tuple[int, int, int, int]:
    fitted = fit_rect(int(rgb.shape[1]), int(rgb.shape[0]), dest_rect)
    if fitted[2] < 1 or fitted[3] < 1:
        return fitted
    surface = rgb_to_surface(rgb, mirror)
    scaled = pygame.transform.smoothscale(surface, (fitted[2], fitted[3]))
    dest.blit(scaled, (fitted[0], fitted[1]))
    return fitted
