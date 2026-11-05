"""Medidas y correspondencias geométricas. No identifican personas."""

from __future__ import annotations

import math

import numpy as np

from motion_art_lab.contracts import MotionRegion


def compute_motion_fraction(mask: np.ndarray) -> float:
    array = np.asarray(mask, dtype=bool)
    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(array)) / float(array.size)


def compute_motion_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def compute_motion_energy(difference: np.ndarray, clean_mask: np.ndarray) -> float:
    """Media de la diferencia dentro de la máscara limpia, dividida por 255."""

    diff = np.asarray(difference)
    mask = np.asarray(clean_mask, dtype=bool)
    if diff.shape != mask.shape:
        raise ValueError("la diferencia y la máscara no coinciden")
    if not mask.any():
        return 0.0
    return float(np.mean(diff[mask]) / 255.0)


def pair_points(
    previous: list[tuple[float, float]],
    current: list[tuple[float, float]],
    max_distance: float,
) -> list[tuple[int, int, float, float]]:
    """Empareja puntos por cercanía. No une saltos mayores que `max_distance`.

    Cada punto se usa como máximo una vez. El resultado es
    (índice previo, índice actual, dx, dy).
    """

    if max_distance < 0:
        raise ValueError("la distancia máxima no puede ser negativa")
    candidates: list[tuple[float, int, int]] = []
    limit = float(max_distance)
    for i, (x1, y1) in enumerate(current):
        for j, (x0, y0) in enumerate(previous):
            dist = math.hypot(x1 - x0, y1 - y0)
            if dist <= limit:
                candidates.append((dist, j, i))
    candidates.sort()
    used_prev: set[int] = set()
    used_curr: set[int] = set()
    pairs: list[tuple[int, int, float, float]] = []
    for _dist, j, i in candidates:
        if j in used_prev or i in used_curr:
            continue
        used_prev.add(j)
        used_curr.add(i)
        x0, y0 = previous[j]
        x1, y1 = current[i]
        pairs.append((j, i, x1 - x0, y1 - y0))
    return pairs


def track_region_centroids(
    previous_regions: list[MotionRegion],
    current_regions: list[MotionRegion],
    max_distance: float,
) -> list[tuple[int, int, float, float]]:
    """Asociación temporal de centroides. No es un identificador persistente."""

    previous = [region.centroid_xy for region in previous_regions]
    current = [region.centroid_xy for region in current_regions]
    return pair_points(previous, current, max_distance)


def sample_emission_points(
    mask: np.ndarray,
    regions: list[MotionRegion],
    max_points: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Muestra píxeles activos para emitir partículas, sin crear una por píxel."""

    if max_points <= 0 or not regions:
        return []
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return []
    count = min(int(max_points), int(xs.size))
    if count == int(xs.size):
        chosen = np.arange(count)
    else:
        chosen = rng.choice(int(xs.size), size=count, replace=False)
    return [(float(xs[index]), float(ys[index])) for index in np.atleast_1d(chosen)]


def transform_coordinates(
    xy: tuple[float, float],
    input_size: tuple[int, int],
    canvas_rect: tuple[int, int, int, int],
    mirror: bool,
) -> tuple[float, float]:
    """Pasa de coordenadas de procesamiento (x, y) al rectángulo del lienzo.

    Conserva la escala que ya trae `canvas_rect` (letterbox incluido).
    El espejo invierte x dentro de la imagen de entrada.
    """

    x, y = xy
    in_w, in_h = input_size
    origin_x, origin_y, width, height = canvas_rect
    if in_w <= 0 or in_h <= 0 or width <= 0 or height <= 0:
        raise ValueError("tamaños de transformación inválidos")
    px = (in_w - 1) - x if mirror else x
    return (
        origin_x + (px + 0.5) * (width / in_w),
        origin_y + (y + 0.5) * (height / in_h),
    )
