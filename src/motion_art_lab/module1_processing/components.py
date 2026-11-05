"""Componentes conectados por flood-fill iterativo sobre tramos horizontales.

Cada tramo de píxeles activos es un nodo. Los tramos de filas vecinas se unen si se
tocan (4-conexión) o si también se tocan en diagonal (8-conexión). Después se recorre
ese grafo con una BFS. El resultado es el mismo que un flood-fill píxel a píxel, con
menos iteraciones de Python.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from motion_art_lab.contracts import MotionRegion


def _extract_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    left = np.array([False])
    for y in range(mask.shape[0]):
        row = mask[y]
        if not bool(row.any()):
            continue
        padded = np.concatenate((left, row, left))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        for x0, x1 in zip(starts.tolist(), ends.tolist(), strict=True):
            runs.append((y, int(x0), int(x1)))
    return runs


def _link_runs(
    runs: list[tuple[int, int, int]],
    connectivity: int,
) -> list[list[int]]:
    pad = 0 if connectivity == 4 else 1
    by_row: dict[int, list[int]] = {}
    for index, (y, _x0, _x1) in enumerate(runs):
        by_row.setdefault(y, []).append(index)
    neighbors: list[list[int]] = [[] for _ in runs]
    for y, current in by_row.items():
        above = by_row.get(y - 1)
        if not above:
            continue
        cursor = 0
        for index in current:
            _y, x0, x1 = runs[index]
            left_limit = x0 - pad
            while cursor < len(above) and runs[above[cursor]][2] < left_limit:
                cursor += 1
            probe = cursor
            right_limit = x1 + pad
            while probe < len(above) and runs[above[probe]][1] <= right_limit:
                other = above[probe]
                neighbors[index].append(other)
                neighbors[other].append(index)
                probe += 1
    return neighbors


def label_components(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Etiquetas enteras. El fondo queda en 0 y la primera región útil empieza en 1."""

    if connectivity not in (4, 8):
        raise ValueError("la conectividad debe ser 4 u 8")
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("la máscara debe ser 2D")
    labels = np.zeros(binary.shape, dtype=np.int32)
    if binary.size == 0 or not binary.any():
        return labels

    runs = _extract_runs(binary)
    if not runs:
        return labels
    neighbors = _link_runs(runs, connectivity)
    run_labels = [0] * len(runs)
    current = 0
    for start in range(len(runs)):
        if run_labels[start]:
            continue
        current += 1
        queue: deque[int] = deque((start,))
        run_labels[start] = current
        while queue:
            node = queue.popleft()
            for nxt in neighbors[node]:
                if run_labels[nxt] == 0:
                    run_labels[nxt] = current
                    queue.append(nxt)
    for (y, x0, x1), label in zip(runs, run_labels, strict=True):
        labels[y, x0 : x1 + 1] = label
    return labels


def component_stats(labels: np.ndarray) -> list[MotionRegion]:
    """Área, centroide (media de columnas, media de filas) y caja (x, y, w, h)."""

    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError("las etiquetas deben ser 2D")
    if array.size == 0:
        return []
    count = int(array.max())
    regions: list[MotionRegion] = []
    for comp_id in range(1, count + 1):
        ys, xs = np.nonzero(array == comp_id)
        if xs.size == 0:
            continue
        x_min = int(xs.min())
        y_min = int(ys.min())
        regions.append(
            MotionRegion(
                id=comp_id,
                area_px=int(xs.size),
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                bbox_xywh=(
                    x_min,
                    y_min,
                    int(xs.max()) - x_min + 1,
                    int(ys.max()) - y_min + 1,
                ),
            )
        )
    return regions


def analyze_components(
    mask: np.ndarray,
    min_area: int,
    connectivity: int = 8,
) -> tuple[np.ndarray, list[MotionRegion]]:
    """Devuelve la máscara sin componentes pequeños y las regiones conservadas."""

    if min_area < 1:
        raise ValueError("el área mínima debe ser al menos 1")
    labels = label_components(mask, connectivity)
    if labels.size == 0 or int(labels.max()) == 0:
        return np.zeros(np.asarray(mask).shape, dtype=bool), []
    areas = np.bincount(labels.ravel())
    keep = areas >= int(min_area)
    keep[0] = False
    clean = keep[labels]
    regions = [region for region in component_stats(labels) if region.area_px >= min_area]
    return clean, regions


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
    connectivity: int = 8,
) -> np.ndarray:
    clean, _regions = analyze_components(mask, min_area, connectivity)
    return clean
