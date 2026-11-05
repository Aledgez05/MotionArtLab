"""Morfología binaria con un elemento estructurante explícito.

El borde se trata como fondo (padding en cero): un píxel del borde no sobrevive a la
erosión si su vecindad sale de la imagen. Apertura y cierre no se aplican solas;
el pipeline decide cuántas iteraciones usar, y cero significa omitir la operación.
"""

from __future__ import annotations

import numpy as np


def square_structure(size: int = 3) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("el elemento estructurante debe ser cuadrado, impar y positivo")
    return np.ones((size, size), dtype=bool)


def _offsets(structure: np.ndarray) -> list[tuple[int, int]]:
    element = np.asarray(structure, dtype=bool)
    if element.ndim != 2 or element.shape[0] % 2 == 0 or element.shape[1] % 2 == 0:
        raise ValueError("el elemento estructurante debe ser 2D y de tamaño impar")
    center_y = element.shape[0] // 2
    center_x = element.shape[1] // 2
    offsets: list[tuple[int, int]] = []
    ys, xs = np.nonzero(element)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        offsets.append((int(y) - center_y, int(x) - center_x))
    if not offsets:
        raise ValueError("el elemento estructurante está vacío")
    return offsets


def _neighbor(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Valor de mask[y + dy, x + dx]. Fuera de rango se lee como False."""

    height, width = mask.shape
    view = np.zeros((height, width), dtype=bool)
    y0 = max(0, -dy)
    y1 = min(height, height - dy)
    x0 = max(0, -dx)
    x1 = min(width, width - dx)
    if y1 > y0 and x1 > x0:
        view[y0:y1, x0:x1] = mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return view


def binary_erode(
    mask: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
) -> np.ndarray:
    """Un píxel queda activo solo si todos los vecinos del elemento están activos."""

    if iterations < 0:
        raise ValueError("las iteraciones no pueden ser negativas")
    result = np.asarray(mask, dtype=bool).copy()
    if iterations == 0:
        return result
    offsets = _offsets(square_structure(3) if structure is None else structure)
    for _ in range(iterations):
        acc = np.ones(result.shape, dtype=bool)
        for dy, dx in offsets:
            acc &= _neighbor(result, dy, dx)
        result = acc
    return result


def binary_dilate(
    mask: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
) -> np.ndarray:
    """Un píxel queda activo si algún vecino del elemento simétrico está activo."""

    if iterations < 0:
        raise ValueError("las iteraciones no pueden ser negativas")
    result = np.asarray(mask, dtype=bool).copy()
    if iterations == 0:
        return result
    offsets = _offsets(square_structure(3) if structure is None else structure)
    for _ in range(iterations):
        acc = np.zeros(result.shape, dtype=bool)
        for dy, dx in offsets:
            acc |= _neighbor(result, dy, dx)
        result = acc
    return result


def binary_open(
    mask: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
) -> np.ndarray:
    """Apertura: erosión seguida de dilatación. Quita salpicaduras finas."""

    element = square_structure(3) if structure is None else structure
    return binary_dilate(binary_erode(mask, element, iterations), element, iterations)


def binary_close(
    mask: np.ndarray,
    structure: np.ndarray | None = None,
    iterations: int = 1,
) -> np.ndarray:
    """Cierre: dilatación seguida de erosión. Rellena huecos pequeños."""

    element = square_structure(3) if structure is None else structure
    return binary_erode(binary_dilate(mask, element, iterations), element, iterations)
