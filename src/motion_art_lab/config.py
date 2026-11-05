"""Ajustes del procesamiento y de la experiencia artística.

Los valores iniciales son experimentales. Subir la sensibilidad reduce el umbral T:
más sensibilidad detecta cambios más leves.
"""

from __future__ import annotations

from dataclasses import dataclass

THRESHOLD_LOW_SENSITIVITY = 70
THRESHOLD_HIGH_SENSITIVITY = 10


@dataclass(frozen=True)
class ProcessingSettings:
    detector: str = "frame_diff"
    width: int = 320
    height: int = 240
    threshold: int = 25
    apply_mean_filter: bool = True
    mean_filter_size: int = 3
    morph_kernel_size: int = 3
    opening_iterations: int = 1
    closing_iterations: int = 0
    component_connectivity: int = 8
    min_component_area: int = 50
    # Reservado para un detector posterior. El MVP no lo aplica.
    running_alpha: float = 0.03
    temporal_beta: float = 0.55
    temporal_smoothing_enabled: bool = False


@dataclass(frozen=True)
class ArtisticSettings:
    mode: str = "trails"
    palette: str = "aurora"
    intensity: float = 0.65
    persistence: float = 0.65
    particle_limit: int = 350
    mirror_preview: bool = True
    show_camera_background: bool = False
    show_diagnostics: bool = False


def threshold_from_sensitivity(sensitivity: float) -> int:
    """Convierte sensibilidad 0..1 en umbral T. 0.75 corresponde al T inicial de 25."""

    clamped = min(1.0, max(0.0, float(sensitivity)))
    span = THRESHOLD_LOW_SENSITIVITY - THRESHOLD_HIGH_SENSITIVITY
    return int(round(THRESHOLD_LOW_SENSITIVITY - clamped * span))


def sensitivity_from_threshold(threshold: int) -> float:
    span = THRESHOLD_LOW_SENSITIVITY - THRESHOLD_HIGH_SENSITIVITY
    return min(1.0, max(0.0, (THRESHOLD_LOW_SENSITIVITY - int(threshold)) / span))
