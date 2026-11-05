"""Rastros luminosos que se desvanecen con el tiempo, no con el número de fotogramas."""

from __future__ import annotations

import math

import pygame

from motion_art_lab.module1_processing.motion_features import pair_points


def trail_decay(persistence: float, dt: float) -> float:
    """gamma(dt) = exp(-lambda * dt). Más persistencia, vida media más larga."""

    clamped = min(1.0, max(0.0, persistence))
    half_life = 0.35 + clamped * 3.25
    if dt <= 0:
        return 1.0
    return math.exp(-math.log(2) * dt / half_life)


class TrailEffect:
    def __init__(self) -> None:
        self.surface: pygame.Surface | None = None
        self._fade: pygame.Surface | None = None
        self._previous: list[tuple[float, float]] = []

    def ensure(self, size: tuple[int, int]) -> None:
        if self.surface is not None and self.surface.get_size() == size:
            return
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))
        self._fade = pygame.Surface(size, pygame.SRCALPHA)
        self._previous = []

    def clear(self) -> None:
        self._previous = []
        if self.surface is not None:
            self.surface.fill((0, 0, 0, 0))

    def break_tracks(self) -> None:
        """Corta la continuidad sin borrar lo ya pintado. Evita una línea a través del lienzo."""

        self._previous = []

    def update(
        self,
        points: list[tuple[float, float]],
        dt: float,
        intensity: float,
        persistence: float,
        colors: tuple[tuple[int, int, int], ...],
    ) -> None:
        if self.surface is None or self._fade is None:
            return
        gamma = trail_decay(persistence, dt)
        factor = int(round(min(1.0, max(0.0, gamma)) * 255))
        self._fade.fill((factor, factor, factor, factor))
        self.surface.blit(self._fade, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        max_jump = max(36.0, self.surface.get_width() * 0.07)
        pairs = pair_points(self._previous, points, max_jump)
        strength = min(1.0, max(0.0, intensity))
        width = max(2, int(round(2 + strength * 6)))
        alpha = int(30 + strength * 210)
        palette = colors or ((185, 233, 230), (255, 255, 255))
        light = palette[min(2, len(palette) - 1)]
        for prev_i, curr_i, _dx, _dy in pairs:
            start = _pixel(self._previous[prev_i])
            end = _pixel(points[curr_i])
            color = palette[curr_i % len(palette)]
            pygame.draw.line(self.surface, (*color, max(24, alpha // 3)), start, end, width + 8)
            pygame.draw.line(self.surface, (*color, alpha), start, end, width)
            pygame.draw.line(self.surface, (*light, alpha), start, end, max(1, width // 2))
        self._previous = list(points)


def _pixel(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))
