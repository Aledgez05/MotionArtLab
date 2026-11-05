"""Cuadrícula de diagnóstico. Solo dibuja datos que el pipeline realmente produjo."""

from __future__ import annotations

import numpy as np
import pygame

from motion_art_lab.contracts import MotionResult
from motion_art_lab.module2_experience.effects.compositor import (
    blit_rgb,
    difference_to_rgb,
    mask_to_rgb,
)
from motion_art_lab.module2_experience.theme import Fonts, Theme, blit_text, draw_round_rect


def draw_diagnostics(
    surface: pygame.Surface,
    result: MotionResult | None,
    art: pygame.Surface | None,
    fonts: Fonts,
    theme: Theme,
    mirror: bool,
    lines: list[str],
) -> None:
    bounds = surface.get_rect()
    pad, gap = 12, 10
    cols, rows = 3, 2
    cell_w = (bounds.w - pad * 2 - gap * (cols - 1)) // cols
    cell_h = (bounds.h - pad * 2 - gap * (rows - 1)) // rows
    titles = _titles(result)
    panels: list[np.ndarray | pygame.Surface | None] = _panels(result, art)
    for index in range(6):
        col, row = index % cols, index // cols
        rect = pygame.Rect(
            pad + col * (cell_w + gap),
            pad + row * (cell_h + gap),
            cell_w,
            cell_h,
        )
        draw_round_rect(surface, (14, 22, 28), rect, radius=12)
        blit_text(surface, titles[index], fonts.small, theme.aqua, (rect.x + 10, rect.y + 8))
        body = pygame.Rect(rect.x + 8, rect.y + 28, rect.w - 16, rect.h - 38)
        panel = panels[index]
        if index == 5:
            _draw_lines(surface, body, lines, fonts, theme)
        elif isinstance(panel, pygame.Surface):
            _blit_surface(surface, panel, body)
        elif isinstance(panel, np.ndarray):
            blit_rgb(surface, panel, (body.x, body.y, body.w, body.h), mirror=mirror and index < 4)
        else:
            blit_text(surface, "Sin fotograma", fonts.small, theme.muted, (body.x, body.y + 8))


def _titles(result: MotionResult | None) -> list[str]:
    if result is None or result.difference.size == 0:
        diff = "Diferencia"
    else:
        diff = f"Diferencia (máx {float(np.max(result.difference)):.0f})"
    return [
        "Original",
        diff,
        "Máscara cruda",
        "Máscara limpia",
        "Resultado artístico",
        "Medición",
    ]


def _panels(
    result: MotionResult | None,
    art: pygame.Surface | None,
) -> list[np.ndarray | pygame.Surface | None]:
    if result is None:
        return [None, None, None, None, art, None]
    return [
        result.frame_rgb,
        difference_to_rgb(result.difference),
        mask_to_rgb(result.raw_mask),
        mask_to_rgb(result.clean_mask),
        art,
        None,
    ]


def _blit_surface(dest: pygame.Surface, source: pygame.Surface, rect: pygame.Rect) -> None:
    if rect.w < 2 or rect.h < 2:
        return
    scaled = pygame.transform.smoothscale(source, (rect.w, rect.h))
    dest.blit(scaled, rect.topleft)


def _draw_lines(
    surface: pygame.Surface,
    rect: pygame.Rect,
    lines: list[str],
    fonts: Fonts,
    theme: Theme,
) -> None:
    y = rect.y + 4
    for line in lines:
        if y + 18 > rect.bottom:
            break
        blit_text(surface, line, fonts.small, theme.white, (rect.x, y))
        y += 20
