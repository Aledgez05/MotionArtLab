"""Color, tipografía y trazos de la identidad visual."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


class Theme:
    bg = (247, 248, 246)
    surface = (255, 255, 255)
    aqua = (185, 233, 230)
    teal = (58, 158, 157)
    teal_dark = (36, 122, 121)
    coral = (215, 143, 122)
    coral_soft = (255, 236, 230)
    text = (21, 40, 47)
    muted = (101, 118, 123)
    border = (220, 232, 230)
    canvas = (17, 25, 31)
    white = (255, 255, 255)
    track = (232, 239, 238)
    hint = (90, 112, 116)


@dataclass
class Fonts:
    display: pygame.font.Font
    display_small: pygame.font.Font
    ui: pygame.font.Font
    ui_bold: pygame.font.Font
    small: pygame.font.Font


def load_fonts() -> Fonts:
    pygame.font.init()
    serif = "Cormorant Garamond, Palatino Linotype, Georgia, Times New Roman"
    sans = "Inter, Segoe UI, Calibri, Arial"
    return Fonts(
        display=pygame.font.SysFont(serif, 46),
        display_small=pygame.font.SysFont(serif, 30),
        ui=pygame.font.SysFont(sans, 16),
        ui_bold=pygame.font.SysFont(sans, 16, bold=True),
        small=pygame.font.SysFont(sans, 13),
    )


def draw_round_rect(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    radius: int = 14,
    width: int = 0,
) -> None:
    if rect.w < 2 or rect.h < 2:
        return
    pygame.draw.rect(surface, color, rect, width=width, border_radius=min(radius, rect.h // 2))


def draw_corner_ticks(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    length: int = 16,
    width: int = 2,
    inset: int = 14,
) -> None:
    x0, y0 = rect.x + inset, rect.y + inset
    x1, y1 = rect.right - inset, rect.bottom - inset
    segments = (
        ((x0, y0), (x0 + length, y0)),
        ((x0, y0), (x0, y0 + length)),
        ((x1, y0), (x1 - length, y0)),
        ((x1, y0), (x1, y0 + length)),
        ((x0, y1), (x0 + length, y1)),
        ((x0, y1), (x0, y1 - length)),
        ((x1, y1), (x1 - length, y1)),
        ((x1, y1), (x1, y1 - length)),
    )
    for start, end in segments:
        pygame.draw.line(surface, color, start, end, width)


def blit_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    pos: tuple[int, int],
) -> pygame.Surface:
    rendered = font.render(text, True, color)
    surface.blit(rendered, pos)
    return rendered
