"""Controles propios: botones, sliders, segmentos, interruptores y pasos."""

from __future__ import annotations

from collections.abc import Callable

import pygame

from motion_art_lab.module2_experience.palettes import PALETTE_ORDER, palette_colors
from motion_art_lab.module2_experience.theme import Fonts, Theme, draw_round_rect

_PALETTE_IDS = set(PALETTE_ORDER)


class Button:
    def __init__(self, label: str, kind: str = "secondary") -> None:
        self.label = label
        self.kind = kind
        self.active = False
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.hovered = False

    def place(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        if self.kind == "primary" or self.active:
            fill = theme.teal_dark if self.hovered else theme.teal
            text = theme.white
            border = fill
        elif self.hovered:
            fill = (236, 247, 246)
            text = theme.text
            border = theme.teal
        else:
            fill = theme.surface
            text = theme.text
            border = theme.border
        draw_round_rect(surface, fill, self.rect, radius=12)
        draw_round_rect(surface, border, self.rect, radius=12, width=1)
        label = fonts.ui.render(self.label, True, text)
        surface.blit(label, label.get_rect(center=self.rect.center))


class Slider:
    def __init__(
        self,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        formatter: Callable[[float], str],
    ) -> None:
        self.label = label
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.formatter = formatter
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.track = pygame.Rect(0, 0, 10, 4)
        self.dragging = False

    def place(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)
        self.track = pygame.Rect(rect.x + 8, rect.y + 28, max(1, rect.w - 16), 4)

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.track.inflate(0, 18).collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            self._set_from_x(event.pos[0])
            return True
        return False

    def _set_from_x(self, x: int) -> None:
        span = self.maximum - self.minimum
        if self.track.w <= 0 or span == 0:
            return
        ratio = min(1.0, max(0.0, (x - self.track.x) / self.track.w))
        self.value = self.minimum + ratio * span

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        label = fonts.ui.render(self.label, True, theme.text)
        value = fonts.small.render(self.formatter(self.value), True, theme.teal)
        surface.blit(label, (self.rect.x, self.rect.y))
        surface.blit(value, (self.rect.right - value.get_width(), self.rect.y + 2))
        draw_round_rect(surface, theme.track, self.track, radius=2)
        ratio = (
            0.0
            if self.maximum == self.minimum
            else (self.value - self.minimum) / (self.maximum - self.minimum)
        )
        fill_w = int(self.track.w * min(1.0, max(0.0, ratio)))
        if fill_w > 0:
            fill = pygame.Rect(self.track.x, self.track.y, fill_w, self.track.h)
            draw_round_rect(surface, theme.teal, fill, radius=2)
        knob = (self.track.x + fill_w, self.track.centery)
        pygame.draw.circle(surface, theme.white, knob, 7)
        pygame.draw.circle(surface, theme.teal, knob, 7, 2)


class Segmented:
    def __init__(self, options: list[tuple[str, str]]) -> None:
        self.options = options
        self.selected = options[0][0]
        self.rect = pygame.Rect(0, 0, 10, 10)

    def place(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        if not self.rect.collidepoint(event.pos) or not self.options:
            return False
        cell = self.rect.w / len(self.options)
        index = int((event.pos[0] - self.rect.x) / max(cell, 1))
        index = min(len(self.options) - 1, max(0, index))
        chosen = self.options[index][0]
        if chosen == self.selected:
            return False
        self.selected = chosen
        return True

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        draw_round_rect(surface, theme.surface, self.rect, radius=12)
        draw_round_rect(surface, theme.border, self.rect, radius=12, width=1)
        if not self.options:
            return
        cell_w = self.rect.w / len(self.options)
        for index, (key, label) in enumerate(self.options):
            cell = pygame.Rect(self.rect.x + index * cell_w, self.rect.y, cell_w, self.rect.h)
            if key == self.selected:
                draw_round_rect(surface, theme.teal, cell.inflate(-6, -6), radius=9)
                color = theme.white
            else:
                color = theme.text
            text = fonts.ui.render(label, True, color)
            surface.blit(text, text.get_rect(center=cell.center))


class Toggle:
    def __init__(self, label: str, value: bool = False) -> None:
        self.label = label
        self.value = value
        self.rect = pygame.Rect(0, 0, 10, 10)

    def place(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value
                return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        text = fonts.small.render(self.label, True, theme.text)
        surface.blit(text, (self.rect.x, self.rect.centery - text.get_height() // 2))
        switch = pygame.Rect(self.rect.right - 36, self.rect.centery - 9, 36, 18)
        draw_round_rect(surface, theme.teal if self.value else theme.border, switch, radius=9)
        knob_x = switch.right - 10 if self.value else switch.x + 10
        pygame.draw.circle(surface, theme.white, (knob_x, switch.centery), 6)


class Stepper:
    def __init__(self, label: str, value: int, minimum: int, maximum: int, step: int = 1) -> None:
        self.label = label
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.minus = pygame.Rect(0, 0, 22, 22)
        self.plus = pygame.Rect(0, 0, 22, 22)

    def place(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)
        self.plus = pygame.Rect(rect.right - 22, rect.centery - 11, 22, 22)
        self.minus = pygame.Rect(rect.right - 78, rect.centery - 11, 22, 22)

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        if self.minus.collidepoint(event.pos) and self.value - self.step >= self.minimum:
            self.value -= self.step
            return True
        if self.plus.collidepoint(event.pos) and self.value + self.step <= self.maximum:
            self.value += self.step
            return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        text = fonts.small.render(self.label, True, theme.text)
        surface.blit(text, (self.rect.x, self.rect.centery - text.get_height() // 2))
        for rect, glyph in ((self.minus, "−"), (self.plus, "+")):
            draw_round_rect(surface, theme.surface, rect, radius=6)
            draw_round_rect(surface, theme.border, rect, radius=6, width=1)
            mark = fonts.ui.render(glyph, True, theme.text)
            surface.blit(mark, mark.get_rect(center=rect.center))
        value = fonts.small.render(str(self.value), True, theme.teal)
        center_x = (self.minus.right + self.plus.left) // 2
        surface.blit(value, value.get_rect(center=(center_x, self.rect.centery)))


class PalettePicker:
    def __init__(self, selected: str = "aurora") -> None:
        self.ids = list(PALETTE_ORDER)
        self.selected = selected if selected in _PALETTE_IDS else "aurora"
        self.rects: list[pygame.Rect] = []

    def place(self, rect: pygame.Rect) -> None:
        gap = 8
        count = len(self.ids)
        size = (rect.w - gap * (count - 1)) // count
        self.rects = [
            pygame.Rect(rect.x + index * (size + gap), rect.y, size, rect.h)
            for index in range(count)
        ]

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        for key, rect in zip(self.ids, self.rects, strict=True):
            if rect.collidepoint(event.pos) and key != self.selected:
                self.selected = key
                return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts, theme: Theme) -> None:
        del fonts
        for key, rect in zip(self.ids, self.rects, strict=True):
            color = palette_colors(key)[0]
            draw_round_rect(surface, color, rect, radius=8)
            if key == self.selected:
                draw_round_rect(surface, theme.teal, rect, radius=8, width=3)
            else:
                draw_round_rect(surface, theme.border, rect, radius=8, width=1)
