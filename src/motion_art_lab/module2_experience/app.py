"""Bucle del estudio: eventos, procesamiento, efectos y dibujo."""

from __future__ import annotations

import math
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from motion_art_lab.config import (
    ArtisticSettings,
    ProcessingSettings,
    sensitivity_from_threshold,
    threshold_from_sensitivity,
)
from motion_art_lab.contracts import MotionResult
from motion_art_lab.metrics import FrameMeter
from motion_art_lab.module1_processing.motion_features import (
    sample_emission_points,
    track_region_centroids,
    transform_coordinates,
)
from motion_art_lab.module1_processing.pipeline import MotionPipeline
from motion_art_lab.module1_processing.sources import CameraSource, SyntheticFrameSource
from motion_art_lab.module2_experience.debug_view import draw_diagnostics
from motion_art_lab.module2_experience.effects.compositor import blit_rgb, fit_rect
from motion_art_lab.module2_experience.effects.particles import ParticleSystem, render_particles
from motion_art_lab.module2_experience.effects.trails import TrailEffect
from motion_art_lab.module2_experience.export import gallery_dir, save_png
from motion_art_lab.module2_experience.palettes import palette_colors, palette_name
from motion_art_lab.module2_experience.theme import (
    Theme,
    blit_text,
    draw_corner_ticks,
    draw_round_rect,
    load_fonts,
)
from motion_art_lab.module2_experience.widgets import (
    Button,
    PalettePicker,
    Segmented,
    Slider,
    Stepper,
    Toggle,
)

_MARGIN = 22
_HEADER_H = 74
_FOOTER_H = 34
_PANEL_W = 304
_GAP = 16


class StudioApp:
    def __init__(self, prefer_demo: bool = False, self_test_frames: int | None = None) -> None:
        self.prefer_demo = prefer_demo
        self.self_test_frames = self_test_frames
        self.running = True
        self.paused = False
        self.advanced_open = False
        self.diagnostics = False
        self.connecting = False
        self.camera_lost = False
        self.source_kind = "synthetic"
        self._resume_pending = False
        self._failed_reads = 0
        self._frames = 0
        self._last_dt = 1.0 / 60.0
        self._closed = False
        self.probe_error: str | None = None
        self._pending_shot: str | None = None
        self.notice: tuple[str, float] | None = None
        self.result: MotionResult | None = None
        self._prev_regions: list = []
        self.layer_size = (0, 0)
        self._round_mask: pygame.Surface | None = None

        self.theme = Theme()
        self.fonts = load_fonts()
        self.screen = pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
        pygame.display.set_caption("Motion Art Lab")
        self.clock = pygame.time.Clock()
        self.meter = FrameMeter()
        self.processing = ProcessingSettings()
        self.artistic = ArtisticSettings()
        self.pipeline = MotionPipeline(self.processing)
        self.camera = CameraSource()
        self.synthetic = SyntheticFrameSource()
        self.trails = TrailEffect()
        self.particles = ParticleSystem(limit=self.artistic.particle_limit)
        self.rng = np.random.default_rng(4)
        self._build_widgets()
        self.canvas_rect = pygame.Rect(0, 0, 10, 10)
        self.panel = pygame.Rect(0, 0, 10, 10)
        self.header = pygame.Rect(0, 0, 10, 10)
        self.footer = pygame.Rect(0, 0, 10, 10)
        self.metrics_rect = pygame.Rect(0, 0, 10, 10)
        self.canvas_layer = pygame.Surface((10, 10))
        self.art = pygame.Surface((10, 10))
        self.particle_layer = pygame.Surface((10, 10), pygame.SRCALPHA)
        self._palette_label_y = 0

    def run(self) -> int:
        self._layout()
        self._open_initial_source()
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            self._frames += 1
            self.meter.tick()
            self._layout()
            self._handle_events()
            self._update(dt)
            self._self_test_hooks()
            self._draw()
            pygame.display.flip()
            if self.self_test_frames is not None and self._frames >= self.self_test_frames:
                self.running = False
        self._report_self_test()
        return 2 if self.probe_error else 0

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.camera.release()
        self.synthetic.release()

    def _build_widgets(self) -> None:
        threshold = self.processing.threshold
        self.sensitivity = Slider(
            "Sensibilidad",
            sensitivity_from_threshold(threshold),
            0.0,
            1.0,
            lambda value: f"T {threshold_from_sensitivity(value)}",
        )
        self.intensity = Slider(
            "Intensidad",
            self.artistic.intensity,
            0.0,
            1.0,
            lambda value: f"{round(value * 100)} %",
        )
        self.persistence = Slider(
            "Persistencia",
            self.artistic.persistence,
            0.0,
            1.0,
            lambda value: f"{round(value * 100)} %",
        )
        self.mode = Segmented([("trails", "Rastros"), ("particles", "Partículas")])
        self.palettes = PalettePicker(self.artistic.palette)
        self.advanced_button = Button("Ajustes avanzados")
        self.filter_toggle = Toggle("Filtro promedio 3×3", self.processing.apply_mean_filter)
        self.opening = Stepper("Apertura", self.processing.opening_iterations, 0, 2)
        self.closing = Stepper("Cierre", self.processing.closing_iterations, 0, 2)
        self.area = Stepper("Área mínima", self.processing.min_component_area, 10, 200, step=10)
        self.mirror = Toggle("Espejo", self.artistic.mirror_preview)
        self.background = Toggle("Fondo de cámara", self.artistic.show_camera_background)
        self.source = Segmented([("camera", "Cámara"), ("synthetic", "Demo")])
        self.source.selected = "synthetic"
        self.pause_button = Button("Pausar", "primary")
        self.clear_button = Button("Limpiar")
        self.save_button = Button("Guardar PNG")
        self.diag_button = Button("Diagnóstico")

    def _open_initial_source(self) -> None:
        if self.prefer_demo:
            self.source.selected = "synthetic"
            self.source_kind = "synthetic"
            return
        self._set_source("camera")

    def _layout(self) -> None:
        width, height = self.screen.get_size()
        if width < 960 or height < 640:
            self.screen = pygame.display.set_mode(
                (max(width, 960), max(height, 640)),
                pygame.RESIZABLE,
            )
            width, height = self.screen.get_size()
        self.header = pygame.Rect(_MARGIN, 8, width - 2 * _MARGIN, _HEADER_H)
        self.footer = pygame.Rect(_MARGIN, height - _FOOTER_H - 6, width - 2 * _MARGIN, _FOOTER_H)
        body_top = self.header.bottom + 4
        body_h = self.footer.top - 8 - body_top
        self.panel = pygame.Rect(width - _MARGIN - _PANEL_W, body_top, _PANEL_W, body_h)
        stage = pygame.Rect(
            _MARGIN,
            body_top,
            self.panel.left - _MARGIN - _GAP,
            body_h,
        )
        self.buttons_rect = pygame.Rect(stage.x, stage.bottom - 42, stage.w, 42)
        self.metrics_rect = pygame.Rect(stage.x, self.buttons_rect.y - 24, stage.w, 18)
        self.canvas_rect = pygame.Rect(
            stage.x,
            stage.y,
            stage.w,
            max(2, self.metrics_rect.y - stage.y - 8),
        )
        self._place_buttons()
        self._place_panel()
        self._ensure_layers()

    def _place_buttons(self) -> None:
        buttons = [self.pause_button, self.clear_button, self.save_button, self.diag_button]
        gap = 8
        width = (self.buttons_rect.w - gap * (len(buttons) - 1)) // len(buttons)
        for index, button in enumerate(buttons):
            button.place(
                pygame.Rect(
                    self.buttons_rect.x + index * (width + gap),
                    self.buttons_rect.y,
                    width,
                    self.buttons_rect.h,
                )
            )

    def _place_panel(self) -> None:
        pad = 18
        x = self.panel.x + pad
        width = self.panel.w - 2 * pad
        y = self.panel.y + 86
        self.mode.place(pygame.Rect(x, y, width, 36))
        y += 44
        self.sensitivity.place(pygame.Rect(x, y, width, 46))
        y += 50
        self.intensity.place(pygame.Rect(x, y, width, 46))
        y += 50
        self.persistence.place(pygame.Rect(x, y, width, 46))
        y += 52
        self._palette_label_y = y
        y += 18
        self.palettes.place(pygame.Rect(x, y, width, 32))
        y += 42
        self.advanced_button.place(pygame.Rect(x, y, width, 32))
        y += 44
        if self.advanced_open:
            for control in (
                self.filter_toggle,
                self.opening,
                self.closing,
                self.area,
                self.mirror,
                self.background,
            ):
                control.place(pygame.Rect(x, y, width, 26))
                y += 28
            y += 8
        source_y = min(y + 16, self.panel.bottom - 52)
        self.source.place(pygame.Rect(x, source_y, width, 36))

    def _ensure_layers(self) -> None:
        size = (max(2, self.canvas_rect.w), max(2, self.canvas_rect.h))
        if self.layer_size == size:
            return
        self.layer_size = size
        self.canvas_layer = pygame.Surface(size)
        self.art = pygame.Surface(size)
        self.particle_layer = pygame.Surface(size, pygame.SRCALPHA)
        self.trails.ensure(size)
        self.particles.reset()
        mask = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=18)
        self._round_mask = mask

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._on_key(event)
            else:
                self._dispatch(event)

    def _on_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_SPACE:
            self._toggle_pause()
        elif event.key == pygame.K_1:
            self.mode.selected = "trails"
        elif event.key == pygame.K_2:
            self.mode.selected = "particles"
        elif event.key == pygame.K_c:
            self._clear_canvas()
        elif event.key == pygame.K_s:
            self._save()
        elif event.key == pygame.K_d:
            self.diagnostics = not self.diagnostics
        elif event.key == pygame.K_ESCAPE:
            if self.advanced_open:
                self.advanced_open = False
            else:
                self.running = False

    def _dispatch(self, event: pygame.event.Event) -> None:
        if self.pause_button.handle(event):
            self._toggle_pause()
        if self.clear_button.handle(event):
            self._clear_canvas()
        if self.save_button.handle(event):
            self._save()
        if self.diag_button.handle(event):
            self.diagnostics = not self.diagnostics
        self.sensitivity.handle(event)
        self.intensity.handle(event)
        self.persistence.handle(event)
        self.mode.handle(event)
        self.palettes.handle(event)
        if self.advanced_button.handle(event):
            self.advanced_open = not self.advanced_open
        if self.advanced_open:
            self.filter_toggle.handle(event)
            self.opening.handle(event)
            self.closing.handle(event)
            self.area.handle(event)
            self.mirror.handle(event)
            self.background.handle(event)
        if self.source.handle(event):
            self._set_source(self.source.selected)

    def _apply_controls(self) -> None:
        proc = ProcessingSettings(
            threshold=threshold_from_sensitivity(self.sensitivity.value),
            apply_mean_filter=self.filter_toggle.value,
            mean_filter_size=3,
            opening_iterations=self.opening.value,
            closing_iterations=self.closing.value,
            min_component_area=self.area.value,
        )
        if proc != self.processing:
            self.processing = proc
            self.pipeline.update_settings(proc)
        self.artistic = ArtisticSettings(
            mode=self.mode.selected,
            palette=self.palettes.selected,
            intensity=self.intensity.value,
            persistence=self.persistence.value,
            particle_limit=350,
            mirror_preview=self.mirror.value,
            show_camera_background=self.background.value,
            show_diagnostics=self.diagnostics,
        )
        self.particles.limit = self.artistic.particle_limit

    def _update(self, dt: float) -> None:
        self._apply_controls()
        self._last_dt = dt
        if self.paused:
            return
        if self._resume_pending:
            self._resume_pending = False
            self.pipeline.reset()
            self._prev_regions = []
            self.trails.break_tracks()
        ok, frame, timestamp = self._acquire()
        if not ok or frame is None:
            self._note_read_failure()
            return
        self._failed_reads = 0
        if self.source_kind == "camera":
            self.camera_lost = False
        try:
            result = self.pipeline.process_bgr(frame, timestamp)
        except ValueError as exc:
            self._notify(str(exc))
            return
        self.result = result
        if result.status != "ready":
            self._prev_regions = []
            return
        self._drive_effects(result, dt)

    def _acquire(self) -> tuple[bool, np.ndarray | None, float]:
        if self.source_kind == "camera":
            return self.camera.read()
        return self.synthetic.read()

    def _note_read_failure(self) -> None:
        self._failed_reads += 1
        if self.source_kind == "camera" and self._failed_reads == 12:
            self.camera_lost = True
            self._notify("La cámara dejó de responder. La interfaz sigue activa.")

    def _drive_effects(self, result: MotionResult, dt: float) -> None:
        fitted = self._fitted_rect(result)
        colors = palette_colors(self.artistic.palette)
        if self.artistic.mode == "trails":
            # Solo las regiones grandes: los fragmentos del contorno no dibujan saltos.
            stroke_regions = sorted(
                result.regions, key=lambda region: region.area_px, reverse=True
            )[:3]
            points = [
                transform_coordinates(
                    region.centroid_xy,
                    (result.frame_rgb.shape[1], result.frame_rgb.shape[0]),
                    fitted,
                    self.artistic.mirror_preview,
                )
                for region in stroke_regions
            ]
            self.trails.update(
                points,
                dt,
                self.artistic.intensity,
                self.artistic.persistence,
                colors,
            )
        else:
            self.trails.break_tracks()
            origins, velocities = self._emissions(result, fitted)
            self.particles.spawn(
                origins,
                velocities,
                self.artistic.intensity,
                self.artistic.persistence,
                colors,
                result.motion_fraction,
            )
        self.particles.update(dt)
        self._prev_regions = list(result.regions)

    def _fitted_rect(self, result: MotionResult) -> tuple[int, int, int, int]:
        height, width = result.frame_rgb.shape[:2]
        return fit_rect(width, height, (0, 0, self.art.get_width(), self.art.get_height()))

    def _emissions(
        self,
        result: MotionResult,
        fitted: tuple[int, int, int, int],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        samples = sample_emission_points(result.clean_mask, result.regions, 16, self.rng)
        if not samples:
            samples = [region.centroid_xy for region in result.regions]
        if not samples:
            return [], []
        src_w = result.frame_rgb.shape[1]
        src_h = result.frame_rgb.shape[0]
        scale_x = fitted[2] / src_w if src_w else 1.0
        scale_y = fitted[3] / src_h if src_h else 1.0
        pairs = track_region_centroids(self._prev_regions, result.regions, 70)
        velocity_by_region: dict[int, tuple[float, float]] = {}
        dt = max(self._last_dt, 1e-3)
        for _prev_i, curr_i, dx, dy in pairs:
            vx = dx / dt * scale_x
            vy = dy / dt * scale_y
            speed = math.hypot(vx, vy)
            if speed > 480:
                vx *= 480 / speed
                vy *= 480 / speed
            velocity_by_region[curr_i] = (vx, vy)
        centroids = [region.centroid_xy for region in result.regions]
        origins: list[tuple[float, float]] = []
        velocities: list[tuple[float, float]] = []
        for sx, sy in samples:
            origins.append(
                transform_coordinates(
                    (sx, sy), (src_w, src_h), fitted, self.artistic.mirror_preview
                )
            )
            if not centroids:
                velocities.append((0.0, 0.0))
                continue
            nearest = min(
                range(len(centroids)),
                key=lambda i: (centroids[i][0] - sx) ** 2 + (centroids[i][1] - sy) ** 2,
            )
            velocities.append(velocity_by_region.get(nearest, (0.0, 0.0)))
        return origins, velocities

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        if not self.paused:
            self._resume_pending = True

    def _clear_canvas(self) -> None:
        self.trails.clear()
        self.particles.reset()

    def _save(self) -> None:
        try:
            shot = self.screen.subsurface(self.canvas_rect).copy()
            mode = "diagnostico" if self.diagnostics else self.artistic.mode
            path = save_png(shot, mode)
        except (OSError, pygame.error) as exc:
            self._notify(f"No se pudo guardar: {exc}")
            return
        self._notify(f"Imagen guardada: {path.name}")

    def _set_source(self, kind: str) -> None:
        if kind == "camera":
            self.connecting = True
            self._draw()
            pygame.display.flip()
            opened = self.camera.open()
            self.connecting = False
            pygame.event.clear()
            if not opened:
                self.camera_lost = self.source_kind == "camera"
                self.source.selected = "camera" if self.source_kind == "camera" else "synthetic"
                message = self.camera.last_error or "No se pudo abrir la cámara."
                if self.source_kind != "camera":
                    self._notify(f"{message} Se activó la demo sintética.")
                else:
                    self._notify(message)
                return
            self.source_kind = "camera"
            self.source.selected = "camera"
            self.camera_lost = False
            self._failed_reads = 0
            self._reset_tracking()
            return
        self.camera.release()
        self.source_kind = "synthetic"
        self.source.selected = "synthetic"
        self.camera_lost = False
        self.synthetic.reset()
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self.pipeline.reset()
        self._prev_regions = []
        self.trails.break_tracks()

    def _notify(self, text: str, seconds: float = 3.4) -> None:
        self.notice = (text, time.perf_counter() + seconds)

    def _draw(self) -> None:
        self.screen.fill(self.theme.bg)
        self._draw_header()
        self._draw_canvas()
        self._draw_metrics()
        self._draw_buttons()
        self._draw_panel()
        self._draw_footer()
        self._draw_notice()
        if self._pending_shot:
            self._write_shot(self._pending_shot)
            self._pending_shot = None

    def _draw_header(self) -> None:
        blit_text(
            self.screen,
            "Motion Art Lab",
            self.fonts.display,
            self.theme.text,
            (self.header.x, self.header.y - 6),
        )
        pygame.draw.line(
            self.screen,
            self.theme.coral,
            (self.header.x, self.header.y + 44),
            (self.header.x + 42, self.header.y + 44),
            2,
        )
        blit_text(
            self.screen,
            "Convierte tus movimientos en arte",
            self.fonts.ui,
            self.theme.muted,
            (self.header.x, self.header.y + 50),
        )
        label, fg, bg = self._status_style()
        text = self.fonts.ui_bold.render(label, True, fg)
        pill = pygame.Rect(0, 0, text.get_width() + 36, 32)
        pill.right = self.header.right
        pill.centery = self.header.centery
        draw_round_rect(self.screen, bg, pill, radius=16)
        pygame.draw.circle(self.screen, fg, (pill.x + 14, pill.centery), 4)
        self.screen.blit(text, (pill.x + 24, pill.centery - text.get_height() // 2))

    def _status_style(self) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
        if self.connecting:
            return "Conectando…", self.theme.teal, self.theme.aqua
        if self.source_kind == "synthetic" and self.paused:
            return "Demo · en pausa", self.theme.text, self.theme.coral_soft
        if self.source_kind == "synthetic":
            return "Demo sintética", self.theme.text, self.theme.coral_soft
        if self.camera_lost:
            return "Sin cámara", self.theme.text, self.theme.coral_soft
        if self.paused:
            return "En pausa", self.theme.text, self.theme.aqua
        if self.result is None or self.result.status == "warmup":
            return "Calentando", self.theme.teal, self.theme.aqua
        return "En vivo", self.theme.white, self.theme.teal

    def _draw_canvas(self) -> None:
        self._compose_art()
        layer = self.canvas_layer
        layer.fill(self.theme.canvas)
        if self.diagnostics:
            draw_diagnostics(
                layer,
                self.result,
                self.art,
                self.fonts,
                self.theme,
                self.artistic.mirror_preview,
                self._measure_lines(),
            )
        else:
            layer.blit(self.art, (0, 0))
            draw_corner_ticks(layer, layer.get_rect(), self.theme.coral)
        if self.source_kind == "synthetic":
            self._draw_banner(layer)
        if self.connecting:
            self._draw_center_message(layer, "Conectando cámara…")
        elif self.source_kind == "camera" and self.camera_lost:
            self._draw_center_message(layer, "No llega imagen de la cámara.")
        self._blit_rounded(layer)

    def _compose_art(self) -> None:
        self.art.fill(self.theme.canvas)
        result = self.result
        if result is not None and self.artistic.show_camera_background:
            blit_rgb(
                self.art,
                result.frame_rgb,
                (0, 0, self.art.get_width(), self.art.get_height()),
                mirror=self.artistic.mirror_preview,
            )
        if self.artistic.mode == "trails" and self.trails.surface is not None:
            self.art.blit(self.trails.surface, (0, 0))
            return
        self.particle_layer.fill((0, 0, 0, 0))
        render_particles(self.particle_layer, self.particles.particles)
        self.art.blit(self.particle_layer, (0, 0))

    def _blit_rounded(self, layer: pygame.Surface) -> None:
        rounded = layer.convert_alpha()
        if self._round_mask is not None:
            rounded.blit(self._round_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(rounded, self.canvas_rect.topleft)

    def _draw_banner(self, layer: pygame.Surface) -> None:
        text = self.fonts.small.render("Demo sintética", True, self.theme.text)
        rect = pygame.Rect(16, 16, text.get_width() + 16, text.get_height() + 8)
        draw_round_rect(layer, self.theme.coral, rect, radius=10)
        layer.blit(text, (rect.x + 8, rect.y + 4))

    def _draw_center_message(self, layer: pygame.Surface, text: str) -> None:
        label = self.fonts.ui.render(text, True, self.theme.white)
        rect = label.get_rect(center=(layer.get_width() // 2, layer.get_height() // 2))
        draw_round_rect(layer, (28, 40, 48), rect.inflate(28, 16), radius=12)
        layer.blit(label, rect)

    def _draw_metrics(self) -> None:
        blit_text(
            self.screen,
            self._metric_text(),
            self.fonts.small,
            self.theme.muted,
            (self.metrics_rect.x, self.metrics_rect.y),
        )

    def _metric_text(self) -> str:
        fps = self.meter.fps()
        fps_txt = "—" if fps is None else f"{fps:.1f}"
        if self.result is None:
            proc, motion, regions = "—", "—", "—"
        else:
            proc = f"{self.result.processing_ms:.1f} ms"
            motion = f"{self.result.motion_fraction * 100:.1f} %"
            regions = str(len(self.result.regions))
        return (
            f"FPS {fps_txt}    Proceso {proc}    Movimiento {motion}"
            f"    Regiones {regions}    T {self.processing.threshold}"
        )

    def _measure_lines(self) -> list[str]:
        fps = self.meter.fps()
        result = self.result
        return [
            f"FPS  {'—' if fps is None else f'{fps:.1f}'}",
            f"Proceso  {'—' if result is None else f'{result.processing_ms:.1f} ms'}",
            f"Movimiento  {'—' if result is None else f'{result.motion_fraction * 100:.1f} %'}",
            f"Regiones  {'—' if result is None else str(len(result.regions))}",
            f"Energía  {'—' if result is None else f'{result.mean_motion_energy:.3f}'}",
            f"Umbral T  {self.processing.threshold}",
            f"Filtro 3×3  {'sí' if self.processing.apply_mean_filter else 'no'}",
            f"Apertura  {self.processing.opening_iterations}",
            f"Cierre  {self.processing.closing_iterations}",
            f"Estado  {'—' if result is None else result.status}",
            f"Detector  {'—' if result is None else result.detector_name}",
        ]

    def _draw_buttons(self) -> None:
        self.pause_button.label = "Reanudar" if self.paused else "Pausar"
        self.diag_button.active = self.diagnostics
        for button in (self.pause_button, self.clear_button, self.save_button, self.diag_button):
            button.draw(self.screen, self.fonts, self.theme)

    def _draw_panel(self) -> None:
        draw_round_rect(self.screen, self.theme.surface, self.panel, radius=18)
        draw_round_rect(self.screen, self.theme.border, self.panel, radius=18, width=1)
        x = self.panel.x + 18
        y = self.panel.y + 14
        blit_text(self.screen, "TU LIENZO", self.fonts.small, self.theme.muted, (x, y))
        title = "Rastros de luz" if self.mode.selected == "trails" else "Partículas"
        blit_text(self.screen, title, self.fonts.display_small, self.theme.text, (x, y + 16))
        blit_text(
            self.screen,
            "Más sensibilidad, menor umbral.",
            self.fonts.small,
            self.theme.hint,
            (x, y + 50),
        )
        self.mode.draw(self.screen, self.fonts, self.theme)
        self.sensitivity.draw(self.screen, self.fonts, self.theme)
        self.intensity.draw(self.screen, self.fonts, self.theme)
        self.persistence.draw(self.screen, self.fonts, self.theme)
        blit_text(
            self.screen,
            f"Paleta   {palette_name(self.palettes.selected)}",
            self.fonts.small,
            self.theme.muted,
            (x, self._palette_label_y),
        )
        self.palettes.draw(self.screen, self.fonts, self.theme)
        self.advanced_button.label = (
            "Ocultar ajustes" if self.advanced_open else "Ajustes avanzados"
        )
        self.advanced_button.draw(self.screen, self.fonts, self.theme)
        if self.advanced_open:
            self.filter_toggle.draw(self.screen, self.fonts, self.theme)
            self.opening.draw(self.screen, self.fonts, self.theme)
            self.closing.draw(self.screen, self.fonts, self.theme)
            self.area.draw(self.screen, self.fonts, self.theme)
            self.mirror.draw(self.screen, self.fonts, self.theme)
            self.background.draw(self.screen, self.fonts, self.theme)
        blit_text(
            self.screen,
            "Fuente",
            self.fonts.small,
            self.theme.muted,
            (x, self.source.rect.y - 18),
        )
        self.source.draw(self.screen, self.fonts, self.theme)

    def _draw_footer(self) -> None:
        left = self.fonts.small.render(
            "Local · el video no sale de este equipo",
            True,
            self.theme.muted,
        )
        shortcuts = (
            "Espacio pausa   ·   1 rastros   ·   2 partículas   ·   "
            "C limpiar   ·   S guardar   ·   D diagnóstico"
        )
        right = self.fonts.small.render(shortcuts, True, self.theme.muted)
        y = self.footer.y + 8
        self.screen.blit(left, (self.footer.x, y))
        if self.footer.x + left.get_width() + 24 + right.get_width() <= self.footer.right:
            self.screen.blit(right, (self.footer.right - right.get_width(), y))

    def _draw_notice(self) -> None:
        if self.notice is None:
            return
        text, until = self.notice
        if time.perf_counter() > until:
            self.notice = None
            return
        label = self.fonts.ui.render(text, True, self.theme.text)
        rect = label.get_rect()
        rect.inflate_ip(28, 14)
        rect.centerx = self.canvas_rect.centerx
        rect.bottom = self.canvas_rect.bottom - 18
        draw_round_rect(self.screen, self.theme.coral_soft, rect, radius=12)
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _self_test_hooks(self) -> None:
        if self.self_test_frames is None:
            return
        if self._frames == 8:
            x = self.sensitivity.track.right - 1
            y = self.sensitivity.track.centery
            self.sensitivity.handle(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1)
            )
            self.sensitivity.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))
            self._apply_controls()
            if self.processing.threshold != 10:
                self._fail(f"el slider de sensibilidad dejó T={self.processing.threshold}")
            self.sensitivity.value = sensitivity_from_threshold(25)
            self._apply_controls()
        if self._frames == 12:
            rect = self.mode.rect
            pos = (int(rect.x + rect.w * 0.75), rect.centery)
            self.mode.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
            self._apply_controls()
            if self.artistic.mode != "particles":
                self._fail("el selector no cambió a partículas")
            self.mode.selected = "trails"
            self._apply_controls()
        if self._frames == 30:
            self._toggle_pause()
        if self._frames == 33:
            self._toggle_pause()
        if self._frames == 55:
            if self.result is None:
                self._fail("no hubo fotograma procesado")
            elif self.result.motion_fraction <= 0 or self.result.status != "ready":
                self._fail(
                    "la demo sintética no produjo movimiento "
                    f"(estado={self.result.status}, fracción={self.result.motion_fraction:.4f})"
                )
        if self._frames == 70:
            self._pending_shot = "_selftest_estudio.png"
        if self._frames == 74:
            self.diagnostics = True
        if self._frames == 88:
            self._pending_shot = "_selftest_diagnostico.png"
        if self._frames == 90:
            self.diagnostics = False
            self.mode.selected = "particles"
        if self._frames == 118:
            self.particles.spawn([(12.0, 12.0)], [(0.0, 0.0)], 1.0, 0.5, ((255, 255, 255),), 1.0)
            self._clear_canvas()
            if self.particles.particles or self.trails._previous:
                self._fail("limpiar no vació el lienzo")
        if self._frames == 126:
            self._pending_shot = "_selftest_particulas.png"

    def _write_shot(self, name: str) -> None:
        folder = gallery_dir()
        canvas = self.screen.subsurface(self.canvas_rect).copy()
        pygame.image.save(canvas, str(folder / name))
        pygame.image.save(self.screen, str(folder / name.replace(".png", "_ventana.png")))

    def _fail(self, message: str) -> None:
        if self.probe_error is None:
            self.probe_error = message

    def _report_self_test(self) -> None:
        if self.self_test_frames is None:
            return
        fps = self.meter.fps()
        result = self.result
        fps_txt = "—" if fps is None else f"{fps:.1f}"
        if result is None:
            proc, motion, regions, status = "—", "—", "—", "—"
        else:
            proc = f"{result.processing_ms:.2f}ms"
            motion = f"{result.motion_fraction:.4f}"
            regions = str(len(result.regions))
            status = result.status
        print(
            "SELFTEST",
            "FAIL" if self.probe_error else "OK",
            f"fps={fps_txt}",
            f"proceso={proc}",
            f"movimiento={motion}",
            f"regiones={regions}",
            f"estado={status}",
        )
        if self.probe_error:
            print(f"SELFTEST DETAIL: {self.probe_error}")


def run_app(prefer_demo: bool = False, self_test_frames: int | None = None) -> int:
    pygame.init()
    app: StudioApp | None = None
    try:
        app = StudioApp(prefer_demo=prefer_demo, self_test_frames=self_test_frames)
        return app.run()
    except pygame.error as exc:
        print(f"No se pudo abrir la interfaz gráfica: {exc}")
        return 1
    finally:
        if app is not None:
            app.shutdown()
        pygame.quit()
