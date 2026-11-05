"""Guardado PNG del lienzo visible. No graba video ni envía la imagen a ningún servicio."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pygame


def gallery_dir() -> Path:
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        root = cwd
    else:
        root = Path(__file__).resolve().parents[3]
    path = root / "outputs" / "gallery"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_png(surface: pygame.Surface, mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe = "".join(ch if ch.isalnum() else "-" for ch in mode) or "lienzo"
    path = gallery_dir() / f"mal_{safe}_{stamp}.png"
    pygame.image.save(surface, str(path))
    return path
