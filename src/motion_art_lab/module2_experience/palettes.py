"""Paletas artísticas. El color no indica certeza de detección."""

from __future__ import annotations

PALETTES: dict[str, dict[str, object]] = {
    "aurora": {
        "name": "Aurora",
        "colors": (
            (125, 229, 219),
            (185, 233, 230),
            (255, 250, 242),
            (58, 158, 157),
        ),
    },
    "coral": {
        "name": "Coral Bloom",
        "colors": (
            (215, 143, 122),
            (244, 194, 184),
            (255, 244, 236),
            (232, 168, 150),
        ),
    },
    "nocturno": {
        "name": "Nocturno",
        "colors": (
            (110, 150, 255),
            (176, 130, 255),
            (90, 230, 236),
            (214, 226, 255),
        ),
    },
    "monocromo": {
        "name": "Monocromo",
        "colors": (
            (255, 255, 255),
            (214, 220, 223),
            (168, 176, 180),
            (236, 236, 236),
        ),
    },
}

PALETTE_ORDER = ("aurora", "coral", "nocturno", "monocromo")


def palette_colors(name: str) -> tuple[tuple[int, int, int], ...]:
    entry = PALETTES.get(name, PALETTES["aurora"])
    colors = entry["colors"]
    return tuple(colors)  # type: ignore[arg-type]


def palette_name(name: str) -> str:
    entry = PALETTES.get(name, PALETTES["aurora"])
    return str(entry["name"])
