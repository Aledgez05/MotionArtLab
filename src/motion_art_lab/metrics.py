"""Medición de FPS a partir de marcas de tiempo reales."""

from __future__ import annotations

import time
from collections import deque


class FrameMeter:
    def __init__(self, window: int = 30) -> None:
        self.samples: deque[float] = deque(maxlen=window)

    def tick(self, now: float | None = None) -> None:
        self.samples.append(time.perf_counter() if now is None else now)

    def fps(self) -> float | None:
        if len(self.samples) < 2:
            return None
        elapsed = self.samples[-1] - self.samples[0]
        if elapsed <= 0:
            return None
        return (len(self.samples) - 1) / elapsed

    def reset(self) -> None:
        self.samples.clear()
