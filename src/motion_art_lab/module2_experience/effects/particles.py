"""Partículas con vida, velocidad y desvanecimiento dependientes de dt."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    lifetime: float
    size: float
    color: tuple[int, int, int]

    def alpha(self) -> float:
        if self.lifetime <= 0:
            return 0.0
        remain = max(0.0, 1.0 - self.age / self.lifetime)
        return remain**1.15


class ParticleSystem:
    def __init__(self, limit: int = 350, seed: int = 7) -> None:
        self.limit = limit
        self.particles: list[Particle] = []
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self.particles.clear()

    def spawn(
        self,
        origins: list[tuple[float, float]],
        velocities: list[tuple[float, float]],
        intensity: float,
        persistence: float,
        colors: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
        motion_fraction: float = 1.0,
    ) -> int:
        if not origins or not colors:
            return 0
        energy = min(1.0, max(0.0, motion_fraction) * 12.0)
        if energy <= 0:
            return 0
        room = self.limit - len(self.particles)
        strength = min(1.0, max(0.0, intensity))
        count = int(round((4 + 16 * strength) * energy))
        count = max(0, min(count, room, 18))
        if count <= 0:
            return 0
        life_base = 0.45 + min(1.0, max(0.0, persistence)) * 1.7
        palette = list(colors)
        for _ in range(count):
            index = self._rng.randrange(len(origins))
            x, y = origins[index]
            vx, vy = velocities[index] if index < len(velocities) else (0.0, 0.0)
            angle = self._rng.random() * math.tau
            spread = 24.0 + 80.0 * strength
            jitter = 8.0 + 18.0 * strength
            self.particles.append(
                Particle(
                    x=x + self._rng.uniform(-jitter, jitter),
                    y=y + self._rng.uniform(-jitter, jitter),
                    vx=vx * 0.35 + math.cos(angle) * spread,
                    vy=vy * 0.35 + math.sin(angle) * spread - 10.0,
                    age=0.0,
                    lifetime=life_base * self._rng.uniform(0.75, 1.2),
                    size=1.6 + self._rng.random() * (1.8 + 3.5 * strength),
                    color=palette[self._rng.randrange(len(palette))],
                )
            )
        return count

    def update(self, dt: float, drag: float = 1.4) -> None:
        if dt < 0:
            raise ValueError("dt no puede ser negativo")
        decay = math.exp(-drag * dt)
        alive: list[Particle] = []
        for particle in self.particles:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vx *= decay
            particle.vy *= decay
            particle.age += dt
            if particle.age < particle.lifetime:
                alive.append(particle)
        self.particles = alive


def render_particles(surface: object, particles: list[Particle]) -> None:
    import pygame

    width, height = surface.get_size()
    for particle in particles:
        alpha = int(round(255 * particle.alpha()))
        if alpha <= 0:
            continue
        outside = (
            particle.x < -30
            or particle.y < -30
            or particle.x > width + 30
            or particle.y > height + 30
        )
        if outside:
            continue
        radius = max(1, int(round(particle.size)))
        pos = (int(round(particle.x)), int(round(particle.y)))
        pygame.draw.circle(surface, (*particle.color, max(18, alpha // 4)), pos, radius + 3)
        pygame.draw.circle(surface, (*particle.color, alpha), pos, radius)
