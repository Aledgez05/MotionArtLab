from motion_art_lab.module2_experience.effects.particles import Particle, ParticleSystem


def test_particles_move_with_dt_and_are_removed_when_lifetime_ends():
    system = ParticleSystem(limit=10, seed=1)
    system.particles.append(Particle(0, 0, 100, 0, 0, 0.25, 2, (255, 255, 255)))
    system.update(0.1, drag=0)
    assert system.particles[0].x == 10
    assert system.particles[0].vx == 100
    assert system.particles[0].age == 0.1
    system.update(0.1, drag=0)
    assert len(system.particles) == 1
    system.update(0.1, drag=0)
    assert system.particles == []


def test_drag_reduces_velocity():
    system = ParticleSystem(limit=5, seed=1)
    system.particles.append(Particle(0, 0, 50, 0, 0, 5, 2, (1, 2, 3)))
    system.update(0.0, drag=1.4)
    assert system.particles[0].vx == 50
    system.update(1.0, drag=1.4)
    assert system.particles[0].vx < 50
    assert system.particles[0].x == 50


def test_spawn_respects_the_active_limit():
    system = ParticleSystem(limit=2, seed=3)
    origins = [(0.0, 0.0), (4.0, 4.0), (8.0, 1.0)]
    velocities = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    colors = ((10, 20, 30), (40, 50, 60))
    created = system.spawn(origins, velocities, 1.0, 0.5, colors, motion_fraction=1.0)
    assert created == 2
    assert len(system.particles) == 2
    extra = system.spawn(origins, velocities, 1.0, 0.5, colors, motion_fraction=1.0)
    assert extra == 0
    assert len(system.particles) == 2


def test_no_spawn_without_motion():
    system = ParticleSystem(limit=10, seed=1)
    created = system.spawn([(0.0, 0.0)], [(0.0, 0.0)], 1.0, 0.5, ((1, 1, 1),), 0.0)
    assert created == 0
    assert system.particles == []
