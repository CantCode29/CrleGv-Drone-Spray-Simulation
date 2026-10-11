import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import grid
import trajectory
import params as P


def test_drag_regimes_are_continuous_enough():
    assert trajectory.drag_coefficient(0.5) == 24.0 / 0.5
    assert abs(trajectory.drag_coefficient(1.0) - 24.0) < 1e-9
    assert trajectory.drag_coefficient(1e6) == 0.18


def test_mass_is_conserved():
    c = grid.build()
    dep, escaped = trajectory.run(c, n=200)
    assert abs(dep.sum() / 200 + escaped - 1.0) < 1e-6


def test_terminal_velocity_is_physical():
    """A 1 mm water droplet falls at roughly 4 m/s in still air. The Stokes
    closed form does not apply here - Re is order 250, in the intermediate
    branch of the piecewise drag law."""
    c = grid.build(depth=20.0)
    dt = 0.001
    rho_p, rho_a, mu_a, g = (P.SPRAY_DENSITY.value, P.AIR_DENSITY.value,
                             P.AIR_VISCOSITY.value, P.GRAVITY.value)
    r_p = 0.5e-3
    d, area = 2 * r_p, np.pi * r_p ** 2
    mass = rho_p * (4 / 3) * np.pi * r_p ** 3
    v = 0.0
    for _ in range(20000):
        speed = abs(v)
        re = rho_a * speed * d / mu_a
        cd = trajectory.drag_coefficient(re)
        a = g - (0.5 * rho_a * cd * area * speed * speed) / mass
        v += a * dt
    re_final = rho_a * v * d / mu_a
    assert 3.0 < v < 5.5, f"terminal velocity {v:.2f} m/s outside plausible band"
    assert 1.0 < re_final < 400.0, f"Re {re_final:.0f} not in intermediate regime"


def test_deeper_canopy_reduces_penetration():
    sparse = grid.build(lai_total=2.0)
    dense = grid.build(lai_total=3.6)
    dep_s, esc_s = trajectory.run(sparse, n=200)
    dep_d, esc_d = trajectory.run(dense, n=200)
    assert esc_d < esc_s


def test_droplet_at_rest_in_still_air_feels_only_gravity():
    acc = trajectory.droplet_acceleration(np.zeros(2), np.zeros(2), P.DROPLET_RADIUS.value)
    assert np.allclose(acc, [0.0, P.GRAVITY.value])


def test_drag_opposes_motion_relative_to_the_air():
    acc = trajectory.droplet_acceleration(np.array([3.0, 0.0]), np.zeros(2), P.DROPLET_RADIUS.value)
    assert acc[0] < 0.0
    assert acc[1] == P.GRAVITY.value
