import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import params as P
import trajectory
import validate


def force_balance_speed(r_p):
    rho_p, rho_a, mu_a, g = (P.SPRAY_DENSITY.value, P.AIR_DENSITY.value,
                             P.AIR_VISCOSITY.value, P.GRAVITY.value)
    mass = rho_p * (4 / 3) * np.pi * r_p ** 3
    area = np.pi * r_p ** 2
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        v = 0.5 * (lo + hi)
        re = rho_a * v * 2 * r_p / mu_a
        drag = 0.5 * rho_a * trajectory.drag_coefficient(re) * area * v * v
        lo, hi = (v, hi) if drag < mass * g else (lo, v)
    return 0.5 * (lo + hi)


def test_integrated_fall_speed_reaches_the_force_balance():
    for r_p in validate.GUNN_KINZER_1949:
        assert abs(validate.fall_speed(r_p) / force_balance_speed(r_p) - 1.0) < 1e-4


def test_fall_speed_rises_with_droplet_size():
    speeds = [validate.fall_speed(r) for r in sorted(validate.GUNN_KINZER_1949)]
    assert speeds == sorted(speeds)
