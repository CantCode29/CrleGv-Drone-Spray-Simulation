import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import params
import trajectory
import validate


def speed_where_drag_balances_weight(droplet_radius):
    liquid_density, air_density, air_viscosity, gravity = (
        params.SPRAY_LIQUID_DENSITY.value, params.AIR_DENSITY.value,
        params.AIR_VISCOSITY.value, params.GRAVITY.value)
    mass = liquid_density * (4 / 3) * np.pi * droplet_radius ** 3
    cross_section_area = np.pi * droplet_radius ** 2
    too_slow, too_fast = 1e-6, 50.0
    for _ in range(200):
        speed = 0.5 * (too_slow + too_fast)
        reynolds_number = air_density * speed * 2 * droplet_radius / air_viscosity
        drag = (0.5 * air_density * trajectory.drag_coefficient_for(reynolds_number)
                * cross_section_area * speed * speed)
        too_slow, too_fast = (speed, too_fast) if drag < mass * gravity else (too_slow, speed)
    return 0.5 * (too_slow + too_fast)


def test_integrated_fall_speed_reaches_the_force_balance():
    for droplet_radius in validate.GUNN_KINZER_1949_FALL_SPEEDS:
        simulated = validate.steady_fall_speed(droplet_radius)
        assert abs(simulated / speed_where_drag_balances_weight(droplet_radius) - 1.0) < 1e-4


def test_fall_speed_rises_with_droplet_size():
    speeds = [validate.steady_fall_speed(radius) for radius in sorted(validate.GUNN_KINZER_1949_FALL_SPEEDS)]
    assert speeds == sorted(speeds)
