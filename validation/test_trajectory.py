import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import grid
import params
import trajectory


def test_drag_regimes_are_continuous_enough():
    assert trajectory.drag_coefficient_for(0.5) == 24.0 / 0.5
    assert abs(trajectory.drag_coefficient_for(1.0) - 24.0) < 1e-9
    assert trajectory.drag_coefficient_for(1e6) == 0.18


def test_all_spray_is_accounted_for():
    canopy = grid.build_canopy()
    spray_landed, share_passing_through = trajectory.spray_canopy(canopy, droplets_released=200)
    assert abs(spray_landed.sum() / 200 + share_passing_through - 1.0) < 1e-6


def test_a_1_mm_drop_falls_at_a_realistic_speed():
    time_step = 0.001
    liquid_density, air_density, air_viscosity, gravity = (
        params.SPRAY_LIQUID_DENSITY.value, params.AIR_DENSITY.value,
        params.AIR_VISCOSITY.value, params.GRAVITY.value)
    droplet_radius = 0.5e-3
    diameter, cross_section_area = 2 * droplet_radius, np.pi * droplet_radius ** 2
    mass = liquid_density * (4 / 3) * np.pi * droplet_radius ** 3
    speed = 0.0
    for _ in range(20000):
        reynolds_number = air_density * abs(speed) * diameter / air_viscosity
        drag_coefficient = trajectory.drag_coefficient_for(reynolds_number)
        drag = 0.5 * air_density * drag_coefficient * cross_section_area * abs(speed) * abs(speed)
        speed += (gravity - drag / mass) * time_step
    final_reynolds_number = air_density * speed * diameter / air_viscosity
    assert 3.0 < speed < 5.5, f"fall speed {speed:.2f} m/s outside plausible band"
    assert 1.0 < final_reynolds_number < 400.0, f"Re {final_reynolds_number:.0f} not in intermediate regime"


def test_denser_canopy_lets_less_spray_through():
    sparse = grid.build_canopy(leaf_area_index=2.0)
    dense = grid.build_canopy(leaf_area_index=3.6)
    _, passing_through_sparse = trajectory.spray_canopy(sparse, droplets_released=200)
    _, passing_through_dense = trajectory.spray_canopy(dense, droplets_released=200)
    assert passing_through_dense < passing_through_sparse


def test_droplet_at_rest_in_still_air_feels_only_gravity():
    acceleration = trajectory.droplet_acceleration(np.zeros(2), np.zeros(2), params.DROPLET_RADIUS.value)
    assert np.allclose(acceleration, [0.0, params.GRAVITY.value])


def test_drag_opposes_motion_relative_to_the_air():
    acceleration = trajectory.droplet_acceleration(np.array([3.0, 0.0]), np.zeros(2), params.DROPLET_RADIUS.value)
    assert acceleration[0] < 0.0
    assert acceleration[1] == params.GRAVITY.value
