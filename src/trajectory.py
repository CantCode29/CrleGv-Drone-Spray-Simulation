"""Stage 2: where the spray lands.

Follows each droplet from the drone down through the canopy in small time
steps (forward Euler), pulled by gravity and slowed by air drag, using the
Betancourt et al. (2024) equation of motion and a drag coefficient that
changes with the Reynolds number. In every cell the droplet passes through,
the leaves and fruit catch part of the spray it still carries. Design
decisions D2 (catching a share of the spray in each cell) and D4 (downwash
fading with depth) are implemented here.
"""

import numpy as np

import params
from grid import Canopy

ACROSS = 0
DEPTH = 1
NEGLIGIBLE_SPRAY_LEFT = 1e-6
LONGEST_FLIGHT_S = 60.0


def drag_coefficient_for(reynolds_number: float) -> float:
    if reynolds_number < 1e-9:
        return 0.0
    if reynolds_number <= 1.0:
        return 24.0 / reynolds_number
    if reynolds_number <= 400.0:
        return 24.0 / reynolds_number ** 0.646
    if reynolds_number <= 3e5:
        return 0.5
    return 0.18


def air_velocity_at(depth: float, downwash_speed: float) -> np.ndarray:
    """Downwash from the rotors: full strength above the canopy, fading with
    depth inside it."""
    if depth <= 0.0:
        return np.array([0.0, downwash_speed])
    return np.array([0.0, downwash_speed * np.exp(-depth / params.DOWNWASH_FADE_DEPTH.value)])


def droplet_acceleration(droplet_velocity: np.ndarray, air_velocity: np.ndarray,
                         droplet_radius: float) -> np.ndarray:
    liquid_density = params.SPRAY_LIQUID_DENSITY.value
    air_density = params.AIR_DENSITY.value
    air_viscosity = params.AIR_VISCOSITY.value
    gravity = params.GRAVITY.value

    diameter = 2.0 * droplet_radius
    cross_section_area = np.pi * droplet_radius ** 2
    mass = liquid_density * (4.0 / 3.0) * np.pi * droplet_radius ** 3

    air_relative_to_droplet = air_velocity - droplet_velocity
    relative_speed = np.linalg.norm(air_relative_to_droplet)
    reynolds_number = air_density * relative_speed * diameter / air_viscosity
    drag_coefficient = drag_coefficient_for(reynolds_number)

    drag_force = (0.5 * air_density * drag_coefficient * cross_section_area
                  * relative_speed * air_relative_to_droplet)
    return drag_force / mass + np.array([0.0, gravity])


def follow_one_droplet(release_position: float, canopy: Canopy, drone_height: float,
                       drone_speed: float, downwash_speed: float, droplet_radius: float,
                       time_step: float = None):
    """Follow one droplet and record how much of its spray each cell catches.

    The slice wraps around from side to side. It stands for a piece of a
    continuous orchard row, so a droplet carried past the right-hand edge by
    the drone's forward speed comes back in at the left instead of being
    lost. Without this, the strip at the upwind edge, which droplets drift
    across before they reach the canopy, would receive no spray at all.
    """
    time_step = params.TIME_STEP.value if time_step is None else time_step
    spray_capture_per_leaf_area = canopy.light_extinction_coefficient
    fruit_spray_capture = params.FRUIT_SPRAY_CAPTURE.value
    canopy_width = canopy.width
    canopy_depth = canopy.depth

    position = np.array([release_position, -drone_height])
    velocity = np.array([drone_speed, downwash_speed])
    spray_left = 1.0
    spray_caught_by_cell = {}

    for _ in range(int(LONGEST_FLIGHT_S / time_step)):
        acceleration = droplet_acceleration(
            velocity, air_velocity_at(position[DEPTH], downwash_speed), droplet_radius)

        velocity = velocity + acceleration * time_step
        next_position = position + velocity * time_step
        step = next_position - position
        step_length = np.linalg.norm(step)

        if next_position[DEPTH] > 0.0 and spray_left > NEGLIGIBLE_SPRAY_LEFT and step_length > 0.0:
            pieces = max(1, int(np.ceil(step_length / (0.5 * canopy.cell_size))))
            piece_length = step_length / pieces
            for piece in range(pieces):
                point = position + step * ((piece + 0.5) / pieces)
                depth = point[DEPTH]
                if depth <= 0.0:
                    continue
                if depth >= canopy_depth:
                    break
                across = point[ACROSS] % canopy_width
                column, row = canopy.cell_containing(across, depth)
                capture_rate = canopy.leaf_area_density[column, row] * spray_capture_per_leaf_area
                if canopy.contains_fruit[column, row]:
                    capture_rate += fruit_spray_capture
                spray_caught = spray_left * (1.0 - np.exp(-capture_rate * piece_length))
                spray_caught_by_cell[(column, row)] = spray_caught_by_cell.get((column, row), 0.0) + spray_caught
                spray_left -= spray_caught
                if spray_left <= NEGLIGIBLE_SPRAY_LEFT:
                    break

        position = next_position

        if spray_left <= NEGLIGIBLE_SPRAY_LEFT:
            break
        if position[DEPTH] >= canopy_depth:
            break

    return spray_caught_by_cell, spray_left


def spray_canopy(canopy: Canopy, drone_height: float = None, drone_speed: float = None,
                 downwash_speed: float = None, droplet_radius: float = None,
                 droplets_released: int = None, time_step: float = None):
    """Release droplets evenly across the slice and add up where their spray lands."""
    drone_height = params.DRONE_HEIGHT_ABOVE_CANOPY.value if drone_height is None else drone_height
    drone_speed = params.DRONE_SPEED.value if drone_speed is None else drone_speed
    downwash_speed = params.DOWNWASH_SPEED.value if downwash_speed is None else downwash_speed
    droplet_radius = params.DROPLET_RADIUS.value if droplet_radius is None else droplet_radius
    droplets_released = (int(params.DROPLETS_PER_RUN.value) if droplets_released is None
                         else int(droplets_released))

    spray_landed = np.zeros((canopy.cells_across, canopy.cells_deep))
    spray_passing_through = 0.0
    release_positions = np.linspace(0.0, canopy.width, droplets_released, endpoint=False)

    for release_position in release_positions:
        spray_caught_by_cell, spray_left = follow_one_droplet(
            release_position, canopy, drone_height, drone_speed, downwash_speed,
            droplet_radius, time_step)
        for (column, row), amount in spray_caught_by_cell.items():
            spray_landed[column, row] += amount
        spray_passing_through += spray_left

    return spray_landed, spray_passing_through / droplets_released
