import numpy as np

import params
from grid import Canopy

ACROSS = 0
DEPTH = 1
TINY_AMOUNT_OF_SPRAY = 1e-6
LONGEST_FLIGHT_SECONDS = 60.0


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


def catch_spray_along_step(start: np.ndarray, end: np.ndarray, spray_left: float,
                           canopy: Canopy, spray_caught_by_cell: dict) -> float:
    step = end - start
    step_length = np.linalg.norm(step)
    if step_length == 0.0:
        return spray_left

    spray_caught_per_leaf_area = canopy.light_extinction_coefficient
    pieces = max(1, int(np.ceil(step_length / (0.5 * canopy.cell_size))))
    piece_length = step_length / pieces

    for piece in range(pieces):
        point = start + step * ((piece + 0.5) / pieces)
        depth = point[DEPTH]
        if depth <= 0.0:
            continue
        if depth >= canopy.depth:
            break
        across = point[ACROSS] % canopy.width
        column, row = canopy.cell_containing(across, depth)
        catch_rate = canopy.leaf_area_density[column, row] * spray_caught_per_leaf_area
        if canopy.contains_fruit[column, row]:
            catch_rate += params.FRUIT_SPRAY_CAPTURE.value
        spray_caught = spray_left * (1.0 - np.exp(-catch_rate * piece_length))
        spray_caught_by_cell[(column, row)] = spray_caught_by_cell.get((column, row), 0.0) + spray_caught
        spray_left -= spray_caught
        if spray_left <= TINY_AMOUNT_OF_SPRAY:
            break

    return spray_left


def follow_one_droplet(release_position: float, canopy: Canopy, drone_height: float,
                       drone_speed: float, downwash_speed: float, droplet_radius: float,
                       time_step: float = None):
    time_step = params.value_or_default(time_step, params.TIME_STEP)
    position = np.array([release_position, -drone_height])
    velocity = np.array([drone_speed, downwash_speed])
    spray_left = 1.0
    spray_caught_by_cell = {}

    for _ in range(int(LONGEST_FLIGHT_SECONDS / time_step)):
        air_velocity = air_velocity_at(position[DEPTH], downwash_speed)
        acceleration = droplet_acceleration(velocity, air_velocity, droplet_radius)
        velocity = velocity + acceleration * time_step
        next_position = position + velocity * time_step

        if next_position[DEPTH] > 0.0 and spray_left > TINY_AMOUNT_OF_SPRAY:
            spray_left = catch_spray_along_step(position, next_position, spray_left,
                                                canopy, spray_caught_by_cell)

        position = next_position
        if spray_left <= TINY_AMOUNT_OF_SPRAY or position[DEPTH] >= canopy.depth:
            break

    return spray_caught_by_cell, spray_left


def spray_canopy(canopy: Canopy, drone_height: float = None, drone_speed: float = None,
                 downwash_speed: float = None, droplet_radius: float = None,
                 droplets_released: int = None, time_step: float = None):
    drone_height = params.value_or_default(drone_height, params.DRONE_HEIGHT_ABOVE_CANOPY)
    drone_speed = params.value_or_default(drone_speed, params.DRONE_SPEED)
    downwash_speed = params.value_or_default(downwash_speed, params.DOWNWASH_SPEED)
    droplet_radius = params.value_or_default(droplet_radius, params.DROPLET_RADIUS)
    droplets_released = int(params.value_or_default(droplets_released, params.DROPLETS_PER_RUN))

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
