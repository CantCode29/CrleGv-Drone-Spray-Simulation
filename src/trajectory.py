"""Stage 2 - deposition mechanics.

Forward-Euler integration of the Betancourt et al. (2024) equation of motion
with a Reynolds-dependent piecewise drag coefficient. Design decisions D2
(fractional-mass interception) and D4 (downwash decay) are implemented here.
"""

import numpy as np
import params as P
from grid import Canopy


INTERCEPTION_EFFICIENCY = 1.0
FRUIT_INTERCEPTION = 12.0
DOWNWASH_DECAY_LENGTH = 0.5
MASS_FLOOR = 1e-6


def drag_coefficient(re: float) -> float:
    if re < 1e-9:
        return 0.0
    if re <= 1.0:
        return 24.0 / re
    if re <= 400.0:
        return 24.0 / re ** 0.646
    if re <= 3e5:
        return 0.5
    return 0.18


def air_velocity(z: float, v_dw: float) -> np.ndarray:
    """Downwash field. Full strength above the canopy, decaying with depth."""
    if z <= 0.0:
        return np.array([0.0, v_dw])
    return np.array([0.0, v_dw * np.exp(-z / DOWNWASH_DECAY_LENGTH)])


def droplet_acceleration(vel: np.ndarray, v_air: np.ndarray,
                         r_p: float) -> np.ndarray:
    rho_p = P.SPRAY_DENSITY.value
    rho_a = P.AIR_DENSITY.value
    mu_a = P.AIR_VISCOSITY.value
    g = P.GRAVITY.value

    d = 2.0 * r_p
    area = np.pi * r_p ** 2
    mass = rho_p * (4.0 / 3.0) * np.pi * r_p ** 3

    rel = v_air - vel
    speed = np.linalg.norm(rel)
    re = rho_a * speed * d / mu_a
    cd = drag_coefficient(re)

    f_drag = 0.5 * rho_a * cd * area * speed * rel
    return f_drag / mass + np.array([0.0, g])


def integrate_one(x0: float, canopy: Canopy, altitude: float, v_uav: float,
                  v_dw: float, r_p: float, dt: float = None):
    """Track one droplet, depositing intercepted mass fraction cell by cell.

    The domain wraps horizontally. The cross-section represents a slice of a
    continuous orchard row, so a droplet carried past the right-hand edge by
    the UAV forward velocity re-enters at the left rather than being lost.
    Without this, the strip at the upwind edge that droplets drift across
    before reaching the canopy receives no deposition at all.
    """
    dt = P.TIME_STEP.value if dt is None else dt

    pos = np.array([x0, -altitude])
    vel = np.array([v_uav, v_dw])
    remaining = 1.0
    deposits = {}

    max_steps = int(60.0 / dt)
    for _ in range(max_steps):
        acc = droplet_acceleration(vel, air_velocity(pos[1], v_dw), r_p)

        vel = vel + acc * dt
        new_pos = pos + vel * dt
        step = new_pos - pos
        ds = np.linalg.norm(step)

        if new_pos[1] > 0.0 and remaining > MASS_FLOOR and ds > 0.0:
            sub = max(1, int(np.ceil(ds / (0.5 * canopy.dx))))
            seg = ds / sub
            for s_i in range(sub):
                point = pos + step * ((s_i + 0.5) / sub)
                z = point[1]
                if z <= 0.0:
                    continue
                if z >= canopy.z_edges[-1]:
                    break
                x = point[0] % canopy.x_edges[-1]
                i, j = canopy.cell_of(x, z)
                optical = canopy.lad[i, j] * INTERCEPTION_EFFICIENCY
                if canopy.fruit_mask[i, j]:
                    optical += FRUIT_INTERCEPTION
                captured = remaining * (1.0 - np.exp(-optical * seg))
                deposits[(i, j)] = deposits.get((i, j), 0.0) + captured
                remaining -= captured
                if remaining <= MASS_FLOOR:
                    break

        pos = new_pos

        if remaining <= MASS_FLOOR:
            break
        if pos[1] >= canopy.z_edges[-1]:
            break

    return deposits, remaining


def run(canopy: Canopy, altitude: float = None, v_uav: float = None,
        v_dw: float = None, r_p: float = None, n: int = None,
        dt: float = None):
    """Release n droplets uniformly across the domain width."""
    altitude = P.UAV_ALTITUDE.value if altitude is None else altitude
    v_uav = P.UAV_VELOCITY.value if v_uav is None else v_uav
    v_dw = P.DOWNWASH_VELOCITY.value if v_dw is None else v_dw
    r_p = P.DROPLET_RADIUS.value if r_p is None else r_p
    n = int(P.N_DROPLETS.value) if n is None else int(n)

    deposition = np.zeros((canopy.nx, canopy.nz))
    escaped = 0.0
    width = canopy.x_edges[-1]
    starts = np.linspace(0.0, width, n, endpoint=False)

    for x0 in starts:
        deposits, remaining = integrate_one(
            x0, canopy, altitude, v_uav, v_dw, r_p, dt)
        for (i, j), amount in deposits.items():
            deposition[i, j] += amount
        escaped += remaining

    return deposition, escaped / n
