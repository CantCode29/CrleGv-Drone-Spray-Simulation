"""Validation targets against published data. Results go to runs/validation/."""

import json
import os
from pathlib import Path

import numpy as np

import grid
import params as P
import survival
import trajectory

RUNS_DIR = Path(os.environ.get("CRLEGV_RUNS_DIR", Path(__file__).resolve().parent.parent / "runs")) / "validation"

GUNN_KINZER_1949 = {0.05e-3: 0.27, 0.10e-3: 0.72, 0.25e-3: 2.06}
TARGET1_TOLERANCE = 0.20

HAN_2026_UPPER_LOWER = ((318.85, 168.01), (318.85, 210.64),
                        (600.74, 190.98), (600.74, 409.30))
CUI_2025_DDPR = 0.735
TARGET2_RATIOS = tuple(lower / upper for upper, lower in HAN_2026_UPPER_LOWER) + (CUI_2025_DDPR,)
TARGET2_BAND = (0.8 * min(TARGET2_RATIOS), 1.2 * max(TARGET2_RATIOS))

DEIDDA_1992_LIGHT = (0.24, 0.47)

MWANZA_2015_LD50_RATIO = 15.0
TARGET4_BAND = (MWANZA_2015_LD50_RATIO / 10.0, MWANZA_2015_LD50_RATIO * 10.0)
TARGET4_HOURS = (24.0, 48.0, 72.0, 504.0)


def fall_speed(r_p: float, dt: float = None, t_max: float = 30.0,
               tol: float = 1e-9) -> float:
    dt = P.TIME_STEP.value if dt is None else dt
    still_air = np.zeros(2)
    vel = np.zeros(2)
    for _ in range(int(t_max / dt)):
        new_vel = vel + trajectory.droplet_acceleration(vel, still_air, r_p) * dt
        if abs(new_vel[1] - vel[1]) < tol:
            return float(new_vel[1])
        vel = new_vel
    return float(vel[1])


def target1() -> dict:
    rows = []
    for r_p, measured in GUNN_KINZER_1949.items():
        simulated = fall_speed(r_p)
        deviation = (simulated - measured) / measured
        rows.append({
            "radius_mm": round(r_p * 1e3, 3),
            "diameter_mm": round(2 * r_p * 1e3, 3),
            "measured_m_s": measured,
            "simulated_m_s": round(simulated, 3),
            "deviation": round(deviation, 4),
            "within_tolerance": bool(abs(deviation) <= TARGET1_TOLERANCE),
        })
    return {
        "target": "T1 droplet fall speed vs Gunn and Kinzer (1949)",
        "tolerance": TARGET1_TOLERANCE,
        "time_step_s": P.TIME_STEP.value,
        "rows": rows,
        "passed": all(r["within_tolerance"] for r in rows),
    }


def _sig(x: float) -> float:
    return float(f"{x:.4g}")


def central_run():
    canopy = grid.build()
    deposition, _ = trajectory.run(canopy, n=int(P.N_DROPLETS.value))
    return canopy, deposition, grid.zones(canopy)


def target2(canopy, deposition: np.ndarray, masks: dict) -> dict:
    cell_area = canopy.dx ** 2
    density = {zone: float(deposition[masks[zone]].sum() / (masks[zone].sum() * cell_area))
               for zone in ("outer", "inner")}
    ratio = density["inner"] / density["outer"]
    low, high = TARGET2_BAND
    return {
        "target": "T2 inner / outer deposit per m2 vs UAV citrus lower / upper deposition",
        "benchmark_ratios": [round(r, 3) for r in TARGET2_RATIOS],
        "band": [round(low, 3), round(high, 3)],
        "outer_per_m2": _sig(density["outer"]),
        "inner_per_m2": _sig(density["inner"]),
        "ratio": _sig(ratio),
        "passed": bool(low <= ratio <= high),
    }


def zone_means(canopy, deposition: np.ndarray, masks: dict, hours) -> dict:
    means = {}
    for h in hours:
        surviving = survival.viable_fraction(canopy.intensity_ratio, h)
        summary = survival.zone_summary(deposition, surviving, masks)
        means[h] = {zone: s["mean_viable_fraction"] for zone, s in summary.items()}
    return means


def light_floor() -> dict:
    floors = {}
    for name, lai, k in (("central", P.LAI_TOTAL.value, P.EXTINCTION_COEFF.value),
                         ("most_light", P.LAI_TOTAL.low, P.EXTINCTION_COEFF.low),
                         ("least_light", P.LAI_TOTAL.high, P.EXTINCTION_COEFF.high)):
        canopy = grid.build(lai_total=lai, k=k)
        open_columns = ~(canopy.fruit_mask.any(axis=1) | (canopy.optical_depth == np.inf).any(axis=1))
        floors[name] = _sig(float(canopy.intensity_ratio[open_columns, -1].mean()))
    return {
        "target": "Light at the canopy floor, leaves only, vs Deidda et al. (1992)",
        "benchmark_range": list(DEIDDA_1992_LIGHT),
        "floor_light": floors,
        "comparison_only": True,
    }


def target4(means: dict) -> dict:
    rows = []
    for h in TARGET4_HOURS:
        m = means[h]
        rows.append({
            "hours": h,
            "inner": _sig(m["inner"]),
            "outer": _sig(m["outer"]),
            "microhabitat": _sig(m["microhabitat"]),
            "inner_over_outer": _sig(m["inner"] / m["outer"]),
            "microhabitat_over_outer": _sig(m["microhabitat"] / m["outer"]),
        })
    direction = all(means[h]["inner"] > means[h]["outer"] for h in TARGET4_HOURS)
    ratio_21d = means[TARGET4_HOURS[-1]]["inner"] / means[TARGET4_HOURS[-1]]["outer"]
    low, high = TARGET4_BAND
    magnitude = low <= ratio_21d <= high
    return {
        "target": "T4 inner vs outer viability against the 15x LD50 differential at 21 days",
        "band_21d": list(TARGET4_BAND),
        "rows": rows,
        "direction_holds": bool(direction),
        "ratio_21d": _sig(ratio_21d),
        "magnitude_holds": bool(magnitude),
        "passed": bool(direction and magnitude),
    }


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    canopy, deposition, masks = central_run()
    means = zone_means(canopy, deposition, masks, TARGET4_HOURS)
    results = {"target1": target1(), "target2": target2(canopy, deposition, masks),
               "target4": target4(means), "light": light_floor()}
    for name, result in results.items():
        (RUNS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(results, indent=2))
