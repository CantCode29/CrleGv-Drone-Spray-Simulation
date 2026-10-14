"""Checks of the model against published data. Results go to runs/validation/."""

import json
import os
from pathlib import Path

import numpy as np

import grid
import params
import survival
import trajectory

VALIDATION_RESULTS_DIR = Path(os.environ.get(
    "CRLEGV_RUNS_DIR", Path(__file__).resolve().parent.parent / "runs")) / "validation"

GUNN_KINZER_1949_FALL_SPEEDS = {0.05e-3: 0.27, 0.10e-3: 0.72, 0.25e-3: 2.06}
FALL_SPEED_TOLERANCE = 0.20

HAN_2026_UPPER_AND_LOWER_DEPOSITS = ((318.85, 168.01), (318.85, 210.64),
                                     (600.74, 190.98), (600.74, 409.30))
CUI_2025_LOWER_OVER_UPPER = 0.735
PUBLISHED_LOWER_OVER_UPPER_RATIOS = (tuple(lower / upper for upper, lower in HAN_2026_UPPER_AND_LOWER_DEPOSITS)
                                     + (CUI_2025_LOWER_OVER_UPPER,))
INNER_OVER_OUTER_PASS_BAND = (0.8 * min(PUBLISHED_LOWER_OVER_UPPER_RATIOS),
                              1.2 * max(PUBLISHED_LOWER_OVER_UPPER_RATIOS))

DEIDDA_1992_FLOOR_LIGHT = (0.24, 0.47)

MWANZA_2015_LD50_RATIO = 15.0
SURVIVAL_RATIO_PASS_BAND = (MWANZA_2015_LD50_RATIO / 10.0, MWANZA_2015_LD50_RATIO * 10.0)
SURVIVAL_CHECK_HOURS = (24.0, 48.0, 72.0, 504.0)


def steady_fall_speed(droplet_radius: float, time_step: float = None, longest_time_s: float = 30.0,
                      settled_when_change_below: float = 1e-9) -> float:
    time_step = params.TIME_STEP.value if time_step is None else time_step
    still_air = np.zeros(2)
    velocity = np.zeros(2)
    for _ in range(int(longest_time_s / time_step)):
        next_velocity = velocity + trajectory.droplet_acceleration(velocity, still_air, droplet_radius) * time_step
        if abs(next_velocity[1] - velocity[1]) < settled_when_change_below:
            return float(next_velocity[1])
        velocity = next_velocity
    return float(velocity[1])


def check_fall_speed() -> dict:
    rows = []
    for droplet_radius, measured in GUNN_KINZER_1949_FALL_SPEEDS.items():
        simulated = steady_fall_speed(droplet_radius)
        deviation = (simulated - measured) / measured
        rows.append({
            "radius_mm": round(droplet_radius * 1e3, 3),
            "diameter_mm": round(2 * droplet_radius * 1e3, 3),
            "measured_m_s": measured,
            "simulated_m_s": round(simulated, 3),
            "deviation": round(deviation, 4),
            "within_tolerance": bool(abs(deviation) <= FALL_SPEED_TOLERANCE),
        })
    return {
        "target": "T1 droplet fall speed vs Gunn and Kinzer (1949)",
        "tolerance": FALL_SPEED_TOLERANCE,
        "time_step_s": params.TIME_STEP.value,
        "rows": rows,
        "passed": all(row["within_tolerance"] for row in rows),
    }


def _four_significant_figures(number: float) -> float:
    return float(f"{number:.4g}")


def run_central_settings():
    canopy = grid.build_canopy()
    spray_landed, _ = trajectory.spray_canopy(canopy, droplets_released=int(params.DROPLETS_PER_RUN.value))
    return canopy, spray_landed, grid.canopy_zones(canopy)


def check_spray_penetration(canopy, spray_landed: np.ndarray, zones: dict) -> dict:
    cell_area = canopy.cell_size ** 2
    spray_per_m2 = {zone_name: float(spray_landed[zones[zone_name]].sum()
                                     / (zones[zone_name].sum() * cell_area))
                    for zone_name in ("outer", "inner")}
    inner_over_outer = spray_per_m2["inner"] / spray_per_m2["outer"]
    low, high = INNER_OVER_OUTER_PASS_BAND
    return {
        "target": "T2 inner / outer deposit per m2 vs UAV citrus lower / upper deposition",
        "benchmark_ratios": [round(ratio, 3) for ratio in PUBLISHED_LOWER_OVER_UPPER_RATIOS],
        "band": [round(low, 3), round(high, 3)],
        "outer_per_m2": _four_significant_figures(spray_per_m2["outer"]),
        "inner_per_m2": _four_significant_figures(spray_per_m2["inner"]),
        "ratio": _four_significant_figures(inner_over_outer),
        "passed": bool(low <= inner_over_outer <= high),
    }


def average_share_still_active_by_zone(canopy, spray_landed: np.ndarray, zones: dict, hours) -> dict:
    averages = {}
    for hours_after_spraying in hours:
        still_active = survival.share_still_active(canopy.share_of_full_light, hours_after_spraying)
        summary = survival.summarise_zones(spray_landed, still_active, zones)
        averages[hours_after_spraying] = {zone_name: zone_summary["average_share_still_active"]
                                          for zone_name, zone_summary in summary.items()}
    return averages


def light_at_canopy_floor() -> dict:
    floor_light = {}
    for name, leaf_area_index, light_extinction_coefficient in (
            ("central", params.LEAF_AREA_INDEX.value, params.LIGHT_EXTINCTION_COEFFICIENT.value),
            ("most_light", params.LEAF_AREA_INDEX.low, params.LIGHT_EXTINCTION_COEFFICIENT.low),
            ("least_light", params.LEAF_AREA_INDEX.high, params.LIGHT_EXTINCTION_COEFFICIENT.high)):
        canopy = grid.build_canopy(leaf_area_index=leaf_area_index,
                                   light_extinction_coefficient=light_extinction_coefficient)
        columns_without_fruit = ~(canopy.contains_fruit.any(axis=1)
                                  | (canopy.light_blocking == np.inf).any(axis=1))
        floor_light[name] = _four_significant_figures(
            float(canopy.share_of_full_light[columns_without_fruit, -1].mean()))
    return {
        "target": "Light at the canopy floor, leaves only, vs Deidda et al. (1992)",
        "benchmark_range": list(DEIDDA_1992_FLOOR_LIGHT),
        "floor_light": floor_light,
        "comparison_only": True,
    }


def check_virus_survival(averages: dict) -> dict:
    rows = []
    for hours in SURVIVAL_CHECK_HOURS:
        at_this_time = averages[hours]
        rows.append({
            "hours": hours,
            "inner": _four_significant_figures(at_this_time["inner"]),
            "outer": _four_significant_figures(at_this_time["outer"]),
            "shaded_next_to_fruit": _four_significant_figures(at_this_time["shaded_next_to_fruit"]),
            "inner_over_outer": _four_significant_figures(at_this_time["inner"] / at_this_time["outer"]),
            "shaded_next_to_fruit_over_outer":
                _four_significant_figures(at_this_time["shaded_next_to_fruit"] / at_this_time["outer"]),
        })
    inner_always_higher = all(averages[hours]["inner"] > averages[hours]["outer"]
                              for hours in SURVIVAL_CHECK_HOURS)
    last_check = SURVIVAL_CHECK_HOURS[-1]
    ratio_at_21_days = averages[last_check]["inner"] / averages[last_check]["outer"]
    low, high = SURVIVAL_RATIO_PASS_BAND
    ratio_in_band = low <= ratio_at_21_days <= high
    return {
        "target": "T4 inner vs outer viability against the 15x LD50 differential at 21 days",
        "band_21d": list(SURVIVAL_RATIO_PASS_BAND),
        "rows": rows,
        "direction_holds": bool(inner_always_higher),
        "ratio_21d": _four_significant_figures(ratio_at_21_days),
        "magnitude_holds": bool(ratio_in_band),
        "passed": bool(inner_always_higher and ratio_in_band),
    }


if __name__ == "__main__":
    VALIDATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    canopy, spray_landed, zones = run_central_settings()
    averages = average_share_still_active_by_zone(canopy, spray_landed, zones, SURVIVAL_CHECK_HOURS)
    results = {"target1": check_fall_speed(),
               "target2": check_spray_penetration(canopy, spray_landed, zones),
               "target4": check_virus_survival(averages),
               "light": light_at_canopy_floor()}
    for name, result in results.items():
        (VALIDATION_RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(results, indent=2))
