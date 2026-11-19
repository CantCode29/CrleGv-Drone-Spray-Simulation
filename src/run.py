import json
import os
import time
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

import numpy as np

import grid
import params
import survival
import trajectory

RUNS_DIR = Path(os.environ.get("CRLEGV_RUNS_DIR", Path(__file__).resolve().parent.parent / "runs"))


def run_simulation(run_name: str, drone_height=None, drone_speed=None, downwash_speed=None,
                   droplet_radius=None, leaf_area_index=None, light_extinction_coefficient=None,
                   droplets_released=None, save_to_disk=True) -> dict:
    started = time.time()
    drone_height = params.value_or_default(drone_height, params.DRONE_HEIGHT_ABOVE_CANOPY)
    drone_speed = params.value_or_default(drone_speed, params.DRONE_SPEED)
    downwash_speed = params.value_or_default(downwash_speed, params.DOWNWASH_SPEED)
    droplet_radius = params.value_or_default(droplet_radius, params.DROPLET_RADIUS)
    leaf_area_index = params.value_or_default(leaf_area_index, params.LEAF_AREA_INDEX)
    light_extinction_coefficient = params.value_or_default(light_extinction_coefficient,
                                                           params.LIGHT_EXTINCTION_COEFFICIENT)
    if droplets_released is None:
        droplets_released = int(params.DROPLETS_PER_RUN.value)

    canopy = grid.build_canopy(leaf_area_index=leaf_area_index,
                               light_extinction_coefficient=light_extinction_coefficient)
    spray_landed, share_passing_through = trajectory.spray_canopy(
        canopy, drone_height=drone_height, drone_speed=drone_speed,
        downwash_speed=downwash_speed, droplet_radius=droplet_radius,
        droplets_released=droplets_released)
    still_active_maps = survival.share_still_active_maps(canopy.share_of_full_light)
    zones = grid.canopy_zones(canopy)

    virus_by_zone = {}
    for hours, still_active in still_active_maps.items():
        virus_by_zone[str(hours)] = survival.summarise_zones(spray_landed, still_active, zones)
    spray_by_zone = summarise_spray_by_zone(canopy, spray_landed, zones)
    percent_by_zone = {}
    for zone_name, totals in spray_by_zone.items():
        percent_by_zone[zone_name] = 100.0 * totals["spray_landed"] / droplets_released

    runtime = time.time() - started
    peak_memory_mb = peak_memory_in_mb()

    record = {
        "run_name": run_name,
        "inputs": {
            "drone_height": drone_height,
            "drone_speed": drone_speed,
            "downwash_speed": downwash_speed,
            "droplet_radius": droplet_radius,
            "leaf_area_index": leaf_area_index,
            "light_extinction_coefficient": light_extinction_coefficient,
            "droplets_released": droplets_released,
        },
        "outputs": {
            "share_passing_through_canopy": float(share_passing_through),
            "share_landing_in_canopy": float(1.0 - share_passing_through),
            "depth_spray_reaches_m": depth_spray_reaches(canopy, spray_landed),
            "share_of_cells_in_deep_shade": float(canopy.deep_shade.mean()),
            "inner_spray_per_m2": spray_by_zone["inner"]["spray_per_m2"],
            "percent_of_spray_in_inner_canopy": percent_by_zone["inner"],
            "percent_of_spray_by_zone": percent_by_zone,
            "spray_by_zone": spray_by_zone,
            "virus_by_zone": virus_by_zone,
            "virus_compared_with_outer": compare_virus_with_outer(virus_by_zone),
        },
        "performance": {"runtime_s": round(runtime, 3), "peak_memory_mb": peak_memory_mb},
    }

    if save_to_disk:
        save_run(run_name, record, canopy, spray_landed, still_active_maps)
    return record


def peak_memory_in_mb():
    if resource is None:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def save_run(run_name, record, canopy, spray_landed, still_active_maps):
    folder = RUNS_DIR / run_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "record.json").write_text(json.dumps(record, indent=2))
    still_active_arrays = {f"still_active_{int(hours)}h": still_active
                           for hours, still_active in still_active_maps.items()}
    np.savez_compressed(
        folder / "arrays.npz",
        spray_landed=spray_landed,
        share_of_full_light=canopy.share_of_full_light,
        deep_shade=canopy.deep_shade,
        **still_active_arrays,
    )


def summarise_spray_by_zone(canopy, spray_landed, zones) -> dict:
    cell_area = canopy.cell_size ** 2
    summary = {}
    for zone_name, in_zone in zones.items():
        spray_in_zone = spray_landed[in_zone]
        if spray_in_zone.size:
            average_per_cell = float(spray_in_zone.mean())
            spray_per_m2 = float(spray_in_zone.sum() / (spray_in_zone.size * cell_area))
        else:
            average_per_cell = 0.0
            spray_per_m2 = 0.0
        if average_per_cell > 0:
            unevenness = float(spray_in_zone.std() / average_per_cell)
        else:
            unevenness = None
        summary[zone_name] = {
            "spray_landed": float(spray_in_zone.sum()),
            "spray_per_m2": spray_per_m2,
            "unevenness": unevenness,
        }
    return summary


def compare_virus_with_outer(virus_by_zone: dict) -> dict:
    comparison = {}
    for hours, zones in virus_by_zone.items():
        outer = zones["outer"]["average_share_still_active"]
        comparison[hours] = {}
        for zone_name in ("inner", "shaded_next_to_fruit"):
            zone = zones[zone_name]["average_share_still_active"]
            comparison[hours][zone_name] = {
                "difference_in_percentage_points": 100.0 * (zone - outer),
                "ratio_to_outer": zone / outer,
            }
    return comparison


def depth_spray_reaches(canopy, spray_landed, cutoff_share_of_busiest_row=0.01) -> float:
    spray_by_row = spray_landed.sum(axis=0)
    if spray_by_row.max() <= 0:
        return 0.0
    cutoff = spray_by_row.max() * cutoff_share_of_busiest_row
    rows_above_cutoff = np.nonzero(spray_by_row >= cutoff)[0]
    if len(rows_above_cutoff) == 0:
        return 0.0
    return float(canopy.row_depths[rows_above_cutoff[-1]])


if __name__ == "__main__":
    record = run_simulation("baseline", droplets_released=2000)
    print(json.dumps(record, indent=2))
