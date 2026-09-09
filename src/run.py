"""Driver - executes the three stages for one parameter set and persists it."""

import json
import time
from pathlib import Path

import numpy as np

import params as P
import grid
import trajectory
import survival

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def execute(run_id: str, altitude=None, v_uav=None, v_dw=None, r_p=None,
            lai=None, k=None, n=None, persist=True) -> dict:
    started = time.time()
    canopy = grid.build(lai_total=lai, k=k)
    deposition, escaped = trajectory.run(
        canopy, altitude=altitude, v_uav=v_uav, v_dw=v_dw, r_p=r_p, n=n)
    maps = survival.survival_maps(canopy.intensity_ratio)
    masks = grid.zones(canopy)

    summaries = {
        str(h): survival.zone_summary(deposition, maps[h], masks)
        for h in maps
    }
    runtime = time.time() - started

    record = {
        "run_id": run_id,
        "inputs": {
            "altitude": altitude if altitude is not None else P.UAV_ALTITUDE.value,
            "v_uav": v_uav if v_uav is not None else P.UAV_VELOCITY.value,
            "v_dw": v_dw if v_dw is not None else P.DOWNWASH_VELOCITY.value,
            "r_p": r_p if r_p is not None else P.DROPLET_RADIUS.value,
            "lai": lai if lai is not None else P.LAI_TOTAL.value,
            "k": k if k is not None else P.EXTINCTION_COEFF.value,
            "n_droplets": n if n is not None else int(P.N_DROPLETS.value),
        },
        "outputs": {
            "escaped_fraction": float(escaped),
            "deposited_fraction": float(1.0 - escaped),
            "penetration_depth_m": penetration_depth(canopy, deposition),
            "refuge_cell_fraction": float(canopy.refuge.mean()),
            "zone_summaries": summaries,
        },
        "performance": {"runtime_s": round(runtime, 3)},
    }

    if persist:
        out = RUNS_DIR / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "record.json").write_text(json.dumps(record, indent=2))
        np.savez_compressed(
            out / "arrays.npz",
            deposition=deposition,
            intensity_ratio=canopy.intensity_ratio,
            refuge=canopy.refuge,
            **{f"survival_{int(h)}h": maps[h] for h in maps},
        )

    return record


def penetration_depth(canopy, deposition, threshold_fraction=0.01) -> float:
    by_depth = deposition.sum(axis=0)
    if by_depth.max() <= 0:
        return 0.0
    threshold = by_depth.max() * threshold_fraction
    deep = np.nonzero(by_depth >= threshold)[0]
    return float(canopy.z_centres[deep[-1]]) if len(deep) else 0.0


if __name__ == "__main__":
    r = execute("baseline", n=2000)
    print(json.dumps(r, indent=2))
