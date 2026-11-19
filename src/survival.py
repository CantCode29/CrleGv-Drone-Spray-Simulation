import numpy as np

import params


def breakdown_rates(share_of_full_light: np.ndarray) -> np.ndarray:
    rate_in_full_sun = np.log(2.0) / params.VIRUS_HALF_LIFE_IN_FULL_SUN.value
    rate_in_full_shade = np.log(2.0) / params.VIRUS_HALF_LIFE_IN_FULL_SHADE.value
    return rate_in_full_shade + (rate_in_full_sun - rate_in_full_shade) * share_of_full_light


def share_still_active(share_of_full_light: np.ndarray, hours: float) -> np.ndarray:
    return np.exp(-breakdown_rates(share_of_full_light) * hours)


def share_still_active_maps(share_of_full_light: np.ndarray, checkpoint_hours=None) -> dict:
    if checkpoint_hours is None:
        checkpoint_hours = params.CHECKPOINT_HOURS
    return {hours: share_still_active(share_of_full_light, hours) for hours in checkpoint_hours}


def summarise_zones(spray_landed: np.ndarray, still_active: np.ndarray, zones: dict) -> dict:
    summary = {}
    received_spray = spray_landed > 0.0
    for zone_name, in_zone in zones.items():
        sprayed_cells = in_zone & received_spray
        if sprayed_cells.any():
            average_share_still_active = float(still_active[sprayed_cells].mean())
        else:
            average_share_still_active = float("nan")
        summary[zone_name] = {
            "cells_that_received_spray": int(sprayed_cells.sum()),
            "spray_landed": float(spray_landed[in_zone].sum()),
            "average_share_still_active": average_share_still_active,
        }
    return summary
