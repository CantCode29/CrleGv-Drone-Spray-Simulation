"""Stage 3 - CrleGV viral survival.

The per-cell decay constant interpolates between the exposed and shaded
half-lives according to local UV intensity. The submitted chapter specifies
lambda(x,z) = lambda_base * [I(x,z)/I0], which drives lambda to zero in full
shade and so implies an infinite shaded half-life, contradicting the 72 h
bound the same chapter declares. The corrected form below reproduces both
published bounds at their respective limits. See correction log item B11.
"""

import numpy as np
import params as P


def decay_constants(intensity_ratio: np.ndarray) -> np.ndarray:
    lam_exposed = np.log(2.0) / P.HALF_LIFE_EXPOSED.value
    lam_shaded = np.log(2.0) / P.HALF_LIFE_SHADED.value
    return lam_shaded + (lam_exposed - lam_shaded) * intensity_ratio


def viable_fraction(intensity_ratio: np.ndarray, hours: float) -> np.ndarray:
    return np.exp(-decay_constants(intensity_ratio) * hours)


def survival_maps(intensity_ratio: np.ndarray, checkpoints=None) -> dict:
    checkpoints = P.CHECKPOINTS_H if checkpoints is None else checkpoints
    return {h: viable_fraction(intensity_ratio, h) for h in checkpoints}


def zone_summary(deposition: np.ndarray, survival: np.ndarray,
                 zone_masks: dict) -> dict:
    out = {}
    deposited = deposition > 0.0
    for name, mask in zone_masks.items():
        sel = mask & deposited
        out[name] = {
            "cells_with_deposit": int(sel.sum()),
            "deposit_total": float(deposition[mask].sum()),
            "mean_viable_fraction": float(survival[sel].mean()) if sel.any() else float("nan"),
        }
    return out
