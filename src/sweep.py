"""OFAT sensitivity sweep and the optional confirmation factorial (D5)."""

import itertools
import json
from pathlib import Path

import params as P
import run as runner

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

KEYS = {"h_uav": "altitude", "v_uav": "v_uav", "v_dw": "v_dw", "r_p": "r_p"}


def levels(param):
    low, high = param.bounds()
    return [low, param.value, high]


def ofat(n=None, metric="penetration_depth_m"):
    records = [runner.execute("ofat_baseline", n=n)]
    for param in P.SWEEP_PARAMS:
        kw_name = KEYS[param.symbol]
        for level in levels(param):
            if level == param.value:
                continue
            rid = f"ofat_{param.symbol}_{level:g}"
            records.append(runner.execute(rid, n=n, **{kw_name: level}))
    return records, coefficients(records, metric)


def coefficients(records, metric):
    base = records[0]
    y0 = base["outputs"][metric]
    out = {}
    for param in P.SWEEP_PARAMS:
        kw_name = KEYS[param.symbol]
        x0 = base["inputs"][kw_name]
        values = []
        for r in records[1:]:
            x = r["inputs"][kw_name]
            if x == x0:
                continue
            y = r["outputs"][metric]
            if x0 == 0 or y0 == 0:
                continue
            values.append(((y - y0) / y0) / ((x - x0) / x0))
        if values:
            out[param.symbol] = sum(values) / len(values)
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def factorial(n=None, metric="penetration_depth_m"):
    grids = {p.symbol: levels(p) for p in P.SWEEP_PARAMS}
    combos = list(itertools.product(*grids.values()))
    records = []
    for idx, combo in enumerate(combos):
        kwargs = {KEYS[sym]: val for sym, val in zip(grids.keys(), combo)}
        records.append(runner.execute(f"fact_{idx:03d}", n=n, **kwargs))
    best = max(records, key=lambda r: r["outputs"][metric])
    return records, best


if __name__ == "__main__":
    recs, coeffs = ofat(n=1000)
    print(json.dumps(coeffs, indent=2))
