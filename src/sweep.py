"""81-run full factorial (decision D5) and its descriptive effect analysis."""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

import params as P
import run as runner

RUNS_DIR = Path(os.environ.get("CRLEGV_RUNS_DIR", Path(__file__).resolve().parent.parent / "runs"))

KEYS = {"h_uav": "altitude", "v_uav": "v_uav", "v_dw": "v_dw", "r_p": "r_p"}
METRIC = "inner_share_pct"


def levels(param):
    low, high = param.bounds()
    return [low, param.value, high]


def design():
    grids = {p.symbol: levels(p) for p in P.SWEEP_PARAMS}
    return [{KEYS[sym]: val for sym, val in zip(grids, combo)}
            for combo in itertools.product(*grids.values())]


def factorial(n=None, persist=True, indices=None):
    runs = list(enumerate(design()))
    if indices is not None:
        runs = [(idx, kwargs) for idx, kwargs in runs if idx in indices]
    return [runner.execute(f"fact_{idx:02d}", n=n, persist=persist, **kwargs)
            for idx, kwargs in runs]


def load_records(runs_dir=None):
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    return [json.loads(path.read_text()) for path in sorted(runs_dir.glob("fact_*/record.json"))]


def summarise(records, runs_dir=None):
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    top = best(records)
    summary = {
        "best_inputs": top["inputs"],
        "best_value": top["outputs"][METRIC],
        "effects": effects(records),
    }
    (runs_dir / "factorial_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def best(records, metric=METRIC):
    return max(records, key=lambda r: r["outputs"][metric])


def effects(records, metric=METRIC):
    symbols = [p.symbol for p in P.SWEEP_PARAMS]
    x = {s: np.array([r["inputs"][KEYS[s]] for r in records], dtype=float) for s in symbols}
    y = np.array([r["outputs"][metric] for r in records], dtype=float)
    grand = float(y.mean())
    total = float(((y - grand) ** 2).sum())

    def share(ss):
        return ss / total if total > 0 else 0.0

    main = {}
    for s in symbols:
        values = np.unique(x[s])
        means = np.array([y[x[s] == v].mean() for v in values])
        ss = sum(float((x[s] == v).sum()) * (m - grand) ** 2 for v, m in zip(values, means))
        low, central, high = values
        sensitivity = (((means[2] - means[0]) / means[1]) / ((high - low) / central)
                       if means[1] != 0 else None)
        main[s] = {
            "levels": values.tolist(),
            "level_means": means.tolist(),
            "share": share(ss),
            "sensitivity": sensitivity,
        }

    interactions = {}
    for a, b in itertools.combinations(symbols, 2):
        ss = 0.0
        for va in np.unique(x[a]):
            for vb in np.unique(x[b]):
                cell = (x[a] == va) & (x[b] == vb)
                residual = (y[cell].mean() - y[x[a] == va].mean()
                            - y[x[b] == vb].mean() + grand)
                ss += float(cell.sum()) * residual ** 2
        interactions[f"{a} x {b}"] = share(ss)

    explained = sum(m["share"] for m in main.values()) + sum(interactions.values())
    return {
        "metric": metric,
        "runs": len(records),
        "grand_mean": grand,
        "main": main,
        "interactions": dict(sorted(interactions.items(), key=lambda kv: -kv[1])),
        "higher_order_share": max(0.0, 1.0 - explained),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        chosen = set(range(args.runs[0], args.runs[1] + 1)) if args.runs else None
        factorial(n=int(P.N_DROPLETS.value), indices=chosen)
    if args.runs is None or args.summary_only:
        records = load_records()
        if len(records) == len(design()):
            print(json.dumps(summarise(records), indent=2))
