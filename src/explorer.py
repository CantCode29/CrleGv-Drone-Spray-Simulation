"""Data block for the interactive explorer page, built from saved runs.

The page at explorer/index.html reruns a JavaScript copy of the model in the
browser. This module writes everything the page cannot compute live into its
JSON data block: the parameter values, the 81 saved study runs and their
effect analysis, the validation verdicts, and the central-run figures the page
uses to check that its copy of the model matches the Python model.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

import grid
import params as P
import sweep

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
VALIDATION_DIR = RUNS_DIR / "validation" / "post-sourcing"
PAGE = ROOT / "explorer" / "index.html"
OPEN_TAG = '<script id="crlegv-data" type="application/json">'
CLOSE_TAG = "</script>"
MODEL_FILES = ("src/params.py", "src/grid.py", "src/trajectory.py", "src/survival.py")
ZONES = ("outer", "mid", "inner", "microhabitat")
CHECK_HOURS = ("24.0", "72.0")


def _mm(metres):
    return round(metres * 1e3, 6)


def _records(runs_dir):
    records = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("fact_*/record.json"))]
    expected = len(sweep.design())
    if len(records) != expected:
        raise FileNotFoundError(f"Found {len(records)} of {expected} factorial runs - run src/sweep.py first.")
    return records


def _central_dir(runs_dir):
    central = {"altitude": P.UAV_ALTITUDE.value, "v_uav": P.UAV_VELOCITY.value,
               "v_dw": P.DOWNWASH_VELOCITY.value, "r_p": P.DROPLET_RADIUS.value}
    for record in sorted(Path(runs_dir).glob("fact_*/record.json")):
        inputs = json.loads(record.read_text())["inputs"]
        if all(abs(inputs[k] - v) < 1e-12 for k, v in central.items()):
            return record.parent
    raise FileNotFoundError("No factorial run at the central settings - run src/sweep.py first.")


def parameters():
    out = P.baseline()
    out["FRUIT_FIRST_X_M"] = P.FRUIT_FIRST_X_M
    out["FRUIT_ROW_DEPTHS_M"] = list(P.FRUIT_ROW_DEPTHS_M)
    return out


def levels():
    out = {p.symbol: sweep.levels(p) for p in P.SWEEP_PARAMS}
    out["r_p_mm"] = [_mm(v) for v in out["r_p"]]
    return out


def ranges():
    return {p.symbol: [p.bounds()[0], p.value, p.bounds()[1]]
            for p in (P.LAI_TOTAL, P.EXTINCTION_COEFF)}


def runs(records):
    rows = []
    for record in records:
        inputs, outputs = record["inputs"], record["outputs"]
        zones = outputs["zone_deposition"]
        rows.append([inputs["altitude"], inputs["v_uav"], inputs["v_dw"], _mm(inputs["r_p"]),
                     zones["outer"]["deposit_per_m2"], zones["mid"]["deposit_per_m2"],
                     zones["inner"]["deposit_per_m2"], outputs["escaped_fraction"]])
    return rows


def effects(summary):
    found = summary["effects"]
    return {
        "grand": found["grand_mean"],
        "higher": found["higher_order_share"],
        "main": {key: {"levels": m["levels"], "means": m["level_means"],
                       "share": m["share"], "S": m["sensitivity"]}
                 for key, m in found["main"].items()},
        "inter": found["interactions"],
    }


def check(runs_dir):
    folder = _central_dir(runs_dir)
    record = json.loads((folder / "record.json").read_text())
    arrays = np.load(folder / "arrays.npz")
    canopy = grid.build()
    masks = grid.zones(canopy)
    outputs = record["outputs"]
    return {
        "deposit_sum": float(arrays["deposition"].sum()),
        "inner_per_m2": outputs["zone_deposition"]["inner"]["deposit_per_m2"],
        "outer_per_m2": outputs["zone_deposition"]["outer"]["deposit_per_m2"],
        "escaped": outputs["escaped_fraction"],
        "refuge_cells": int(arrays["refuge"].sum()),
        "micro_cells": int(masks["microhabitat"].sum()),
        "fruit_cells": int(canopy.fruit_mask.sum()),
        "uv_sum": float(arrays["intensity_ratio"].sum()),
        "viable": {h: {z: outputs["zone_summaries"][h][z]["mean_viable_fraction"] for z in ZONES}
                   for h in CHECK_HOURS},
    }


def validation(source):
    t1, t2, t3, t4 = (json.loads((Path(source) / f"target{i}.json").read_text()) for i in range(1, 5))
    return {
        "t1": {"tolerance": t1["tolerance"],
               "rows": [{"d": r["diameter_mm"], "measured": r["measured_m_s"],
                         "simulated": r["simulated_m_s"], "dev": r["deviation"],
                         "ok": r["within_tolerance"]} for r in t1["rows"]]},
        "t2": {"field": [min(t2["benchmark_ratios"]), max(t2["benchmark_ratios"])],
               "ratio": t2["ratio"], "band": t2["band"], "ok": t2["passed"]},
        "t3": {"value": t3["outer_24h"], "band": t3["band"], "ok": t3["passed"]},
        "t4": {"ratio": t4["ratio_21d"], "band": t4["band_21d"], "ok": t4["passed"]},
    }


def model_version():
    try:
        found = subprocess.run(["git", "log", "-1", "--format=%h", "--", *MODEL_FILES],
                               cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return found.stdout.strip() or "unknown"


def study_date(runs_dir):
    manifest = json.loads((Path(runs_dir) / "factorial_manifest.json").read_text())
    finished = datetime.strptime(manifest["finished_utc"], "%Y-%m-%dT%H:%M:%SZ")
    return f"{finished.day} {finished:%b %Y}"


def build(runs_dir=RUNS_DIR, validation_dir=VALIDATION_DIR):
    runs_dir = Path(runs_dir)
    summary = json.loads((runs_dir / "factorial_summary.json").read_text())
    return {
        "params": parameters(),
        "levels": levels(),
        "ranges": ranges(),
        "runs": runs(_records(runs_dir)),
        "effects": effects(summary),
        "check": check(runs_dir),
        "validation": validation(validation_dir),
        "commit": model_version(),
        "built": study_date(runs_dir),
    }


def inject(page_text, data):
    start = page_text.index(OPEN_TAG) + len(OPEN_TAG)
    end = page_text.index(CLOSE_TAG, start)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return page_text[:start] + payload + page_text[end:]


def embedded(page_text):
    start = page_text.index(OPEN_TAG) + len(OPEN_TAG)
    return json.loads(page_text[start:page_text.index(CLOSE_TAG, start)])


def write(page=PAGE, runs_dir=RUNS_DIR, validation_dir=VALIDATION_DIR):
    page = Path(page)
    text = page.read_text(encoding="utf-8")
    page.write_text(inject(text, build(runs_dir, validation_dir)), encoding="utf-8", newline="\n")
    return page


if __name__ == "__main__":
    print(write())
