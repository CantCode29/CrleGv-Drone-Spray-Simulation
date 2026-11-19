import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

import grid
import params
import sweep

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
VALIDATION_DIR = RUNS_DIR / "validation" / "post-sourcing"
PAGE = ROOT / "explorer" / "index.html"
OPEN_TAG = '<script id="crlegv-data" type="application/json">'
CLOSE_TAG = "</script>"
MODEL_FILES = ("src/params.py", "src/grid.py", "src/trajectory.py", "src/survival.py")
PAGE_ZONE_NAMES = {"outer": "outer", "middle": "mid", "inner": "inner",
                   "shaded_next_to_fruit": "microhabitat"}
PAGE_SYMBOLS = {name: setting.symbol for name, setting in params.SETTINGS_TESTED.items()}
CHECK_HOURS = ("24.0", "72.0")


def _millimetres(metres):
    return round(metres * 1e3, 6)


def _saved_runs(runs_dir):
    records = [json.loads(path.read_text()) for path in sorted(Path(runs_dir).glob("fact_*/record.json"))]
    expected = len(sweep.all_combinations())
    if len(records) != expected:
        raise FileNotFoundError(f"Found {len(records)} of {expected} factorial runs - run src/sweep.py first.")
    return records


def _central_run_folder(runs_dir):
    central_settings = {name: setting.value for name, setting in params.SETTINGS_TESTED.items()}
    for record_path in sorted(Path(runs_dir).glob("fact_*/record.json")):
        inputs = json.loads(record_path.read_text())["inputs"]
        if all(abs(inputs[name] - value) < 1e-12 for name, value in central_settings.items()):
            return record_path.parent
    raise FileNotFoundError("No factorial run at the central settings - run src/sweep.py first.")


def parameters():
    values = params.central_values()
    values["FRUIT_FIRST_X_M"] = params.FIRST_FRUIT_FROM_LEFT_EDGE_M
    values["FRUIT_ROW_DEPTHS_M"] = list(params.FRUIT_ROW_DEPTHS_M)
    return values


def levels():
    values = {setting.symbol: sweep.three_levels(setting) for setting in params.SETTINGS_TESTED.values()}
    values["r_p_mm"] = [_millimetres(radius) for radius in values["r_p"]]
    return values


def ranges():
    return {setting.symbol: [setting.low_and_high()[0], setting.value, setting.low_and_high()[1]]
            for setting in (params.LEAF_AREA_INDEX, params.LIGHT_EXTINCTION_COEFFICIENT)}


def runs(records):
    rows = []
    for record in records:
        inputs, outputs = record["inputs"], record["outputs"]
        spray_by_zone = outputs["spray_by_zone"]
        rows.append([inputs["drone_height"], inputs["drone_speed"], inputs["downwash_speed"],
                     _millimetres(inputs["droplet_radius"]),
                     spray_by_zone["outer"]["spray_per_m2"], spray_by_zone["middle"]["spray_per_m2"],
                     spray_by_zone["inner"]["spray_per_m2"], outputs["share_passing_through_canopy"]])
    return rows


def _page_pair_name(pair):
    first, second = pair.split(" x ")
    return f"{PAGE_SYMBOLS[first]} x {PAGE_SYMBOLS[second]}"


def effects(summary):
    found = summary["effects"]
    return {
        "grand": found["overall_mean"],
        "higher": found["higher_order_share"],
        "main": {PAGE_SYMBOLS[name]: {"levels": effect["levels"], "means": effect["level_means"],
                                      "share": effect["share_of_variation"], "S": effect["sensitivity"]}
                 for name, effect in found["main_effects"].items()},
        "inter": {_page_pair_name(pair): share for pair, share in found["interactions"].items()},
    }


def check(runs_dir):
    folder = _central_run_folder(runs_dir)
    record = json.loads((folder / "record.json").read_text())
    arrays = np.load(folder / "arrays.npz")
    canopy = grid.build_canopy()
    zones = grid.canopy_zones(canopy)
    outputs = record["outputs"]
    return {
        "deposit_sum": float(arrays["spray_landed"].sum()),
        "inner_per_m2": outputs["spray_by_zone"]["inner"]["spray_per_m2"],
        "outer_per_m2": outputs["spray_by_zone"]["outer"]["spray_per_m2"],
        "escaped": outputs["share_passing_through_canopy"],
        "refuge_cells": int(arrays["deep_shade"].sum()),
        "micro_cells": int(zones["shaded_next_to_fruit"].sum()),
        "fruit_cells": int(canopy.contains_fruit.sum()),
        "uv_sum": float(arrays["share_of_full_light"].sum()),
        "viable": {hours: {page_zone: outputs["virus_by_zone"][hours][zone_name]["average_share_still_active"]
                           for zone_name, page_zone in PAGE_ZONE_NAMES.items()}
                   for hours in CHECK_HOURS},
    }


def validation(source):
    fall_speed_target, deposit_target, outer_survival_target, survival_ratio_target = (
        json.loads((Path(source) / f"target{number}.json").read_text()) for number in range(1, 5))
    return {
        "t1": {"tolerance": fall_speed_target["tolerance"],
               "rows": [{"d": row["diameter_mm"], "measured": row["measured_m_s"],
                         "simulated": row["simulated_m_s"], "dev": row["deviation"],
                         "ok": row["within_tolerance"]} for row in fall_speed_target["rows"]]},
        "t2": {"field": [min(deposit_target["benchmark_ratios"]), max(deposit_target["benchmark_ratios"])],
               "ratio": deposit_target["ratio"], "band": deposit_target["band"], "ok": deposit_target["passed"]},
        "t3": {"value": outer_survival_target["outer_24h"], "band": outer_survival_target["band"],
               "ok": outer_survival_target["passed"]},
        "t4": {"ratio": survival_ratio_target["ratio_21d"], "band": survival_ratio_target["band_21d"],
               "ok": survival_ratio_target["passed"]},
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


def page_data(runs_dir=RUNS_DIR, validation_dir=VALIDATION_DIR):
    runs_dir = Path(runs_dir)
    summary = json.loads((runs_dir / "factorial_summary.json").read_text())
    return {
        "params": parameters(),
        "levels": levels(),
        "ranges": ranges(),
        "runs": runs(_saved_runs(runs_dir)),
        "effects": effects(summary),
        "check": check(runs_dir),
        "validation": validation(validation_dir),
        "commit": model_version(),
        "built": study_date(runs_dir),
    }


def put_data_into_page(page_text, data):
    start = page_text.index(OPEN_TAG) + len(OPEN_TAG)
    end = page_text.index(CLOSE_TAG, start)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return page_text[:start] + payload + page_text[end:]


def data_in_page(page_text):
    start = page_text.index(OPEN_TAG) + len(OPEN_TAG)
    return json.loads(page_text[start:page_text.index(CLOSE_TAG, start)])


def update_page(page=PAGE, runs_dir=RUNS_DIR, validation_dir=VALIDATION_DIR):
    page = Path(page)
    text = page.read_text(encoding="utf-8")
    page.write_text(put_data_into_page(text, page_data(runs_dir, validation_dir)), encoding="utf-8", newline="\n")
    return page


if __name__ == "__main__":
    print(update_page())
