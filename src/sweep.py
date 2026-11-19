"""The 81-run study (decision D5): every combination of three levels of the
four settings tested, and how much of the change in the result each setting
explains."""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

import params
import run

RUNS_DIR = Path(os.environ.get("CRLEGV_RUNS_DIR", Path(__file__).resolve().parent.parent / "runs"))

RANKING_MEASURE = "percent_of_spray_in_inner_canopy"


def three_levels(setting):
    low, high = setting.low_and_high()
    return [low, setting.value, high]


def all_combinations():
    levels_by_setting = {name: three_levels(setting) for name, setting in params.SETTINGS_TESTED.items()}
    return [dict(zip(levels_by_setting, combination))
            for combination in itertools.product(*levels_by_setting.values())]


def run_all(droplets_released=None, save_to_disk=True, run_numbers=None):
    numbered = list(enumerate(all_combinations()))
    if run_numbers is not None:
        numbered = [(number, settings) for number, settings in numbered if number in run_numbers]
    return [run.run_simulation(f"fact_{number:02d}", droplets_released=droplets_released,
                               save_to_disk=save_to_disk, **settings)
            for number, settings in numbered]


def load_saved_runs(runs_dir=None):
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    return [json.loads(path.read_text()) for path in sorted(runs_dir.glob("fact_*/record.json"))]


def save_study_summary(records, runs_dir=None):
    runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    top = best_run(records)
    summary = {
        "best_inputs": top["inputs"],
        "best_value": top["outputs"][RANKING_MEASURE],
        "effects": setting_effects(records),
    }
    (runs_dir / "factorial_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def best_run(records, measure=RANKING_MEASURE):
    return max(records, key=lambda record: record["outputs"][measure])


def setting_effects(records, measure=RANKING_MEASURE):
    setting_names = list(params.SETTINGS_TESTED)
    setting_values = {name: np.array([record["inputs"][name] for record in records], dtype=float)
                      for name in setting_names}
    results = np.array([record["outputs"][measure] for record in records], dtype=float)
    overall_mean = float(results.mean())
    total_variation = float(((results - overall_mean) ** 2).sum())

    def share_of_variation(sum_of_squares):
        return sum_of_squares / total_variation if total_variation > 0 else 0.0

    main_effects = {}
    for name in setting_names:
        levels = np.unique(setting_values[name])
        level_means = np.array([results[setting_values[name] == level].mean() for level in levels])
        sum_of_squares = sum(float((setting_values[name] == level).sum()) * (mean - overall_mean) ** 2
                             for level, mean in zip(levels, level_means))
        low, central, high = levels
        sensitivity = (((level_means[2] - level_means[0]) / level_means[1]) / ((high - low) / central)
                       if level_means[1] != 0 else None)
        main_effects[name] = {
            "levels": levels.tolist(),
            "level_means": level_means.tolist(),
            "share_of_variation": share_of_variation(sum_of_squares),
            "sensitivity": sensitivity,
        }

    interactions = {}
    for first, second in itertools.combinations(setting_names, 2):
        sum_of_squares = 0.0
        for first_level in np.unique(setting_values[first]):
            for second_level in np.unique(setting_values[second]):
                at_both_levels = ((setting_values[first] == first_level)
                                  & (setting_values[second] == second_level))
                left_unexplained = (results[at_both_levels].mean()
                                    - results[setting_values[first] == first_level].mean()
                                    - results[setting_values[second] == second_level].mean()
                                    + overall_mean)
                sum_of_squares += float(at_both_levels.sum()) * left_unexplained ** 2
        interactions[f"{first} x {second}"] = share_of_variation(sum_of_squares)

    explained = (sum(effect["share_of_variation"] for effect in main_effects.values())
                 + sum(interactions.values()))
    return {
        "measure": measure,
        "runs": len(records),
        "overall_mean": overall_mean,
        "main_effects": main_effects,
        "interactions": dict(sorted(interactions.items(), key=lambda pair: -pair[1])),
        "higher_order_share": max(0.0, 1.0 - explained),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--summary-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.summary_only:
        chosen = set(range(arguments.runs[0], arguments.runs[1] + 1)) if arguments.runs else None
        run_all(droplets_released=int(params.DROPLETS_PER_RUN.value), run_numbers=chosen)
    if arguments.runs is None or arguments.summary_only:
        records = load_saved_runs()
        if len(records) == len(all_combinations()):
            print(json.dumps(save_study_summary(records), indent=2))
