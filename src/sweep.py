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
    combinations = itertools.product(*levels_by_setting.values())
    return [dict(zip(levels_by_setting, combination)) for combination in combinations]


def run_all(droplets_released=None, save_to_disk=True, run_numbers=None):
    records = []
    for number, settings in enumerate(all_combinations()):
        if run_numbers is not None and number not in run_numbers:
            continue
        records.append(run.run_simulation(f"fact_{number:02d}", droplets_released=droplets_released,
                                          save_to_disk=save_to_disk, **settings))
    return records


def load_saved_runs(runs_dir=None):
    runs_dir = RUNS_DIR if runs_dir is None else Path(runs_dir)
    return [json.loads(path.read_text()) for path in sorted(runs_dir.glob("fact_*/record.json"))]


def save_study_summary(records, runs_dir=None):
    runs_dir = RUNS_DIR if runs_dir is None else Path(runs_dir)
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


def share_of_total(part, total):
    return part / total if total > 0 else 0.0


def main_effect(values, results, overall_mean, total_variation):
    levels = np.unique(values)
    level_means = np.array([results[values == level].mean() for level in levels])
    sum_of_squares = sum(float((values == level).sum()) * (mean - overall_mean) ** 2
                         for level, mean in zip(levels, level_means))

    low, central, high = levels
    if level_means[1] != 0:
        change_in_result = (level_means[2] - level_means[0]) / level_means[1]
        change_in_setting = (high - low) / central
        sensitivity = change_in_result / change_in_setting
    else:
        sensitivity = None

    return {
        "levels": levels.tolist(),
        "level_means": level_means.tolist(),
        "share_of_variation": share_of_total(sum_of_squares, total_variation),
        "sensitivity": sensitivity,
    }


def pair_effect(first_values, second_values, results, overall_mean, total_variation):
    sum_of_squares = 0.0
    for first_level in np.unique(first_values):
        for second_level in np.unique(second_values):
            at_first_level = first_values == first_level
            at_second_level = second_values == second_level
            at_both_levels = at_first_level & at_second_level
            left_over = (results[at_both_levels].mean()
                         - results[at_first_level].mean()
                         - results[at_second_level].mean()
                         + overall_mean)
            sum_of_squares += float(at_both_levels.sum()) * left_over ** 2
    return share_of_total(sum_of_squares, total_variation)


def setting_effects(records, measure=RANKING_MEASURE):
    setting_names = list(params.SETTINGS_TESTED)
    values = {name: np.array([record["inputs"][name] for record in records], dtype=float)
              for name in setting_names}
    results = np.array([record["outputs"][measure] for record in records], dtype=float)
    overall_mean = float(results.mean())
    total_variation = float(((results - overall_mean) ** 2).sum())

    main_effects = {name: main_effect(values[name], results, overall_mean, total_variation)
                    for name in setting_names}
    interactions = {f"{first} x {second}": pair_effect(values[first], values[second], results,
                                                      overall_mean, total_variation)
                    for first, second in itertools.combinations(setting_names, 2)}

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
