import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import csv
import json

import numpy as np
import pytest

import figures
import grid
import sweep
import trajectory
import validate


@pytest.fixture(scope="module")
def central():
    canopy = grid.build_canopy()
    spray_landed, _ = trajectory.spray_canopy(canopy, droplets_released=200)
    return canopy, spray_landed, grid.canopy_zones(canopy)


def made_up_summary():
    records = [{"inputs": settings,
                "outputs": {sweep.RANKING_MEASURE: 150.0 - 3.0 * settings["drone_speed"]
                            + 2.0 * settings["drone_height"]}}
               for settings in sweep.all_combinations()]
    return {"effects": sweep.setting_effects(records)}


def test_every_figure_renders(tmp_path, central):
    canopy, spray_landed, zones = central
    written = [
        figures.draw_light_map(canopy, tmp_path / "a.png", dpi=40),
        figures.draw_spray_map(canopy, spray_landed, zones, tmp_path / "b.png", dpi=40),
        figures.draw_virus_maps(canopy, tmp_path / "c.png", dpi=40),
        figures.draw_survival_curves(canopy, spray_landed, zones, tmp_path / "d.png", dpi=40),
        figures.draw_main_effects(made_up_summary(), tmp_path / "e.png", dpi=40),
        figures.draw_effect_summary(made_up_summary(), tmp_path / "f.png", dpi=40),
    ]
    for path in written:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_survival_curves_start_whole_and_only_decline(central):
    curves = figures.survival_curve_by_zone(*central, [0.0, 24.0, 72.0, 504.0])
    for zone_name in ("outer", "middle", "inner"):
        assert np.isfinite(curves[zone_name]).all()
    for values in curves.values():
        if np.isnan(values).all():
            continue
        assert values[0] == pytest.approx(1.0)
        assert np.all(np.diff(values) < 0.0)


def test_survival_curves_match_the_validation_zone_averages(central):
    curves = figures.survival_curve_by_zone(*central, [24.0, 72.0])
    averages = validate.average_share_still_active_by_zone(*central, (24.0, 72.0))
    for zone_name, values in curves.items():
        if np.isnan(values).all():
            assert np.isnan(averages[24.0][zone_name]) and np.isnan(averages[72.0][zone_name])
            continue
        assert values[0] == pytest.approx(averages[24.0][zone_name])
        assert values[1] == pytest.approx(averages[72.0][zone_name])


def test_validation_table_reports_each_verdict(tmp_path, central):
    averages = validate.average_share_still_active_by_zone(*central, validate.SURVIVAL_CHECK_HOURS)
    results = {"target1": validate.check_fall_speed(),
               "target2": validate.check_spray_penetration(*central),
               "target4": validate.check_virus_survival(averages),
               "light": validate.light_at_canopy_floor()}
    for name, result in results.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(result))
    path = figures.write_validation_table(tmp_path, tmp_path / "table.csv")
    with path.open(encoding="utf-8-sig", newline="") as table_file:
        rows = list(csv.reader(table_file))
    assert rows[0] == ["Target", "Quantity", "Benchmark", "Simulated", "Criterion", "Result"]
    verdicts = [row[5] for row in rows[1:]]
    expected = ["Pass" if row["within_tolerance"] else "Fail" for row in results["target1"]["rows"]]
    expected += ["Pass" if results[name]["passed"] else "Fail" for name in ("target2", "target4")]
    expected += ["Comparison"]
    assert verdicts == expected
