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
    canopy = grid.build()
    deposition, _ = trajectory.run(canopy, n=200)
    return canopy, deposition, grid.zones(canopy)


def summary():
    records = [{"inputs": kw, "outputs": {sweep.METRIC: 150.0 - 3.0 * kw["v_uav"] + 2.0 * kw["altitude"]}}
               for kw in sweep.design()]
    return {"effects": sweep.effects(records)}


def test_every_figure_renders(tmp_path, central):
    canopy, deposition, masks = central
    written = [
        figures.uv_field(canopy, tmp_path / "a.png", dpi=40),
        figures.deposition_map(canopy, deposition, masks, tmp_path / "b.png", dpi=40),
        figures.survival_maps(canopy, tmp_path / "c.png", dpi=40),
        figures.survival_curves(canopy, deposition, masks, tmp_path / "d.png", dpi=40),
        figures.main_effects(summary(), tmp_path / "e.png", dpi=40),
        figures.effect_summary(summary(), tmp_path / "f.png", dpi=40),
    ]
    for path in written:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_zone_curves_start_whole_and_only_decline(central):
    curves = figures.zone_curves(*central, [0.0, 24.0, 72.0, 504.0])
    for key in ("outer", "mid", "inner"):
        assert np.isfinite(curves[key]).all()
    for values in curves.values():
        if np.isnan(values).all():
            continue
        assert values[0] == pytest.approx(1.0)
        assert np.all(np.diff(values) < 0.0)


def test_zone_curves_match_the_validation_zone_means(central):
    curves = figures.zone_curves(*central, [24.0, 72.0])
    means = validate.zone_means(*central, (24.0, 72.0))
    for key, values in curves.items():
        if np.isnan(values).all():
            assert np.isnan(means[24.0][key]) and np.isnan(means[72.0][key])
            continue
        assert values[0] == pytest.approx(means[24.0][key])
        assert values[1] == pytest.approx(means[72.0][key])


def test_validation_table_reports_each_verdict(tmp_path, central):
    means = validate.zone_means(*central, validate.TARGET4_HOURS)
    results = {"target1": validate.target1(), "target2": validate.target2(*central),
               "target4": validate.target4(means), "light": validate.light_floor()}
    for name, result in results.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(result))
    path = figures.validation_table(tmp_path, tmp_path / "table.csv")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Target", "Quantity", "Benchmark", "Simulated", "Criterion", "Result"]
    verdicts = [row[5] for row in rows[1:]]
    expected = ["Pass" if r["within_tolerance"] else "Fail" for r in results["target1"]["rows"]]
    expected += ["Pass" if results[t]["passed"] else "Fail" for t in ("target2", "target4")]
    expected += ["Comparison"]
    assert verdicts == expected
