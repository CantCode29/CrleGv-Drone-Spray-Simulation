import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import run


def test_depth_zones_account_for_all_deposited_mass():
    record = run.execute("check", n=100, persist=False)
    out = record["outputs"]
    zones = out["zone_deposition"]
    in_zones = sum(zones[z]["deposit_total"] for z in ("outer", "mid", "inner"))
    assert in_zones == pytest.approx(100 * out["deposited_fraction"])
    assert out["inner_deposit_per_m2"] == zones["inner"]["deposit_per_m2"]


def test_viability_comparison_is_against_the_outer_zone():
    out = run.execute("check", n=100, persist=False)["outputs"]
    for h, zones in out["viability_vs_outer"].items():
        outer = out["zone_summaries"][h]["outer"]["mean_viable_fraction"]
        inner = out["zone_summaries"][h]["inner"]["mean_viable_fraction"]
        assert zones["inner"]["ratio"] == pytest.approx(inner / outer)
        assert zones["inner"]["difference_pp"] == pytest.approx(100 * (inner - outer))
