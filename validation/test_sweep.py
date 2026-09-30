import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import itertools
import pytest
import params as P
import sweep


def synthetic(f):
    return [{"inputs": kw, "outputs": {sweep.METRIC: f(kw)}} for kw in sweep.design()]


def test_design_is_the_full_three_level_factorial():
    runs = sweep.design()
    assert len(runs) == 81
    assert len({tuple(sorted(r.items())) for r in runs}) == 81
    for key in sweep.KEYS.values():
        values = [r[key] for r in runs]
        assert len(set(values)) == 3
        assert all(values.count(v) == 27 for v in set(values))


def test_additive_response_has_no_interactions():
    fx = sweep.effects(synthetic(lambda k: 2 * k["altitude"] + 5 * k["v_uav"] + 1e3 * k["r_p"]))
    assert all(v < 1e-12 for v in fx["interactions"].values())
    assert sum(m["share"] for m in fx["main"].values()) == pytest.approx(1.0)
    assert fx["main"]["v_dw"]["share"] == pytest.approx(0.0, abs=1e-12)


def test_product_response_shows_up_as_that_interaction():
    fx = sweep.effects(synthetic(lambda k: k["altitude"] * k["v_dw"]))
    assert fx["interactions"]["h_uav x v_dw"] > 0.0
    others = {pair: s for pair, s in fx["interactions"].items() if pair != "h_uav x v_dw"}
    assert all(s < 1e-12 for s in others.values())


def test_sensitivity_of_a_proportional_response_is_one():
    fx = sweep.effects(synthetic(lambda k: k["altitude"]))
    assert fx["main"]["h_uav"]["sensitivity"] == pytest.approx(1.0)
    assert fx["main"]["v_uav"]["sensitivity"] == pytest.approx(0.0, abs=1e-12)


def test_best_run_maximises_the_metric():
    central = P.UAV_ALTITUDE.value
    records = synthetic(lambda k: -abs(k["altitude"] - central) - k["v_dw"])
    top = sweep.best(records)
    assert top["inputs"]["altitude"] == central
    assert top["inputs"]["v_dw"] == min(r["inputs"]["v_dw"] for r in records)
