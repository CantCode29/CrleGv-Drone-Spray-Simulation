import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import params
import sweep


def made_up_runs(result_for):
    return [{"inputs": settings, "outputs": {sweep.RANKING_MEASURE: result_for(settings)}}
            for settings in sweep.all_combinations()]


def test_design_is_the_full_three_level_factorial():
    combinations = sweep.all_combinations()
    assert len(combinations) == 81
    assert len({tuple(sorted(settings.items())) for settings in combinations}) == 81
    for name in params.SETTINGS_TESTED:
        values = [settings[name] for settings in combinations]
        assert len(set(values)) == 3
        assert all(values.count(value) == 27 for value in set(values))


def test_additive_response_has_no_interactions():
    effects = sweep.setting_effects(made_up_runs(
        lambda settings: 2 * settings["drone_height"] + 5 * settings["drone_speed"]
        + 1e3 * settings["droplet_radius"]))
    assert all(share < 1e-12 for share in effects["interactions"].values())
    assert sum(effect["share_of_variation"] for effect in effects["main_effects"].values()) == pytest.approx(1.0)
    assert effects["main_effects"]["downwash_speed"]["share_of_variation"] == pytest.approx(0.0, abs=1e-12)


def test_product_response_shows_up_as_that_interaction():
    effects = sweep.setting_effects(made_up_runs(
        lambda settings: settings["drone_height"] * settings["downwash_speed"]))
    assert effects["interactions"]["drone_height x downwash_speed"] > 0.0
    others = {pair: share for pair, share in effects["interactions"].items()
              if pair != "drone_height x downwash_speed"}
    assert all(share < 1e-12 for share in others.values())


def test_sensitivity_of_a_proportional_response_is_one():
    effects = sweep.setting_effects(made_up_runs(lambda settings: settings["drone_height"]))
    assert effects["main_effects"]["drone_height"]["sensitivity"] == pytest.approx(1.0)
    assert effects["main_effects"]["drone_speed"]["sensitivity"] == pytest.approx(0.0, abs=1e-12)


def test_best_run_has_the_highest_ranking_measure():
    central_height = params.DRONE_HEIGHT_ABOVE_CANOPY.value
    records = made_up_runs(
        lambda settings: -abs(settings["drone_height"] - central_height) - settings["downwash_speed"])
    top = sweep.best_run(records)
    assert top["inputs"]["drone_height"] == central_height
    assert top["inputs"]["downwash_speed"] == min(record["inputs"]["downwash_speed"] for record in records)
