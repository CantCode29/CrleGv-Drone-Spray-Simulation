import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import run


def test_depth_zones_account_for_all_spray_that_lands():
    outputs = run.run_simulation("check", droplets_released=100, save_to_disk=False)["outputs"]
    spray_by_zone = outputs["spray_by_zone"]
    in_depth_zones = sum(spray_by_zone[zone_name]["spray_landed"] for zone_name in ("outer", "middle", "inner"))
    assert in_depth_zones == pytest.approx(100 * outputs["share_landing_in_canopy"])
    assert outputs["inner_spray_per_m2"] == spray_by_zone["inner"]["spray_per_m2"]


def test_zone_percentages_add_up_to_the_share_that_lands():
    outputs = run.run_simulation("check", droplets_released=100, save_to_disk=False)["outputs"]
    depth_zones = sum(outputs["percent_of_spray_by_zone"][zone_name] for zone_name in ("outer", "middle", "inner"))
    assert depth_zones == pytest.approx(100 * outputs["share_landing_in_canopy"])
    assert outputs["percent_of_spray_in_inner_canopy"] == outputs["percent_of_spray_by_zone"]["inner"]


def test_virus_comparison_is_against_the_outer_zone():
    outputs = run.run_simulation("check", droplets_released=100, save_to_disk=False)["outputs"]
    for hours, comparison in outputs["virus_compared_with_outer"].items():
        outer = outputs["virus_by_zone"][hours]["outer"]["average_share_still_active"]
        inner = outputs["virus_by_zone"][hours]["inner"]["average_share_still_active"]
        assert comparison["inner"]["ratio_to_outer"] == pytest.approx(inner / outer)
        assert comparison["inner"]["difference_in_percentage_points"] == pytest.approx(100 * (inner - outer))
