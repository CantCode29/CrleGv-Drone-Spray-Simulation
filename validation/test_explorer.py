import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

import explorer
import grid
import params
import trajectory

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
needs_runs = pytest.mark.skipif(not (explorer.RUNS_DIR / "factorial_summary.json").exists(),
                                reason="Saved factorial runs are missing - run src/sweep.py first")

HOURS = (24.0, 72.0, 504.0)
CASES = [
    {"drone_height": 1.25, "drone_speed": 3.0, "downwash_speed": 8.5, "droplet_radius": 0.0005,
     "leaf_area_index": 2.8, "light_extinction_coefficient": 0.575, "half_life_sun": 24.0, "half_life_shade": 72.0},
    {"drone_height": 2.0, "drone_speed": 1.0, "downwash_speed": 10.0, "droplet_radius": 0.0003,
     "leaf_area_index": 3.6, "light_extinction_coefficient": 0.6, "half_life_sun": 24.0, "half_life_shade": 72.0},
    {"drone_height": 0.9, "drone_speed": 2.2, "downwash_speed": 7.3, "droplet_radius": 0.00042,
     "leaf_area_index": 2.3, "light_extinction_coefficient": 0.56, "half_life_sun": 30.0, "half_life_shade": 120.0},
]

HARNESS = r"""
const vm = require("vm"), fs = require("fs");
const page = fs.readFileSync(process.argv[2], "utf8");
const job = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
vm.runInThisContext(page.match(/<script id="crlegv-model">([\s\S]*?)<\/script>/)[1]);
const M = globalThis.CrleGV;
const zoneNames = ["outer", "mid", "inner", "microhabitat"];
const out = job.cases.map(c => {
  const canopy = M.buildCanopy(job.params, c.leaf_area_index, c.light_extinction_coefficient);
  const zones = M.zones(canopy);
  const spray = M.runSpray(job.params, canopy, { altitude: c.drone_height, vUav: c.drone_speed,
                                                 vDw: c.downwash_speed, rp: c.droplet_radius, n: job.n });
  const decay = M.decayConstants(canopy.intensity, c.half_life_sun, c.half_life_shade);
  const viable = job.hours.map(h => zoneNames.map(z => M.zoneViable(decay, spray.deposition, zones[z], h)));
  const perM2 = ["outer", "mid", "inner"].map(z => M.zoneDeposit(canopy, spray.deposition, zones[z]));
  return { deposition: Array.from(spray.deposition), escaped: spray.escaped, intensity: Array.from(canopy.intensity),
           refuge: Array.from(canopy.refuge), fruit: Array.from(canopy.fruitMask), micro: Array.from(zones.microhabitat),
           viable: viable, perM2: perM2 };
});
process.stdout.write(JSON.stringify(out));
"""

ZONE_NAMES = ("outer", "middle", "inner", "shaded_next_to_fruit")


def scripts(text):
    return {match.group(1) or "ui": match.group(2)
            for match in re.finditer(r'<script(?: id="([^"]+)")?[^>]*>(.*?)</script>', text, re.S)}


def run_page_model(tmp_path, cases, droplets_released):
    harness = tmp_path / "harness.js"
    job = tmp_path / "job.json"
    harness.write_text(HARNESS, encoding="utf-8")
    job.write_text(json.dumps({"params": explorer.parameters(), "cases": cases, "hours": HOURS,
                               "n": droplets_released}))
    found = subprocess.run([NODE, str(harness), str(explorer.PAGE), str(job)],
                           capture_output=True, text=True, check=True)
    return json.loads(found.stdout)


def share_still_active_by_zone(canopy, spray_landed, zones, half_life_sun, half_life_shade):
    rate_in_full_sun, rate_in_full_shade = np.log(2.0) / half_life_sun, np.log(2.0) / half_life_shade
    breakdown_rate = rate_in_full_shade + (rate_in_full_sun - rate_in_full_shade) * canopy.share_of_full_light
    return [[np.exp(-breakdown_rate[zones[zone_name] & (spray_landed > 0.0)] * hours).mean()
             for zone_name in ZONE_NAMES] for hours in HOURS]


def as_grid(values, canopy):
    return np.asarray(values, dtype=float).reshape(canopy.cells_across, canopy.cells_deep)


@needs_node
def test_page_model_matches_python(tmp_path):
    droplets_released = 300
    for case, page_result in zip(CASES, run_page_model(tmp_path, CASES, droplets_released)):
        canopy = grid.build_canopy(leaf_area_index=case["leaf_area_index"],
                                   light_extinction_coefficient=case["light_extinction_coefficient"])
        zones = grid.canopy_zones(canopy)
        spray_landed, share_passing_through = trajectory.spray_canopy(
            canopy, drone_height=case["drone_height"], drone_speed=case["drone_speed"],
            downwash_speed=case["downwash_speed"], droplet_radius=case["droplet_radius"],
            droplets_released=droplets_released)
        assert np.array_equal(as_grid(page_result["refuge"], canopy).astype(bool), canopy.deep_shade)
        assert np.array_equal(as_grid(page_result["fruit"], canopy).astype(bool), canopy.contains_fruit)
        assert np.array_equal(as_grid(page_result["micro"], canopy).astype(bool), zones["shaded_next_to_fruit"])
        np.testing.assert_allclose(as_grid(page_result["intensity"], canopy), canopy.share_of_full_light,
                                   rtol=1e-13)
        np.testing.assert_allclose(as_grid(page_result["deposition"], canopy), spray_landed,
                                   rtol=1e-12, atol=1e-15)
        assert page_result["escaped"] == pytest.approx(share_passing_through, rel=1e-12, abs=1e-15)
        spray_per_m2 = [spray_landed[zones[zone_name]].sum() / (zones[zone_name].sum() * canopy.cell_size ** 2)
                        for zone_name in ("outer", "middle", "inner")]
        np.testing.assert_allclose(page_result["perM2"], spray_per_m2, rtol=1e-12)
        np.testing.assert_allclose(page_result["viable"],
                                   share_still_active_by_zone(canopy, spray_landed, zones,
                                                              case["half_life_sun"], case["half_life_shade"]),
                                   rtol=1e-12)


@needs_node
@needs_runs
def test_page_model_reproduces_the_saved_central_run(tmp_path):
    folder = explorer.RUNS_DIR / "fact_40"
    record = json.loads((folder / "record.json").read_text())
    assert (record["inputs"]["drone_height"] == params.DRONE_HEIGHT_ABOVE_CANOPY.value
            and record["inputs"]["droplet_radius"] == params.DROPLET_RADIUS.value)
    page_result = run_page_model(tmp_path, CASES[:1], int(params.DROPLETS_PER_RUN.value))[0]
    canopy = grid.build_canopy()
    saved = np.load(folder / "arrays.npz")["spray_landed"]
    np.testing.assert_allclose(as_grid(page_result["deposition"], canopy), saved, rtol=1e-12, atol=1e-15)
    assert page_result["escaped"] == pytest.approx(record["outputs"]["share_passing_through_canopy"], rel=1e-12)


@needs_node
def test_page_scripts_parse(tmp_path):
    for name, body in scripts(explorer.PAGE.read_text(encoding="utf-8")).items():
        if name == "crlegv-data":
            continue
        path = tmp_path / f"{name}.js"
        path.write_text(body, encoding="utf-8")
        subprocess.run([NODE, "--check", str(path)], check=True, capture_output=True)


def test_putting_data_into_the_page_replaces_only_the_data_block():
    page = "<p>before</p>\n" + explorer.OPEN_TAG + "null" + explorer.CLOSE_TAG + "\n<script>var x = 1;</script>"
    data = {"note": "</script><b>", "values": [1.5, 2]}
    once = explorer.put_data_into_page(page, data)
    assert explorer.put_data_into_page(once, data) == once
    assert once.startswith("<p>before</p>\n") and once.endswith("<script>var x = 1;</script>")
    assert "</script><b>" not in once
    assert explorer.data_in_page(once) == data


@needs_runs
def test_page_data_is_current():
    in_page = explorer.data_in_page(explorer.PAGE.read_text(encoding="utf-8"))
    assert in_page == json.loads(json.dumps(explorer.page_data()))
    assert len(in_page["runs"]) == len(in_page["levels"]["h_uav"]) ** 4
