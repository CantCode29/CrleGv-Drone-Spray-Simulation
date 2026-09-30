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
import params as P
import trajectory

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
needs_runs = pytest.mark.skipif(not (explorer.RUNS_DIR / "factorial_summary.json").exists(),
                                reason="Saved factorial runs are missing - run src/sweep.py first")

HOURS = (24.0, 72.0, 504.0)
CASES = [
    {"h": 1.25, "v": 3.0, "dw": 8.5, "rp": 0.0005, "lai": 2.8, "k": 0.575, "hlE": 24.0, "hlS": 72.0},
    {"h": 2.0, "v": 1.0, "dw": 10.0, "rp": 0.0003, "lai": 3.6, "k": 0.6, "hlE": 24.0, "hlS": 72.0},
    {"h": 0.9, "v": 2.2, "dw": 7.3, "rp": 0.00042, "lai": 2.3, "k": 0.56, "hlE": 30.0, "hlS": 120.0},
]

HARNESS = r"""
const vm = require("vm"), fs = require("fs");
const page = fs.readFileSync(process.argv[2], "utf8");
const job = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
vm.runInThisContext(page.match(/<script id="crlegv-model">([\s\S]*?)<\/script>/)[1]);
const M = globalThis.CrleGV;
const zoneNames = ["outer", "mid", "inner", "microhabitat"];
const out = job.cases.map(c => {
  const canopy = M.buildCanopy(job.params, c.lai, c.k);
  const zones = M.zones(canopy);
  const spray = M.runSpray(job.params, canopy, { altitude: c.h, vUav: c.v, vDw: c.dw, rp: c.rp, n: job.n });
  const decay = M.decayConstants(canopy.intensity, c.hlE, c.hlS);
  const viable = job.hours.map(h => zoneNames.map(z => M.zoneViable(decay, spray.deposition, zones[z], h)));
  const perM2 = ["outer", "mid", "inner"].map(z => M.zoneDeposit(canopy, spray.deposition, zones[z]));
  return { deposition: Array.from(spray.deposition), escaped: spray.escaped, intensity: Array.from(canopy.intensity),
           refuge: Array.from(canopy.refuge), fruit: Array.from(canopy.fruitMask), micro: Array.from(zones.microhabitat),
           viable: viable, perM2: perM2 };
});
process.stdout.write(JSON.stringify(out));
"""


def scripts(text):
    return {m.group(1) or "ui": m.group(2)
            for m in re.finditer(r'<script(?: id="([^"]+)")?[^>]*>(.*?)</script>', text, re.S)}


def run_js(tmp_path, cases, n):
    harness = tmp_path / "harness.js"
    job = tmp_path / "job.json"
    harness.write_text(HARNESS, encoding="utf-8")
    job.write_text(json.dumps({"params": explorer.parameters(), "cases": cases, "hours": HOURS, "n": n}))
    found = subprocess.run([NODE, str(harness), str(explorer.PAGE), str(job)],
                           capture_output=True, text=True, check=True)
    return json.loads(found.stdout)


def viable(canopy, deposition, masks, half_exposed, half_shaded):
    lam_exposed, lam_shaded = np.log(2.0) / half_exposed, np.log(2.0) / half_shaded
    decay = lam_shaded + (lam_exposed - lam_shaded) * canopy.intensity_ratio
    return [[np.exp(-decay[masks[z] & (deposition > 0.0)] * h).mean()
             for z in ("outer", "mid", "inner", "microhabitat")] for h in HOURS]


def shaped(values, canopy):
    return np.asarray(values, dtype=float).reshape(canopy.nx, canopy.nz)


@needs_node
def test_js_model_matches_python(tmp_path):
    n = 300
    for case, js in zip(CASES, run_js(tmp_path, CASES, n)):
        canopy = grid.build(lai_total=case["lai"], k=case["k"])
        masks = grid.zones(canopy)
        deposition, escaped = trajectory.run(canopy, altitude=case["h"], v_uav=case["v"],
                                             v_dw=case["dw"], r_p=case["rp"], n=n)
        assert np.array_equal(shaped(js["refuge"], canopy).astype(bool), canopy.refuge)
        assert np.array_equal(shaped(js["fruit"], canopy).astype(bool), canopy.fruit_mask)
        assert np.array_equal(shaped(js["micro"], canopy).astype(bool), masks["microhabitat"])
        np.testing.assert_allclose(shaped(js["intensity"], canopy), canopy.intensity_ratio, rtol=1e-13)
        np.testing.assert_allclose(shaped(js["deposition"], canopy), deposition, rtol=1e-12, atol=1e-15)
        assert js["escaped"] == pytest.approx(escaped, rel=1e-12, abs=1e-15)
        per_m2 = [deposition[masks[z]].sum() / (masks[z].sum() * canopy.dx ** 2) for z in ("outer", "mid", "inner")]
        np.testing.assert_allclose(js["perM2"], per_m2, rtol=1e-12)
        np.testing.assert_allclose(js["viable"], viable(canopy, deposition, masks, case["hlE"], case["hlS"]), rtol=1e-12)


@needs_node
@needs_runs
def test_js_model_reproduces_the_saved_central_run(tmp_path):
    folder = explorer.RUNS_DIR / "fact_40"
    record = json.loads((folder / "record.json").read_text())
    assert record["inputs"]["altitude"] == P.UAV_ALTITUDE.value and record["inputs"]["r_p"] == P.DROPLET_RADIUS.value
    js = run_js(tmp_path, CASES[:1], int(P.N_DROPLETS.value))[0]
    canopy = grid.build()
    saved = np.load(folder / "arrays.npz")["deposition"]
    np.testing.assert_allclose(shaped(js["deposition"], canopy), saved, rtol=1e-12, atol=1e-15)
    assert js["escaped"] == pytest.approx(record["outputs"]["escaped_fraction"], rel=1e-12)


@needs_node
def test_page_scripts_parse(tmp_path):
    for name, body in scripts(explorer.PAGE.read_text(encoding="utf-8")).items():
        if name == "crlegv-data":
            continue
        path = tmp_path / f"{name}.js"
        path.write_text(body, encoding="utf-8")
        subprocess.run([NODE, "--check", str(path)], check=True, capture_output=True)


def test_inject_replaces_only_the_data_block():
    page = "<p>before</p>\n" + explorer.OPEN_TAG + "null" + explorer.CLOSE_TAG + "\n<script>var x = 1;</script>"
    data = {"note": "</script><b>", "values": [1.5, 2]}
    once = explorer.inject(page, data)
    assert explorer.inject(once, data) == once
    assert once.startswith("<p>before</p>\n") and once.endswith("<script>var x = 1;</script>")
    assert "</script><b>" not in once
    assert explorer.embedded(once) == data


@needs_runs
def test_page_data_is_current():
    embedded = explorer.embedded(explorer.PAGE.read_text(encoding="utf-8"))
    assert embedded == json.loads(json.dumps(explorer.build()))
    assert len(embedded["runs"]) == len(embedded["levels"]["h_uav"]) ** 4
