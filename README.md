# crlegv-sim

2D deterministic in-silico simulation of UAV-delivered CrleGV biopesticide
deposition and viral survival in a citrus canopy cross-section.

Supports DIS 481 (Belgium Campus iTversity), dissertation due 12 November 2026.
See `02_Projects/DIS Final Submission - Action Plan.md` in the vault.

## Layout

| Path | Purpose |
|---|---|
| `src/params.py` | Every model input with its citation. Single source of truth |
| `src/grid.py` | Stage 1 — canopy grid, Beer–Lambert UV field, refuge classification |
| `src/trajectory.py` | Stage 2 — droplet integration and fractional-mass interception |
| `src/survival.py` | Stage 3 — per-cell CrleGV decay and viable fraction |
| `src/run.py` | Executes all three stages for one parameter set, persists to `runs/` |
| `src/sweep.py` | 81-run full factorial and its effect analysis (D5) |
| `src/figures.py` | Figure generation for the results chapter |
| `src/validate.py` | Validation targets against published data, written to `runs/validation/` |
| `validation/` | Sanity checks that must pass before any result is trusted |

## Run

```
pip install -r requirements.txt
python -m pytest validation -q
PYTHONPATH=src python src/run.py
PYTHONPATH=src python src/validate.py
PYTHONPATH=src python src/sweep.py
```

## Design decisions encoded here

These were open in the submitted methodology and are implemented as described
below. Each needs a corresponding paragraph in Chapter 3.

**D1 — vertical LAI distribution.** Leaf area density is uniform,
`LAD = LAI_total / canopy_depth`, so `LAI(z) = LAD·z`. The submitted chapter
defines `LAI(z)` as cumulative but never states the profile.

**D2 — deposition capture.** A droplet carries a mass fraction that decays as
`exp(−LAD·ds)` along its path, depositing the intercepted portion into each
cell it crosses. Deterministic, conserves mass exactly, and avoids the
stochastic capture rule the study excludes. Fruit bodies carry a much higher
interception coefficient and cast a deposition shadow.

**D3 — horizontal structure.** Discrete fruit bodies add local optical depth.
Without them `I(x,z) = I(z)`, every cell at a given depth is identical, and the
model cannot distinguish a calyx recess from open inner canopy — which RQ1
requires. This also relaxes the boundary assumption in §3.9 from "equivalent
depth" to "equivalent depth and equivalent local occlusion".

**D4 — downwash field.** Full strength above the canopy, decaying as
`exp(−z/0.5 m)` with depth. An assumption, not a sourced profile.

**D5 — sensitivity design.** An 81-run full factorial (three levels of each
of the four sweep parameters) replaces the one-factor-at-a-time sweep. Runs
are ranked by inner-zone deposit per m², the quantity RQ2 asks to maximise.
Effects are reported descriptively — each factor's and each two-factor
interaction's share of the total variation, plus a normalised sensitivity
from the level means — with no inferential tests, as §3.8 requires.

**Horizontal wrapping.** The domain wraps in x. At 3 m/s forward velocity and a
5 m release height the drift exceeds the 4 m domain width, so without wrapping
the upwind third of the canopy receives no deposition at all.

## Findings that affect the chapter

**The refuge threshold is unreachable by leaf attenuation alone.** At the most
attenuating sourced combination (LAI 3.6, k 0.60) the canopy floor sits at
`I/I₀ = 0.115`, above the τ = 0.10 refuge threshold. No cell anywhere in the
sourced citrus range qualifies as refuge on leaf area alone. Refuge exists in
this model only because of discrete fruit shading, which makes D3 load-bearing
rather than a refinement. Covered by `test_leaf_attenuation_alone_never_reaches_refuge_threshold`.

**The decay law in §3.7.1 contradicts the 72 h bound.** The chapter gives
`λ(x,z) = λ_base · [I(x,z)/I₀]`, which sends λ to zero in full shade and so
implies an infinite shaded half-life. The chapter separately declares a 72 h
shaded half-life. `survival.py` interpolates between the two published bounds
instead, reproducing 24 h at full exposure and 72 h in full shade.

**Air viscosity is missing from the parameter table.** The Reynolds number in
the piecewise drag law needs it. Added to `params.py` as a standard value.
