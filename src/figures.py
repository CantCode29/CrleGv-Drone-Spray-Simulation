"""Figure set for the results chapter (plan item W4.4), drawn from saved runs.

Colour is assigned by the job it does. Fields are magnitude, so each uses a
single-hue sequential ramp running light to dark: blue for deposit, orange
for UV and aqua for viability. The depth zones are ordered, so outer, mid and
inner take steps of one blue ramp, with the microhabitat in orange. Signed
effects use a blue-red diverging pair. Every palette was checked with the
colour validator before use, and series are directly labelled so identity
never rests on colour alone.
"""

import csv
import functools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

import grid
import params as P
import survival

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("CRLEGV_RUNS_DIR", ROOT / "runs"))
FIGURES_DIR = Path(os.environ.get("CRLEGV_FIGURES_DIR", ROOT / "figures"))
VALIDATION_DIR = Path(os.environ.get("CRLEGV_VALIDATION_DIR", RUNS_DIR / "validation" / "post-sourcing"))
DPI = 300
TEXT_WIDTH_IN = 6.3

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
ORANGE_RAMP = ("#fbd7ca", "#f5c4b2", "#f2b098", "#eb9c7f", "#e68764", "#df7249", "#d95923",
               "#c94908", "#b44005", "#9f3600", "#8a2e00", "#752600", "#611e00")
AQUA_RAMP = ("#cce8d9", "#b4dbc6", "#9ad0b4", "#80c3a2", "#62b890", "#43ac7f", "#009f6d",
             "#008f62", "#007f56", "#006f4b", "#006040", "#005136", "#00432b")

SERIES = "#2a78d6"
NEGATIVE = "#e34948"
DEEMPHASIS = MUTED
ZONES = (("outer", "Outer", "#86b6ef"), ("mid", "Mid", "#2a78d6"),
         ("inner", "Inner", "#104281"), ("microhabitat", "Microhabitat", "#eb6834"))
FACTORS = (("v_uav", "Flight speed", "Flight speed", "m/s", 1.0),
           ("r_p", "Droplet size", "Droplet size", "µm diameter", 2e6),
           ("h_uav", "Height above canopy", "Height", "m above canopy", 1.0),
           ("v_dw", "Downwash", "Downwash", "m/s", 1.0))

STYLE = {
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
}


def _styled(draw):
    @functools.wraps(draw)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return draw(*args, **kwargs)
    return wrapper


def _ramp(colours, name):
    cmap = LinearSegmentedColormap.from_list(name, colours)
    cmap.set_bad(SURFACE)
    return cmap


def _save(fig, path, dpi):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def _minus(text):
    return text.replace("-", "−")


def _field(ax, canopy, values, cmap, norm):
    width, depth = canopy.x_edges[-1], canopy.z_edges[-1]
    image = ax.imshow(values.T, cmap=cmap, norm=norm, origin="upper", aspect="equal",
                      extent=(0.0, width, depth, 0.0), interpolation="nearest")
    ax.set_xlim(0.0, width)
    ax.set_ylim(depth, 0.0)
    ax.set_xticks(np.arange(0.0, width + 1e-9, 1.0))
    ax.set_yticks(np.arange(0.0, depth + 1e-9, 1.0))
    return image


def _zone_bounds(canopy):
    third = canopy.nz // 3
    return canopy.z_edges[third], canopy.z_edges[2 * third]


def _zone_lines(ax, canopy):
    for z in _zone_bounds(canopy):
        ax.axhline(z, color=INK_SECONDARY, lw=0.7)


def _zone_labels(ax, canopy, labels):
    upper, lower = _zone_bounds(canopy)
    depth, width = canopy.z_edges[-1], canopy.x_edges[-1]
    for z, text in zip((upper / 2, (upper + lower) / 2, (lower + depth) / 2), labels):
        ax.text(width * 1.025, z, text, va="center", ha="left", color=INK_SECONDARY,
                fontsize=7.5, clip_on=False)


def _cell_outline(ax, canopy, mask):
    xe, ze = canopy.x_edges, canopy.z_edges
    padded = np.pad(mask, 1)
    segments = []
    for i, j in zip(*np.nonzero(mask)):
        if not padded[i, j + 1]:
            segments.append(((xe[i], ze[j]), (xe[i], ze[j + 1])))
        if not padded[i + 2, j + 1]:
            segments.append(((xe[i + 1], ze[j]), (xe[i + 1], ze[j + 1])))
        if not padded[i + 1, j]:
            segments.append(((xe[i], ze[j]), (xe[i + 1], ze[j])))
        if not padded[i + 1, j + 2]:
            segments.append(((xe[i], ze[j + 1]), (xe[i + 1], ze[j + 1])))
    ax.add_collection(LineCollection(segments, colors=INK, linewidths=0.6))


def _fruits(ax, canopy):
    for fruit in canopy.fruits:
        ax.add_patch(Circle((fruit.x, fruit.z), fruit.radius, fill=False, lw=0.5, ec=INK))


def _position_labels(ax):
    ax.set_xlabel("Position across the row (m)")
    ax.set_ylabel("Depth below canopy top (m)")


@_styled
def uv_field(canopy, path, dpi=DPI):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.1))
    image = _field(ax, canopy, canopy.intensity_ratio, _ramp(ORANGE_RAMP, "uv"), Normalize(0.0, 1.0))
    _cell_outline(ax, canopy, canopy.refuge)
    _fruits(ax, canopy)
    _zone_lines(ax, canopy)
    _zone_labels(ax, canopy, ("Outer", "Mid", "Inner"))
    _position_labels(ax)
    handles = [Line2D([0], [0], color=INK, lw=0.6,
                      label=f"Refuge cells, I/I₀ ≤ {P.REFUGE_THRESHOLD.value:.2f}"),
               Line2D([0], [0], ls="none", marker="o", ms=5, mfc="none", mec=INK, mew=0.5,
                      label="Fruit body")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              borderaxespad=0.3, handlelength=1.6)
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.16, aspect=45)
    bar.set_label("UV intensity relative to the canopy top, I/I₀")
    bar.outline.set_visible(False)
    return _save(fig, path, dpi)


@_styled
def deposition_map(canopy, deposition, masks, path, dpi=DPI):
    released = P.N_DROPLETS.value
    per_m2 = 100.0 * deposition / (canopy.dx ** 2 * released)
    positive = per_m2[per_m2 > 0.0]
    norm = LogNorm(positive.min(), positive.max())
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.0))
    image = _field(ax, canopy, np.where(per_m2 > 0.0, per_m2, np.nan), _ramp(BLUE_RAMP, "deposit"), norm)
    _fruits(ax, canopy)
    _zone_lines(ax, canopy)
    shares = [100.0 * deposition[masks[z]].sum() / released for z in ("outer", "mid", "inner")]
    _zone_labels(ax, canopy, [f"{name}\n{share:.1f}% of spray"
                              for name, share in zip(("Outer", "Mid", "Inner"), shares)])
    _position_labels(ax)
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.14, aspect=45)
    bar.set_label("Spray landing, % of the spray released per m² (log scale)")
    bar.ax.xaxis.set_major_locator(LogLocator(base=10.0))
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bar.outline.set_visible(False)
    return _save(fig, path, dpi)


@_styled
def survival_maps(canopy, path, hours=None, dpi=DPI):
    hours = P.CHECKPOINTS_H if hours is None else hours
    fig, axes = plt.subplots(1, len(hours), figsize=(TEXT_WIDTH_IN, 2.5), sharey=True)
    cmap, norm = _ramp(AQUA_RAMP, "viability"), Normalize(0.0, 1.0)
    for ax, h in zip(axes, hours):
        image = _field(ax, canopy, survival.viable_fraction(canopy.intensity_ratio, h), cmap, norm)
        _zone_lines(ax, canopy)
        ax.set_title(f"{h:g} h after spraying")
    axes[0].set_ylabel("Depth below\ncanopy top (m)")
    axes[len(axes) // 2].set_xlabel("Position across the row (m)")
    _zone_labels(axes[-1], canopy, ("Outer", "Mid", "Inner"))
    bar = fig.colorbar(image, ax=list(axes), orientation="horizontal", fraction=0.06, pad=0.2, aspect=50)
    bar.set_label("Share of deposited CrleGV still active")
    bar.outline.set_visible(False)
    return _save(fig, path, dpi)


def zone_curves(canopy, deposition, masks, hours):
    decay = survival.decay_constants(canopy.intensity_ratio)
    deposited = deposition > 0.0
    hours = np.asarray(hours, dtype=float)
    curves = {}
    for key, _, _ in ZONES:
        selected = decay[masks[key] & deposited]
        curves[key] = (np.exp(-np.outer(hours, selected)).mean(axis=1) if selected.size
                       else np.full(hours.shape, np.nan))
    return curves


@_styled
def survival_curves(canopy, deposition, masks, path, dpi=DPI):
    early = np.linspace(0.0, 72.0, 289)
    late_days = np.linspace(0.0, 21.0, 421)
    first = zone_curves(canopy, deposition, masks, early)
    second = zone_curves(canopy, deposition, masks, late_days * 24.0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7))
    for key, label, colour in ZONES:
        if np.isnan(first[key]).all():
            continue
        a.plot(early, first[key], color=colour, lw=1.5, label=label)
        b.plot(late_days, second[key], color=colour, lw=1.5, label=label)
        a.text(74.0, first[key][-1], label, va="center", ha="left", color=INK_SECONDARY,
               fontsize=7.5, clip_on=False)
    a.axhline(0.5, color=MUTED, lw=0.6, zorder=0)
    a.text(1.5, 0.52, "Half remaining", color=MUTED, fontsize=7, va="bottom")
    a.set_xlim(0.0, 72.0)
    a.set_xticks([0, 24, 48, 72])
    a.set_ylim(0.0, 1.0)
    a.set_xlabel("Hours after spraying")
    a.set_ylabel("Share still active")
    a.set_title("(a) First 72 hours")
    b.set_yscale("log")
    lowest = min(np.nanmin(values) for values in second.values() if np.isfinite(values).any())
    b.set_ylim(10.0 ** np.floor(np.log10(lowest)), 1.0)
    b.set_xlim(0.0, 21.0)
    b.set_xticks([0, 7, 14, 21])
    b.set_xlabel("Days after spraying")
    b.set_title("(b) Three weeks, log scale")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.yaxis.set_minor_locator(NullLocator())
    b.legend(loc="lower left", handlelength=1.6)
    for ax in (a, b):
        ax.grid(axis="y", color=GRIDLINE, lw=0.5)
        ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.62)
    return _save(fig, path, dpi)


@_styled
def main_effects(summary, path, dpi=DPI):
    effects = summary["effects"]
    grand = effects["grand_mean"]
    main = effects["main"]
    change = {key: [100.0 * (m - grand) / grand for m in main[key]["level_means"]] for key in main}
    limit = np.ceil(1.3 * max(abs(v) for values in change.values() for v in values))
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(TEXT_WIDTH_IN, 2.4), sharey=True)
    for ax, (key, _, label, unit, scale) in zip(axes, FACTORS):
        levels = np.asarray(main[key]["levels"], dtype=float) * scale
        pad = 0.18 * (levels.max() - levels.min())
        ax.axhline(0.0, color=AXIS, lw=0.6)
        ax.plot(levels, change[key], color=SERIES, lw=1.5, marker="o", ms=5.5,
                mfc=SERIES, mec=SURFACE, mew=1.2)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{v:g}" for v in levels])
        ax.set_xlim(levels.min() - pad, levels.max() + pad)
        ax.set_xlabel(unit)
        ax.set_title(label, fontsize=8.5)
        ax.text(0.03, 0.97, f"{100.0 * main[key]['share']:.0f}% of variation", transform=ax.transAxes,
                va="top", ha="left", color=INK_SECONDARY, fontsize=7.5)
        ax.grid(axis="y", color=GRIDLINE, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylim(-limit, limit)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: _minus(f"{v:+.0f}") if v else "0"))
    axes[0].set_ylabel("Inner-canopy spray vs mean\nof all 81 runs (%)")
    fig.subplots_adjust(wspace=0.12)
    return _save(fig, path, dpi)


@_styled
def effect_summary(summary, path, dpi=DPI):
    effects = summary["effects"]
    main = effects["main"]
    shares = [(label, 100.0 * main[key]["share"], SERIES) for key, label, _, _, _ in FACTORS]
    shares += [("Two-factor interactions", 100.0 * sum(effects["interactions"].values()), DEEMPHASIS),
               ("Higher-order terms", 100.0 * effects["higher_order_share"], DEEMPHASIS)]
    shares.sort(key=lambda row: row[1])
    sens = sorted(((label, main[key]["sensitivity"]) for key, label, _, _, _ in FACTORS),
                  key=lambda row: abs(row[1]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.5),
                               gridspec_kw={"width_ratios": [1.15, 1.0]})
    rows = np.arange(len(shares))
    values = [value for _, value, _ in shares]
    a.barh(rows, values, height=0.55, color=[colour for _, _, colour in shares])
    for y, value in zip(rows, values):
        a.text(value + 1.5, y, f"{value:.1f}%", va="center", ha="left", color=INK_SECONDARY, fontsize=7.5)
    a.set_yticks(rows)
    a.set_yticklabels([label for label, _, _ in shares])
    a.set_xlim(0.0, 1.25 * max(values))
    a.set_xlabel("Share of variation in inner-canopy spray (%)")
    a.set_title("(a) Where the variation comes from")
    a.tick_params(axis="y", length=0)
    rows = np.arange(len(sens))
    values = [value for _, value in sens]
    span = max(abs(v) for v in values)
    b.barh(rows, values, height=0.55, color=[NEGATIVE if v < 0 else SERIES for v in values])
    b.axvline(0.0, color=AXIS, lw=0.6)
    for y, v in zip(rows, values):
        b.text(v + 0.06 * span * (1 if v >= 0 else -1), y, _minus(f"{v:+.3f}"), va="center",
               ha="left" if v >= 0 else "right", color=INK_SECONDARY, fontsize=7.5)
    b.set_yticks(rows)
    b.set_yticklabels([label for label, _ in sens])
    b.set_xlim(-1.9 * span, 1.4 * span)
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _minus(f"{v:+.2f}") if v else "0"))
    b.set_xlabel("Normalised sensitivity, S")
    b.set_title("(b) Sensitivity, ordered by |S|")
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_ylim(-0.6, len(sens) - 0.05)
    b.text(-0.05 * span, len(sens) - 0.3, "← lowers", ha="right", va="center", color=MUTED, fontsize=7.5)
    b.text(0.05 * span, len(sens) - 0.3, "raises →", ha="left", va="center", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(wspace=0.75)
    return _save(fig, path, dpi)


def validation_table(source, path):
    source = Path(source)
    t1, t2, t4 = (json.loads((source / f"target{i}.json").read_text()) for i in (1, 2, 4))
    rows = [("Target", "Quantity", "Benchmark", "Simulated", "Criterion", "Result")]
    for r in t1["rows"]:
        rows.append(("1", f"Fall speed of a {r['diameter_mm']:g} mm drop (m/s)", f"{r['measured_m_s']:.2f}",
                     f"{r['simulated_m_s']:.3f} ({100.0 * r['deviation']:+.1f}%)",
                     f"Within ±{100.0 * t1['tolerance']:.0f}%",
                     "Pass" if r["within_tolerance"] else "Fail"))
    low, high = t2["band"]
    rows.append(("2", "Inner ÷ outer deposit per m²",
                 f"{min(t2['benchmark_ratios']):.3f}–{max(t2['benchmark_ratios']):.3f}",
                 f"{t2['ratio']:.3f}", f"{low:.3f}–{high:.3f}", "Pass" if t2["passed"] else "Fail"))
    low, high = t4["band_21d"]
    rows.append(("4", "Inner ÷ outer share still active at 21 days", "15", f"{t4['ratio_21d']:.2f}",
                 f"{low:g}–{high:g}, and inner above outer at every checkpoint",
                 "Pass" if t4["passed"] else "Fail"))
    light_path = source / "light.json"
    if light_path.exists():
        light = json.loads(light_path.read_text())
        low, high = light["benchmark_range"]
        floor = light["floor_light"]
        rows.append(("Light", "Light at the canopy floor, leaves only (share of full light)",
                     f"{low:.2f}–{high:.2f}",
                     f"{floor['central']:.3f} ({floor['least_light']:.3f}–{floor['most_light']:.3f})",
                     "Comparison only", "Comparison"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(rows)
    return path


def central_deposition(runs_dir=RUNS_DIR):
    central = {"altitude": P.UAV_ALTITUDE.value, "v_uav": P.UAV_VELOCITY.value,
               "v_dw": P.DOWNWASH_VELOCITY.value, "r_p": P.DROPLET_RADIUS.value}
    for record in sorted(Path(runs_dir).glob("fact_*/record.json")):
        inputs = json.loads(record.read_text())["inputs"]
        if all(abs(inputs[k] - v) < 1e-12 for k, v in central.items()):
            return np.load(record.parent / "arrays.npz")["deposition"]
    raise FileNotFoundError("No factorial run at the central settings - run src/sweep.py first.")


def render_all(out_dir=FIGURES_DIR, runs_dir=RUNS_DIR, validation_dir=VALIDATION_DIR, dpi=DPI):
    out_dir = Path(out_dir)
    canopy = grid.build()
    masks = grid.zones(canopy)
    deposition = central_deposition(runs_dir)
    summary = json.loads((Path(runs_dir) / "factorial_summary.json").read_text())
    return [
        uv_field(canopy, out_dir / "uv_field.png", dpi),
        deposition_map(canopy, deposition, masks, out_dir / "deposition_central.png", dpi),
        survival_maps(canopy, out_dir / "survival_maps.png", dpi=dpi),
        survival_curves(canopy, deposition, masks, out_dir / "survival_curves.png", dpi),
        main_effects(summary, out_dir / "main_effects.png", dpi),
        effect_summary(summary, out_dir / "effect_summary.png", dpi),
        validation_table(validation_dir, out_dir / "validation_table.csv"),
    ]


if __name__ == "__main__":
    for written in render_all():
        print(written)
