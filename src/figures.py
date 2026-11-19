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
import params
import survival

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("CRLEGV_RUNS_DIR", ROOT / "runs"))
FIGURES_DIR = Path(os.environ.get("CRLEGV_FIGURES_DIR", ROOT / "figures"))
VALIDATION_DIR = Path(os.environ.get("CRLEGV_VALIDATION_DIR", RUNS_DIR / "validation"))
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
ZONE_STYLES = (("outer", "Outer", "#86b6ef"), ("middle", "Mid", "#2a78d6"),
               ("inner", "Inner", "#104281"), ("shaded_next_to_fruit", "Microhabitat", "#eb6834"))
SETTING_STYLES = (("drone_speed", "Flight speed", "Flight speed", "m/s", 1.0),
                  ("droplet_radius", "Droplet size", "Droplet size", "µm diameter", 2e6),
                  ("drone_height", "Height above canopy", "Height", "m above canopy", 1.0),
                  ("downwash_speed", "Downwash", "Downwash", "m/s", 1.0))

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


def _in_our_style(draw):
    @functools.wraps(draw)
    def draw_in_style(*args, **kwargs):
        with plt.rc_context(STYLE):
            return draw(*args, **kwargs)
    return draw_in_style


def _colour_ramp(colours, name):
    colour_map = LinearSegmentedColormap.from_list(name, colours)
    colour_map.set_bad(SURFACE)
    return colour_map


def _save_figure(figure, path, dpi):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return path


def _with_true_minus_sign(text):
    return text.replace("-", "−")


def _draw_cell_values(panel, canopy, values, colour_map, colour_scale):
    width, depth = canopy.width, canopy.depth
    image = panel.imshow(values.T, cmap=colour_map, norm=colour_scale, origin="upper", aspect="equal",
                         extent=(0.0, width, depth, 0.0), interpolation="nearest")
    panel.set_xlim(0.0, width)
    panel.set_ylim(depth, 0.0)
    panel.set_xticks(np.arange(0.0, width + 1e-9, 1.0))
    panel.set_yticks(np.arange(0.0, depth + 1e-9, 1.0))
    return image


def _zone_boundary_depths(canopy):
    rows_per_third = canopy.cells_deep // 3
    return canopy.row_boundaries[rows_per_third], canopy.row_boundaries[2 * rows_per_third]


def _draw_zone_lines(panel, canopy):
    for boundary_depth in _zone_boundary_depths(canopy):
        panel.axhline(boundary_depth, color=INK_SECONDARY, lw=0.7)


def _label_zones(panel, canopy, labels):
    outer_to_middle, middle_to_inner = _zone_boundary_depths(canopy)
    depth, width = canopy.depth, canopy.width
    label_depths = (outer_to_middle / 2, (outer_to_middle + middle_to_inner) / 2,
                    (middle_to_inner + depth) / 2)
    for label_depth, text in zip(label_depths, labels):
        panel.text(width * 1.025, label_depth, text, va="center", ha="left", color=INK_SECONDARY,
                   fontsize=7.5, clip_on=False)


def _outline_cells(panel, canopy, marked):
    column_edges, row_edges = canopy.column_boundaries, canopy.row_boundaries
    padded = np.pad(marked, 1)
    segments = []
    for column, row in zip(*np.nonzero(marked)):
        left, right = column_edges[column], column_edges[column + 1]
        top, bottom = row_edges[row], row_edges[row + 1]
        marked_on_left = padded[column, row + 1]
        marked_on_right = padded[column + 2, row + 1]
        marked_above = padded[column + 1, row]
        marked_below = padded[column + 1, row + 2]
        if not marked_on_left:
            segments.append(((left, top), (left, bottom)))
        if not marked_on_right:
            segments.append(((right, top), (right, bottom)))
        if not marked_above:
            segments.append(((left, top), (right, top)))
        if not marked_below:
            segments.append(((left, bottom), (right, bottom)))
    panel.add_collection(LineCollection(segments, colors=INK, linewidths=0.6))


def _draw_fruits(panel, canopy):
    for fruit in canopy.fruits:
        panel.add_patch(Circle((fruit.position_across, fruit.depth), fruit.radius, fill=False, lw=0.5, ec=INK))


def _label_position_axes(panel):
    panel.set_xlabel("Position across the row (m)")
    panel.set_ylabel("Depth below canopy top (m)")


@_in_our_style
def draw_light_map(canopy, path, dpi=DPI):
    figure, panel = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.1))
    image = _draw_cell_values(panel, canopy, canopy.share_of_full_light,
                              _colour_ramp(ORANGE_RAMP, "uv"), Normalize(0.0, 1.0))
    _outline_cells(panel, canopy, canopy.deep_shade)
    _draw_fruits(panel, canopy)
    _draw_zone_lines(panel, canopy)
    _label_zones(panel, canopy, ("Outer", "Mid", "Inner"))
    _label_position_axes(panel)
    legend_entries = [Line2D([0], [0], color=INK, lw=0.6,
                             label=f"Refuge cells, I/I₀ ≤ {params.DEEP_SHADE_LIGHT_LIMIT.value:.2f}"),
                      Line2D([0], [0], ls="none", marker="o", ms=5, mfc="none", mec=INK, mew=0.5,
                             label="Fruit body")]
    panel.legend(handles=legend_entries, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
                 borderaxespad=0.3, handlelength=1.6)
    colour_bar = figure.colorbar(image, ax=panel, orientation="horizontal", fraction=0.045, pad=0.16, aspect=45)
    colour_bar.set_label("UV intensity relative to the canopy top, I/I₀")
    colour_bar.outline.set_visible(False)
    return _save_figure(figure, path, dpi)


@_in_our_style
def draw_spray_map(canopy, spray_landed, zones, path, dpi=DPI):
    released = params.DROPLETS_PER_RUN.value
    percent_per_m2 = 100.0 * spray_landed / (canopy.cell_size ** 2 * released)
    above_zero = percent_per_m2[percent_per_m2 > 0.0]
    colour_scale = LogNorm(above_zero.min(), above_zero.max())
    figure, panel = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.0))
    image = _draw_cell_values(panel, canopy, np.where(percent_per_m2 > 0.0, percent_per_m2, np.nan),
                              _colour_ramp(BLUE_RAMP, "deposit"), colour_scale)
    _draw_fruits(panel, canopy)
    _draw_zone_lines(panel, canopy)
    percent_by_zone = [100.0 * spray_landed[zones[zone_name]].sum() / released
                       for zone_name in ("outer", "middle", "inner")]
    _label_zones(panel, canopy, [f"{label}\n{percent:.1f}% of spray"
                                 for label, percent in zip(("Outer", "Mid", "Inner"), percent_by_zone)])
    _label_position_axes(panel)
    colour_bar = figure.colorbar(image, ax=panel, orientation="horizontal", fraction=0.045, pad=0.14, aspect=45)
    colour_bar.set_label("Spray landing, % of the spray released per m² (log scale)")
    colour_bar.ax.xaxis.set_major_locator(LogLocator(base=10.0))
    colour_bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    colour_bar.outline.set_visible(False)
    return _save_figure(figure, path, dpi)


@_in_our_style
def draw_virus_maps(canopy, path, checkpoint_hours=None, dpi=DPI):
    checkpoint_hours = params.CHECKPOINT_HOURS if checkpoint_hours is None else checkpoint_hours
    figure, panels = plt.subplots(1, len(checkpoint_hours), figsize=(TEXT_WIDTH_IN, 2.5), sharey=True)
    colour_map, colour_scale = _colour_ramp(AQUA_RAMP, "viability"), Normalize(0.0, 1.0)
    for panel, hours in zip(panels, checkpoint_hours):
        image = _draw_cell_values(panel, canopy, survival.share_still_active(canopy.share_of_full_light, hours),
                                  colour_map, colour_scale)
        _draw_zone_lines(panel, canopy)
        panel.set_title(f"{hours:g} h after spraying")
    panels[0].set_ylabel("Depth below\ncanopy top (m)")
    panels[len(panels) // 2].set_xlabel("Position across the row (m)")
    _label_zones(panels[-1], canopy, ("Outer", "Mid", "Inner"))
    colour_bar = figure.colorbar(image, ax=list(panels), orientation="horizontal", fraction=0.06, pad=0.2,
                                 aspect=50)
    colour_bar.set_label("Share of deposited CrleGV still active")
    colour_bar.outline.set_visible(False)
    return _save_figure(figure, path, dpi)


def survival_curve_by_zone(canopy, spray_landed, zones, hours):
    breakdown_rate = survival.breakdown_rates(canopy.share_of_full_light)
    received_spray = spray_landed > 0.0
    hours = np.asarray(hours, dtype=float)
    curves = {}
    for zone_name, _, _ in ZONE_STYLES:
        rates_in_zone = breakdown_rate[zones[zone_name] & received_spray]
        curves[zone_name] = (np.exp(-np.outer(hours, rates_in_zone)).mean(axis=1) if rates_in_zone.size
                             else np.full(hours.shape, np.nan))
    return curves


@_in_our_style
def draw_survival_curves(canopy, spray_landed, zones, path, dpi=DPI):
    first_72_hours = np.linspace(0.0, 72.0, 289)
    three_weeks_in_days = np.linspace(0.0, 21.0, 421)
    early = survival_curve_by_zone(canopy, spray_landed, zones, first_72_hours)
    late = survival_curve_by_zone(canopy, spray_landed, zones, three_weeks_in_days * 24.0)
    figure, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7))
    for zone_name, label, colour in ZONE_STYLES:
        if np.isnan(early[zone_name]).all():
            continue
        left.plot(first_72_hours, early[zone_name], color=colour, lw=1.5, label=label)
        right.plot(three_weeks_in_days, late[zone_name], color=colour, lw=1.5, label=label)
        left.text(74.0, early[zone_name][-1], label, va="center", ha="left", color=INK_SECONDARY,
                  fontsize=7.5, clip_on=False)
    left.axhline(0.5, color=MUTED, lw=0.6, zorder=0)
    left.text(1.5, 0.52, "Half remaining", color=MUTED, fontsize=7, va="bottom")
    left.set_xlim(0.0, 72.0)
    left.set_xticks([0, 24, 48, 72])
    left.set_ylim(0.0, 1.0)
    left.set_xlabel("Hours after spraying")
    left.set_ylabel("Share still active")
    left.set_title("(a) First 72 hours")
    right.set_yscale("log")
    lowest = min(np.nanmin(values) for values in late.values() if np.isfinite(values).any())
    right.set_ylim(10.0 ** np.floor(np.log10(lowest)), 1.0)
    right.set_xlim(0.0, 21.0)
    right.set_xticks([0, 7, 14, 21])
    right.set_xlabel("Days after spraying")
    right.set_title("(b) Three weeks, log scale")
    right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    right.yaxis.set_minor_locator(NullLocator())
    right.legend(loc="lower left", handlelength=1.6)
    for panel in (left, right):
        panel.grid(axis="y", color=GRIDLINE, lw=0.5)
        panel.set_axisbelow(True)
    figure.subplots_adjust(wspace=0.62)
    return _save_figure(figure, path, dpi)


@_in_our_style
def draw_main_effects(summary, path, dpi=DPI):
    effects = summary["effects"]
    overall_mean = effects["overall_mean"]
    main_effects = effects["main_effects"]
    percent_change = {name: [100.0 * (mean - overall_mean) / overall_mean for mean in effect["level_means"]]
                      for name, effect in main_effects.items()}
    limit = np.ceil(1.3 * max(abs(change) for changes in percent_change.values() for change in changes))
    figure, panels = plt.subplots(1, len(SETTING_STYLES), figsize=(TEXT_WIDTH_IN, 2.4), sharey=True)
    for panel, (name, _, label, unit, to_display_unit) in zip(panels, SETTING_STYLES):
        levels = np.asarray(main_effects[name]["levels"], dtype=float) * to_display_unit
        positions = np.arange(len(levels), dtype=float) if levels.max() / levels.min() > 4.0 else levels
        margin = 0.18 * (positions.max() - positions.min())
        panel.axhline(0.0, color=AXIS, lw=0.6)
        panel.plot(positions, percent_change[name], color=SERIES, lw=1.5, marker="o", ms=5.5,
                   mfc=SERIES, mec=SURFACE, mew=1.2)
        panel.set_xlim(positions.min() - margin, positions.max() + margin)
        panel.set_xticks(positions)
        panel.set_xticklabels([f"{level:g}" for level in levels])
        panel.set_xlabel(unit)
        panel.set_title(label, fontsize=8.5)
        panel.text(0.03, 0.97, f"{100.0 * main_effects[name]['share_of_variation']:.0f}% of variation",
                   transform=panel.transAxes, va="top", ha="left", color=INK_SECONDARY, fontsize=7.5)
        panel.grid(axis="y", color=GRIDLINE, lw=0.5)
        panel.set_axisbelow(True)
    panels[0].set_ylim(-limit, limit)
    panels[0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: _with_true_minus_sign(f"{value:+.0f}") if value else "0"))
    panels[0].set_ylabel("Inner-canopy spray vs mean\nof all 81 runs (%)")
    figure.subplots_adjust(wspace=0.12)
    return _save_figure(figure, path, dpi)


@_in_our_style
def draw_effect_summary(summary, path, dpi=DPI):
    effects = summary["effects"]
    main_effects = effects["main_effects"]
    shares = [(label, 100.0 * main_effects[name]["share_of_variation"], SERIES)
              for name, label, _, _, _ in SETTING_STYLES]
    shares += [("Two-factor interactions", 100.0 * sum(effects["interactions"].values()), MUTED),
               ("Higher-order terms", 100.0 * effects["higher_order_share"], MUTED)]
    shares.sort(key=lambda bar: bar[1])
    sensitivities = sorted(((label, main_effects[name]["sensitivity"]) for name, label, _, _, _ in SETTING_STYLES),
                           key=lambda bar: abs(bar[1]))
    figure, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.5),
                                         gridspec_kw={"width_ratios": [1.15, 1.0]})
    bar_positions = np.arange(len(shares))
    values = [value for _, value, _ in shares]
    left.barh(bar_positions, values, height=0.55, color=[colour for _, _, colour in shares])
    for position, value in zip(bar_positions, values):
        left.text(value + 1.5, position, f"{value:.1f}%", va="center", ha="left", color=INK_SECONDARY,
                  fontsize=7.5)
    left.set_yticks(bar_positions)
    left.set_yticklabels([label for label, _, _ in shares])
    left.set_xlim(0.0, 1.25 * max(values))
    left.set_xlabel("Share of variation in inner-canopy spray (%)")
    left.set_title("(a) Where the variation comes from")
    left.tick_params(axis="y", length=0)
    bar_positions = np.arange(len(sensitivities))
    values = [value for _, value in sensitivities]
    largest = max(abs(value) for value in values)
    right.barh(bar_positions, values, height=0.55, color=[NEGATIVE if value < 0 else SERIES for value in values])
    right.axvline(0.0, color=AXIS, lw=0.6)
    for position, value in zip(bar_positions, values):
        right.text(value + 0.06 * largest * (1 if value >= 0 else -1), position,
                   _with_true_minus_sign(f"{value:+.3f}"), va="center",
                   ha="left" if value >= 0 else "right", color=INK_SECONDARY, fontsize=7.5)
    right.set_yticks(bar_positions)
    right.set_yticklabels([label for label, _ in sensitivities])
    right.set_xlim(-1.9 * largest, 1.4 * largest)
    right.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: _with_true_minus_sign(f"{value:+.2f}") if value else "0"))
    right.set_xlabel("Normalised sensitivity, S")
    right.set_title("(b) Sensitivity, ordered by |S|")
    right.tick_params(axis="y", length=0)
    right.spines["left"].set_visible(False)
    right.set_ylim(-0.6, len(sensitivities) - 0.05)
    right.text(-0.05 * largest, len(sensitivities) - 0.3, "← lowers", ha="right", va="center", color=MUTED,
               fontsize=7.5)
    right.text(0.05 * largest, len(sensitivities) - 0.3, "raises →", ha="left", va="center", color=MUTED,
               fontsize=7.5)
    figure.subplots_adjust(wspace=0.75)
    return _save_figure(figure, path, dpi)


def write_validation_table(source, path):
    source = Path(source)
    fall_speed_check, spray_check, virus_check = (
        json.loads((source / f"target{number}.json").read_text()) for number in (1, 2, 4))
    rows = [("Target", "Quantity", "Benchmark", "Simulated", "Criterion", "Result")]
    for row in fall_speed_check["rows"]:
        rows.append(("1", f"Fall speed of a {row['diameter_mm']:g} mm drop (m/s)", f"{row['measured_m_s']:.2f}",
                     f"{row['simulated_m_s']:.3f} ({100.0 * row['deviation']:+.1f}%)",
                     f"Within ±{100.0 * fall_speed_check['tolerance']:.0f}%",
                     "Pass" if row["within_tolerance"] else "Fail"))
    low, high = spray_check["band"]
    rows.append(("2", "Inner ÷ outer deposit per m²",
                 f"{min(spray_check['benchmark_ratios']):.3f}–{max(spray_check['benchmark_ratios']):.3f}",
                 f"{spray_check['ratio']:.3f}", f"{low:.3f}–{high:.3f}",
                 "Pass" if spray_check["passed"] else "Fail"))
    low, high = virus_check["band_21d"]
    rows.append(("4", "Inner ÷ outer share still active at 21 days", "15", f"{virus_check['ratio_21d']:.2f}",
                 f"{low:g}–{high:g}, and inner above outer at every checkpoint",
                 "Pass" if virus_check["passed"] else "Fail"))
    light_path = source / "light.json"
    if light_path.exists():
        light_check = json.loads(light_path.read_text())
        low, high = light_check["benchmark_range"]
        floor_light = light_check["floor_light"]
        rows.append(("Light", "Light at the canopy floor, leaves only (share of full light)",
                     f"{low:.2f}–{high:.2f}",
                     f"{floor_light['central']:.3f} ({floor_light['least_light']:.3f}–{floor_light['most_light']:.3f})",
                     "Comparison only", "Comparison"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as table_file:
        csv.writer(table_file).writerows(rows)
    return path


def load_central_spray_map(runs_dir=RUNS_DIR):
    central_settings = {name: setting.value for name, setting in params.SETTINGS_TESTED.items()}
    for record_path in sorted(Path(runs_dir).glob("fact_*/record.json")):
        inputs = json.loads(record_path.read_text())["inputs"]
        if all(abs(inputs[name] - value) < 1e-12 for name, value in central_settings.items()):
            return np.load(record_path.parent / "arrays.npz")["spray_landed"]
    raise FileNotFoundError("No factorial run at the central settings - run src/sweep.py first.")


def draw_all(figures_dir=FIGURES_DIR, runs_dir=RUNS_DIR, validation_dir=VALIDATION_DIR, dpi=DPI):
    figures_dir = Path(figures_dir)
    canopy = grid.build_canopy()
    zones = grid.canopy_zones(canopy)
    spray_landed = load_central_spray_map(runs_dir)
    summary = json.loads((Path(runs_dir) / "factorial_summary.json").read_text())
    return [
        draw_light_map(canopy, figures_dir / "uv_field.png", dpi),
        draw_spray_map(canopy, spray_landed, zones, figures_dir / "deposition_central.png", dpi),
        draw_virus_maps(canopy, figures_dir / "survival_maps.png", dpi=dpi),
        draw_survival_curves(canopy, spray_landed, zones, figures_dir / "survival_curves.png", dpi),
        draw_main_effects(summary, figures_dir / "main_effects.png", dpi),
        draw_effect_summary(summary, figures_dir / "effect_summary.png", dpi),
        write_validation_table(validation_dir, figures_dir / "validation_table.csv"),
    ]


if __name__ == "__main__":
    for written in draw_all():
        print(written)
