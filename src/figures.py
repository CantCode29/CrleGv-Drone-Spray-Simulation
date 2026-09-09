"""Figure generation for the results chapter.

Colour is assigned by the job it does. The three field plots are magnitude,
so each uses a single-hue sequential ramp running light to dark. The zone
series are identity, so they use a fixed categorical order validated for
colour-vision deficiency; that order never changes between figures, and every
series is directly labelled so identity is never carried by colour alone.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

ZONE_ORDER = ["outer", "mid", "inner", "microhabitat"]
ZONE_COLOURS = {
    "outer": "#0072B2",
    "mid": "#009E73",
    "inner": "#E69F00",
    "microhabitat": "#CC79A7",
}

FIELD_RAMPS = {
    "deposition": "Blues",
    "intensity": "Purples",
    "survival": "Greens",
}

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#e4e4e2"
SURFACE = "#fcfcfb"
DPI = 300


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    return ax


def field(array, canopy, kind, title, label, path, log=False):
    """Render one 2D field over the canopy cross-section."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    data = array.T
    norm = matplotlib.colors.LogNorm(vmin=max(data[data > 0].min(), 1e-6),
                                     vmax=data.max()) if log else None
    im = ax.imshow(
        data, origin="upper", aspect="auto",
        extent=[0, canopy.x_edges[-1], canopy.z_edges[-1], 0],
        cmap=FIELD_RAMPS[kind], norm=norm,
    )
    ax.set_xlabel("Horizontal position x (m)", fontsize=9)
    ax.set_ylabel("Canopy depth z (m)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", pad=10)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=8, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, path)


def survival_curves(hours, series, path,
                    title="Mean viable CrleGV fraction by canopy zone"):
    """One line per zone, directly labelled at the right-hand end."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for zone in ZONE_ORDER:
        if zone not in series:
            continue
        y = series[zone]
        ax.plot(hours, y, color=ZONE_COLOURS[zone], linewidth=2.0,
                marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=zone, zorder=3)
        ax.annotate(zone, xy=(hours[-1], y[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)

    ax.set_xlabel("Hours after deposition", fontsize=9)
    ax.set_ylabel("Mean viable fraction", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", pad=10)
    ax.set_xlim(0, hours[-1] * 1.22)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower left",
              labelcolor=INK, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def sensitivity_bars(coefficients, path,
                     title="Normalised sensitivity coefficients"):
    """Ranked by magnitude. One measure, one hue - this is not identity."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    names = list(coefficients.keys())
    values = [coefficients[n] for n in names]
    ax.barh(names, values, color="#0072B2", height=0.55, zorder=3)
    for name, value in zip(names, values):
        ax.annotate(f"{value:+.2f}", xy=(value, name),
                    xytext=(6 if value >= 0 else -6, 0),
                    textcoords="offset points", fontsize=8, color=INK,
                    va="center", ha="left" if value >= 0 else "right")

    ax.axvline(0, color=INK_MUTED, linewidth=1.0)
    ax.set_xlabel("S = (dOutput/Output) / (dInput/Input)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left", pad=10)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path):
    out = FIGURES_DIR / path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out
