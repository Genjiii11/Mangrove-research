import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {
    "year",
    "class_name",
    "area_ha",
    "annual_esv_2020usd",
    "adjusted_annual_esv_2020usd",
}


def plot_esv_time_series(input_csv: str | Path, output_file: str | Path) -> None:
    """
    Reads the yearly ESV output CSV, extracts mangrove time series,
    and generates a 3-panel publication-quality figure.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing_text}")

    # Filter for mangrove and sort chronologically
    df_mangrove = df.loc[df["class_name"] == "mangrove"].sort_values(by="year").copy()

    if df_mangrove.empty:
        raise ValueError("No rows found with class_name == 'mangrove'")

    # Convert units to publication-friendly scales
    df_mangrove["area_km2"] = df_mangrove["area_ha"] / 100.0
    df_mangrove["esv_million"] = df_mangrove["annual_esv_2020usd"] / 1e6
    df_mangrove["adj_esv_million"] = df_mangrove["adjusted_annual_esv_2020usd"] / 1e6

    # Calculate interval changes
    df_mangrove["prev_area"] = df_mangrove["area_km2"].shift(1)
    df_mangrove["prev_esv"] = df_mangrove["esv_million"].shift(1)
    df_mangrove["prev_adj_esv"] = df_mangrove["adj_esv_million"].shift(1)

    df_mangrove["area_diff"] = df_mangrove["area_km2"] - df_mangrove["prev_area"]
    df_mangrove["esv_diff"] = df_mangrove["esv_million"] - df_mangrove["prev_esv"]
    df_mangrove["adj_esv_diff"] = (
        df_mangrove["adj_esv_million"] - df_mangrove["prev_adj_esv"]
    )

    df_mangrove["area_pct"] = (
        df_mangrove["area_diff"] / df_mangrove["prev_area"]
    ) * 100
    df_mangrove["esv_pct"] = (df_mangrove["esv_diff"] / df_mangrove["prev_esv"]) * 100
    df_mangrove["adj_esv_pct"] = (
        df_mangrove["adj_esv_diff"] / df_mangrove["prev_adj_esv"]
    ) * 100

    # Calculate Adjusted to Basic ESV ratio
    df_mangrove["esv_ratio"] = (
        df_mangrove["adj_esv_million"] / df_mangrove["esv_million"]
    )

    # Nature-style Matplotlib Configuration
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    # Editorial Palette
    COLOR_AREA = "#d5d5d5"
    COLOR_BASIC = "#2c3e50"  # Deep charcoal/blue
    COLOR_ADJ = "#c96a52"  # Muted terracotta
    COLOR_RATIO = "#4a90e2"  # Subtle teal/blue
    COLOR_ZERO = "#888888"

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.2, 1.2, 0.8], hspace=0.35)

    # General styling function
    def style_axes(ax, hide_top_right=True):
        if hide_top_right:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)

    # ==========================================
    # Panel (a): Area as bars + ESV lines
    # ==========================================
    ax1 = fig.add_subplot(gs[0])
    years = df_mangrove["year"].values
    area = df_mangrove["area_km2"].values
    esv = df_mangrove["esv_million"].values
    adj_esv = df_mangrove["adj_esv_million"].values

    bar_width = 1.0
    ax1.bar(
        years,
        area,
        color=COLOR_AREA,
        edgecolor="none",
        label="Area (km²)",
        width=bar_width,
        zorder=2,
    )
    ax1.set_ylabel("Mangrove Area (km²)")
    ax1.set_xticks(years)
    ax1.set_xticklabels(years)

    style_axes(ax1, hide_top_right=False)
    ax1.spines["top"].set_visible(False)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        years,
        esv,
        marker="o",
        markersize=5,
        color=COLOR_BASIC,
        label="Basic ESV",
        linewidth=1.5,
        zorder=3,
    )
    ax1_twin.plot(
        years,
        adj_esv,
        marker="s",
        markersize=5,
        color=COLOR_ADJ,
        label="MHI-adjusted ESV",
        linewidth=1.5,
        zorder=3,
    )
    ax1_twin.set_ylabel("ESV (Million 2020 USD)")
    ax1_twin.spines["top"].set_visible(False)

    # Unified legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
    )

    ax1.text(
        -0.05,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )

    # ==========================================
    # Panel (b): Interval changes (Lollipop chart)
    # ==========================================
    ax2 = fig.add_subplot(gs[1])

    intervals = []
    area_diffs, esv_diffs, adj_esv_diffs = [], [], []
    area_pcts, esv_pcts, adj_esv_pcts = [], [], []

    for i in range(1, len(years)):
        intervals.append(f"{int(years[i - 1])}-{int(years[i])}")
        area_diffs.append(df_mangrove["area_diff"].iloc[i])
        esv_diffs.append(df_mangrove["esv_diff"].iloc[i])
        adj_esv_diffs.append(df_mangrove["adj_esv_diff"].iloc[i])
        area_pcts.append(df_mangrove["area_pct"].iloc[i])
        esv_pcts.append(df_mangrove["esv_pct"].iloc[i])
        adj_esv_pcts.append(df_mangrove["adj_esv_pct"].iloc[i])

    x = np.arange(len(intervals))
    w = 0.22  # offset

    # Zero line
    ax2.axhline(0, color=COLOR_ZERO, linewidth=0.8, zorder=1)

    # Lollipops: Percentage change
    ax2.vlines(x - w, 0, area_pcts, color="#a0a0a0", linewidth=1.5, zorder=2)
    ax2.plot(
        x - w,
        area_pcts,
        "o",
        color="#a0a0a0",
        markersize=5,
        label="Area Change (%)",
        zorder=3,
    )

    ax2.vlines(x, 0, esv_pcts, color=COLOR_BASIC, linewidth=1.5, zorder=2)
    ax2.plot(
        x,
        esv_pcts,
        "o",
        color=COLOR_BASIC,
        markersize=5,
        label="Basic ESV Change (%)",
        zorder=3,
    )

    ax2.vlines(x + w, 0, adj_esv_pcts, color=COLOR_ADJ, linewidth=1.5, zorder=2)
    ax2.plot(
        x + w,
        adj_esv_pcts,
        "s",
        color=COLOR_ADJ,
        markersize=5,
        label="MHI-adj ESV Change (%)",
        zorder=3,
    )

    ax2.set_ylabel("Percentage Change (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(intervals)
    style_axes(ax2)

    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

    # Annotate absolute values
    y_min, y_max = ax2.get_ylim()
    y_range = y_max - y_min
    ax2.set_ylim(y_min - y_range * 0.5, y_max + y_range * 0.5)

    def annotate_abs(ax, x_coords, pcts, diffs, unit=""):
        for xi, pct, diff in zip(x_coords, pcts, diffs):
            va = "bottom" if pct >= 0 else "top"
            offset = 4 if pct >= 0 else -4
            text = f"{diff:+.1f}{unit}"
            ax.annotate(
                text,
                xy=(xi, pct),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=9.5,
                color="#444444",
                rotation=90,
            )

    annotate_abs(ax2, x - w, area_pcts, area_diffs, " km²")
    annotate_abs(ax2, x, esv_pcts, esv_diffs, " M")
    annotate_abs(ax2, x + w, adj_esv_pcts, adj_esv_diffs, " M")

    ax2.text(
        -0.05,
        1.05,
        "(b)",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )

    # ==========================================
    # Panel (c): Adjusted/Basic ESV ratio
    # ==========================================
    ax3 = fig.add_subplot(gs[2])
    ratio = df_mangrove["esv_ratio"].values

    ax3.plot(
        years,
        ratio,
        marker="D",
        markersize=4,
        color=COLOR_RATIO,
        linewidth=1.5,
        linestyle="-",
        zorder=3,
    )

    # Subtle baseline for ratio=1.0 if it fits within plot context
    if min(ratio) < 1.0 and max(ratio) > 1.0:
        ax3.axhline(1.0, color=COLOR_ZERO, linewidth=0.8, linestyle=":", zorder=1)

    ax3.set_ylabel("Adjusted / Basic Ratio")
    ax3.set_xlabel("Year")
    ax3.set_xticks(years)
    ax3.set_xticklabels(years)
    style_axes(ax3)

    ax3.text(
        -0.05,
        1.05,
        "(c)",
        transform=ax3.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )

    out_path = Path(output_file)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    plt.savefig(out_path)
    plt.close(fig)
    print(f"Figure successfully saved to: {out_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Mangrove ESV Time Series figure for publication."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT /"yearly_class_area_esv.csv"),
        help="Input CSV file containing ESV and area classes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT /"mangrove_esv_time_series.png"),
        help="Output figure file path.",
    )

    args = parser.parse_args()
    plot_esv_time_series(args.input, args.output)
