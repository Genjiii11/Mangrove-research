import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {
    "year",
    "group",
    "item",
    "annual_esv_2020usd",
    "adjusted_annual_esv_2020usd",
}

GROUP_ORDER = ["Provisioning", "Regulating", "Cultural", "Supporting"]

# Editorial Palette for Groups (Nature-like style)
GROUP_COLORS = {
    "Provisioning": "#5e81ac",  # Nord blue
    "Regulating": "#a3be8c",  # Nord green
    "Cultural": "#d08770",  # Nord orange/terracotta
    "Supporting": "#ebcb8b",  # Nord yellow/gold
}


def plot_service_composition(
    input_csv: str | Path, output_file: str | Path, value_type: str
) -> None:
    """
    Reads the yearly ESV output CSV, extracts mangrove time series,
    and generates a 2-panel publication-quality figure.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing_text}")

    # Determine value column based on CLI
    val_col = (
        "adjusted_annual_esv_2020usd"
        if value_type == "adjusted"
        else "annual_esv_2020usd"
    )

    # Convert to millions for publication
    df["esv_million"] = df[val_col] / 1e6

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

    fig = plt.figure(figsize=(7.5, 9))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5], hspace=0.3)

    # General styling function
    def style_axes(ax, hide_top_right=True):
        if hide_top_right:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)

    # ==========================================
    # Panel (a): Stacked bar chart of Groups
    # ==========================================
    ax1 = fig.add_subplot(gs[0])

    # Aggregate by year and group
    df_group = pd.DataFrame(
        df.groupby(["year", "group"], as_index=False).agg(
            esv_million=("esv_million", "sum")
        )
    )
    years = sorted(df["year"].unique())

    bottoms = np.zeros(len(years))
    bar_width = 1.0

    for group in GROUP_ORDER:
        group_data = df_group[df_group["group"] == group]
        # Align with years securely
        group_vals = [
            group_data.loc[group_data["year"] == y, "esv_million"].sum() for y in years
        ]

        ax1.bar(
            years,
            group_vals,
            bottom=bottoms,
            color=GROUP_COLORS.get(group, "#888888"),
            edgecolor="none",
            label=group,
            width=bar_width,
            zorder=2,
        )
        bottoms += np.array(group_vals)

    ax1.set_ylabel("Ecosystem Service Value\n(Million 2020 USD)")
    ax1.set_xticks(years)
    ax1.set_xticklabels(years)
    style_axes(ax1, hide_top_right=True)

    # Add legend outside
    ax1.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        labelspacing=1.5
    )

    # Panel A Label
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
    # Panel (b): Heatmap of Detailed Items
    # ==========================================
    ax2 = fig.add_subplot(gs[1])

    # Aggregate by item across all years to find order
    item_means = pd.DataFrame(
        df.groupby(["group", "item"], as_index=False).agg(
            esv_million=("esv_million", "mean")
        )
    )

    # Sort: first by GROUP_ORDER, then by mean value descending
    item_means["group_cat"] = pd.Categorical(
        item_means["group"], categories=GROUP_ORDER, ordered=True
    )
    item_means = item_means.sort_values(
        by=["group_cat", "esv_million"], ascending=[True, False]
    )

    ordered_items = item_means["item"].tolist()
    ordered_groups = item_means["group"].tolist()

    # Create pivot table
    df_pivot = df.pivot_table(
        index="item", columns="year", values="esv_million", aggfunc="sum", fill_value=0
    )

    # Reindex pivot table to match ordered items
    df_pivot = df_pivot.reindex(ordered_items)
    years_heatmap = df_pivot.columns.tolist()
    heatmap_data = df_pivot.values

    # Custom colormap (white to deep editorial blue)
    cmap = LinearSegmentedColormap.from_list(
        "custom_heat", ["#f4f6f8", "#d1deec", "#799bc0", "#3a5b82", "#1d2c40"]
    )

    im = ax2.imshow(heatmap_data, cmap=cmap, aspect="auto", interpolation="nearest")

    # Add horizontal colorbar below the heatmap
    cbar = fig.colorbar(im, ax=ax2, orientation="horizontal", pad=0.14, fraction=0.1,aspect=100)
    cbar.set_label("ESV (Million 2020 USD)", labelpad=6)
    cbar.outline.set_linewidth(0.8)

    # Formatting
    ax2.set_yticks(np.arange(len(ordered_items)))
    ax2.set_yticklabels(ordered_items, fontsize=8)
    ax2.set_xticks(np.arange(len(years_heatmap)))
    ax2.set_xticklabels(years_heatmap)

    # Add horizontal lines to separate groups visually
    current_group = ordered_groups[0]
    for i, group in enumerate(ordered_groups):
        if group != current_group:
            ax2.axhline(
                i - 0.5, color="#333333", linewidth=1.2, linestyle="-", zorder=3
            )
            current_group = group

    ax2.tick_params(axis="both", which="both", length=3, color="#bbbbbb")

    # Group Labels as secondary Y axis
    ax2_left = ax2.twinx()
    ax2_left.set_ylim(ax2.get_ylim())

    group_centers = []
    group_labels = []

    # Calculate center of each group for label placement
    unique_groups = []
    for g in ordered_groups:
        if g not in unique_groups:
            unique_groups.append(g)

    for g in unique_groups:
        indices = [i for i, x in enumerate(ordered_groups) if x == g]
        center = sum(indices) / len(indices)
        group_centers.append(center)
        group_labels.append(g)

    ax2_left.set_yticks(group_centers)
    ax2_left.set_yticklabels(
        group_labels, rotation=90, va="center", fontsize=9, fontweight="bold"
    )
    ax2_left.tick_params(axis="y", which="both", length=0)
    ax2_left.spines["top"].set_visible(False)
    ax2_left.spines["right"].set_visible(False)
    ax2_left.spines["bottom"].set_visible(False)
    ax2_left.spines["left"].set_visible(False)

    # Position twin axis on the left, shift it outwards
    ax2_left.yaxis.set_label_position("left")
    ax2_left.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2_left.spines["left"].set_position(("outward", 10))

    # Panel B Label
    ax2.text(
        -0.05,
        1.05,
        "(b)",
        transform=ax2.transAxes,
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
        description="Plot Mangrove Ecosystem Service Composition."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "yearly_mangrove_esv_details.csv"),
        help="Input CSV file containing ESV details.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "mangrove_service_composition.png"),
        help="Output figure file path.",
    )
    parser.add_argument(
        "--value-type",
        type=str,
        choices=["adjusted", "basic"],
        default="adjusted",
        help="Which ESV value to plot (adjusted or basic). Default is adjusted.",
    )

    args = parser.parse_args()
    plot_service_composition(args.input, args.output, args.value_type)
