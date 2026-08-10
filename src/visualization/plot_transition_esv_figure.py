import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "period",
    "from_class_id",
    "from_class_name",
    "to_class_id",
    "to_class_name",
    "transit_area_ha",
    "esv_change_2020usd",
}

CLASS_ORDER = [
    "cropland",
    "forest",
    "shrub",
    "grassland",
    "water",
    "bare_land",
    "impervious_surface",
    "mangrove",
]

DISPLAY_NAMES = {
    "cropland": "Cropland",
    "forest": "Forest",
    "shrub": "Shrub",
    "grassland": "Grassland",
    "water": "Water",
    "bare_land": "Bare land",
    "impervious_surface": "Impervious",
    "mangrove": "Mangrove",
}

TARGET_PERIOD = "2000-2024"


def _style_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.labelpad": 6,
            "xtick.major.pad": 3,
            "ytick.major.pad": 3,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _ordered_periods(df: pd.DataFrame) -> list[str]:
    return df["period"].astype(str).drop_duplicates().tolist()


def _class_order_from_data(df: pd.DataFrame) -> list[str]:
    present = set(df["from_class_name"]).union(set(df["to_class_name"]))
    ordered = [name for name in CLASS_ORDER if name in present]
    extras = sorted(name for name in present if name not in CLASS_ORDER)
    return ordered + extras


def _pretty_class_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _build_period_matrix(
    df: pd.DataFrame, value_column: str, class_order: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    block = df.pivot_table(
        index="from_class_name",
        columns="to_class_name",
        values=value_column,
        aggfunc="sum",
    ).reindex(index=class_order, columns=class_order)

    data = block.to_numpy(dtype=float)
    row_labels = [_pretty_class_name(c) for c in class_order]
    col_labels = row_labels
    return data, row_labels, col_labels


def _plot_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    cmap,
    cbar_label: str,
    norm=None,
) -> None:
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(
        masked, cmap=cmap, aspect="auto", interpolation="nearest", norm=norm
    )

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center")
    ax.tick_params(axis="both", which="both", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        image, ax=ax, orientation="horizontal", pad=0.22, fraction=0.05, shrink=1.0, aspect=80
    )
    vmin, vmax = image.get_clim()
    ticks = np.linspace(vmin, vmax, 4)
    cbar.set_ticks(ticks)
    cbar.set_label(cbar_label, labelpad=6)
    cbar.outline.set_linewidth(0.8)


def _format_transition_label(period: str, from_name: str, to_name: str) -> str:
    return f"{_pretty_class_name(from_name)}\n↓\n{_pretty_class_name(to_name)}"


def _plot_waterfall(ax: plt.Axes, df: pd.DataFrame, top_n: int) -> None:
    df_sorted = df.sort_values(by="esv_change_2020usd", ascending=False).copy()
    gains = df_sorted.head(top_n).sort_values(by="esv_change_2020usd", ascending=False)
    losses = df_sorted.tail(top_n).sort_values(by="esv_change_2020usd", ascending=True)

    gains = gains.copy()
    losses = losses.copy()
    gains["plot_value"] = gains["esv_change_2020usd"] / 1e6
    losses["plot_value"] = losses["esv_change_2020usd"] / 1e6

    spacing = 1.6
    gain_x = np.arange(-len(gains), 0) * spacing
    loss_x = np.arange(1, len(losses) + 1) * spacing
    width = 0.4

    gain_bottoms = np.zeros(len(gains))
    gain_cumulative = 0.0
    gain_values = gains["plot_value"].to_numpy()
    for i in range(len(gain_values) - 1, -1, -1):
        gain_bottoms[i] = gain_cumulative
        gain_cumulative += gain_values[i]

    loss_bottoms = np.zeros(len(losses))
    loss_cumulative = 0.0
    loss_values = losses["plot_value"].to_numpy()
    for i, value in enumerate(loss_values):
        loss_bottoms[i] = loss_cumulative
        loss_cumulative += value

    if len(gains) > 0:
        ax.bar(
            gain_x,
            gain_values,
            bottom=gain_bottoms,
            color="#5b8f72",
            width=width,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    if len(losses) > 0:
        ax.bar(
            loss_x,
            loss_values,
            bottom=loss_bottoms,
            color="#b85a4d",
            width=width,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    for i in range(len(gain_x) - 1):
        y_level = gain_bottoms[i + 1]
        ax.plot(
            [gain_x[i] + width / 2, gain_x[i + 1] - width / 2],
            [y_level, y_level],
            color="#8f8f8f",
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )
    for i in range(len(loss_x) - 1):
        y_level = loss_bottoms[i + 1]
        ax.plot(
            [loss_x[i] + width / 2, loss_x[i + 1] - width / 2],
            [y_level, y_level],
            color="#8f8f8f",
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )

    ax.axhline(0, color="#666666", linewidth=0.9, zorder=2)
    ax.axvline(0, color="#c7c7c7", linewidth=0.8, linestyle=":", zorder=1)

    xticks = list(gain_x) + [0] + list(loss_x)
    xticklabels = (
        [
            _format_transition_label(
                str(row["period"]),
                str(row["from_class_name"]),
                str(row["to_class_name"]),
            )
            for _, row in gains.iterrows()
        ]
        + [" "]
        + [
            _format_transition_label(
                str(row["period"]),
                str(row["from_class_name"]),
                str(row["to_class_name"]),
            )
            for _, row in losses.iterrows()
        ]
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0, ha="center")
    ax.set_ylabel("ESV Change\n(Million 2020 USD)")
    ax.set_ylim(-8000, 3000)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_abs = max(
        float(np.nanmax(np.abs(gain_values))) if len(gain_values) else 0.0,
        float(np.nanmax(np.abs(loss_values))) if len(loss_values) else 0.0,
        1.0,
    )
    offset = 0.01 * max_abs

    for xi, value, bottom in zip(gain_x, gain_values, gain_bottoms):
        y_pos = bottom + value
        ax.text(
            xi,
            y_pos + offset,
            f"{value:+.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    for xi, value, bottom in zip(loss_x, loss_values, loss_bottoms):
        y_pos = bottom + value
        va = "bottom" if value >= 0 else "top"
        y_text = y_pos + offset if value >= 0 else y_pos - offset
        ax.text(
            xi, y_text, f"{value:+.1f}", ha="center", va=va, fontsize=7, rotation=90
        )

    if len(gains) > 0:
        ax.text(
            float(np.mean(gain_x)),
            1.02,
            "",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    if len(losses) > 0:
        ax.text(
            float(np.mean(loss_x)),
            1.02,
            "",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )


def plot_transition_figure(
    input_csv: str | Path, output_file: str | Path, top_n: int
) -> None:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing_text}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    _style_rcparams()

    periods = _ordered_periods(df)
    if TARGET_PERIOD not in periods:
        raise ValueError(f"Target period not found in input CSV: {TARGET_PERIOD}")

    period = TARGET_PERIOD
    period_df = df.loc[df["period"].astype(str) == period].copy()
    if period_df.empty:
        raise ValueError(f"No rows found for target period: {TARGET_PERIOD}")

    class_order = _class_order_from_data(df)

    out_path = Path(output_file)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    area_data, area_rows, area_cols = _build_period_matrix(
        period_df, "transit_area_ha", class_order
    )
    esv_data, esv_rows, esv_cols = _build_period_matrix(
        period_df, "esv_change_2020usd", class_order
    )

    area_data = area_data / 100.0
    esv_data = esv_data / 1e6

    area_cmap = LinearSegmentedColormap.from_list(
        "area_heat", ["#faf9f7", "#dbe6ef", "#87a8c1", "#2f5977"]
    )
    change_cmap = LinearSegmentedColormap.from_list(
        "change_diverging", ["#b55749", "#f7efe9", "#edf5ef", "#5b8f72"]
    )
    change_norm = TwoSlopeNorm(
        vmin=np.nanmin(esv_data), vcenter=0.0, vmax=np.nanmax(esv_data)+500
    )

    fig = plt.figure(figsize=(12, 10), facecolor="white")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    _plot_waterfall(ax1, period_df, top_n=top_n)
    ax1.text(
        -0.055, 1.04, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold"
    )
    #ax1.set_title(f"Top ESV Changes ({period})", fontsize=11, fontweight="bold", pad=12)

    ax2 = fig.add_subplot(gs[1])
    _plot_heatmap(
        fig,
        ax2,
        area_data,
        area_rows,
        area_cols,
        area_cmap,
        "Transition Area (km²)",
    )
    ax2.set_ylabel("From class")
    ax2.set_xlabel("To class")
    ax2.text(
        -0.055, 1.04, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold"
    )
    #ax2.set_title(f"Transition Area ({period})", fontsize=11, fontweight="bold", pad=12)

    ax3 = fig.add_subplot(gs[2])
    _plot_heatmap(
        fig,
        ax3,
        esv_data,
        esv_rows,
        esv_cols,
        change_cmap,
        "ESV Change (Million 2020 USD)",
        norm=change_norm,
    )
    ax3.set_ylabel("From class")
    ax3.set_xlabel("To class")
    ax3.text(
        -0.055, 1.04, "(c)", transform=ax3.transAxes, fontsize=12, fontweight="bold"
    )
    #ax3.set_title(f"ESV Change ({period})", fontsize=11, fontweight="bold", pad=12)

    plt.savefig(out_path)
    plt.close(fig)
    print(f"Figure successfully saved to: {out_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot transition area and transition-induced ESV change figures."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "all_transition_esv.csv"),
        help="Input CSV file containing transition ESV records.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "transition_esv_figure.png"),
        help="Output figure file path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of highest-gain and highest-loss transitions to show in panel (c).",
    )

    args = parser.parse_args()
    plot_transition_figure(args.input, args.output, args.top_n)
