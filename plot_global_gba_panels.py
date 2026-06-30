"""
Plot double-panel figure from precomputed global and GBA outputs.

依赖输入：
- global_gba_pa_analysis_outputs/global_country_stats_2000_2020.csv
- global_gba_pa_analysis_outputs/gba_hex_delta_mhi_inside_outside.csv
- global_gba_pa_analysis_outputs/gba_summary_metrics_2000_2020.csv

输出：
- global_gba_pa_analysis_outputs/global_scatter_gba_delta_mhi_panel.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Desktop\Mangrove")
FINAL_DIR = ROOT / "Final"
OUTPUT_DIR = FINAL_DIR / "global_gba_pa_analysis_outputs"

GLOBAL_COUNTRY_CSV = OUTPUT_DIR / "global_country_stats_2000_2020.csv"
GBA_HEX_CSV = OUTPUT_DIR / "gba_hex_delta_mhi_inside_outside.csv"
GBA_SUMMARY_CSV = OUTPUT_DIR / "gba_summary_metrics_2000_2020.csv"
FIGURE_PATH = OUTPUT_DIR / "global_scatter_gba_delta_mhi_panel.png"


def build_panel_a_dataframe(
    global_df: pd.DataFrame, gba_summary_df: pd.DataFrame
) -> pd.DataFrame:
    gba_row = pd.DataFrame(
        [
            {
                "country_na": "GBA",
                "country_co": "GBA",
                "baseline_km2": float(gba_summary_df.loc[0, "baseline_km2"]),
                "endpoint_km2": float(gba_summary_df.loc[0, "endpoint_km2"]),
                "protected_endpoint_km2": np.nan,
                "extent_change_pct": float(gba_summary_df.loc[0, "extent_change_pct"]),
                "protected_coverage_pct": float(
                    gba_summary_df.loc[0, "protected_coverage_pct"]
                ),
            }
        ]
    )
    return pd.concat([global_df, gba_row], ignore_index=True)


def plot_double_panel(
    global_df: pd.DataFrame, gba_hex_df: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    ax = axes[0]
    country_mask = global_df["country_na"].fillna("") != "GBA"
    gba_mask = global_df["country_na"].fillna("") == "GBA"
    china_mask = (
        global_df["country_na"].fillna("").str.contains("China", case=False, na=False)
    )

    others = global_df[country_mask & ~china_mask]
    china = global_df[country_mask & china_mask]
    gba = global_df[gba_mask]

    ax.scatter(
        others["protected_coverage_pct"],
        others["extent_change_pct"],
        s=28,
        color="#bdbdbd",
        alpha=0.7,
        edgecolors="none",
        label="Other countries/regions",
    )

    if not china.empty:
        ax.scatter(
            china["protected_coverage_pct"],
            china["extent_change_pct"],
            s=80,
            color="#d73027",
            edgecolors="black",
            linewidths=0.5,
            label="China",
            zorder=4,
        )
        for _, row in china.iterrows():
            ax.annotate(
                row["country_na"],
                (row["protected_coverage_pct"], row["extent_change_pct"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="#8c1d18",
            )

    if not gba.empty:
        ax.scatter(
            gba["protected_coverage_pct"],
            gba["extent_change_pct"],
            s=150,
            marker="*",
            color="#2b83ba",
            edgecolors="black",
            linewidths=0.6,
            label="GBA",
            zorder=5,
        )
        for _, row in gba.iterrows():
            ax.annotate(
                "GBA",
                (row["protected_coverage_pct"], row["extent_change_pct"]),
                xytext=(7, -10),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#1f4e79",
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Protected mangrove coverage in 2020 (%)")
    ax.set_ylabel("Mangrove extent change, 2000-2020 (%)")
    ax.set_title("Panel A. Global benchmark with China and GBA highlighted")
    ax.legend(frameon=False, fontsize=9, loc="best")

    ax = axes[1]
    inside = gba_hex_df.loc[gba_hex_df["inside_pa"], "delta_mhi_mean"].dropna().values
    outside = gba_hex_df.loc[~gba_hex_df["inside_pa"], "delta_mhi_mean"].dropna().values

    box = ax.boxplot(
        [inside, outside],
        labels=[f"Inside PA\n(n={len(inside)})", f"Outside PA\n(n={len(outside)})"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], ["#91bfdb", "#fdae61"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(42)
    for idx, values in enumerate([inside, outside], start=1):
        if len(values) == 0:
            continue
        jitter = rng.normal(loc=0, scale=0.04, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=10,
            alpha=0.35,
            color="black",
            edgecolors="none",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel("ΔMHI (2020 - 2000)")
    ax.set_title("Panel B. GBA hex-level ΔMHI inside vs outside PA")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global_df = pd.read_csv(GLOBAL_COUNTRY_CSV)
    gba_hex_df = pd.read_csv(GBA_HEX_CSV)
    gba_summary_df = pd.read_csv(GBA_SUMMARY_CSV)

    panel_a_df = build_panel_a_dataframe(global_df, gba_summary_df)
    plot_double_panel(panel_a_df, gba_hex_df, FIGURE_PATH)
    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
