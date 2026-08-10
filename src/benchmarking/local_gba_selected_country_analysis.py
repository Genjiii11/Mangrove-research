"""
Local processing for selected-country vs GBA mangrove protection analysis.

用途
----
1. 读取从 Google Drive 下载/同步到本地的国家级统计 CSV；
2. 本地计算 GBA hex-level ΔMHI 和 2020 protected coverage；
3. 构建精选国家 vs GBA 的 Panel A；
4. 生成双 panel 图并输出本地结果表。

前置条件
--------
Prepare the exported country statistics first:
`global_gba_pa_analysis_outputs/global_country_stats_2000_2020_raw.csv`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================
# Configuration
# =========================

ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT
OUTPUT_DIR = FINAL_DIR / "global_gba_pa_analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VI_2000_PATH = FINAL_DIR / "VI" / "mangrove_2000_90conf_indices.tif"
VI_2020_PATH = FINAL_DIR / "VI" / "mangrove_2020_90conf_indices.tif"
HEX_GRID_PATH = FINAL_DIR / "buffer" / "mangrove_2020_500m_hex_grid.shp"
RESERVE_PATH = FINAL_DIR / "buffer" / "landuse_nature_reserve.shp"
GBA_AREA_CSV = ROOT / "mangrove_area_gba_2000_2024.csv"

GLOBAL_COUNTRY_CSV = OUTPUT_DIR / "global_country_stats_2000_2020.csv"
GLOBAL_COUNTRY_RAW_CSV = OUTPUT_DIR / "global_country_stats_2000_2020_raw.csv"
GBA_HEX_CSV = OUTPUT_DIR / "gba_hex_delta_mhi_inside_outside.csv"
GBA_CLIPPED_RESERVE_PATH = OUTPUT_DIR / "gba_reserves_clipped.shp"
FIGURE_PATH =  "Plot/global_scatter_gba_delta_mhi_panel.png"

BASELINE_YEAR = 2000
ENDPOINT_YEAR = 2020

VI_BAND_NAMES = ["NDVI", "EVI", "MVI", "EMVI", "CMRI", "kNDVI"]
VI_BAND_INDICES = [2, 3, 4, 5, 6, 7]

SELECTED_PANEL_A_COUNTRY_CODES = {
   "NI",
   "JA",
   "ID",
   "MY",
   "AS",
   "US",
   "BR",
   "CH"
}
SELECTED_PANEL_A_COUNTRY_NAMES = {
    "Nigeria",
    "Japan",
    "Indonesia",
    "Malaysia",
    "Australia",
    "United States",
    "Brazil",
    "China"
}

COUNTRY_ABBR3_BY_NAME = {
    "Nigeria": "NGA",
    "Japan": "JPN",
    "Indonesia": "IDN",
    "Malaysia": "MYS",
    "Australia": "AUS",
    "United States": "USA",
    "Brazil": "BRA",
    "China": "CHN",
    "GBA": "GBA",
}

COUNTRY_ABBR3_BY_CODE = {
    "NI": "NGA",
    "NGA": "NGA",
    "JA": "JPN",
    "JPN": "JPN",
    "ID": "IDN",
    "IDN": "IDN",
    "MY": "MYS",
    "MYS": "MYS",
    "AS": "AUS",
    "AUS": "AUS",
    "US": "USA",
    "USA": "USA",
    "BR": "BRA",
    "BRA": "BRA",
    "CH": "CHN",
    "CN": "CHN",
    "CHN": "CHN",
    "GBA": "GBA",
}


@dataclass
class RasterVI:
    mask: np.ndarray
    bands: list[np.ndarray]
    transform: rasterio.Affine
    crs: Any
    nodata_values: tuple[float | None, ...]
    profile: dict


def postprocess_global_country_stats(df: pd.DataFrame) -> pd.DataFrame:
    """对导出的国家统计 CSV 做本地清洗与指标计算。"""
    if "country_na" not in df.columns:
        for alt in ["COUNTRY_NA", "name", "NAME", "country_name"]:
            if alt in df.columns:
                df["country_na"] = df[alt]
                break
    if "country_co" not in df.columns:
        for alt in ["COUNTRY_CO", "iso_a2", "ISO_A2", "iso2"]:
            if alt in df.columns:
                df["country_co"] = df[alt]
                break

    for col in ["baseline_m2", "endpoint_m2", "protected_endpoint_m2"]:
        if col not in df.columns:
            df[col] = np.nan

    baseline_series = pd.Series(
        pd.to_numeric(df["baseline_m2"], errors="coerce"), dtype="float64"
    )
    endpoint_series = pd.Series(
        pd.to_numeric(df["endpoint_m2"], errors="coerce"), dtype="float64"
    )
    protected_series = pd.Series(
        pd.to_numeric(df["protected_endpoint_m2"], errors="coerce"), dtype="float64"
    )

    df["baseline_km2"] = baseline_series / 1e6
    df["endpoint_km2"] = endpoint_series / 1e6
    df["protected_endpoint_km2"] = protected_series / 1e6

    df["extent_change_pct"] = np.where(
        df["baseline_km2"] > 0,
        (df["endpoint_km2"] - df["baseline_km2"]) / df["baseline_km2"] * 100,
        np.nan,
    )
    df["protected_coverage_pct"] = np.where(
        df["endpoint_km2"] > 0,
        df["protected_endpoint_km2"] / df["endpoint_km2"] * 100,
        np.nan,
    )

    keep_cols = [
        "country_na",
        "country_co",
        "baseline_km2",
        "endpoint_km2",
        "protected_endpoint_km2",
        "extent_change_pct",
        "protected_coverage_pct",
    ]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df.loc[:, available_cols].copy()
    valid_mask = pd.DataFrame(df[["baseline_km2", "endpoint_km2"]]).notna().any(axis=1)
    df = df.loc[valid_mask].copy()
    df = df.sort_values("endpoint_km2", ascending=False)
    return df


def load_exported_global_country_stats() -> pd.DataFrame:
    """读取原始国家统计 CSV，并做本地后处理。"""
    if not GLOBAL_COUNTRY_RAW_CSV.exists():
        raise FileNotFoundError(
            f"原始国家统计 CSV 不存在，请确认文件位于: {GLOBAL_COUNTRY_RAW_CSV}"
        )

    raw_df = pd.read_csv(GLOBAL_COUNTRY_RAW_CSV)
    return postprocess_global_country_stats(raw_df)


def normalize_crs_for_geopandas(crs: Any) -> Any:
    """尽量把 rasterio / pyproj CRS 转成 geopandas 可接受形式。"""
    if crs is None:
        raise ValueError("CRS is missing.")
    if hasattr(crs, "to_string"):
        return crs.to_string()
    return crs


def read_vi_raster(path: Path) -> RasterVI:
    """读取 VI 栅格，band1 为 mangrove mask，band2-7 为植被指数。"""
    with rasterio.open(path) as src:
        mask = src.read(1) == 1
        bands: list[np.ndarray] = []

        for band_idx in VI_BAND_INDICES:
            arr = src.read(band_idx).astype(np.float32)
            nodata = None
            if src.nodatavals and len(src.nodatavals) >= band_idx:
                nodata = src.nodatavals[band_idx - 1]
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            bands.append(arr)

        return RasterVI(
            mask=mask,
            bands=bands,
            transform=src.transform,
            crs=src.crs,
            nodata_values=src.nodatavals,
            profile=src.profile.copy(),
        )


def ensure_aligned_rasters(r1: RasterVI, r2: RasterVI) -> None:
    """确认两个 VI 栅格已对齐。"""
    same_shape = r1.mask.shape == r2.mask.shape
    same_transform = r1.transform == r2.transform
    same_crs = str(r1.crs) == str(r2.crs)

    if not (same_shape and same_transform and same_crs):
        raise ValueError(
            "2000 和 2020 VI 栅格未对齐。请先统一 shape / transform / CRS 后再运行脚本。"
        )


def extract_valid_pixel_matrix(
    bands: list[np.ndarray], mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """提取 mask 内有效像元，返回 (N, 6) 矩阵与 valid_rows 标记。"""
    pixel_matrix = np.column_stack([band[mask] for band in bands]).astype(np.float64)
    valid_rows = np.all(np.isfinite(pixel_matrix), axis=1)
    return pixel_matrix, valid_rows


def fit_shared_mhi_model(
    r2000: RasterVI, r2020: RasterVI
) -> tuple[StandardScaler, PCA]:
    """用 2000 + 2020 合并样本拟合统一的 scaler + PCA。"""
    m2000, v2000 = extract_valid_pixel_matrix(r2000.bands, r2000.mask)
    m2020, v2020 = extract_valid_pixel_matrix(r2020.bands, r2020.mask)

    combined = np.vstack([m2000[v2000], m2020[v2020]])
    if combined.shape[0] < 2:
        raise ValueError("有效红树林像元不足，无法拟合共享 MHI 模型。")

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    pca = PCA(n_components=1)
    pca.fit(combined_scaled)

    loadings = pca.components_[0]
    if np.nanmean(loadings) < 0:
        pca.components_[0] = -pca.components_[0]

    return scaler, pca


def transform_raster_to_mhi(
    raster_vi: RasterVI, scaler: StandardScaler, pca: PCA
) -> np.ndarray:
    """将某一年 VI 栅格投影到共享 PCA 轴上，得到 MHI。"""
    out = np.full(raster_vi.mask.shape, np.nan, dtype=np.float32)

    matrix, valid_rows = extract_valid_pixel_matrix(raster_vi.bands, raster_vi.mask)
    if valid_rows.sum() == 0:
        return out

    transformed = pca.transform(scaler.transform(matrix[valid_rows])).flatten()

    mangrove_indices = np.where(raster_vi.mask.ravel())[0]
    valid_indices = mangrove_indices[valid_rows]
    out_flat = out.ravel()
    out_flat[valid_indices] = transformed.astype(np.float32)
    return out_flat.reshape(raster_vi.mask.shape)


def safe_union(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """兼容 geopandas / shapely 不同版本的 union 调用。"""
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.unary_union


def clip_reserves_to_hex_study_area(
    reserve_path: Path,
    hex_path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """将保护区裁剪到 GBA 研究区（hex grid union）。"""
    hex_gdf = gpd.read_file(hex_path)
    reserve_gdf = gpd.read_file(reserve_path)

    if hex_gdf.crs is None:
        raise ValueError("Hex grid CRS is missing.")
    if reserve_gdf.crs is None:
        raise ValueError("Reserve shapefile CRS is missing.")

    if reserve_gdf.crs != hex_gdf.crs:
        reserve_gdf = reserve_gdf.to_crs(hex_gdf.crs)

    study_area = gpd.GeoDataFrame(geometry=[safe_union(hex_gdf)], crs=hex_gdf.crs)
    clipped = gpd.clip(reserve_gdf, study_area)
    clipped = clipped.loc[~clipped.geometry.is_empty].copy()

    return hex_gdf, clipped


def classify_hex_inside_pa(
    hex_gdf: gpd.GeoDataFrame, clipped_reserves: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """给 hex 标记 inside_pa。"""
    if clipped_reserves.empty:
        hex_out = hex_gdf.copy()
        hex_out["inside_pa"] = False
        return hex_out

    try:
        target_crs = hex_gdf.estimate_utm_crs()
    except Exception:
        target_crs = "EPSG:32649"

    hex_proj = hex_gdf.to_crs(target_crs).copy()
    reserve_proj = clipped_reserves.to_crs(target_crs)
    reserve_union = safe_union(reserve_proj)

    centroids = hex_proj.geometry.centroid
    hex_proj["inside_pa"] = centroids.within(reserve_union)

    if hex_gdf.crs is None:
        raise ValueError("Hex grid CRS is missing.")
    hex_out = hex_proj.to_crs(hex_gdf.crs)
    return hex_out


def zonal_mean_from_array(
    array: np.ndarray,
    transform: rasterio.Affine,
    geoms: Iterable[BaseGeometry],
) -> list[float]:
    """对每个 polygon 计算数组均值。"""
    means: list[float] = []
    for geom in geoms:
        mask = geometry_mask(
            [geom],
            out_shape=array.shape,
            transform=transform,
            invert=True,
        )
        values = array[mask]
        if values.size == 0:
            means.append(np.nan)
        else:
            finite_vals = values[np.isfinite(values)]
            means.append(
                float(np.nanmean(finite_vals)) if finite_vals.size > 0 else np.nan
            )
    return means


def compute_gba_protected_coverage_pct(
    mangrove_mask: np.ndarray,
    transform: rasterio.Affine,
    reserves_in_raster_crs: gpd.GeoDataFrame,
) -> float:
    """计算 2020 年 GBA 红树林中，位于保护区内的比例（%）。"""
    total_mangrove_pixels = int(mangrove_mask.sum())
    if total_mangrove_pixels == 0 or reserves_in_raster_crs.empty:
        return np.nan

    reserve_mask = geometry_mask(
        list(reserves_in_raster_crs.geometry),
        out_shape=mangrove_mask.shape,
        transform=transform,
        invert=True,
    )

    inside_pixels = int(np.logical_and(mangrove_mask, reserve_mask).sum())
    return inside_pixels / total_mangrove_pixels * 100


def compute_local_gba_stats() -> tuple[pd.DataFrame, float, gpd.GeoDataFrame]:
    """计算本地 GBA hex 级别 ΔMHI，以及 GBA 的 2020 protected coverage (%)."""
    r2000 = read_vi_raster(VI_2000_PATH)
    r2020 = read_vi_raster(VI_2020_PATH)
    ensure_aligned_rasters(r2000, r2020)
    raster_crs = normalize_crs_for_geopandas(r2020.crs)

    scaler, pca = fit_shared_mhi_model(r2000, r2020)
    mhi_2000 = transform_raster_to_mhi(r2000, scaler, pca)
    mhi_2020 = transform_raster_to_mhi(r2020, scaler, pca)
    delta_mhi = mhi_2020 - mhi_2000

    hex_gdf, clipped_reserves = clip_reserves_to_hex_study_area(
        RESERVE_PATH, HEX_GRID_PATH
    )
    hex_gdf = classify_hex_inside_pa(hex_gdf, clipped_reserves)

    hex_in_raster_crs = hex_gdf.to_crs(raster_crs)
    delta_mhi_mean = zonal_mean_from_array(
        delta_mhi, r2020.transform, hex_in_raster_crs.geometry
    )
    mhi_2000_mean = zonal_mean_from_array(
        mhi_2000, r2020.transform, hex_in_raster_crs.geometry
    )
    mhi_2020_mean = zonal_mean_from_array(
        mhi_2020, r2020.transform, hex_in_raster_crs.geometry
    )

    result = hex_gdf.copy()
    if "GridID" not in result.columns:
        result = result.reset_index(drop=True)
        result.insert(0, "GridID", result.index + 1)

    result["mhi_2000_mean"] = mhi_2000_mean
    result["mhi_2020_mean"] = mhi_2020_mean
    result["delta_mhi_mean"] = delta_mhi_mean
    result["pa_group"] = np.where(result["inside_pa"], "Inside PA", "Outside PA")

    reserves_in_raster_crs = clipped_reserves.to_crs(raster_crs)
    gba_protected_coverage_pct = compute_gba_protected_coverage_pct(
        r2020.mask,
        r2020.transform,
        reserves_in_raster_crs,
    )

    return result, gba_protected_coverage_pct, clipped_reserves


def load_gba_extent_change_from_csv() -> dict[str, float]:
    """从现成 CSV 读取 GBA 2000 和 2020 面积，并计算变化率。"""
    df = pd.read_csv(GBA_AREA_CSV)
    df.columns = [c.strip() for c in df.columns]

    area_col = [c for c in df.columns if "Area" in c][0]
    baseline = float(df.loc[df["Year"] == BASELINE_YEAR, area_col].iloc[0])
    endpoint = float(df.loc[df["Year"] == ENDPOINT_YEAR, area_col].iloc[0])
    change_pct = (endpoint - baseline) / baseline * 100 if baseline > 0 else np.nan

    return {
        "baseline_km2": baseline,
        "endpoint_km2": endpoint,
        "extent_change_pct": change_pct,
    }


def select_panel_a_countries(panel_a_df: pd.DataFrame) -> pd.DataFrame:
    """筛选 Panel A 中用于与 GBA 对比的代表性国家。"""
    df = panel_a_df.copy()

    country_codes = df["country_co"].fillna("").astype(str).str.upper().str.strip()
    country_names = df["country_na"].fillna("").astype(str).str.strip()

    selected_mask = country_codes.isin(
        SELECTED_PANEL_A_COUNTRY_CODES
    ) | country_names.isin(SELECTED_PANEL_A_COUNTRY_NAMES)
    gba_mask = country_names.eq("GBA")

    selected_df = df.loc[selected_mask | gba_mask].copy()
    selected_df = selected_df.sort_values("endpoint_km2", ascending=False)
    return selected_df


def to_country_abbr3(country_name: Any, country_code: Any) -> str:
    """将国家名称/代码标准化为三字母缩写。"""
    name = str(country_name).strip()
    code = str(country_code).strip().upper()

    if name in COUNTRY_ABBR3_BY_NAME:
        return COUNTRY_ABBR3_BY_NAME[name]
    if code in COUNTRY_ABBR3_BY_CODE:
        return COUNTRY_ABBR3_BY_CODE[code]

    letters = "".join(ch for ch in name.upper() if "A" <= ch <= "Z")
    if len(letters) >= 3:
        return letters[:3]
    if len(code) >= 3:
        return code[:3]
    if len(code) > 0:
        return (code + "XXX")[:3]
    return "UNK"


def plot_double_panel(
    global_df: pd.DataFrame, gba_hex_df: pd.DataFrame, output_path: Path
) -> None:
    """绘制双 panel 图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # ---------- Panel A ----------
    ax = axes[0]
    panel_df = global_df.copy()
    panel_df["country_name_norm"] = (
        panel_df["country_na"].fillna("").astype(str).str.strip()
    )
    panel_df["country_code_norm"] = (
        panel_df["country_co"].fillna("").astype(str).str.upper().str.strip()
    )
    panel_df["label_abbr3"] = panel_df.apply(
        lambda row: to_country_abbr3(row["country_na"], row["country_co"]), axis=1
    )

    is_gba = panel_df["country_name_norm"].eq("GBA") | panel_df[
        "country_code_norm"
    ].eq("GBA")
    is_china = panel_df["country_name_norm"].eq("China") | panel_df[
        "country_code_norm"
    ].isin(["CH", "CN", "CHN"])

    base_points = panel_df.loc[~is_gba & ~is_china]
    china_points = panel_df.loc[is_china & ~is_gba]
    gba_points = panel_df.loc[is_gba]

    ax.scatter(
        base_points["protected_coverage_pct"],
        base_points["extent_change_pct"],
        s=60,
        color="#a1b4c7",
        alpha=0.85,
        edgecolors="#4b5e73",
        linewidths=0.6,
        label="Selected countries",
        zorder=2,
    )

    if not china_points.empty:
        ax.scatter(
            china_points["protected_coverage_pct"],
            china_points["extent_change_pct"],
            s=120,
            marker="D",
            color="#1f6fb2",
            edgecolors="#0c3f69",
            linewidths=0.8,
            label="China",
            zorder=4,
        )

    if not gba_points.empty:
        ax.scatter(
            gba_points["protected_coverage_pct"],
            gba_points["extent_change_pct"],
            s=180,
            marker="*",
            color="#d6604d",
            edgecolors="#8c2d1b",
            linewidths=0.8,
            label="GBA",
            zorder=5,
        )

    for _, row in panel_df.iterrows():
        label_color = "#555555"
        label_weight = "normal"

        if row["country_name_norm"] == "China" or row["country_code_norm"] in {
            "CH",
            "CN",
            "CHN",
        }:
            label_color = "#0c3f69"
            label_weight = "bold"
        if row["country_name_norm"] == "GBA" or row["country_code_norm"] == "GBA":
            label_color = "#8c2d1b"
            label_weight = "bold"

        ax.annotate(
            row["label_abbr3"],
            (row["protected_coverage_pct"], row["extent_change_pct"]),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8,
            color=label_color,
            fontweight=label_weight,
            ha="center",
            va="bottom",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Protected mangrove coverage (%)", fontsize=10)
    ax.set_ylabel("Mangrove extent change (%)", fontsize=10)
    #ax.set_title("Selected-country benchmark", fontsize=11, pad=10)
    ax.text(
        -0.1,
        1.05,
        "(a)",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    # ---------- Panel B ----------
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
    # Publication-style muted colors for boxplots
    colors = ["#8caba1", "#d9c5b2"]
    for patch, color in zip(box["boxes"], colors):
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
            alpha=0.3,
            color="#333333",
            edgecolors="none",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("ΔMHI", fontsize=10)
    #ax.set_title("GBA hex-level ΔMHI by protection status", fontsize=11, pad=10)
    ax.text(
        -0.1,
        1.05,
        "(b)",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_gba_panel_a_row(gba_protected_coverage_pct: float) -> pd.DataFrame:
    """构建 Panel A 中的 GBA 单独点。"""
    gba_extent = load_gba_extent_change_from_csv()
    return pd.DataFrame(
        [
            {
                "country_na": "GBA",
                "country_co": "GBA",
                "baseline_km2": gba_extent["baseline_km2"],
                "endpoint_km2": gba_extent["endpoint_km2"],
                "protected_endpoint_km2": np.nan,
                "extent_change_pct": gba_extent["extent_change_pct"],
                "protected_coverage_pct": gba_protected_coverage_pct,
            }
        ]
    )


def main() -> None:
    print("[1/4] Loading exported global country statistics...")
    global_df = load_exported_global_country_stats()
    global_df.to_csv(GLOBAL_COUNTRY_CSV, index=False)
    print(f"Saved processed global country table: {GLOBAL_COUNTRY_CSV}")

    print("[2/4] Computing local GBA hex-level ΔMHI and PA grouping...")
    gba_hex_gdf, gba_protected_coverage_pct, clipped_reserves = (
        compute_local_gba_stats()
    )
    gba_hex_gpkg = GBA_HEX_CSV.with_suffix(".gpkg")
    gba_hex_gpkg_gdf = gba_hex_gdf.rename(
        columns={
            col: "source_fid" for col in gba_hex_gdf.columns if col.lower() == "fid"
        }
    )
    if gba_hex_gpkg.exists():
        gba_hex_gpkg.unlink()
    gba_hex_gpkg_gdf.reset_index(drop=True).to_file(
        gba_hex_gpkg,
        driver="GPKG",
        index=False,
    )
    clipped_reserves.to_file(GBA_CLIPPED_RESERVE_PATH)

    gba_hex_df = pd.DataFrame(gba_hex_gdf.drop(columns="geometry"))
    gba_hex_df.to_csv(GBA_HEX_CSV, index=False)
    print(f"Saved GBA hex table: {GBA_HEX_CSV}")
    print(f"Saved clipped reserves: {GBA_CLIPPED_RESERVE_PATH}")

    print("[3/4] Appending GBA and selecting comparison countries for Panel A...")
    gba_row = build_gba_panel_a_row(gba_protected_coverage_pct)
    panel_a_df = pd.concat([global_df, gba_row], ignore_index=True)
    panel_a_df = select_panel_a_countries(panel_a_df)

    print("[4/4] Plotting double-panel figure...")
    plot_double_panel(panel_a_df, gba_hex_df, FIGURE_PATH)
    print(f"Saved figure: {FIGURE_PATH}")

    print("\nDone.")
    print(f"GBA 2020 protected mangrove coverage (%): {gba_protected_coverage_pct:.2f}")


if __name__ == "__main__":
    main()
