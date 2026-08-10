"""
GBA local analysis: hex-level ΔMHI and protection grouping.

输出：
- global_gba_pa_analysis_outputs/gba_hex_delta_mhi_inside_outside.csv
- global_gba_pa_analysis_outputs/gba_hex_delta_mhi_inside_outside.gpkg
- global_gba_pa_analysis_outputs/gba_reserves_clipped.shp
- global_gba_pa_analysis_outputs/gba_summary_metrics_2000_2020.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT
OUTPUT_DIR = FINAL_DIR / "global_gba_pa_analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VI_2000_PATH = FINAL_DIR / "VI" / "mangrove_2000_90conf_indices.tif"
VI_2020_PATH = FINAL_DIR / "VI" / "mangrove_2020_90conf_indices.tif"
HEX_GRID_PATH = FINAL_DIR / "buffer" / "mangrove_2020_500m_hex_grid.shp"
RESERVE_PATH = FINAL_DIR / "buffer" / "landuse_nature_reserve.shp"
GBA_AREA_CSV = ROOT / "mangrove_area_gba_2000_2024.csv"

GBA_HEX_CSV = OUTPUT_DIR / "gba_hex_delta_mhi_inside_outside.csv"
GBA_HEX_GPKG = OUTPUT_DIR / "gba_hex_delta_mhi_inside_outside.gpkg"
GBA_CLIPPED_RESERVE_PATH = OUTPUT_DIR / "gba_reserves_clipped.shp"
GBA_SUMMARY_CSV = OUTPUT_DIR / "gba_summary_metrics_2000_2020.csv"

BASELINE_YEAR = 2000
ENDPOINT_YEAR = 2020
VI_BAND_INDICES = [2, 3, 4, 5, 6, 7]


@dataclass
class RasterVI:
    mask: np.ndarray
    bands: list[np.ndarray]
    transform: rasterio.Affine
    crs: Any
    nodata_values: tuple[float | None, ...]
    profile: dict


def normalize_crs_for_geopandas(crs: Any) -> Any:
    if crs is None:
        raise ValueError("CRS is missing.")
    if hasattr(crs, "to_string"):
        return crs.to_string()
    return crs


def read_vi_raster(path: Path) -> RasterVI:
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
            mask, bands, src.transform, src.crs, src.nodatavals, src.profile.copy()
        )


def ensure_aligned_rasters(r1: RasterVI, r2: RasterVI) -> None:
    if not (
        r1.mask.shape == r2.mask.shape
        and r1.transform == r2.transform
        and str(r1.crs) == str(r2.crs)
    ):
        raise ValueError("2000 and 2020 VI rasters are not aligned.")


def extract_valid_pixel_matrix(
    bands: list[np.ndarray], mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pixel_matrix = np.column_stack([band[mask] for band in bands]).astype(np.float64)
    valid_rows = np.all(np.isfinite(pixel_matrix), axis=1)
    return pixel_matrix, valid_rows


def fit_shared_mhi_model(
    r2000: RasterVI, r2020: RasterVI
) -> tuple[StandardScaler, PCA]:
    m2000, v2000 = extract_valid_pixel_matrix(r2000.bands, r2000.mask)
    m2020, v2020 = extract_valid_pixel_matrix(r2020.bands, r2020.mask)
    combined = np.vstack([m2000[v2000], m2020[v2020]])
    if combined.shape[0] < 2:
        raise ValueError("Not enough valid mangrove pixels to fit shared MHI.")

    scaler = StandardScaler()
    pca = PCA(n_components=1)
    pca.fit(scaler.fit_transform(combined))
    if np.nanmean(pca.components_[0]) < 0:
        pca.components_[0] = -pca.components_[0]
    return scaler, pca


def transform_raster_to_mhi(
    raster_vi: RasterVI, scaler: StandardScaler, pca: PCA
) -> np.ndarray:
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
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.unary_union


def clip_reserves_to_hex_study_area(
    reserve_path: Path, hex_path: Path
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
    if clipped_reserves.empty:
        out = hex_gdf.copy()
        out["inside_pa"] = False
        return out
    try:
        target_crs = hex_gdf.estimate_utm_crs()
    except Exception:
        target_crs = "EPSG:32649"
    hex_proj = hex_gdf.to_crs(target_crs).copy()
    reserve_proj = clipped_reserves.to_crs(target_crs)
    reserve_union = safe_union(reserve_proj)
    hex_proj["inside_pa"] = hex_proj.geometry.centroid.within(reserve_union)
    if hex_gdf.crs is None:
        raise ValueError("Hex grid CRS is missing.")
    return hex_proj.to_crs(hex_gdf.crs)


def zonal_mean_from_array(
    array: np.ndarray, transform: rasterio.Affine, geoms: Iterable[BaseGeometry]
) -> list[float]:
    means: list[float] = []
    for geom in geoms:
        mask = geometry_mask(
            [geom], out_shape=array.shape, transform=transform, invert=True
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


def load_gba_extent_change_from_csv() -> dict[str, float]:
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


def compute_local_gba_stats() -> tuple[gpd.GeoDataFrame, float, gpd.GeoDataFrame]:
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

    result = cast(gpd.GeoDataFrame, hex_gdf.copy())
    if "GridID" not in result.columns:
        result = cast(gpd.GeoDataFrame, result.reset_index(drop=True))
        result.insert(0, "GridID", result.index + 1)

    result["mhi_2000_mean"] = zonal_mean_from_array(
        mhi_2000, r2020.transform, hex_in_raster_crs.geometry
    )
    result["mhi_2020_mean"] = zonal_mean_from_array(
        mhi_2020, r2020.transform, hex_in_raster_crs.geometry
    )
    result["delta_mhi_mean"] = zonal_mean_from_array(
        delta_mhi, r2020.transform, hex_in_raster_crs.geometry
    )
    result["pa_group"] = np.where(result["inside_pa"], "Inside PA", "Outside PA")

    reserves_in_raster_crs = clipped_reserves.to_crs(raster_crs)
    gba_protected_coverage_pct = compute_gba_protected_coverage_pct(
        r2020.mask, r2020.transform, reserves_in_raster_crs
    )
    return result, gba_protected_coverage_pct, clipped_reserves


def build_gba_summary_row(
    gba_protected_coverage_pct: float, gba_hex_df: pd.DataFrame
) -> pd.DataFrame:
    extent = load_gba_extent_change_from_csv()
    inside_mean = float(
        gba_hex_df.loc[gba_hex_df["inside_pa"], "delta_mhi_mean"].dropna().mean()
    )
    outside_mean = float(
        gba_hex_df.loc[~gba_hex_df["inside_pa"], "delta_mhi_mean"].dropna().mean()
    )
    return pd.DataFrame(
        [
            {
                "unit": "GBA",
                "baseline_km2": extent["baseline_km2"],
                "endpoint_km2": extent["endpoint_km2"],
                "extent_change_pct": extent["extent_change_pct"],
                "protected_coverage_pct": gba_protected_coverage_pct,
                "delta_mhi_inside_mean": inside_mean,
                "delta_mhi_outside_mean": outside_mean,
            }
        ]
    )


def main() -> None:
    print("Computing GBA local ΔMHI and PA grouping...")
    gba_hex_gdf, gba_protected_coverage_pct, clipped_reserves = (
        compute_local_gba_stats()
    )
    gba_hex_gdf.to_file(GBA_HEX_GPKG, driver="GPKG")
    clipped_reserves.to_file(GBA_CLIPPED_RESERVE_PATH)

    gba_hex_df = pd.DataFrame(gba_hex_gdf.drop(columns="geometry"))
    gba_hex_df.to_csv(GBA_HEX_CSV, index=False)

    gba_summary_df = build_gba_summary_row(gba_protected_coverage_pct, gba_hex_df)
    gba_summary_df.to_csv(GBA_SUMMARY_CSV, index=False)

    print(f"Saved: {GBA_HEX_CSV}")
    print(f"Saved: {GBA_HEX_GPKG}")
    print(f"Saved: {GBA_CLIPPED_RESERVE_PATH}")
    print(f"Saved: {GBA_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
