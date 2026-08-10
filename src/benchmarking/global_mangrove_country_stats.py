"""
Global mangrove country statistics (2000-2020).

输出：
- global_gba_pa_analysis_outputs/global_country_stats_2000_2020.csv

说明：
- 仅负责全球国家/地区尺度红树林面积变化和 2020 受保护覆盖率统计。
- 使用当前 WDPA，因此支持的是 current protection status comparison。
"""

from __future__ import annotations

from pathlib import Path

import ee
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT
OUTPUT_DIR = FINAL_DIR / "global_gba_pa_analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_COUNTRY_CSV = OUTPUT_DIR / "global_country_stats_2000_2020.csv"

BASELINE_YEAR = 2000
ENDPOINT_YEAR = 2020
GLOBAL_SCALE = 30

GMW_IMAGECOLLECTION_ASSET = (
    "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/GMW_V3"
)
GMW_BASELINE_VECTOR_ASSET: str | None = None
GMW_ENDPOINT_VECTOR_ASSET = "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/gmw_v3_2020_vec"

WDPA_POLYGON_ASSET = "WCMC/WDPA/current/polygons"
COUNTRY_BOUNDARY_ASSET = "USDOS/LSIB_SIMPLE/2017"
WDPA_VALID_STATUS = ["designated", "established", "inscribed"]


def initialize_earth_engine() -> None:
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def feature_collection_to_mask(fc: ee.FeatureCollection) -> ee.Image:
    return ee.Image.constant(0).byte().paint(fc, 1).selfMask()


def try_load_gmw_from_imagecollection(year: int) -> ee.Image | None:
    ic = ee.ImageCollection(GMW_IMAGECOLLECTION_ASSET)
    candidates = [
        ic.filter(ee.Filter.eq("year", year)),
        ic.filter(ee.Filter.eq("Year", year)),
        ic.filter(ee.Filter.eq("YEAR", year)),
        ic.filter(ee.Filter.calendarRange(year, year, "year")),
        ic.filter(ee.Filter.stringContains("system:index", str(year))),
        ic.filter(ee.Filter.stringContains("system:id", str(year))),
    ]
    for candidate in candidates:
        size = candidate.size().getInfo()
        if size and size > 0:
            return ee.Image(candidate.first()).gt(0).selfMask()
    return None


def load_gmw_mask(year: int, vector_asset: str | None = None) -> ee.Image:
    if vector_asset:
        return feature_collection_to_mask(ee.FeatureCollection(vector_asset))

    image = try_load_gmw_from_imagecollection(year)
    if image is not None:
        return image

    raise RuntimeError(
        f"Could not auto-detect GMW {year}. Set GMW_BASELINE_VECTOR_ASSET manually."
    )


def load_wdpa_mask() -> ee.Image:
    wdpa = ee.FeatureCollection(WDPA_POLYGON_ASSET).filter(
        ee.Filter.inList("STATUS", WDPA_VALID_STATUS)
    )
    return feature_collection_to_mask(wdpa)


def load_countries() -> ee.FeatureCollection:
    return ee.FeatureCollection(COUNTRY_BOUNDARY_ASSET)


def ee_feature_collection_to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
    info = fc.getInfo()
    features = info.get("features", [])
    rows = [feature.get("properties", {}) for feature in features]
    return pd.DataFrame(rows)


def summarize_global_country_stats() -> pd.DataFrame:
    baseline_mask = load_gmw_mask(BASELINE_YEAR, GMW_BASELINE_VECTOR_ASSET)
    endpoint_mask = load_gmw_mask(ENDPOINT_YEAR, GMW_ENDPOINT_VECTOR_ASSET)
    wdpa_mask = load_wdpa_mask()
    countries = load_countries()

    baseline_area = ee.Image.pixelArea().updateMask(baseline_mask).rename("baseline_m2")
    endpoint_area = ee.Image.pixelArea().updateMask(endpoint_mask).rename("endpoint_m2")
    protected_endpoint_area = (
        ee.Image.pixelArea()
        .updateMask(endpoint_mask)
        .updateMask(wdpa_mask)
        .rename("protected_endpoint_m2")
    )

    stacked = baseline_area.addBands(endpoint_area).addBands(protected_endpoint_area)
    reduced = stacked.reduceRegions(
        collection=countries,
        reducer=ee.Reducer.sum(),
        scale=GLOBAL_SCALE,
        crs="EPSG:4326",
    )
    df = ee_feature_collection_to_dataframe(reduced)

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
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    valid_mask = pd.DataFrame(df[["baseline_km2", "endpoint_km2"]]).notna().any(axis=1)
    df = df.loc[valid_mask].copy().sort_values("endpoint_km2", ascending=False)
    return df


def main() -> None:
    print("Initializing Earth Engine...")
    initialize_earth_engine()

    print("Computing global country statistics...")
    global_df = summarize_global_country_stats()
    global_df.to_csv(GLOBAL_COUNTRY_CSV, index=False)

    print(f"Saved: {GLOBAL_COUNTRY_CSV}")


if __name__ == "__main__":
    main()
