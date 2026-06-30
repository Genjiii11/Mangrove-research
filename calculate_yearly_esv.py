from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

INPUT_FILES = {
    2000: ROOT
    / "LULC_convert_mangrove/CLCD_v01_2000_mangrove.tif",
    2005: ROOT
    / "LULC_convert_mangrove/CLCD_v01_2005_mangrove.tif",
    2010: ROOT
    / "LULC_convert_mangrove/CLCD_v01_2010_mangrove.tif",
    2015: ROOT
    / "LULC_convert_mangrove/CLCD_v01_2015_mangrove.tif",
    2020: ROOT
    / "LULC_convert_mangrove/CLCD_v01_2020_mangrove.tif",
    2024: ROOT
    / "LULC_convert_mangrove/CLCD_v01_2024_mangrove.tif",
}

CLASS_NAME = {
    1: "cropland",
    2: "forest",
    3: "shrub",
    4: "grassland",
    5: "water",
    7: "bare_land",
    8: "impervious_surface",
    100: "mangrove",
}

COEF = {
    1: 30894.16999,
    2: 4923.740518,
    3: 4298.731109,
    4: 3673.7217,
    5: 29409.53668,
    7: 0.0,
    8: 683.2253096,
    100: 597839.6238,
}

CLASS_IDS = list(CLASS_NAME.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate yearly class area and ecosystem service value (ESV) "
            "from multi-band LULC rasters. Band 1 is LULC class, band 2 is MHI."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for yearly_class_area_esv.csv and yearly_total_esv.csv (default: project root).",
    )
    return parser.parse_args()


def normalize_mhi_to_unit_interval(
    mhi: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Min-max normalize valid MHI values to [0, 1]."""
    normalized = np.full(mhi.shape, np.nan, dtype=np.float32)

    if not np.any(valid_mask):
        return normalized

    valid_values = mhi[valid_mask].astype(np.float32)
    valid_values = valid_values[np.isfinite(valid_values)]

    if valid_values.size == 0:
        return normalized

    min_value = float(valid_values.min())
    max_value = float(valid_values.max())

    if np.isclose(min_value, max_value):
        normalized[valid_mask] = 0.0
        return normalized

    normalized[valid_mask] = (mhi[valid_mask] - min_value) / (max_value - min_value)
    return normalized


def get_valid_class_mask(
    lulc: np.ndarray, nodata_value: float | int | None
) -> np.ndarray:
    valid_mask = np.isin(lulc, CLASS_IDS)

    if nodata_value is None:
        return valid_mask

    if np.issubdtype(lulc.dtype, np.floating) and np.isnan(nodata_value):
        return valid_mask & ~np.isnan(lulc)

    return valid_mask & (lulc != nodata_value)


def get_valid_mhi_mask(
    lulc: np.ndarray,
    mhi: np.ndarray,
    valid_class_mask: np.ndarray,
    mhi_nodata: float | int | None,
) -> np.ndarray:
    valid_mhi_mask = valid_class_mask & (lulc == 100) & np.isfinite(mhi)

    if mhi_nodata is None:
        return valid_mhi_mask

    if np.isnan(mhi_nodata):
        return valid_mhi_mask & ~np.isnan(mhi)

    return valid_mhi_mask & (mhi != mhi_nodata)


def summarize_one_year(
    year: int, raster_path: Path, mangrove_items_df: pd.DataFrame
) -> tuple[list[dict], dict, list[dict]]:
    try:
        rasterio = importlib.import_module("rasterio")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rasterio is required to read the input GeoTIFF files. "
            "Install it in the active Python environment before running this script."
        ) from exc

    with rasterio.open(raster_path) as src:
        lulc = src.read(1)
        mhi = src.read(2).astype(np.float32)
        class_nodata = src.nodatavals[0] if len(src.nodatavals) >= 1 else src.nodata
        mhi_nodata = src.nodatavals[1] if len(src.nodatavals) >= 2 else src.nodata
        pixel_area_ha = abs(src.transform.a * src.transform.e) / 10000.0

    valid_class_mask = get_valid_class_mask(lulc, class_nodata)
    valid_mhi_mask = get_valid_mhi_mask(lulc, mhi, valid_class_mask, mhi_nodata)

    # 将 band 2 的 MHI 标准化到 [0, 1]
    _normalized_mhi = normalize_mhi_to_unit_interval(mhi, valid_mhi_mask)

    # 计算加权后的红树林有效面积（仅用于计算ESV，不改变原始面积统计）
    if valid_mhi_mask.any():
        norm_mhi_vals = _normalized_mhi[valid_mhi_mask]
        
        # 直接使用归一化后的MHI给定权重 映射到 [0.8, 1.2]
        weights = 0.8 + norm_mhi_vals * 0.4
        
        adjusted_mangrove_area_ha = float(np.sum(weights)) * pixel_area_ha
    else:
        adjusted_mangrove_area_ha = 0.0

    # 将那些属于红树林，但是没有有效 MHI 的像元，以默认的 1.0 权重加入回来
    mangrove_mask = (lulc == 100) & valid_class_mask
    missing_mhi_mask = mangrove_mask & ~valid_mhi_mask
    adjusted_mangrove_area_ha += float(np.sum(missing_mhi_mask)) * pixel_area_ha

    valid_lulc = lulc[valid_class_mask].astype(np.int32)
    present_classes, present_counts = np.unique(valid_lulc, return_counts=True)
    count_by_class = dict(zip(present_classes.tolist(), present_counts.tolist()))

    records: list[dict] = []
    total_area_ha = 0.0
    total_esv = 0.0
    total_adjusted_esv = 0.0

    mangrove_raw_area_ha = 0.0

    for class_id in CLASS_IDS:
        pixel_count = int(count_by_class.get(class_id, 0))
        area_ha = pixel_count * pixel_area_ha
        coef = COEF[class_id]
        annual_esv = area_ha * coef

        if class_id == 100:
            adjusted_area_ha = area_ha
            mangrove_raw_area_ha = area_ha
            adjusted_annual_esv = adjusted_mangrove_area_ha * coef
        else:
            adjusted_area_ha = area_ha
            adjusted_annual_esv = area_ha * coef

        total_area_ha += area_ha
        total_esv += annual_esv
        total_adjusted_esv += adjusted_annual_esv

        records.append(
            {
                "year": year,
                "class_id": class_id,
                "class_name": CLASS_NAME[class_id],
                "area_ha": area_ha,
                "adjusted_area_ha": adjusted_area_ha,
                "coef_2020usd_ha_yr": coef,
                "annual_esv_2020usd": annual_esv,
                "adjusted_annual_esv_2020usd": adjusted_annual_esv,
            }
        )

    for record in records:
        record["area_share"] = (
            record["area_ha"] / total_area_ha if total_area_ha > 0 else np.nan
        )
        record["adjusted_area_share"] = (
            record["adjusted_area_ha"] / total_area_ha if total_area_ha > 0 else np.nan
        )
        record["value_share"] = (
            record["annual_esv_2020usd"] / total_esv if total_esv > 0 else np.nan
        )
        record["adjusted_value_share"] = (
            record["adjusted_annual_esv_2020usd"] / total_adjusted_esv if total_adjusted_esv > 0 else np.nan
        )

    total_record = {
        "year": year,
        "total_area_ha": total_area_ha,
        "total_esv_2020usd": total_esv,
        "total_adjusted_esv_2020usd": total_adjusted_esv,
    }

    mangrove_records = []
    if not mangrove_items_df.empty:
        for _, row in mangrove_items_df.iterrows():
            mangrove_records.append({
                "year": year,
                "group": row["group"],
                "item": row["item"],
                "coef_2020usd_ha_yr": row["coef_2020usd_ha_yr"],
                "area_ha": mangrove_raw_area_ha,
                "adjusted_area_ha": mangrove_raw_area_ha,
                "annual_esv_2020usd": mangrove_raw_area_ha * row["coef_2020usd_ha_yr"],
                "adjusted_annual_esv_2020usd": adjusted_mangrove_area_ha * row["coef_2020usd_ha_yr"],
            })

    return records, total_record, mangrove_records


def build_outputs():
    class_rows: list[dict] = []
    total_rows: list[dict] = []
    mangrove_detail_rows: list[dict] = []

    mangrove_items_csv = ROOT / "mangrove_esv_items.csv"
    if mangrove_items_csv.exists():
        mangrove_items_df = pd.read_csv(mangrove_items_csv)
    else:
        mangrove_items_df = pd.DataFrame()

    for year, raster_path in INPUT_FILES.items():
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found for year {year}: {raster_path}")

        yearly_class_rows, yearly_total_row, yearly_mangrove_rows = summarize_one_year(
            year, raster_path, mangrove_items_df
        )
        class_rows.extend(yearly_class_rows)
        total_rows.append(yearly_total_row)
        mangrove_detail_rows.extend(yearly_mangrove_rows)

    class_df = pd.DataFrame.from_records(class_rows)
    total_df = pd.DataFrame.from_records(total_rows)
    mangrove_df = pd.DataFrame.from_records(mangrove_detail_rows)

    class_df = class_df[
        [
            "year",
            "class_id",
            "class_name",
            "area_ha",
            "adjusted_area_ha",
            "coef_2020usd_ha_yr",
            "annual_esv_2020usd",
            "adjusted_annual_esv_2020usd",
            "area_share",
            "adjusted_area_share",
            "value_share",
            "adjusted_value_share",
        ]
    ]
    total_df = total_df[
        [
            "year", 
            "total_area_ha", 
            "total_esv_2020usd", 
            "total_adjusted_esv_2020usd"
        ]
    ]
    if not mangrove_df.empty:
        mangrove_df = mangrove_df[
            [
                "year",
                "group",
                "item",
                "area_ha",
                "adjusted_area_ha",
                "coef_2020usd_ha_yr",
                "annual_esv_2020usd",
                "adjusted_annual_esv_2020usd",
            ]
        ]

    return class_df, total_df, mangrove_df


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_df, total_df, mangrove_df = build_outputs()

    class_output = output_dir / "yearly_class_area_esv.csv"
    total_output = output_dir / "yearly_total_esv.csv"
    mangrove_output = output_dir / "yearly_mangrove_esv_details.csv"

    class_df.to_csv(class_output, index=False)
    total_df.to_csv(total_output, index=False)

    print(f"Saved: {class_output}")
    print(f"Saved: {total_output}")

    if not mangrove_df.empty:
        mangrove_df.to_csv(mangrove_output, index=False)
        print(f"Saved: {mangrove_output}")


if __name__ == "__main__":
    main()
