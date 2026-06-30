import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

INPUT_FILES = {
    2000: ROOT / "LULC_convert_mangrove/CLCD_v01_2000_mangrove.tif",
    2005: ROOT / "LULC_convert_mangrove/CLCD_v01_2005_mangrove.tif",
    2010: ROOT / "LULC_convert_mangrove/CLCD_v01_2010_mangrove.tif",
    2015: ROOT / "LULC_convert_mangrove/CLCD_v01_2015_mangrove.tif",
    2020: ROOT / "LULC_convert_mangrove/CLCD_v01_2020_mangrove.tif",
    2024: ROOT / "LULC_convert_mangrove/CLCD_v01_2024_mangrove.tif",
}

PERIODS = [(2000, 2010), (2010, 2024),(2000,2024)]

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
        description="Calculate ESV changes caused by transitions and MHI changes."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for output CSV (default: project root).",
    )
    return parser.parse_args()


def normalize_mhi_to_unit_interval(mhi: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
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


def get_valid_class_mask(lulc: np.ndarray, nodata_value: float | int | None) -> np.ndarray:
    valid_mask = np.isin(lulc, CLASS_IDS)
    if nodata_value is None:
        return valid_mask
    if np.issubdtype(lulc.dtype, np.floating) and np.isnan(nodata_value):
        return valid_mask & ~np.isnan(lulc)
    return valid_mask & (lulc != nodata_value)


def get_valid_mhi_mask(lulc: np.ndarray, mhi: np.ndarray, valid_class_mask: np.ndarray, mhi_nodata: float | int | None) -> np.ndarray:
    valid_mhi_mask = valid_class_mask & (lulc == 100) & np.isfinite(mhi)
    if mhi_nodata is None:
        return valid_mhi_mask
    if np.isnan(mhi_nodata):
        return valid_mhi_mask & ~np.isnan(mhi)
    return valid_mhi_mask & (mhi != mhi_nodata)


def read_raster_data(raster_path: Path):
    try:
        rasterio = importlib.import_module("rasterio")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rasterio is required to read the input GeoTIFF files."
        ) from exc

    with rasterio.open(raster_path) as src:
        lulc = src.read(1)
        mhi = src.read(2).astype(np.float32)
        class_nodata = src.nodatavals[0] if len(src.nodatavals) >= 1 else src.nodata
        mhi_nodata = src.nodatavals[1] if len(src.nodatavals) >= 2 else src.nodata
        pixel_area_ha = abs(src.transform.a * src.transform.e) / 10000.0

    valid_class_mask = get_valid_class_mask(lulc, class_nodata)
    valid_mhi_mask = get_valid_mhi_mask(lulc, mhi, valid_class_mask, mhi_nodata)
    
    # 按照之前的逻辑对MHI进行标准化，限定于 [0, 1]
    normalized_mhi = normalize_mhi_to_unit_interval(mhi, valid_mhi_mask)

    return lulc, normalized_mhi, valid_class_mask, pixel_area_ha


def compute_transition(y1: int, y2: int) -> list[dict]:
    path1 = INPUT_FILES[y1]
    path2 = INPUT_FILES[y2]

    print(f"Loading data for {y1}...")
    l1, mhi1, vm1, pa1 = read_raster_data(path1)
    
    print(f"Loading data for {y2}...")
    l2, mhi2, vm2, pa2 = read_raster_data(path2)
    
    pixel_area_ha = pa1

    # 两个年份都具有合法 LULC 值的交集 Mask
    valid_mask = vm1 & vm2

    # 拉平处理以加速计算
    valid_mask_flat = valid_mask.ravel()
    
    l1_valid = l1.ravel()[valid_mask_flat]
    l2_valid = l2.ravel()[valid_mask_flat]
    mhi1_valid = mhi1.ravel()[valid_mask_flat]
    mhi2_valid = mhi2.ravel()[valid_mask_flat]
    
    df = pd.DataFrame({
        'c1': l1_valid,
        'c2': l2_valid,
        'mhi1': mhi1_valid,
        'mhi2': mhi2_valid
    })
    
    # 根据 MHI 为每一年计算权重，映射到 [0.8, 1.2]
    df['w1'] = 1.0
    valid_mhi1 = (df['c1'] == 100) & df['mhi1'].notna()
    df.loc[valid_mhi1, 'w1'] = 0.8 + df.loc[valid_mhi1, 'mhi1'] * 0.4

    df['w2'] = 1.0
    valid_mhi2 = (df['c2'] == 100) & df['mhi2'].notna()
    df.loc[valid_mhi2, 'w2'] = 0.8 + df.loc[valid_mhi2, 'mhi2'] * 0.4
    
    print(f"Processing transitions for {y1}-{y2}...")
    results = []
    
    # groupby 时默认排除 nan 但我们需要保持索引计算所有，实际上这不需要我们做什么控制，因为 c1 c2 现在是 valid class
    grouped = df.groupby(['c1', 'c2'])
    
    for (c1, c2), group in grouped:
        count = len(group)
        if count == 0:
            continue
            
        area_ha = count * pixel_area_ha
        adjusted_area1_ha = group['w1'].sum() * pixel_area_ha
        adjusted_area2_ha = group['w2'].sum() * pixel_area_ha
        
        # 按照转移面积 * 系数计算
        esv1 = area_ha * COEF[c1]
        esv2 = area_ha * COEF[c2]
        esv_change = area_ha * (COEF[c2] - COEF[c1])
        
        adjusted_esv1 = adjusted_area1_ha * COEF[c1]
        adjusted_esv2 = adjusted_area2_ha * COEF[c2]
        adjusted_esv_change = adjusted_esv2 - adjusted_esv1
        
        # 仅记录真是发生地类转移的部分
        if c1 == c2:
            continue
            
        results.append({
            "period": f"{y1}-{y2}",
            "from_class_id": int(c1),
            "from_class_name": CLASS_NAME[int(c1)],
            "to_class_id": int(c2),
            "to_class_name": CLASS_NAME[int(c2)],
            "transit_area_ha": area_ha,
            "adjusted_transit_area1_ha": adjusted_area1_ha,
            "adjusted_transit_area2_ha": adjusted_area2_ha,
            "esv_year1_2020usd": esv1,
            "esv_year2_2020usd": esv2,
            "esv_change_2020usd": esv_change,
            "adjusted_esv_year1_2020usd": adjusted_esv1,
            "adjusted_esv_year2_2020usd": adjusted_esv2,
            "adjusted_esv_change_2020usd": adjusted_esv_change,
        })
        
    return results


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_rows = []
    
    for y1, y2 in PERIODS:
        print(f"--- Computing {y1} to {y2} ---")
        period_rows = compute_transition(y1, y2)
        all_rows.extend(period_rows)
        
    if all_rows:
        df_out = pd.DataFrame.from_records(all_rows)
        
        out_csv = output_dir / "all_transition_esv.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"Saved transitions data to: {out_csv}")
    else:
        print("No valid transition data found.")


if __name__ == "__main__":
    main()
