"""
将 VI 红树林分布覆盖到 LULC_convert 土地利用数据上，
基于 VI 的植被指数波段计算 MHI（红树林健康指数），
重投影到目标坐标系，并计算各年份红树林面积。

处理流程:
  1. 读取 VI tif（波段1: 1=红树林, 0=非红树林；波段2~7: 植被指数）
  2. 读取对应年份的 LULC_convert tif
  3. 将 VI 重采样对齐到 LULC 的空间范围和分辨率
  4. 在 VI==1 的位置，将 LULC 的值赋为 100（红树林）
  5. 基于 VI 波段2~7 (NDVI, EVI, MVI, EMVI, CMRI, kNDVI) 通过 PCA 计算 MHI
  6. 输出双波段 TIF：波段1=LULC，波段2=MHI（float32）
  7. 重投影到目标坐标系
  8. 计算红树林面积（公顷）并输出 CSV
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import (
    calculate_default_transform,
    reproject,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

VI_DIR = ROOT / "VI"
LULC_DIR = ROOT / "LULC_convert"
OUTPUT_DIR = ROOT / "LULC_convert_mangrove"

DST_CRS = "EPSG:9822"

# VI tif 波段定义
# 波段 1: 红树林分布 (1=红树林, 0=非红树林)
# 波段 2: NDVI
# 波段 3: EVI
# 波段 4: MVI
# 波段 5: EMVI
# 波段 6: CMRI
# 波段 7: kNDVI
VI_BAND_NAMES = ["NDVI", "EVI", "MVI", "EMVI", "CMRI", "kNDVI"]
VI_BAND_INDICES = [2, 3, 4, 5, 6, 7]  # 对应波段编号

# VI 和 LULC 的年份匹配关系
YEAR_MAPPING = {
    2000: {
        "vi": VI_DIR / "mangrove_2000_90conf_indices.tif",
        "lulc": LULC_DIR / "CLCD_v01_2000_wgs84_clipped.tif",
    },
    2005: {
        "vi": VI_DIR / "mangrove_2005_90conf_indices.tif",
        "lulc": LULC_DIR / "CLCD_v01_2005_wgs84_clipped.tif",
    },
    2010: {
        "vi": VI_DIR / "mangrove_2010_90conf_indices.tif",
        "lulc": LULC_DIR / "CLCD_v01_2010_wgs84_clipped.tif",
    },
    2015: {
        "vi": VI_DIR / "mangrove_2015_90conf_indices.tif",
        "lulc": LULC_DIR / "CLCD_v01_2015_wgs84_clipped.tif",
    },
    2020: {
        "vi": VI_DIR / "mangrove_2020_90conf_indices.tif",
        "lulc": LULC_DIR / "CLCD_v01_2020_wgs84_clipped.tif",
    },
    2024: {
        "vi": VI_DIR / "mangrove_2024_90conf_indices.tif",
        "lulc": LULC_DIR / "CLCD_v01_2024_wgs84_clipped.tif",
    },
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


def compute_mhi_from_vi_bands(
    vi_bands: list[np.ndarray],
    mangrove_mask: np.ndarray,
) -> np.ndarray:
    """
    基于 6 个植被指数波段，使用 PCA 第一主成分计算 MHI（红树林健康指数）。
    仅在红树林像素上计算，非红树林区域赋值为 NaN。

    参数:
        vi_bands: 6 个对齐后的植被指数数组 (NDVI, EVI, MVI, EMVI, CMRI, kNDVI)
        mangrove_mask: 红树林掩膜 (True=红树林)

    返回:
        MHI 数组，非红树林区域为 NaN
    """
    height, width = mangrove_mask.shape
    mhi = np.full((height, width), np.nan, dtype=np.float32)

    if not np.any(mangrove_mask):
        print("    警告：未检测到红树林像素，MHI 全部为 NaN")
        return mhi

    # 提取红树林区域的植被指数值，构建 (N, 6) 矩阵
    n_mangrove = int(np.sum(mangrove_mask))
    vi_matrix = np.column_stack(
        [band[mangrove_mask].astype(np.float64) for band in vi_bands]
    )
    print(f"    红树林像素数: {n_mangrove}, 植被指数矩阵形状: {vi_matrix.shape}")

    # 检查缺失值和无效值
    valid_rows = np.all(np.isfinite(vi_matrix), axis=1)
    missing_count = n_mangrove - int(np.sum(valid_rows))
    if missing_count > 0:
        print(f"    警告：检测到 {missing_count} 个含缺失/无效值的像素")
        for i, name in enumerate(VI_BAND_NAMES):
            nan_count = int(np.sum(~np.isfinite(vi_matrix[:, i])))
            if nan_count > 0:
                print(f"      {name}: {nan_count} 个无效值")

    vi_clean = vi_matrix[valid_rows]
    if vi_clean.shape[0] < 2:
        print("    警告：有效红树林像素不足，无法执行 PCA")
        return mhi

    print(f"    有效数据行数: {vi_clean.shape[0]}")

    # 数据标准化
    scaler = StandardScaler()
    vi_scaled = scaler.fit_transform(vi_clean)
    print("    植被指数数据已标准化")

    # 执行 PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(vi_scaled)

    # 获取 PCA 载荷
    loadings = pca.components_[0]
    explained_var = pca.explained_variance_ratio_[0]
    print(f"\n    PCA 第一主成分解释方差比: {explained_var:.4f}")
    print("    PCA 第一主成分载荷:")
    for name, loading in zip(VI_BAND_NAMES, loadings):
        print(f"      {name}: {loading:.4f}")

    # 方向校正：确保 MHI 与植被健康度正相关
    avg_loading = np.mean(loadings)
    if avg_loading < 0:
        print("\n    检测到主成分方向与植被健康度负相关，正在翻转方向...")
        principal_component = -principal_component
        loadings = -loadings
        print("    方向已校正，MHI 与植被健康度正相关")
    else:
        print("\n    主成分方向正常，MHI 与植被健康度正相关")

    print("    校正后的载荷:")
    for name, loading in zip(VI_BAND_NAMES, loadings):
        print(f"      {name}: {loading:.4f}")

    # 将 PCA 结果写回红树林像素位置
    # 注意：只在 valid_rows 对应的位置赋值
    mangrove_indices = np.where(mangrove_mask.ravel())[0]
    valid_indices = mangrove_indices[valid_rows]
    mhi_flat = mhi.ravel()
    mhi_flat[valid_indices] = principal_component.flatten().astype(np.float32)
    mhi = mhi_flat.reshape(height, width)

    print(f"    MHI 计算完成, 有效值范围: [{np.nanmin(mhi):.4f}, {np.nanmax(mhi):.4f}]")
    return mhi


def overlay_mangrove_on_lulc(
    vi_path: Path,
    lulc_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    将 VI 红树林掩膜覆盖到 LULC 上，同时计算 MHI。
    返回: (合并后的 LULC 数组, MHI 数组, LULC profile)
    """
    with rasterio.open(lulc_path) as lulc_src:
        lulc_data = lulc_src.read(1)
        lulc_profile = lulc_src.profile.copy()
        lulc_transform = lulc_src.transform
        lulc_crs = lulc_src.crs
        lulc_width = lulc_src.width
        lulc_height = lulc_src.height

    with rasterio.open(vi_path) as vi_src:
        vi_band_count = vi_src.count
        print(f"    VI 波段数: {vi_band_count}")

        # 将 VI 波段1（红树林分布）重采样对齐到 LULC
        vi_band1_aligned = np.zeros((lulc_height, lulc_width), dtype=vi_src.dtypes[0])
        reproject(
            source=rasterio.band(vi_src, 1),
            destination=vi_band1_aligned,
            src_transform=vi_src.transform,
            src_crs=vi_src.crs,
            dst_transform=lulc_transform,
            dst_crs=lulc_crs,
            dst_width=lulc_width,
            dst_height=lulc_height,
            resampling=Resampling.nearest,
        )

        # 将 VI 波段2~7（植被指数）重采样对齐到 LULC
        vi_bands_aligned = []
        for band_idx in VI_BAND_INDICES:
            if band_idx > vi_band_count:
                print(f"    警告：VI 缺少波段 {band_idx} ({VI_BAND_NAMES[band_idx - 2]})")
                vi_bands_aligned.append(
                    np.full((lulc_height, lulc_width), np.nan, dtype=np.float32)
                )
                continue
            band_aligned = np.zeros((lulc_height, lulc_width), dtype=np.float32)
            reproject(
                source=rasterio.band(vi_src, band_idx),
                destination=band_aligned,
                src_transform=vi_src.transform,
                src_crs=vi_src.crs,
                dst_transform=lulc_transform,
                dst_crs=lulc_crs,
                dst_width=lulc_width,
                dst_height=lulc_height,
                resampling=Resampling.bilinear,
            )
            vi_bands_aligned.append(band_aligned)

    # 在 VI==1（红树林）处将 LULC 赋值为 100
    mangrove_mask = vi_band1_aligned == 1
    merged = lulc_data.copy()
    merged[mangrove_mask] = 100

    # 计算 MHI
    print("  计算 MHI（红树林健康指数）...")
    mhi = compute_mhi_from_vi_bands(vi_bands_aligned, mangrove_mask)

    return merged, mhi, lulc_profile


def reproject_to_target(
    lulc_data: np.ndarray,
    mhi_data: np.ndarray,
    src_profile: dict,
    dst_crs: str,
    output_path: Path,
) -> Path:
    """
    将双波段栅格（波段1: LULC uint8, 波段2: MHI float32）重投影到目标坐标系并写出文件。
    """
    src_crs_val = src_profile["crs"]
    src_transform = src_profile["transform"]
    src_width = src_profile["width"]
    src_height = src_profile["height"]

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs_val, dst_crs, src_width, src_height,
        left=src_transform.c,
        bottom=src_transform.f + src_transform.e * src_height,
        right=src_transform.c + src_transform.a * src_width,
        top=src_transform.f,
    )

    # 波段1: LULC (uint8), 波段2: MHI (float32)
    # 使用 float32 作为统一 dtype 以兼容两个波段
    dst_profile = src_profile.copy()
    dst_profile.update(
        {
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "dtype": "float32",
            "count": 2,
            "compress": "lzw",
            "nodata": None,
        }
    )

    # 波段1: LULC 重投影
    dst_lulc = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=lulc_data.astype(np.float32),
        destination=dst_lulc,
        src_transform=src_transform,
        src_crs=src_crs_val,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    # 波段2: MHI 重投影
    dst_mhi = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=mhi_data.astype(np.float32),
        destination=dst_mhi,
        src_transform=src_transform,
        src_crs=src_crs_val,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(dst_lulc, 1)
        dst.write(dst_mhi, 2)

    return output_path


def calculate_mangrove_area(raster_path: Path) -> float:
    """
    从投影栅格中计算红树林面积（公顷）。
    投影坐标系的单位是米，所以像素面积 = |res_x * res_y| 平方米。
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        pixel_area_m2 = abs(src.transform.a * src.transform.e)
        mangrove_pixels = int(np.sum(data == 100))
        area_ha = mangrove_pixels * pixel_area_m2 / 10000.0
    return area_ha


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    area_records = []

    for year, paths in sorted(YEAR_MAPPING.items()):
        vi_path = paths["vi"]
        lulc_path = paths["lulc"]

        if not vi_path.exists():
            print(f"[跳过] {year}: VI 文件不存在 {vi_path}")
            continue
        if not lulc_path.exists():
            print(f"[跳过] {year}: LULC 文件不存在 {lulc_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[{year}] 开始处理...")
        print(f"{'='*60}")

        # 步骤1: 覆盖红树林到 LULC 并计算 MHI
        merged_data, mhi_data, lulc_profile = overlay_mangrove_on_lulc(vi_path, lulc_path)
        mangrove_pixels_wgs84 = int(np.sum(merged_data == 100))
        print(f"  覆盖完成, 红树林像素数: {mangrove_pixels_wgs84}")

        # 步骤2: 重投影到目标坐标系
        output_name = f"CLCD_v01_{year}_mangrove.tif"
        output_path = OUTPUT_DIR / output_name
        reproject_to_target(merged_data, mhi_data, lulc_profile, DST_CRS, output_path)
        print(f"  已保存: {output_path}")

        # 步骤3: 计算红树林面积
        area_ha = calculate_mangrove_area(output_path)
        print(f"  红树林面积: {area_ha:.4f} 公顷 ({area_ha / 100:.4f} 平方千米)")

        area_records.append(
            {
                "year": year,
                "mangrove_area_ha": round(area_ha, 4),
                "mangrove_area_km2": round(area_ha / 100, 4),
            }
        )

    # 输出面积汇总 CSV
    if area_records:
        area_df = pd.DataFrame(area_records)
        csv_path = ROOT / "mangrove_area_by_year.csv"
        area_df.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"面积汇总已保存: {csv_path}")
        print(f"{'='*60}")
        print("\n各年份红树林面积:")
        print(area_df.to_string(index=False))


if __name__ == "__main__":
    main()
