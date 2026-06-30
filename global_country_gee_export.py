"""
Export global mangrove country statistics from Google Earth Engine to Drive.

用途
----
1. 在 GEE 服务器端计算 2000->2020 国家级红树林面积变化；
2. 计算 2020 年红树林受保护覆盖率；
3. 以 batch task 方式导出 CSV 到 Google Drive；
4. 可选：在本地轮询等待任务完成。

说明
----
- 该脚本不做任何本地 GBA 栅格分析。
- 该脚本不会把完整国家统计表通过 getInfo() 拉回本地。
- 导出完成后，请将 Drive 中的 CSV 下载/同步到本地，再运行
  `local_gba_selected_country_analysis.py`。
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import ee


# =========================
# Configuration
# =========================

ROOT = Path(r"D:\Desktop\Mangrove")
FINAL_DIR = ROOT / "Final"
OUTPUT_DIR = FINAL_DIR / "global_gba_pa_analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_YEAR = 2000
ENDPOINT_YEAR = 2020

GMW_IMAGECOLLECTION_ASSET = (
    "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/GMW_V3"
)
GMW_BASELINE_VECTOR_ASSET: str | None = (
    "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/gmw_v3_1996_vec"
)
GMW_ENDPOINT_VECTOR_ASSET = "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/gmw_v3_2020_vec"

WDPA_POLYGON_ASSET = "WCMC/WDPA/current/polygons"
COUNTRY_BOUNDARY_ASSET = "USDOS/LSIB/2017"
WDPA_VALID_STATUS = ["Designated", "Established", "Inscribed"]
CHINA_COUNTRY_NAME = "China"
CHINA_NATURE_RESERVE_ASSET = "projects/ee-lbwnb331161/assets/china_nature_reserve"
GLOBAL_SCALE = 30
# Web Mercator cannot represent polar geometries well; use geographic CRS for global reduceRegions.
GLOBAL_REDUCE_CRS = "EPSG:4326"

TARGET_COUNTRY_NAMES = [
    "Indonesia",
    "Brazil",
    "Australia",
    "Nigeria",
    "Malaysia",
    "United States",
    "United States of America",
    "Japan",
    CHINA_COUNTRY_NAME,
]

GEE_PROJECT = "ee-lbwnb331161"
GEE_HIGHVOLUME_URL = "https://earthengine-highvolume.googleapis.com"
GEE_DRIVE_FOLDER = "mangrove_exports"
GEE_GLOBAL_COUNTRY_EXPORT_PREFIX = "global_country_stats_2000_2020"

WAIT_FOR_GEE_EXPORT = True
GEE_TASK_POLL_SECONDS = 15
GEE_TASK_TIMEOUT_SECONDS = 60 * 60 * 6


# =========================
# Earth Engine helpers
# =========================


def initialize_earth_engine() -> None:
    """初始化 Earth Engine。"""
    try:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT, opt_url=GEE_HIGHVOLUME_URL)
    except Exception as exc:
        print("Earth Engine initialization failed:", exc)
        raise


def feature_collection_to_mask(fc: ee.FeatureCollection) -> ee.Image:
    """将 FeatureCollection 光栅化为二值掩膜。"""
    return ee.Image.constant(0).byte().paint(fc, 1).selfMask()


def try_load_gmw_from_imagecollection(year: int) -> ee.Image | None:
    """尝试从 GMW ImageCollection 中提取指定年份。"""
    ic = ee.ImageCollection(GMW_IMAGECOLLECTION_ASSET)

    candidates = [
        ic.filter(ee.Filter.eq("year", year)),
        ic.filter(ee.Filter.eq("Year", year)),
        ic.filter(ee.Filter.eq("YEAR", year)),
        ic.filter(ee.Filter.calendarRange(year, year, "year")),
        ic.filter(ee.Filter.stringContains("system:index", str(year))),
    ]

    for candidate in candidates:
        size = candidate.size().getInfo()
        if size and size > 0:
            return ee.Image(candidate.first()).gt(0).selfMask()

    return None


def load_gmw_mask(year: int, vector_asset: str | None = None) -> ee.Image:
    """加载 GMW 掩膜。"""
    if vector_asset:
        return feature_collection_to_mask(ee.FeatureCollection(vector_asset))

    img = try_load_gmw_from_imagecollection(year)
    if img is not None:
        return img

    raise RuntimeError(
        f"无法自动从 GMW ImageCollection 中识别 {year} 年数据。"
        f"请手动设置对应年份的 vector asset。"
    )


def load_wdpa_mask() -> ee.Image:
    """加载 WDPA 当前多边形并栅格化为二值掩膜。"""
    wdpa = ee.FeatureCollection(WDPA_POLYGON_ASSET).filter(
        ee.Filter.inList("STATUS", WDPA_VALID_STATUS)
    )
    return feature_collection_to_mask(wdpa)


def load_china_nature_reserve_mask() -> ee.Image:
    """加载中国自然保护区边界并栅格化为二值掩膜。"""
    return feature_collection_to_mask(
        ee.FeatureCollection(CHINA_NATURE_RESERVE_ASSET)
    )


def load_countries() -> ee.FeatureCollection:
    """加载并筛选目标国家边界。"""
    countries = ee.FeatureCollection(COUNTRY_BOUNDARY_ASSET)
    return countries.filter(ee.Filter.inList("COUNTRY_NA", TARGET_COUNTRY_NAMES))


def reduce_country_stats(
    countries: ee.FeatureCollection,
    baseline_area: ee.Image,
    endpoint_area: ee.Image,
    protected_mask: ee.Image,
) -> ee.FeatureCollection:
    """按国家统计红树林面积与受保护面积。"""
    protected_endpoint_area = (
        ee.Image.pixelArea()
        .updateMask(endpoint_area.mask())
        .updateMask(protected_mask)
        .rename("protected_endpoint_m2")
    )
    stacked = baseline_area.addBands(endpoint_area).addBands(protected_endpoint_area)
    reduced = stacked.reduceRegions(
        collection=countries,
        reducer=ee.Reducer.sum(),
        scale=GLOBAL_SCALE,
        crs=GLOBAL_REDUCE_CRS,
        tileScale=16,
    )

    return reduced.map(lambda f: ee.Feature(None, ee.Feature(f).toDictionary()))


def build_global_country_stats_feature_collection() -> ee.FeatureCollection:
    """构建全球国家级红树林统计的 EE FeatureCollection。"""
    baseline_mask = load_gmw_mask(BASELINE_YEAR, GMW_BASELINE_VECTOR_ASSET)
    endpoint_mask = load_gmw_mask(ENDPOINT_YEAR, GMW_ENDPOINT_VECTOR_ASSET)
    countries = load_countries()
    china_country = countries.filter(ee.Filter.eq("COUNTRY_NA", CHINA_COUNTRY_NAME))
    non_china_countries = countries.filter(
        ee.Filter.neq("COUNTRY_NA", CHINA_COUNTRY_NAME)
    )

    baseline_area = ee.Image.pixelArea().updateMask(baseline_mask).rename("baseline_m2")
    endpoint_area = ee.Image.pixelArea().updateMask(endpoint_mask).rename("endpoint_m2")

    non_china_stats = reduce_country_stats(
        countries=non_china_countries,
        baseline_area=baseline_area,
        endpoint_area=endpoint_area,
        protected_mask=load_wdpa_mask(),
    )
    china_stats = reduce_country_stats(
        countries=china_country,
        baseline_area=baseline_area,
        endpoint_area=endpoint_area,
        protected_mask=load_china_nature_reserve_mask(),
    )

    return ee.FeatureCollection(non_china_stats).merge(china_stats)


def export_global_country_stats_to_drive() -> ee.batch.Task:
    """将国家统计结果导出到 Google Drive。"""
    reduced = build_global_country_stats_feature_collection()
    task = ee.batch.Export.table.toDrive(
        collection=reduced,
        description=GEE_GLOBAL_COUNTRY_EXPORT_PREFIX,
        folder=GEE_DRIVE_FOLDER,
        fileNamePrefix=GEE_GLOBAL_COUNTRY_EXPORT_PREFIX,
        fileFormat="CSV",
    )
    task.start()
    return task


def wait_for_gee_task(task: ee.batch.Task) -> dict[str, Any]:
    """轮询等待 GEE batch task 完成。"""
    start_time = time.time()
    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")

        if state == "COMPLETED":
            return status
        if state in {"FAILED", "CANCELLED"}:
            raise RuntimeError(
                f"GEE export task ended with state={state}: "
                f"{status.get('error_message', 'No error message returned.')}"
            )

        elapsed = time.time() - start_time
        if elapsed > GEE_TASK_TIMEOUT_SECONDS:
            raise TimeoutError(
                "Timed out while waiting for GEE Drive export task to finish. "
                f"Task status: {status}"
            )

        print(
            f"GEE export task state: {state}. "
            f"Waiting {GEE_TASK_POLL_SECONDS}s before next check..."
        )
        time.sleep(GEE_TASK_POLL_SECONDS)


def main() -> None:
    print("[1/3] Initializing Earth Engine...")
    initialize_earth_engine()

    print("[2/3] Submitting Drive export task for global country statistics...")
    task = export_global_country_stats_to_drive()
    print(
        "Started GEE task "
        f"'{GEE_GLOBAL_COUNTRY_EXPORT_PREFIX}' in Drive folder '{GEE_DRIVE_FOLDER}'."
    )

    if WAIT_FOR_GEE_EXPORT:
        print("[3/3] Waiting for GEE export task to finish...")
        wait_for_gee_task(task)
        print("GEE Drive export task completed.")
    else:
        print("[3/3] Task submitted. Not waiting for completion.")

    print("\nNext step:")
    print(
        "Download or sync the Drive CSV, then run local_gba_selected_country_analysis.py"
    )


if __name__ == "__main__":
    main()
