from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds


ROOT = Path(__file__).resolve().parents[2]
VI_DIR = ROOT / "VI"
OUTPUT_DIR = ROOT / "Plot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2000, 2010, 2020, 2024]
RASTER_PATHS = {year: VI_DIR / f"mangrove_{year}_90conf_indices.tif" for year in YEARS}

WGS84_CRS = "EPSG:4326"
DEFAULT_OUTPUT = OUTPUT_DIR / "four_region_mangrove_change_landsat.png"
DEFAULT_EE_PROJECT = "ee-lbwnb331161"
LANDSAT_VIS_PARAMS = {"bands": ["R", "G", "B"], "min": 0.03, "max": 0.35, "gamma": 1.1}


@dataclass(frozen=True)
class Region:
    name: str
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class CartoeeStack:
    ee: Any
    cartoee: Any
    ccrs: Any


REGIONS = [
    Region(
        name="Qi'ao Island",
        west=113.59500827357967,
        south=22.377241215650638,
        east=113.67315481936014,
        north=22.455387761431116,
    ),
    Region(
        name="Mai Po and Futian\nNature Reserve",
        west=113.97788756549916,
        south=22.46159294620773,
        east=114.0539980350248,
        north=22.537703415733375,
    ),
    Region(
        name="Shenzhen Bay",
        west=113.93054879257447,
        south=22.487868025496205,
        east=113.97214292178022,
        north=22.52946215470194,
    ),
    Region(
        name="Zhenhai Bay",
        west=112.2924042505516,
        south=21.719006849268958,
        east=112.6395067739934,
        north=22.066109372710763,
    ),
]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 4x4 figure of mangrove binary maps for four regions and "
            "four years over year-matched Landsat basemap imagery."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--zoom-level",
        type=int,
        default=13,
        help="cartoee map zoom level for Earth Engine rendering. Default: 13",
    )
    parser.add_argument(
        "--ee-project",
        type=str,
        default=DEFAULT_EE_PROJECT,
        help=(
            "Optional Google Earth Engine project ID passed to geemap.ee_initialize. "
            f"Default: {DEFAULT_EE_PROJECT}"
        ),
    )
    parser.add_argument(
        "--strict-cartoee-basemap",
        action="store_true",
        help=(
            "Fail when cartoee Landsat basemap cannot be initialized. "
            "By default, the script falls back to plotting without basemap."
        ),
    )
    return parser


def validate_inputs() -> None:
    missing_paths = [path for path in RASTER_PATHS.values() if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing VI TIFF files:\n{missing_text}")


def initialize_cartoee_stack(ee_project: str | None) -> CartoeeStack:
    try:
        import cartopy.crs as ccrs
        import ee
        import geemap
        from geemap import cartoee
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for cartoee Landsat basemap. Install with: "
            "conda install -c conda-forge geemap cartopy earthengine-api"
        ) from exc

    try:
        if ee_project:
            geemap.ee_initialize(project=ee_project)
        else:
            geemap.ee_initialize()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Google Earth Engine for cartoee. "
            "Try --ee-project <YOUR_GEE_PROJECT_ID>."
        ) from exc

    return CartoeeStack(ee=ee, cartoee=cartoee, ccrs=ccrs)


def region_to_cartoee_bbox(region: Region) -> list[float]:
    # cartoee region ordering is [E, S, W, N].
    return [region.east, region.south, region.west, region.north]


def _prepare_landsat_tm_etm(image: Any) -> Any:
    qa_pixel = image.select("QA_PIXEL")
    clear_mask = (
        qa_pixel.bitwiseAnd(1 << 0).eq(0)
        .And(qa_pixel.bitwiseAnd(1 << 1).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 2).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 3).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))
    )
    non_saturated_mask = image.select("QA_RADSAT").eq(0)

    optical = (
        image.select(
            ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
            ["B", "G", "R", "NIR", "SWIR1", "SWIR2"],
        )
        .multiply(0.0000275)
        .add(-0.2)
    )
    return optical.updateMask(clear_mask).updateMask(non_saturated_mask)


def _prepare_landsat_oli(image: Any) -> Any:
    qa_pixel = image.select("QA_PIXEL")
    clear_mask = (
        qa_pixel.bitwiseAnd(1 << 0).eq(0)
        .And(qa_pixel.bitwiseAnd(1 << 1).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 2).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 3).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))
    )
    non_saturated_mask = image.select("QA_RADSAT").eq(0)

    optical = (
        image.select(
            ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            ["B", "G", "R", "NIR", "SWIR1", "SWIR2"],
        )
        .multiply(0.0000275)
        .add(-0.2)
    )
    return optical.updateMask(clear_mask).updateMask(non_saturated_mask)


def _build_landsat_collection(
    stack: CartoeeStack,
    year: int,
    start_date: str,
    end_date: str,
    region_geom: Any,
) -> Any:
    ee = stack.ee

    if year <= 2011:
        return (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"))
            .filterDate(start_date, end_date)
            .filterBounds(region_geom)
            .map(_prepare_landsat_tm_etm)
        )
    if year <= 2020:
        return (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(region_geom)
            .map(_prepare_landsat_oli)
        )
    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterDate(start_date, end_date)
        .filterBounds(region_geom)
        .map(_prepare_landsat_oli)
    )


def build_landsat_composite(stack: CartoeeStack, year: int, region: Region) -> Any:
    ee = stack.ee
    region_geom = ee.Geometry.Rectangle(
        [region.west, region.south, region.east, region.north], proj=WGS84_CRS, geodesic=False
    )

    primary_collection = _build_landsat_collection(
        stack=stack,
        year=year,
        start_date=f"{year}-01-01",
        end_date=f"{year + 1}-01-01",
        region_geom=region_geom,
    )
    fallback_collection = _build_landsat_collection(
        stack=stack,
        year=year,
        start_date=f"{year - 1}-01-01",
        end_date=f"{year + 2}-01-01",
        region_geom=region_geom,
    )

    selected_collection = ee.ImageCollection(
        ee.Algorithms.If(
            primary_collection.size().gt(0),
            primary_collection,
            fallback_collection,
        )
    )
    return selected_collection.median().clip(region_geom)


def _format_longitude(value: float) -> str:
    hemisphere = "E" if value >= 0 else "W"
    return f"{abs(value):.3f}°{hemisphere}"


def _format_latitude(value: float) -> str:
    hemisphere = "N" if value >= 0 else "S"
    return f"{abs(value):.3f}°{hemisphere}"


def _apply_lon_lat_ticks(
    ax: Any,
    region: Region,
    year: int,
    row_index: int,
    col_index: int,
    stack: CartoeeStack | None,
) -> None:
    xticks = np.linspace(region.west, region.east, 4)[1:3]
    yticks = np.linspace(region.south, region.north, 4)[1:3]

    if stack is None:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
    else:
        plate = stack.ccrs.PlateCarree()
        ax.set_xticks(xticks, crs=plate)
        ax.set_yticks(yticks, crs=plate)

    if row_index == len(YEARS) - 1:
        ax.set_xticklabels([_format_longitude(value) for value in xticks], fontsize=11)
    else:
        ax.set_xticklabels([])

    if col_index == 0:
        ax.set_yticklabels([_format_latitude(value) for value in yticks], fontsize=11, rotation=90, va="center")
        ax.set_ylabel(f"{year}", fontsize=14, fontweight="bold")
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", which="major", length=2.5, pad=1)


def read_region_mask(
    raster_path: Path, region: Region
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster CRS is missing: {raster_path}")
        if str(src.crs) != WGS84_CRS:
            raise ValueError(
                f"Expected WGS84 raster for direct lon/lat plotting, got {src.crs}: {raster_path}"
            )

        window = window_from_bounds(
            region.west,
            region.south,
            region.east,
            region.north,
            transform=src.transform,
        )
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window, boundless=True, fill_value=0)
        transform = src.window_transform(window)
        extent = (
            transform.c,
            transform.c + transform.a * data.shape[1],
            transform.f + transform.e * data.shape[0],
            transform.f,
        )
        return data == 1, extent


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def create_axes_with_cartoee_basemap(
    stack: CartoeeStack,
    zoom_level: int,
) -> tuple[plt.Figure, list[list[Any]]]:
    fig = plt.figure(figsize=(15, 15), dpi=300)
    grid = fig.add_gridspec(len(YEARS), len(REGIONS), wspace=0.15, hspace=0.15)

    axes_matrix: list[list[Any]] = []
    for row_index, year in enumerate(YEARS):
        row_axes: list[Any] = []
        for col_index, region in enumerate(REGIONS):
            landsat_image = build_landsat_composite(stack, year, region)
            ax = stack.cartoee.get_map(
                landsat_image,
                region=region_to_cartoee_bbox(region),
                vis_params=LANDSAT_VIS_PARAMS,
                zoom_level=zoom_level,
            )
            ax.set_position(grid[row_index, col_index].get_position(fig))
            ax.set_extent(
                [region.west, region.east, region.south, region.north],
                crs=stack.ccrs.PlateCarree(),
            )
            row_axes.append(ax)
        axes_matrix.append(row_axes)
    return fig, axes_matrix


def create_axes_without_basemap() -> tuple[plt.Figure, list[list[Any]]]:
    fig, axes = plt.subplots(
        nrows=len(YEARS),
        ncols=len(REGIONS),
        figsize=(15, 15),
        sharex="col",
        sharey="col",
        dpi=300,
    )
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    return fig, axes.tolist()


def plot_four_region_figure(
    output_path: Path,
    zoom_level: int,
    ee_project: str | None,
    strict_cartoee_basemap: bool = False,
) -> Path:
    configure_matplotlib()

    stack: CartoeeStack | None = None
    try:
        stack = initialize_cartoee_stack(ee_project)
        fig, axes_matrix = create_axes_with_cartoee_basemap(stack, zoom_level)
    except Exception as exc:
        if strict_cartoee_basemap:
            raise RuntimeError(
                "cartoee Landsat basemap failed. "
                "Check dependency installation, Earth Engine authentication, and --ee-project."
            ) from exc

        warnings.warn(
            "cartoee Landsat basemap unavailable; plotting mangrove masks without basemap. "
            f"Reason: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        stack = None
        fig, axes_matrix = create_axes_without_basemap()

    cmap = ListedColormap(["#e31a1c"])

    for row_index, year in enumerate(YEARS):
        raster_path = RASTER_PATHS[year]
        for col_index, region in enumerate(REGIONS):
            ax = axes_matrix[row_index][col_index]
            if stack is None:
                ax.set_facecolor("#eef2f5")

            mask, mask_extent = read_region_mask(raster_path, region)
            masked = np.ma.masked_where(~mask, mask.astype(np.uint8))
            imshow_kwargs: dict[str, Any] = {
                "extent": mask_extent,
                "origin": "upper",
                "cmap": cmap,
                "interpolation": "nearest",
                "alpha": 0.82,
                "zorder": 2,
            }
            if stack is not None:
                imshow_kwargs["transform"] = stack.ccrs.PlateCarree()

            ax.imshow(
                masked,
                **imshow_kwargs,
            )

            if stack is None:
                ax.set_xlim(region.west, region.east)
                ax.set_ylim(region.south, region.north)
                ax.set_aspect("equal")
            else:
                ax.set_extent(
                    [region.west, region.east, region.south, region.north],
                    crs=stack.ccrs.PlateCarree(),
                )

            if row_index == 0:
                ax.set_title(region.name, pad=10, fontsize=15, fontweight="semibold")
            _apply_lon_lat_ticks(
                ax=ax,
                region=region,
                year=year,
                row_index=row_index,
                col_index=col_index,
                stack=stack,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    validate_inputs()
    output_path = plot_four_region_figure(
        args.output,
        args.zoom_level,
        args.ee_project,
        strict_cartoee_basemap=args.strict_cartoee_basemap,
    )
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
