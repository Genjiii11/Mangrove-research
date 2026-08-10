# Mangrove Recovery in the Greater Bay Area

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Data](https://img.shields.io/badge/Data-Zenodo-1682D4?logo=zenodo&logoColor=white)](https://zenodo.org/records/21064003)

Research code and reproducibility materials for an integrated assessment of mangrove extent, ecological condition, spatial heterogeneity, protection status, and ecosystem service value in the Guangdong–Hong Kong–Macao Greater Bay Area (GBA) from 2000 to 2024.

The project combines multi-source remote sensing, Google Earth Engine, spatial statistics, machine learning interpretation, and condition-sensitive ecosystem service valuation. It links annual mangrove mapping with a PCA-derived Mangrove Health Index (MHI), spatial trend analysis, protected-area comparisons, and global country-level benchmarking.

## Research scope

The workflow addresses three connected questions:

1. Did mangrove expansion coincide with improvement in ecological condition?
2. Which climatic, landscape, and human-pressure factors are associated with spatially heterogeneous change?
3. How do local recovery patterns translate into ecosystem service value and compare with national trajectories worldwide?

## Research capabilities demonstrated

| Capability | Implementation in this repository |
|---|---|
| Remote-sensing analysis | Annual Landsat-based mangrove mapping, spectral indices, raster alignment, reprojection, and zonal extraction |
| Cloud geospatial computing | Google Earth Engine workflows for classification, global aggregation, and export |
| Ecological indicator development | PCA-based MHI derived from six vegetation indices |
| Spatial statistics | Global Moran's I, bivariate LISA, geographically weighted regression, and multiscale GWR |
| Robust trend analysis | Mann–Kendall testing and Theil–Sen slope estimation for annual panel data |
| Machine learning | Random forest classification and regression, spatial random forest, SHAP interactions, and partial dependence analysis |
| Ecosystem-service assessment | Condition-adjusted annual and land-transition ESV accounting |
| Scientific visualization | Publication-ready maps, multi-panel figures, Sankey diagrams, time series, and spatial comparison plots |
| Reproducible research | Script-based pipeline, explicit data manifests, Zenodo archive, and documented stage-level execution |

## Analytical framework

```text
Multi-source Earth observation and socioeconomic data
                         |
                         v
Annual mangrove mapping and predictor extraction
                         |
          +--------------+--------------+
          |                             |
          v                             v
 Mangrove Health Index          Extent and landscape metrics
          |                             |
          +--------------+--------------+
                         |
                         v
Trend, spatial dependence, and nonlinear driver analysis
                         |
          +--------------+--------------+
          |                             |
          v                             v
Protected-area benchmarking     Ecosystem service valuation
                         |
                         v
            Publication-ready synthesis
```

## Core code

The essential scripts are grouped below by analytical stage. Jupyter notebooks provide transparent exploratory and intermediate workflows, while the Python scripts support repeatable production runs.

### 1. Mapping and spatial data engineering

| Script | Purpose |
|---|---|
| `RF_opt.py` | Builds the multi-source predictor stack, trains and evaluates the Earth Engine random forest, and exports annual mangrove products |
| `create_hex_grid.py` | Creates the regular hexagonal analysis grid |
| `extract_population_data.py` | Extracts multi-year population summaries to the spatial grid |
| `overlay_mangrove_lulc.py` | Integrates mangrove masks, land cover, and raster MHI products |
| `impute_data.py` | Performs multivariate random-forest imputation |
| `kriging_interpolation.py` | Interpolates spatially incomplete environmental variables |

### 2. Mangrove condition and spatiotemporal analysis

| Script | Purpose |
|---|---|
| `calculate_mhi.py` | Calculates the PCA-based Mangrove Health Index |
| `calculate_slopes.py` | Estimates Mann–Kendall trends and Theil–Sen slopes |
| `global_moran_analysis.py` | Quantifies annual global spatial autocorrelation |
| `lisa_analysis.py` | Maps bivariate local spatial clusters and outliers |
| `mgwr_multiyear.py` | Assesses spatially varying relationships across focal years |
| `PyGRF.py` | Implements geographically weighted random forest modelling |
| `rf_shap_regression.py` | Fits random forest regression and produces SHAP-based interpretation |
| `SHAPInteraction.py` | Evaluates nonlinear feature interactions |

### 3. Protected areas and global benchmarking

| Script | Purpose |
|---|---|
| `global_country_gee_export.py` | Exports country-level mangrove extent and protection statistics from Earth Engine |
| `local_gba_selected_country_analysis.py` | Integrates global country statistics with local GBA condition and protection results |
| `gba_mhi_pa_analysis.py` | Compares hex-level MHI change inside and outside protected areas |
| `global_mangrove_country_stats.py` | Produces global country summaries of mangrove area and protected coverage |
| `plot_global_gba_panels.py` | Creates the global-to-local benchmark figure |

### 4. Ecosystem service valuation and figures

| Script | Purpose |
|---|---|
| `calculate_yearly_esv.py` | Calculates annual class area and condition-adjusted ecosystem service value |
| `calculate_transition_esv.py` | Attributes ESV change to land transitions and MHI change |
| `create_sankey_diagrams.py` | Visualizes health and land-use transitions |
| `plot_esv_time_series.py` | Produces the mangrove ESV time-series figure |
| `plot_mangrove_service_composition.py` | Visualizes the composition of mangrove ecosystem services |
| `plot_transition_esv_figure.py` | Plots transition area and transition-induced ESV change |
| `plot_four_region_mangrove_change_google.py` | Generates comparative multi-region mangrove maps |

## Repository contents

```text
.
├── *.py                         # Reusable analysis and visualization scripts
├── *.ipynb                      # Documented exploratory and intermediate workflows
├── DATA_MANIFEST.csv            # Complete local research-data inventory
├── ZENODO_UPLOAD_MANIFEST.csv   # Files selected for the archived data release
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and reproducibility guide
```

Large rasters, vector datasets, model objects, tabular intermediates, and generated figures are distributed through the associated Zenodo record rather than GitHub.

## Data

Download the research archive from [Zenodo record 21064003](https://zenodo.org/records/21064003), then restore the archived folders and files at the repository root. `DATA_MANIFEST.csv` records the complete data inventory and `ZENODO_UPLOAD_MANIFEST.csv` identifies the archived release contents.

The workflow integrates Landsat surface reflectance, land-cover products, vegetation indices, nighttime lights, population density, protected areas, coastline information, and ecosystem-service coefficients.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment and install the scientific stack:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Earth Engine stages also require an authenticated Google Earth Engine account:

```bash
earthengine authenticate
```

## Reproducing the main workflow

Run scripts from the repository root after restoring the Zenodo data structure. The main sequence is:

1. Generate annual mangrove products with `RF_opt.py` in Earth Engine.
2. Build the analysis grid and extract environmental variables.
3. Calculate MHI, temporal slopes, spatial statistics, and spatially varying models.
4. Export global country statistics and integrate them with the local GBA analysis.
5. Calculate annual and transition-based ecosystem service values.
6. Generate the publication figures from the derived tables and rasters.

Representative commands:

```bash
python calculate_yearly_esv.py --output-dir .
python calculate_transition_esv.py --output-dir .
python plot_esv_time_series.py \
  --input yearly_class_area_esv.csv \
  --output mangrove_esv_time_series.png
python plot_mangrove_service_composition.py \
  --input yearly_mangrove_esv_details.csv \
  --output mangrove_service_composition.png
```

Each production script performs its own input, schema, CRS, or raster-alignment checks at the relevant analytical boundary.

## Selected outputs

The repository contains code for producing:

* annual mangrove extent and MHI products
* spatial trend and autocorrelation maps
* nonlinear driver-response and interaction plots
* protected-area and international benchmark comparisons
* annual, service-specific, and transition-based ESV summaries
* publication-ready multi-panel figures

## Citation and reuse

For reproducible reuse, cite both this GitHub repository and the accompanying [Zenodo research archive](https://zenodo.org/records/21064003). The Zenodo record provides the persistent data reference, while this repository contains the executable analytical workflows.
