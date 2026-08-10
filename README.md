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

## Code organization

Production scripts are organized by analytical responsibility. Jupyter notebooks are stored separately as source-only research records with generated outputs removed.

| Directory | Scope | Representative scripts |
|---|---|---|
| `src/mapping/` | Earth Engine classification, grids, raster overlay, and variable extraction | `RF_opt.py`, `create_hex_grid.py`, `overlay_mangrove_lulc.py` |
| `src/condition/` | MHI construction, temporal slopes, spatial interpolation, and imputation | `calculate_mhi.py`, `calculate_slopes.py`, `kriging_interpolation.py` |
| `src/spatial_modeling/` | Spatial dependence, MGWR, geographic random forest, and SHAP interpretation | `global_moran_analysis.py`, `mgwr_multiyear.py`, `PyGRF.py`, `rf_shap_regression.py` |
| `src/benchmarking/` | Protected-area analysis and global country comparison | `global_country_gee_export.py`, `gba_mhi_pa_analysis.py` |
| `src/valuation/` | Annual and transition-based ecosystem service valuation | `calculate_yearly_esv.py`, `calculate_transition_esv.py` |
| `src/visualization/` | Publication figures, maps, time series, and Sankey diagrams | `plot_esv_time_series.py`, `plot_global_gba_panels.py`, `create_sankey_diagrams.py` |

Notebook groups mirror the same research stages under `notebooks/condition/`, `notebooks/mapping/`, `notebooks/spatial_modeling/`, and `notebooks/visualization/`.

## Repository contents

```text
.
├── src/
│   ├── mapping/                 # Mapping and spatial data engineering
│   ├── condition/               # MHI and environmental preprocessing
│   ├── spatial_modeling/        # Spatial statistics and interpretable modelling
│   ├── benchmarking/            # Protected-area and global comparisons
│   ├── valuation/               # Ecosystem service calculations
│   └── visualization/           # Scientific figures and maps
├── notebooks/                   # Output-free research notebooks by stage
├── DATA_MANIFEST.csv            # Local research-data inventory metadata
├── ZENODO_UPLOAD_MANIFEST.csv   # Archived-data release manifest
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and reproducibility guide
```

No research datasets, model binaries, generated figures, or derived outputs are tracked in GitHub. The repository contains code, dependency metadata, and data manifests only.

## Data

Download the research archive from [Zenodo record 21064003](https://zenodo.org/records/21064003), then restore the archived folders and files at the repository root. The repository's `.gitignore` keeps these data and generated products outside version control. `DATA_MANIFEST.csv` records the inventory metadata and `ZENODO_UPLOAD_MANIFEST.csv` identifies the archived release contents.

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

1. Generate annual mangrove products with `src/mapping/RF_opt.py` in Earth Engine.
2. Build the analysis grid and extract environmental variables.
3. Calculate MHI, temporal slopes, spatial statistics, and spatially varying models.
4. Export global country statistics and integrate them with the local GBA analysis.
5. Calculate annual and transition-based ecosystem service values.
6. Generate the publication figures from the derived tables and rasters.

Representative commands:

```bash
python src/valuation/calculate_yearly_esv.py --output-dir .
python src/valuation/calculate_transition_esv.py --output-dir .
python src/visualization/plot_esv_time_series.py \
  --input yearly_class_area_esv.csv \
  --output mangrove_esv_time_series.png
python src/visualization/plot_mangrove_service_composition.py \
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
