"""
Combine mangrove area trend plot and vegetation indices boxplots.
Creates a publication-quality combined figure.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Define file paths and years
base_dir = r"d:\Desktop\Mangrove\Final\VI"
files = [
    ("mangrove_2000_90conf_indices.tif", 2000),
    ("mangrove_2005_90conf_indices.tif", 2005),
    ("mangrove_2010_90conf_indices.tif", 2010),
    ("mangrove_2015_90conf_indices.tif", 2015),
    ("mangrove_2020_90conf_indices.tif", 2020),
    ("mangrove_2024_90conf_indices.tif", 2024),
]

dst_crs = 'EPSG:32649'

# Color palette - professional ecological journal style
year_colors = {
    2000: '#2E5A87',
    2005: '#3D7EAA',
    2010: '#4AA168',
    2015: '#7BC87B',
    2020: '#E8963A',
    2024: '#C44E52',
}

indices = ['NDVI', 'EVI', 'MVI', 'EMVI', 'CMRI', 'kNDVI']
band_indices = [1, 2, 3, 4, 5, 6]

# ============================================================
# PART 1: Calculate mangrove areas
# ============================================================
print("Calculating mangrove areas...")

def calculate_mangrove_area(filepath, dst_crs):
    with rasterio.open(filepath) as src:
        src_data = src.read(1)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst_data = np.zeros((height, width), dtype=src_data.dtype)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        pixel_area_m2 = pixel_width * pixel_height
        mangrove_pixels = np.sum(dst_data == 1)
        area_km2 = (mangrove_pixels * pixel_area_m2) / 1e6
        return area_km2

years = []
areas = []
for filename, year in files:
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        area = calculate_mangrove_area(filepath, dst_crs)
        years.append(year)
        areas.append(area)
        print(f"  {year}: {area:.2f} km²")

# ============================================================
# PART 2: Extract vegetation indices
# ============================================================
print("\nExtracting vegetation indices...")

def extract_vegetation_indices(filepath):
    with rasterio.open(filepath) as src:
        mangrove_mask = src.read(1) == 1
        data = {}
        for idx_name, band_idx in zip(indices, band_indices):
            band_data = src.read(band_idx + 1)
            values = band_data[mangrove_mask]
            values = values[~np.isnan(values)]
            values = values[(values > -10) & (values < 10)]
            if len(values) > 5000:
                np.random.seed(42)
                values = np.random.choice(values, 5000, replace=False)
            data[idx_name] = values
        return data

all_data = {}
for filename, year in files:
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        all_data[year] = extract_vegetation_indices(filepath)
        print(f"  {year}: extracted")

# ============================================================
# PART 3: Create combined figure
# ============================================================
print("\nCreating combined figure...")

# Create figure with GridSpec for better control
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12, 7))
gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1.2], hspace=0.35, wspace=0.5)

# Top row: Mangrove area trend (spanning all columns)
ax_trend = fig.add_subplot(gs[0, :])

# Calculate trend
years_arr = np.array(years)
areas_arr = np.array(areas)
z = np.polyfit(years_arr, areas_arr, 1)
p = np.poly1d(z)
slope = z[0]

# Colors for trend plot
data_color = '#1a6b6b'
trend_color = '#7fb3b3'
marker_edge = '#0d4d4d'

# Plot trend line
trend_x = np.linspace(years_arr[0] - 1, years_arr[-1] + 1, 100)
trend_y = p(trend_x)
ax_trend.plot(trend_x, trend_y, color=trend_color, linestyle='--', 
              linewidth=1.5, alpha=0.8, zorder=1)

# Plot data points
ax_trend.plot(years, areas, color=data_color, linewidth=2.0, 
              marker='o', markersize=8, markerfacecolor='white',
              markeredgecolor=marker_edge, markeredgewidth=2, zorder=2)

#ax_trend.set_xlabel('Year')
ax_trend.set_ylabel('Mangrove Area (km²)')

y_range = max(areas) - min(areas)
y_margin = y_range * 0.15 if y_range > 0 else max(areas) * 0.1
ax_trend.set_ylim(min(areas) - y_margin, max(areas) + y_margin)
ax_trend.set_xlim(1998, 2026)
ax_trend.set_xticks(years)
ax_trend.set_xticklabels([str(y) for y in years])
ax_trend.set_axisbelow(True)



# Legend for trend
legend_elements_trend = [
    Line2D([0], [0], color=data_color, linewidth=2, marker='o',
           markersize=7, markerfacecolor='white', markeredgecolor=marker_edge,
           markeredgewidth=1.5, label='Mangrove Area'),
    Line2D([0], [0], color=trend_color, linewidth=1.5, linestyle='--',
           label=f'Linear Trend ({slope:+.2f} km²/yr)')
]
ax_trend.legend(handles=legend_elements_trend, loc='lower right', frameon=False)

# Add subplot label (a)
ax_trend.text(-0.03, 1.1, '(a)', transform=ax_trend.transAxes, 
              fontsize=12, fontweight='bold', va='top')

# Bottom row: Boxplots for each vegetation index
for i, vi_name in enumerate(indices):
    ax = fig.add_subplot(gs[1, i])
    
    data_list = []
    positions = []
    colors_list = []
    
    for j, year in enumerate(years):
        if year in all_data and vi_name in all_data[year]:
            data_list.append(all_data[year][vi_name])
            positions.append(j)
            colors_list.append(year_colors[year])
    
    bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, showcaps=True, showmeans=False)
    
    for j, (box, median, whisker1, whisker2, cap1, cap2) in enumerate(zip(
            bp['boxes'], bp['medians'], bp['whiskers'][::2], bp['whiskers'][1::2],
            bp['caps'][::2], bp['caps'][1::2])):
        color = colors_list[j]
        box.set_facecolor(color)
        box.set_alpha(0.6)
        box.set_edgecolor(color)
        box.set_linewidth(1.5)
        median.set_color('white')
        median.set_linewidth(1.5)
        whisker1.set_color(color)
        whisker2.set_color(color)
        whisker1.set_linewidth(1.2)
        whisker2.set_linewidth(1.2)
        cap1.set_color(color)
        cap2.set_color(color)
        cap1.set_linewidth(1.2)
        cap2.set_linewidth(1.2)
    
    # Scatter points with consistent colors
    for j, (data, pos) in enumerate(zip(data_list, positions)):
        sample_size = min(100, len(data))
        np.random.seed(42)
        sample_data = np.random.choice(data, sample_size, replace=False)
        jitter = np.random.uniform(-0.15, 0.15, sample_size)
        ax.scatter(pos + jitter, sample_data, c=colors_list[j], 
                   s=5, alpha=0.4, edgecolors='none', zorder=0)
    
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=90, ha='center')
    #ax.set_xlabel('Year')
    ax.set_ylabel(vi_name, labelpad=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.set_axisbelow(True)
    
    # Add subplot labels (b), (c), (d), (e), (f)
    label = chr(ord('b') + i)
    ax.text(-0.2, 1.1, f'({label})', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

# Create shared legend for boxplots (lower right corner of the figure)
legend_elements_box = [Patch(facecolor=year_colors[y], edgecolor=year_colors[y], 
                             alpha=0.6, label=str(y)) for y in years]
# fig.legend(handles=legend_elements_box, loc='lower right', 
#            bbox_to_anchor=(0.97, 0.02), ncol=6, frameon=False, fontsize=9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# Save figure
output_path = r"d:\Desktop\Mangrove\Final\mangrove_combined_figure.png"
plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
plt.close()

print(f"\nCombined figure saved to: {output_path}")
print("\nDone!")
