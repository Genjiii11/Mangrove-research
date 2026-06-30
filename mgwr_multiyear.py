"""
MGWR多年份主导因子分区组合图 (六边形标记)
Years: 2000, 2010, 2024
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

# MGWR相关库
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR, MGWR

# ============================================================================
# 1. 数据加载与预处理 (保持不变)
# ============================================================================
df = pd.read_csv('final_environmental_vars_with_MHI.csv')

dependent_var = 'MHI'
independent_vars = [
    'mean_annual_sst', 'proportion_of_landscape', 'population_mean',
    'water_occurrence', 'water_percent', 'mean_annual_temp',
    'cropland_percent', 'impervious_surface_percent', 'dist_to_coast',
    'patch_density'
]

factor_groups = {
    'mean_annual_sst': '气候敏感区', 'mean_annual_temp': '气候敏感区',
    'population_mean': '人类干扰主导区', 'impervious_surface_percent': '人类干扰主导区',
    'cropland_percent': '人类干扰主导区', 'dist_to_coast': '人类干扰主导区',
    'proportion_of_landscape': '生境破碎化主导区', 'water_occurrence': '生境破碎化主导区',
    'water_percent': '生境破碎化主导区', 'patch_density': '生境破碎化主导区'
}

target_years = [2000,2005,2010,2015,2020,2024]
FIXED_BANDWIDTH = 40
all_results = {}

# ============================================================================
# 2. 对每个年份分别进行GWR分析 (逻辑保持不变)
# ============================================================================
for year in target_years:
    df_year = df[df['year'] == year].copy().dropna(subset=['MHI'])
    df_analysis = df_year[['GridID', 'lon', 'lat', dependent_var] + independent_vars].dropna()
    
    if len(df_analysis) < 50: continue
    
    coords = df_analysis[['lon', 'lat']].values
    y = df_analysis[dependent_var].values.reshape(-1, 1)
    X_raw = df_analysis[independent_vars].values
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    
    gwr_model = GWR(coords, y, X, bw=FIXED_BANDWIDTH, constant=True)
    gwr_results = gwr_model.fit()
    
    coef_abs = np.abs(gwr_results.params[:, 1:])
    dominant_var_idx = np.argmax(coef_abs, axis=1)
    dominant_zones = [factor_groups[independent_vars[idx]] for idx in dominant_var_idx]
    
    results_df = df_analysis[['GridID', 'lon', 'lat']].copy()
    results_df['dominant_zone'] = dominant_zones
    all_results[year] = results_df

# ============================================================================
# 3. 绘制组合图 (a)(b)(c)
# ============================================================================
print("生成组合图...")

# 设置样式
plt.rcParams['font.family'] = 'Arial'
colors = ['#D62728', '#1F77B4', '#2CA02C']  # 红、蓝、绿
zone_mapping = {'气候敏感区': 0, '人类干扰主导区': 1, '生境破碎化主导区': 2}
HEX_RADIUS = 0.008 

# 加载底图
basemap_path = 'buffer/中国_市.geojson'
try:
    basemap_gdf = gpd.read_file(basemap_path)
    # Ensure CRS matches if necessary, though simpler to assume compatible coordinates for now
except Exception as e:
    print(f"Warning: Could not load basemap: {e}")
    basemap_gdf = None
 

# 创建 1行3列 的画布
fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=600, sharex='col', sharey='row')
plt.subplots_adjust(wspace=0.04, hspace=0.06)
sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# 确定全局坐标范围，保证三张图尺度一致
all_lons = pd.concat([res['lon'] for res in all_results.values()])
all_lats = pd.concat([res['lat'] for res in all_results.values()])
lon_min, lon_max = all_lons.min() - 0.05, all_lons.max() + 0.05
lat_min, lat_max = all_lats.min() - 0.05, all_lats.max() + 0.05

axes = axes.flatten()

for i, year in enumerate(target_years):
    ax = axes[i]
    res_df = all_results[year]
    
    # Step 1: Fill the entire background with light blue (ocean color)
    ocean_color = '#d4e7f7'  # Light blue for ocean
    ax.set_facecolor(ocean_color)
    
    # Step 2: Draw the base map (land areas in light gray/white)
    if basemap_gdf is not None and not basemap_gdf.empty:
        basemap_gdf.plot(
            ax=ax,
            facecolor='#f5f5f5',  # Light gray for land
            edgecolor='#cccccc',  # Gray border for city boundaries
            linewidth=0.3,
            zorder=1
        )

    
    patches = []
    patch_colors = []
    
    for _, row in res_df.iterrows():
        hexagon = RegularPolygon(
            (row['lon'], row['lat']), 
            numVertices=6, 
            radius=HEX_RADIUS,
            orientation=np.radians(30)
        )
        patches.append(hexagon)
        patch_colors.append(colors[zone_mapping[row['dominant_zone']]])
    
    collection = PatchCollection(patches, facecolors=patch_colors, edgecolors='#333333', linewidths=0.1, alpha=0.9)
    ax.add_collection(collection)
    
    # 设置子图属性
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal')
    ax.set_title(f'{year}', fontsize=15, fontweight='bold', pad=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加左上角标 (a, b, c)
    # x=0.02, y=0.95 表示在子图内部左上角
    ax.text(-0.1, 1.1, sub_labels[i], transform=ax.transAxes, 
            fontsize=18, fontweight='bold', va='top', ha='left')

    if i >= 3:
        ax.set_xlabel('Longitude', fontsize=12)
    if i % 3 == 0:
        ax.set_ylabel('Latitude', fontsize=12)

# 添加共享坐标轴标签 (已移至子图循环中)
# fig.supxlabel('Longitude', fontsize=15)
# fig.supylabel('Latitude', fontsize=15)

# 创建全局统一图例
legend_labels = ['Climate-sensitive', 'Human disturbance', 'Habitat fragmentation']
legend_handles = [plt.scatter([], [], c=colors[i], s=100, marker='h', edgecolors='#333333') for i in range(3)]

fig.legend(
    handles=legend_handles, 
    labels=legend_labels,
    loc='lower center', 
    bbox_to_anchor=(0.5, 0.0),
    ncol=3, 
    fontsize=14,
    frameon=False
)

# 保存
plt.savefig('combined_mgwr_zones.png', dpi=600, bbox_inches='tight')
plt.show()

print("组合图已保存为 combined_mgwr_zones.png")
