import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from esda.moran import Moran_Local_BV
import libpysal
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 18    # X轴刻度标签大小
plt.rcParams['ytick.labelsize'] = 18    # Y轴刻度标签大小
plt.rcParams['axes.labelsize'] = 20     # 坐标轴标签大小
def get_lisa_cluster_label(row):
    """
    Classifies LISA results into human-readable categories based on quadrant and significance.
    Uses the user-provided logic.
    """
    if row['p_value'] > 0.05:
        return 'Not Significant'
    if row['cluster'] == 1:
        return 'HH (High MHI, High Impervious)'
    elif row['cluster'] == 2:
        return 'LH (Low MHI, High Impervious)'
    elif row['cluster'] == 3:
        return 'LL (Low MHI, Low Impervious)'
    elif row['cluster'] == 4:
        return 'HL (High MHI, Low Impervious)'
    else:
        return 'Not Significant'

def main():
    """
    Main function to perform bivariate LISA analysis and generate a cluster map.
    """
    # --- 1. Load and Merge Data ---
    print("Step 1: Loading and merging data...")
    
    shp_path = 'buffer/mangrove_1km_hexgrid.shp'
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return
        
    slopes_path = 'slopes.csv'
    try:
        slopes_df = pd.read_csv(slopes_path)
    except FileNotFoundError:
        print(f"Error: {slopes_path} not found. Please run calculate_slopes.py first.")
        return

    gdf['GridID'] = gdf['GridID'].astype(slopes_df['GridID'].dtype)
    merged_gdf = gdf.merge(slopes_df, on='GridID')
    
    merged_gdf.dropna(subset=['MHI_slope', 'Impervious_Slope'], inplace=True)
    
    if merged_gdf.empty:
        print("The merged GeoDataFrame is empty. Check if 'GridID' columns match.")
        return
        
    print("Data loading and merging complete.")
    
    # --- 2. Bivariate LISA Analysis ---
    print("\nStep 2: Performing Bivariate LISA analysis using KNN (k=6)...")
    
    y = merged_gdf['MHI_slope']
    x = merged_gdf['Impervious_Slope']
    
    w = libpysal.weights.KNN.from_dataframe(merged_gdf, k=6)
    w.transform = 'r'
    
    lisa = Moran_Local_BV(y, x, w)
    
    merged_gdf['p_value'] = lisa.p_sim
    merged_gdf['cluster'] = lisa.q
    
    print("Bivariate LISA analysis complete.")
    
    # --- 3. Visualization with New Classification ---
    print("\nStep 3: Creating the visualization with new classification logic...")
    
    # Load the base map (China city boundaries)
    basemap_path = 'buffer/中国_市.geojson'
    try:
        # Get the extent of the study area
        minx, miny, maxx, maxy = merged_gdf.total_bounds
        
        # Use bbox filtering when reading if possible, or read and then clip
        # slicing with cx is efficient for in-memory geodataframes
        basemap_gdf = gpd.read_file(basemap_path)
        
        # Ensure CRS matches
        if basemap_gdf.crs != merged_gdf.crs:
            basemap_gdf = basemap_gdf.to_crs(merged_gdf.crs)
            
        # Clip basemap to the extent of the study area using spatial indexing
        # Adding a small buffer for visual context
        x_range = maxx - minx
        y_range = maxy - miny
        padding_x = x_range * 0.05
        padding_y = y_range * 0.05
        
        basemap_gdf = basemap_gdf.cx[minx-padding_x:maxx+padding_x, miny-padding_y:maxy+padding_y]
        
        print(f"Base map loaded and clipped: {len(basemap_gdf)} features")
    except Exception as e:
        print(f"Warning: Could not load base map: {e}")
        basemap_gdf = None
    
    # Apply the new classification logic
    merged_gdf['cluster_label'] = merged_gdf.apply(get_lisa_cluster_label, axis=1)

    # Define labels and colors for the clusters
    # The order is important for the colormap
    category_order = [
        'Not Significant',
        'HH (High MHI, High Impervious)',
        'LL (Low MHI, Low Impervious)',
        'LH (Low MHI, High Impervious)',
        'HL (High MHI, Low Impervious)'
    ]
    colors = [
        'lightgrey', # Not Significant
        '#d7191c',   # HH: Red
        '#abd9e9',   # LL: Light Blue
        '#fdae61',   # LH: Orange
        '#2c7bb6'    # HL: Blue
    ]
    
    # Enforce the order of categories to ensure colors match labels
    merged_gdf['cluster_label'] = pd.Categorical(
        merged_gdf['cluster_label'], 
        categories=category_order, 
        ordered=True
    )
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
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
    
    # Step 3: Draw the LISA cluster map on top
    # Note: When plottig categorical data, Geopandas uses the order of categories
    merged_gdf.plot(
        column='cluster_label',
        categorical=True,
        k=len(colors),
        cmap=mcolors.ListedColormap(colors),
        linewidth=0.5,
        edgecolor='white',
        ax=ax,
        legend=True,
        legend_kwds={
            'fontsize': 15,
            'loc': 'lower right', 
            'title_fontsize': 14,
            'title': 'Cluster Type', 
            'prop': {'size': 12},
            'frameon': False
        },
        zorder=2  # Ensure LISA results are on top
    )
    
    # Set plot extent to the study area
    ax.set_xlim(minx-padding_x, maxx+padding_x)
    ax.set_ylim(miny-padding_y, maxy+padding_y)
    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    # --- Final Touches for Publication Quality ---

    # Save the figure
    output_fig_path = r'Plot\bivariate_lisa_cluster_map.png'
    plt.savefig(output_fig_path, dpi=600, bbox_inches='tight')
    
    print(f"Visualization saved to {output_fig_path}")
    plt.show()

if __name__ == "__main__":
    main()