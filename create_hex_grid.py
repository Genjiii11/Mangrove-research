import geopandas as gpd
from shapely.geometry import Polygon
import math
import operator

def create_hex_grid(gdf, size):
    bounds = gdf.total_bounds
    
    r = size
    width = 2 * r
    height = math.sqrt(3) * r
    
    x_step = 1.5 * r
    y_step = height
    
    xmin = operator.sub(bounds[0], x_step)
    ymin = operator.sub(bounds[1], y_step)
    xmax = bounds[2] + x_step
    ymax = bounds[3] + y_step
    
    polygons = []
    
    width_diff = operator.sub(xmax, xmin)
    height_diff = operator.sub(ymax, ymin)
    
    cols = int(math.ceil(width_diff / x_step)) + 1
    rows = int(math.ceil(height_diff / y_step)) + 1
    
    for col in range(cols):
        x_center = xmin + col * x_step
        
        y_offset = (height / 2.0) if col % 2 != 0 else 0.0
        
        for row in range(rows):
            y_center = ymin + row * y_step + y_offset
            
            vertices = []
            for i in range(6):
                angle_deg = 60 * i
                angle_rad = math.radians(angle_deg)
                x = x_center + r * math.cos(angle_rad)
                y = y_center + r * math.sin(angle_rad)
                vertices.append((x, y))
            
            polygons.append(Polygon(vertices))
            
    grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=gdf.crs)
    return grid_gdf

def main():
    input_file = "buffer/mangrove_2020_1km_buffer.shp"
    output_file = "buffer/mangrove_2020_500m_hex_grid.shp"
    
    print("Reading shapefile...")
    gdf = gpd.read_file(input_file)
    original_crs = gdf.crs
    
    print("Projecting to local UTM coordinate system...")
    try:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    except Exception:
        print("Fallback to EPSG:32649")
        gdf_proj = gdf.to_crs(epsg=32649)
        
    print("Creating 500m hexagon grid...")
    hex_grid_proj = create_hex_grid(gdf_proj, 500)
    
    print("Intersecting grid with original boundaries...")
    intersected_proj = gpd.sjoin(hex_grid_proj, gdf_proj, how="inner", predicate="intersects")
    intersected_proj = intersected_proj.drop_duplicates(subset="geometry")
    
    print("Projecting back to original CRS...")
    final_grid = intersected_proj.to_crs(original_crs)
    
    print("Adding GridID...")
    final_grid = final_grid.reset_index(drop=True)
    final_grid.insert(0, 'GridID', final_grid.index + 1)
    
    print("Saving to file...")
    final_grid.to_file(output_file)
    print("Success! Grid saved.")

if __name__ == "__main__":
    main()
