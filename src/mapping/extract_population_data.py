import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import rasterio
import os
import numpy as np

def extract_population_data():
    """
    根据1km六边形缓冲区网格提取多年的人口数据，并保存为CSV。
    此版本会动态读取栅格的nodata值，并对结果进行清洗。
    """
    # 1. 定义文件路径
    shapefile_path = 'buffer/mangrove_2020_500m_hex_grid.shp'
    output_csv_path = 'population_data.csv'
    
    # 人口栅格数据和对应的年份
    population_files = {
        2000: 'POP/chn_ppp_2000_1km_Aggregated.tif',
        2005: 'POP/chn_ppp_2005_1km_Aggregated.tif',
        2010: 'POP/chn_ppp_2010_1km_Aggregated.tif',
        2015: 'POP/chn_ppp_2015_1km_Aggregated.tif',
        2020: 'POP/chn_ppp_2020_1km_Aggregated.tif',
        2024: 'POP/chn_ppp_2024_1km_Aggregated.tif'
    }

    # 检查所有输入文件是否存在
    if not os.path.exists(shapefile_path):
        print(f"错误: Shapefile 未找到 at '{shapefile_path}'")
        return
    for year, path in population_files.items():
        if not os.path.exists(path):
            print(f"错误: {year} 年的人口栅格文件未找到 at '{path}'")
            return

    # 2. 加载六边形网格 shapefile
    print(f"正在加载 shapefile: {shapefile_path}")
    grid = gpd.read_file(shapefile_path)

    # 3. 确保坐标参考系 (CRS) 为 WGS84 (EPSG:4326)
    if grid.crs.to_epsg() != 4326:
        print("正在将 shapefile 的 CRS 转换为 EPSG:4326...")
        grid = grid.to_crs(epsg=4326)

    # 4. 计算每个六边形的质心坐标
    print("正在计算质心...")
    grid['lon'] = grid.geometry.centroid.x
    grid['lat'] = grid.geometry.centroid.y
    
    grid_info = grid[['GridID', 'lon', 'lat']].copy()

    # 5. 循环处理每年的栅格数据
    all_years_data = []
    
    for year, raster_path in population_files.items():
        print(f"正在处理 {year} 年的人口数据: {raster_path}")
        
        # 动态地从栅格文件的元数据中获取 nodata 值
        nodata_val = None
        try:
            with rasterio.open(raster_path) as src:
                nodata_val = src.nodata
                if nodata_val is not None:
                    print(f"  从元数据中读取到 nodata 值为: {nodata_val}")
                else:
                    # 如果元数据中没有定义，则使用默认值
                    nodata_val = -9999
                    print(f"  警告: 栅格文件元数据中未定义 nodata 值，将使用默认值: {nodata_val}")
        except Exception as e:
            print(f"  读取栅格元数据时出错: {e}。将使用默认 nodata 值 -9999。")
            nodata_val = -9999

        # 6. 执行区域统计 (Zonal Statistics)
        try:
            stats = zonal_stats(
                grid, 
                raster_path, 
                stats="mean", 
                geojson_out=True,
                nodata=nodata_val
            )
            
            # 7. 提取统计结果并格式化
            year_data = []
            for feature in stats:
                properties = feature['properties']
                grid_id = properties['GridID']
                pop_mean = properties.get('mean')
                
                # 数据清洗：如果人口为负数，则视为无效数据 (NaN)
                if isinstance(pop_mean, (int, float)) and pop_mean < 0:
                    pop_mean = np.nan
                
                year_data.append({
                    'GridID': grid_id,
                    'year': year,
                    'population_mean': pop_mean
                })
            
            all_years_data.extend(year_data)
            print(f"{year} 年数据处理完成。")

        except Exception as e:
            print(f"处理 {year} 年数据时出错: {e}")

    # 8. 整合数据
    if not all_years_data:
        print("没有数据被处理，无法生成CSV。")
        return
        
    print("正在整合所有年份的数据...")
    final_df = pd.DataFrame(all_years_data)
    
    # 合并质心坐标
    final_df = pd.merge(final_df, grid_info, on='GridID')
    
    # 调整列顺序
    final_df = final_df[['GridID', 'year', 'lon', 'lat', 'population_mean']]

    # 9. 保存为 CSV 文件
    print(f"正在将结果保存到: {output_csv_path}")
    final_df.to_csv(output_csv_path, index=False, na_rep='NaN') # 将NaN明确写入CSV
    
    print(f"任务完成！CSV文件已保存至 {output_csv_path}")

if __name__ == '__main__':
    extract_population_data()