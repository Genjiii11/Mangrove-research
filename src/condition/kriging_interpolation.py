import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import os
import argparse

def interpolate_missing_csv(input_csv, output_csv, target_cols):
    """
    使用普通克里金插值填补 CSV 中指定列的缺失值 (NaN)
    """
    print(f"正在读取数据: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 确保经纬度不存在缺失
    if df['lon'].isna().any() or df['lat'].isna().any():
        print("警告: 存在经度或纬度缺失的行，将被忽略。")
    
    for col in target_cols:
        if col not in df.columns:
            print(f"跳过 {col}: 列不存在于数据中")
            continue
            
        valid_data = df.dropna(subset=['lon', 'lat', col])
        missing_data = df[df[col].isna() & df['lon'].notna() & df['lat'].notna()]
        
        if len(missing_data) > 0:
            print(f"正在对 {col} 的 {len(missing_data)} 个缺失值进行普通克里金插值...")
            
            x_valid = valid_data['lon'].values
            y_valid = valid_data['lat'].values
            z_valid = valid_data[col].values
            
            x_missing = missing_data['lon'].values
            y_missing = missing_data['lat'].values
            
            # 使用普通克里金插值，半方差模型默认使用球面模型(spherical)
            # 根据数据特点，也可选择 linear, power, gaussian, exponential
            try:
                OK = OrdinaryKriging(
                    x_valid, y_valid, z_valid, 
                    variogram_model='spherical',
                    verbose=False,
                    enable_plotting=False,
                    coordinates_type='geographic'
                )
                
                # 'points' 模式表示仅计算指定散点的插值
                # 限制只使用最近的 15 个点来预测，有效减少产生诡异负值的可能
                z_pred, ss_pred = OK.execute('points', x_missing, y_missing, backend='loop', n_closest_points=15)
                
                # 智能截断：如果原数据中该列没有负数，则将插值结果的负数强制置为0
                # 如果是诸如降水、人口等，最低就是0；如果是温度，允许原本就有的负数
                min_valid_val = z_valid.min()
                if min_valid_val >= 0:
                    z_pred = np.clip(z_pred, 0, None)
                else:
                    # 如果允许负数，可以考虑限制在历史最小值的合理范围内
                    z_pred = np.clip(z_pred, min_valid_val - (abs(min_valid_val) * 0.1), None)
                
                # 填补回原数据
                df.loc[missing_data.index, col] = z_pred
                print(f"{col} 插值完成。")
            except Exception as e:
                print(f"{col} 插值失败，错误信息: {e}")
        else:
            print(f"{col} 没有检测到缺失值，无需插值。")

    print(f"正在保存插值结果到: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("完成！")

def create_raster_surface(input_csv, target_cols, resolution=0.01):
    """
    生成连续的栅格表面 (可用于导出 TIF )
    """
    import rasterio
    from rasterio.transform import from_origin
    
    print(f"读取数据: {input_csv}")
    df = pd.read_csv(input_csv)
    
    min_x, max_x = df['lon'].min(), df['lon'].max()
    min_y, max_y = df['lat'].min(), df['lat'].max()
    
    # 构建网格
    grid_x = np.arange(min_x, max_x, resolution)
    grid_y = np.arange(min_y, max_y, resolution)
    
    for col in target_cols:
        if col not in df.columns:
            continue
            
        valid_data = df.dropna(subset=['lon', 'lat', col])
        if len(valid_data) == 0:
            continue
            
        print(f"正在为 {col} 生成空间网格表面 (克里金插值)...")
        x_valid = valid_data['lon'].values
        y_valid = valid_data['lat'].values
        z_valid = valid_data[col].values
        
        OK = OrdinaryKriging(
            x_valid, y_valid, z_valid, 
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
        )
        
        # 执行网格级别的插值
        z_grid, ss_grid = OK.execute('grid', grid_x, grid_y)
        
        # 导出为 TIF
        out_tif = f"{col}_kriging_surface.tif"
        print(f"导出 {out_tif} ...")
        
        transform = from_origin(min_x, max_y, resolution, resolution)
        with rasterio.open(
            out_tif,
            'w',
            driver='GTiff',
            height=z_grid.shape[0],
            width=z_grid.shape[1],
            count=1,
            dtype=str(z_grid.dtype),
            crs='EPSG:4326',  # 假设为 WGS84
            transform=transform,
        ) as dst:
            dst.write(np.flipud(z_grid), 1)

if __name__ == "__main__":
    # 需要插值的列
    target_columns = [
        'mean_annual_temp', 'coldest_month_temp', 'total_annual_precip',
        'mean_annual_lst_day', 'mean_annual_sst', 'dist_to_coast', 'water_occurrence',
        'population_mean'
    ]
    
    input_file = "final_environment.csv"
    output_file = "final_environment_interpolated.csv"
    
    # 默认执行缺失值填补。
    # 如果用户的目的是要生成完整的表面栅格TIF，则可以调用 create_raster_surface
    
    if os.path.exists(input_file):
        interpolate_missing_csv(input_file, output_file, target_columns)
        
        # 若需要生成连续栅格地图，请取消下方代码注释，并确保已安装 rasterio：
        # create_raster_surface(input_file, target_columns, resolution=0.005)
    else:
        print("Done")
