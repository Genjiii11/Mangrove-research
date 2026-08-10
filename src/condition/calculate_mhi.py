import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 定义要进行PCA的植被指数列
vi_columns = [
    'CMRI_mean_mangrove',
    'EMVI_mean_mangrove',
    'EVI_mean_mangrove',
    'MVI_mean_mangrove',
    'NDVI_mean_mangrove',
    'kNDVI_mean_mangrove'
]

# 读取数据
try:
    df = pd.read_csv('final_environmental_vars_imputed.csv')
    print("文件 'final_environmental_vars_imputed.csv' 读取成功。")
    
    # 提取植被指数数据
    vi_data = df.loc[:, vi_columns].copy()
    
    # 检查缺失值
    missing_count = vi_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n警告：检测到 {missing_count} 个缺失值。")
        print("缺失值详情：")
        print(vi_data.isnull().sum())
        
        # 删除含有缺失值的行
        valid_indices = vi_data.dropna().index
        vi_data_clean = vi_data.loc[valid_indices].copy()
        print(f"\n跳过 {len(df) - len(valid_indices)} 行含缺失值的数据。")
        print(f"有效数据行数：{len(valid_indices)}")
    else:
        print("\n未检测到缺失值。")
        valid_indices = vi_data.index
        vi_data_clean = vi_data
    
    # 数据标准化
    scaler = StandardScaler()
    vi_data_scaled = scaler.fit_transform(vi_data_clean)
    print("植被指数数据已标准化。")

    # 执行PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(vi_data_scaled)
    print("主成分分析（PCA）执行完毕。")

    # 创建MHI列（初始化为NaN）
    df['MHI'] = np.nan
    # 仅为有效行赋值
    df.loc[valid_indices, 'MHI'] = principal_component.flatten()
    print("已创建 'MHI' 列。")

    # 保存结果
    output_filename = 'final_environmental_vars_with_MHI.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n处理完成！更新后的数据已保存到 '{output_filename}'。")
    print("新数据框的前几行（包含MHI）：")
    print(df.head())
    print(f"\nMHI 统计信息：")
    print(f"- 有效值数量：{df['MHI'].notna().sum()}")
    print(f"- 缺失值数量：{df['MHI'].isna().sum()}")

except FileNotFoundError:
    print("错误：'final_environmental_vars_imputed.csv' 文件未找到。请确保文件位于正确的目录中。")
except Exception as e:
    print(f"处理过程中发生错误：{e}")