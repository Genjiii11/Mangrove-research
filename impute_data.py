import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 定义文件路径
input_file_path = 'final_environmental_vars_cleaned.csv'
output_file_path = 'final_environmental_vars_imputed.csv'

try:
    # 读取CSV文件
    df = pd.read_csv(input_file_path)
    print(f"成功读取文件: {input_file_path}")

    # 识别所有数值类型的列
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # 找出含有缺失值的数值列
    cols_with_missing_values = [col for col in numeric_cols if df[col].isnull().any()]

    if not cols_with_missing_values:
        print("文件中没有找到需要插值的数值型空值。")
    else:
        print(f"将在以下数值列中进行机器学习插值: {cols_with_missing_values}")

        # 创建一个IterativeImputer实例，使用随机森林回归作为估计器
        # 随机森林对于捕捉复杂的非线性关系非常有效
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            max_iter=500, 
            random_state=42,
            #verbose=2,  # 打印每一轮的迭代信息
            imputation_order='roman'
        )

        # 仅对数值列进行插值
        df_numeric = df[numeric_cols]

        print("开始进行插值计算...")
        # 对数值列进行拟合和转换
        imputed_numeric_data = imputer.fit_transform(df_numeric)
        print("插值计算完成。")

        # 将插值后的数据（numpy数组）转换回DataFrame
        df_imputed_numeric = pd.DataFrame(imputed_numeric_data, columns=numeric_cols, index=df_numeric.index)

        # 更新原DataFrame中的数值列
        df.update(df_imputed_numeric)

        # 将处理后的完整DataFrame保存到新文件
        df.to_csv(output_file_path, index=False)

        print(f"成功对缺失值进行机器学习插值，并已将结果保存到 '{output_file_path}'。")

except FileNotFoundError:
    print(f"错误: 文件 '{input_file_path}' 未找到。")
except Exception as e:
    print(f"处理文件时发生错误: {e}")