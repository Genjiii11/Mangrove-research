import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from matplotlib.lines import Line2D
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
COLOR_SCHEMES = {
    1: {'nodes': plt.cm.RdBu_r, 'edges': plt.cm.PRGn},
}
scheme_index = 1

# 样式方案
STYLE_SCHEMES = {
    1: {'marker': 'o', 'linestyle': '-'},
}
style_index = 1

current_color_scheme = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])
current_style_scheme = STYLE_SCHEMES.get(style_index, STYLE_SCHEMES[1])

# --- 1. 定义变量 ---
dependent_var = 'MHI'
independent_vars = [
    'cropland_percent', 
    'impervious_surface_percent', 'water_percent',
    'ntl_mean', 'proportion_of_landscape', 'patch_density', 'edge_density',
    'mean_annual_temp', 'coldest_month_temp', 'total_annual_precip',
    'mean_annual_lst_day', 'mean_annual_sst', 'dist_to_coast',
    'water_occurrence', 'population_mean'
]
data_path = 'final_environmental_vars_with_MHI.csv'

print("--- 步骤 1/5: 准备工作完成 ---")

# --- 2. 数据加载与预处理 ---
print("\n--- 步骤 2/5: 数据加载与预处理 ---")
try:
    df = pd.read_csv(data_path)
    cols_to_keep = independent_vars + [dependent_var]
    df = df[cols_to_keep]
    df.dropna(subset=[dependent_var], inplace=True)
    X = df[independent_vars]
    y = df[dependent_var]
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=independent_vars)
    print("数据预处理完成。")
except FileNotFoundError:
    print(f"错误: 数据文件未找到 '{data_path}'")
    exit()

# --- 3. 模型训练 ---
print("\n--- 步骤 3/5: 模型训练 ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("模型训练完成。")

# --- 4. SHAP交互分析 ---
print("\n--- 步骤 4/5: SHAP交互分析 ---")
explainer = shap.TreeExplainer(rf_model)
# 使用 batch 计算以通过 tqdm 显示进度
batch_size = 100
n_samples = X_test.shape[0]

shap_values_list = []
for i in tqdm(range(0, n_samples, batch_size), desc="Calculating SHAP Values"):
    batch = X_test.iloc[i:i+batch_size]
    shap_values_list.append(explainer.shap_values(batch))
shap_values = np.concatenate(shap_values_list, axis=0)

shap_interaction_values_list = []
for i in tqdm(range(0, n_samples, batch_size), desc="Calculating SHAP Interaction Values"):
    batch = X_test.iloc[i:i+batch_size]
    shap_interaction_values_list.append(explainer.shap_interaction_values(batch))
shap_interaction_values = np.concatenate(shap_interaction_values_list, axis=0)
print("SHAP 交互值计算完成。")

# 节点数据
feature_importance_abs = np.abs(shap_values).mean(axis=0)
feature_importance_signed = shap_values.mean(axis=0)

# 边线数据
mean_interaction_matrix_abs = np.abs(shap_interaction_values).mean(axis=0)
np.fill_diagonal(mean_interaction_matrix_abs, 0)

mean_interaction_matrix_signed = shap_interaction_values.mean(axis=0)
np.fill_diagonal(mean_interaction_matrix_signed, 0)

# --- 5. 绘制网络图 ---
print("\n--- 步骤 5/5: 绘制交互网络图 ---")

# --- 变量重命名映射 ---
rename_dict = {
    'cropland_percent': 'Cropland',
    'impervious_surface_percent': 'Impervious Surface',
    'water_percent': 'Water Body',
    'ntl_mean': 'Nighttime Light',
    'proportion_of_landscape': 'Landscape Proportion',
    'patch_density': 'Patch Density',
    'edge_density': 'Edge Density',
    'mean_annual_temp': 'Mean Annual Temp',
    'coldest_month_temp': 'Min Temp of Coldest Month',
    'total_annual_precip': 'Annual Precipitation',
    'mean_annual_lst_day': 'Mean Ann. LST Day',
    'mean_annual_sst': 'Mean Ann. SST',
    'dist_to_coast': 'Dist. to Coast',
    'water_occurrence': 'Water Occurrence',
    'population_mean': 'Population Density'
}

def plot_circular_interaction(features, importance_abs, importance_signed,
                              interaction_matrix_abs, interaction_matrix_signed):
    cmap_nodes = current_color_scheme['nodes']
    cmap_edges = current_color_scheme['edges']
    node_marker = current_style_scheme['marker']
    edge_linestyle = current_style_scheme['linestyle']
    
    # 获取显示用的特征名称
    display_features = [rename_dict.get(f, f) for f in features]
    
    fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'aspect': 'equal'})
    
    n_features = len(features)
    G = nx.Graph()
    # 使用显示名称作为节点
    G.add_nodes_from(display_features)
    
    pos = nx.circular_layout(G)
    # 稍微增加标签距离，避免遮挡
    label_pos = {k: (v * 1.15) for k, v in pos.items()}
    
    # 颜色归一化
    norm_edges = mcolors.Normalize(vmin=interaction_matrix_signed.min(),
                                   vmax=interaction_matrix_signed.max())
    
    max_interaction_abs = np.max(interaction_matrix_abs)
    max_importance_abs = np.max(importance_abs)
    
    # 收集交互
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            strength_abs = interaction_matrix_abs[i, j]
            strength_signed = interaction_matrix_signed[i, j]
            if strength_abs > 0:
                interactions.append((display_features[i], display_features[j], strength_abs, strength_signed))
    
    interactions.sort(key=lambda x: x[2])
    
    # 绘制边
    for u, v, strength_abs, strength_signed in interactions:
        color = cmap_edges(norm_edges(strength_signed))
        # 调整线条宽度范围，最大宽度从8减小到5，使其更精致
        width = 0.5 + (strength_abs / max_interaction_abs) * 5
        alpha = 0.3 + (strength_abs / max_interaction_abs) * 0.7
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                              edge_color=[color], style=edge_linestyle,
                              alpha=alpha, ax=ax)
    
    # 绘制节点
    norm_nodes = mcolors.Normalize(vmin=importance_signed.min(),
                                   vmax=importance_signed.max())
    
    node_colors = []
    node_sizes = []
    for i, feat in enumerate(features):
        imp_sign = importance_signed[i]
        imp_abs = importance_abs[i]
        node_colors.append(cmap_nodes(norm_nodes(imp_sign)))
        # 调整节点大小范围，增加基础大小，稍微减小最大增量
        node_sizes.append(300 + (imp_abs / max_importance_abs) * 1500)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          node_shape=node_marker, ax=ax)
    
    # 绘制标签
    for node, (x, y) in label_pos.items():
        ha = 'left' if x > 0 else 'right'
        # 增大字体
        plt.text(x, y, node, size=14, horizontalalignment=ha, verticalalignment='center')
    
    ax.axis('off')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    #plt.title('SHAP Interaction Network - Mangrove Health Index', y=0.95, fontsize=16)
    
    # 线条粗细图例
    line_levels = [max_interaction_abs, max_interaction_abs * 0.5, max_interaction_abs * 0.1]
    line_labels = [f"{val:.2f}" for val in line_levels]
    legend_lines = [Line2D([0], [0], color='black', linewidth=0.5 + (val/max_interaction_abs)*5, linestyle=edge_linestyle) 
                   for val in line_levels]
    
    legend1 = ax.legend(legend_lines, line_labels, loc='center left',
                       bbox_to_anchor=(-0.15, 0.8),
                       title="Interaction Strength\n(Line Width)",
                       frameon=False, labelspacing=1.5, fontsize=12, title_fontsize=13)
    ax.add_artist(legend1)
    
    # 节点大小图例
    node_levels = [max_importance_abs, max_importance_abs * 0.5, max_importance_abs * 0.1]
    node_labels = [f"{val:.2f}" for val in node_levels]
    legend_nodes = []
    for val in node_levels:
        # 更新图例中节点大小计算公式以匹配绘图
        s = np.sqrt((300 + (val/max_importance_abs)*1500) / np.pi) * 2
        legend_nodes.append(Line2D([0], [0], marker=node_marker, color='w',
                                   markerfacecolor='black', markersize=s, linestyle='None'))
    
    ax.legend(legend_nodes, node_labels, loc='center left',
             bbox_to_anchor=(-0.15, 0.32),
             title="Feature Importance\n(Node Size)",
             frameon=False, labelspacing=3, fontsize=12, title_fontsize=13,
             handletextpad=2)
    
    # 边颜色条
    cbar_edge_pos = [0.85, 0.55, 0.015, 0.25]
    cax_edge = fig.add_axes(cbar_edge_pos)
    sm_edge = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges)
    sm_edge.set_array([])
    cbar_edge = plt.colorbar(sm_edge, cax=cax_edge)
    cbar_edge.set_label('Interaction Value (Signed)', rotation=270, labelpad=15, fontsize=12)
    cbar_edge.outline.set_visible(False)
    
    # 节点颜色条
    cbar_node_pos = [0.85, 0.20, 0.015, 0.25]
    cax_node = fig.add_axes(cbar_node_pos)
    sm_node = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
    sm_node.set_array([])
    cbar_node = plt.colorbar(sm_node, cax=cax_node)
    cbar_node.set_label('Feature Value (Signed)', rotation=270, labelpad=15, fontsize=12)
    cbar_node.outline.set_visible(False)
    
    # 保存
    save_path_png = f"rf_shap_interaction_style{style_index}_scheme{scheme_index}.png"
    save_path_pdf = f"rf_shap_interaction_style{style_index}_scheme{scheme_index}.pdf"
    plt.savefig(save_path_png, dpi=600, bbox_inches='tight')

    print(f"图像已保存至: {save_path_png} 和 {save_path_pdf}")
    plt.show()

if __name__ == "__main__":
    print("-" * 30)
    print("特征重要性排序")
    print("-" * 30)
    df_importance = pd.DataFrame({
        '特征': independent_vars,
        '重要性 (平均绝对值)': feature_importance_abs,
        '影响方向 (平均值)': feature_importance_signed
    })
    df_importance = df_importance.sort_values(by='重要性 (平均绝对值)', ascending=False)
    print(df_importance.to_string(index=False))
    
    print("-" * 30)
    print("SHAP 交互作用强度排序")
    print("-" * 30)
    interaction_list = []
    n_features = len(independent_vars)
    
    for i in tqdm(range(n_features), desc="Sorting Interactions"):
        for j in range(i + 1, n_features):
            strength = mean_interaction_matrix_abs[i, j]
            direction = mean_interaction_matrix_signed[i, j]
            if strength > 0:
                interaction_list.append({
                    '特征 1': independent_vars[i],
                    '特征 2': independent_vars[j],
                    '交互作用强度 (平均绝对值)': strength,
                    '交互作用影响方向 (平均值)': direction
                })
    
    df_interactions = pd.DataFrame(interaction_list)
    if not df_interactions.empty:
        df_interactions = df_interactions.sort_values(by='交互作用强度 (平均绝对值)', ascending=False)
        print(df_interactions.head(15).to_string(index=False))
    else:
        print("无显著交互作用。")
    
    plot_circular_interaction(independent_vars,
                             feature_importance_abs,
                             feature_importance_signed,
                             mean_interaction_matrix_abs,
                             mean_interaction_matrix_signed)