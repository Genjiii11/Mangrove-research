import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
COLOR_SCHEMES = {
    1: {'nodes': plt.cm.RdBu_r, 'edges': plt.cm.PRGn},
}
scheme_index = 1  # 颜色方案选择
# 获取当前颜色方案
current_color_scheme = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])

STYLE_SCHEMES = {
    1: {'marker': 'o', 'linestyle': '-'},
}
style_index = 1 # 形状标记方案
# 获取样式方案
current_style_scheme = STYLE_SCHEMES.get(style_index, STYLE_SCHEMES[1])
# 原始数据路径
file_path = r'mock_data.xlsx'
# 读取数据
df = pd.read_excel(file_path)
# 目标变量
y = df.iloc[:, -1]
# 特征变量
X = df.iloc[:, :-1]
# 获取特征列的名称并转换为列表
features = X.columns.tolist()
print(f"特征: {features}")
print(f"数据类型: {X.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 超参数网格
param_grid = {
    'max_depth': [4, 6, 8],
    # 'learning_rate': [0.05, 0.1, 0.2],
    # 'n_estimators': [50, 100, 150]
}
# 初始化XGBoost 回归模型
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
# 初始化网格搜索对象
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',verbose=1)
# 在训练集上拟合
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
# 获取最佳模型
best_model = grid_search.best_estimator_

# 使用最佳模型创建SHAP树解释器
explainer = shap.TreeExplainer(best_model)
# 测试集的SHAP交互值
shap_interaction_values = explainer.shap_interaction_values(X_test)
# 测试集的SHAP值
shap_values = explainer.shap_values(X_test)

#节点数据
#特征重要性，平均绝对值，点大小
feature_importance_abs = np.abs(shap_values.mean(axis=0))
#特征影响方向性，实际值平均，点颜色
feature_importance_signed = shap_values.mean(axis=0)

#边线数据
#交互强度，平均绝对值，用于控制连线粗细
mean_interaction_matrix_abs = np.abs(shap_interaction_values.mean(axis=0))
np.fill_diagonal(mean_interaction_matrix_abs, 0)  # 忽略自身交互

#交互影响方向，实际值平均，用于控制连线颜色
mean_interaction_matrix_signed = shap_interaction_values.mean(axis=0)
np.fill_diagonal(mean_interaction_matrix_signed, 0)  # 忽略自身交互

def plot_circular_interaction(features, importance_abs, importance_signed,interaction_matrix_abs, interaction_matrix_signed):
    #获取颜色方案
    cmap_nodes = current_color_scheme['nodes']
    cmap_edges = current_color_scheme['edges']
    # 获取节点形状标记
    node_marker = current_style_scheme['marker']
    # 获取连线样式
    edge_linestyle = current_style_scheme['linestyle']
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'aspect': 'equal'})
    # 获取特征的数量
    n_features = len(features)
    # 向图中添加节点
    G.add_nodes_from(features)
    # 生成节点的环形布局坐标
    pos = nx.circular_layout(G)
    # 标签的坐标
    label_pos = {k: (v * 1.1) for k, v in pos.items()}
    # 颜色归一化
    norm_edges = mcolors.Normalize(vmin=interaction_matrix_signed.min(),
                                   vmax=interaction_matrix_signed.max())
    #宽度/大小归一化基准
    max_interaction_abs = np.max(interaction_matrix_abs)
    max_importance_abs = np.max(importance_abs)
    # 初始化交互列表
    interactions = []
    # 遍历特征
    for i in range(n_features):
        for j in range(i + 1, n_features):

            # 如果绝对强度大于 0 (显示阈值)
            if strength_abs > 0:
                # 将交互对、绝对强度、实际强度添加到列表中
                interactions.append((features[i], features[j], strength_abs, strength_signed))
    # 根据绝对强度对交互列表进行排序
    interactions.sort(key=lambda x: x[2])
    # 遍历排序后的交互列表
    for u, v, strength_abs, strength_signed in interactions:
        # 修正点：根据实际值和当前边颜色方案获取线的颜色
        color = cmap_edges(norm_edges(strength_signed))

        # 根据绝对值计算线的粗细
        width = 0.5 + (strength_abs / max_interaction_abs) * 8
        # 线的透明度
        alpha = 0.3 + (strength_abs / max_interaction_abs) * 0.7
        # 绘制线
        nx.draw_networkx_edges(G,
                               pos,
                               edgelist=[(u, v)],
                               width=width,
                               edge_color=[color],
                               style=edge_linestyle,
                               alpha=alpha, ax=ax)
        # --- 节点的处理 ---
    # 节点颜色归一化
    norm_nodes = mcolors.Normalize(vmin=importance_signed.min(),
                                   vmax=importance_signed.max())

    # 遍历每个特征
    for i, feat in enumerate(features):
        # 获取该特征的实际值
        imp_sign = importance_signed[i]
        # 获取该特征的绝对重要性
        imp_abs = importance_abs[i]
        #计算并添加节点颜色
        node_colors.append(cmap_nodes(norm_nodes(imp_sign)))
        # 绘制标签文本
        plt.text(x,
                 y,
                 node,
                 size=12,
                 horizontalalignment=ha,
                 verticalalignment='center')
    # 关闭坐标轴
    ax.axis('off')
    # x轴显示范围
    ax.set_xlim(-1.5, 1.5)
    # y轴显示范围
    ax.set_ylim(-1.5, 1.5)
    # 标题
    plt.title('(a) Green Ecological -> Agricultural Production', y=0.95, fontsize=16)

    # ---------------------------左侧图例-----------------------
    #线条粗细图例的三个等级数值
    line_levels = [max_interaction_abs, max_interaction_abs * 0.5, max_interaction_abs * 0.1]
    #将数值转换为字符串标签，用于图例显示具体数值
    line_labels = [f"{val:.2f}" for val in line_levels]
    legend1 = ax.legend(legend_lines,
                        line_labels,  # 传入格式化后的数值标签列表
                        loc='center left',  #左侧居中
                        bbox_to_anchor=(-0.1, 0.8),  #精确位置
                        title="Interaction Strength\n(Line Width)",  #图例标题
                        frameon=False,  #去掉图例边框
                        labelspacing=1.5)  #图例垂直间距
    #添加到轴上
    ax.add_artist(legend1)
    #定义节点大小等级数值
    node_levels = [max_importance_abs,
                   max_importance_abs * 0.5,
                   max_importance_abs * 0.1]
    #用于图例显示具体数值
    node_labels = [f"{val:.2f}" for val in node_levels]
    legend_nodes.append(Line2D([0],
                                   [0],
                                   marker=node_marker,  #标记形状
                                   color='w',  #线条颜色
                                   markerfacecolor='black',  #点的填充颜色
                                   markersize=s,  #标记点的大小
                                   linestyle='None'))  #不绘制连接线，只显示点
    #添加点图例
    ax.legend(legend_nodes,
              node_labels,  #数值标签列表
              loc='center left',  #左侧居中
              bbox_to_anchor=(-0.1, 0.32),  #精确位置
              title="Feature Importance\n(Node Size)",  #图例标题
              frameon=False,  #去掉图例边框
              labelspacing=3)  #图例垂直间距
    # --- 颜色条 ---
    #定义边颜色条的位置，左,下,宽,高
    cbar_edge_pos = [0.82, 0.55, 0.015, 0.25]
    # 创建一个新的轴用于放颜色条
    cax_edge = fig.add_axes(cbar_edge_pos)
    #设置线的颜色条的标签
    cbar_edge.set_label('Interaction Value (Signed)', rotation=270, labelpad=15, fontsize=10)
    #去掉线的颜色条的轮廓线
    cbar_edge.outline.set_visible(False)
    # --- 颜色条---
    #节点颜色条的位置
    cbar_node_pos = [0.82, 0.20, 0.015, 0.25]
    #绘制节点颜色条
    cbar_node = plt.colorbar(sm_node, cax=cax_node)
    #设置节点颜色条的标签
    cbar_node.set_label('Feature Value (Signed)', rotation=270, labelpad=15, fontsize=10)
    #去掉节点颜色条的轮廓线
    cbar_node.outline.set_visible(False)
    # 保存
    save_path_png = fr"{style_index}_scheme{scheme_index}.png"
    save_path_pdf = fr"{style_index}_scheme{scheme_index}.pdf"
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')

if __name__ == "__main__":
    print("-" * 30)
    print("特征重要性排序")
    print("-" * 30)
    #创建DataFrame对象，用于展示分析结果
    df_importance = pd.DataFrame({
        '特征': features,  #特征列
        '重要性 (平均绝对值)': feature_importance_abs,  #重要性
        '影响方向 (平均值)': feature_importance_signed  #影响方向
    })
    #根据重要性进行降序排序
    df_importance = df_importance.sort_values(by='重要性 (平均绝对值)', ascending=False)
    print(df_importance.to_string(index=False))
    print("-" * 30)
    print("SHAP 交互作用强度排序")
    print("-" * 30)
    # 初始化一个空列表，用于后续存储筛选出来的交互作用数据字典
    interaction_list = []
    # 获取特征的总数量，用于控制后续循环的次数
    n_features = len(features)
    # 开始外层循环，遍历每一个特征的索引 i，范围从 0 到 特征总数-1
    for i in range(n_features):
        # 开始内层循环，遍历i之后的每一个特征索引j，确保只计算组合（不重复计算且不含自身）
        for j in range(i + 1, n_features):
            # 从平均交互矩阵的绝对值中，获取第 i 个和第 j 个特征之间的交互强度
            strength = mean_interaction_matrix_abs[i, j]
            # 从平均交互矩阵的原始值中，获取第 i 个和第 j 个特征之间的交互方向（正负）
            direction = mean_interaction_matrix_signed[i, j]
            # 条件判断：如果交互强度大于0（即存在有效的交互作用），则执行以下代码块
            if strength > 0:
                #向interaction_list列表中追加一个包含当前交互对详细信息的字典
                interaction_list.append({
                    '特征 1': features[i],  # 记录第一个特征的名称
                    '特征 2': features[j],  # 记录第二个特征的名称
                    '交互作用强度 (平均绝对值)': strength,  # 记录该对特征的交互强度
                    '交互作用影响方向 (平均值)': direction  # 记录该对特征的交互方向
                })
    # 将收集了所有交互信息的列表转换为一个Pandas DataFrame，方便后续处理
    df_interactions = pd.DataFrame(interaction_list)
    #如果df_interactions不为空，即找到了至少一个交互作用
    if not df_interactions.empty:
        # 根据交互作用强度这一列进行降序排序，让强交互排在前面
        df_interactions = df_interactions.sort_values(by='交互作用强度 (平均绝对值)', ascending=False)
        print(df_interactions.head(15).to_string(index=False))
    else:
        print("无显著交互作用。")
    #调用绘图函数
    plot_circular_interaction(features,#特征
                              feature_importance_abs,  #节点大小依据
                              feature_importance_signed,  #节点颜色依据
                              mean_interaction_matrix_abs,  #连线粗细依据
                              mean_interaction_matrix_signed)  #连线颜色依据