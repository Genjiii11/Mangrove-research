# PDP
from pygam import LinearGAM, s
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# ======================
# 科研风格参数设置
# ======================
plt.style.use('default')
rcParams.update({
    'font.family': 'Times New Roman',  # 改为Times New Roman字体
    'font.size': 10,  # 基础字体加大1号
    'axes.labelsize': 14,  # 坐标轴标签字体加大
    'axes.titlesize': 16,  # 标题字体加大
    'xtick.labelsize': 14,  # x轴刻度标签加大
    'ytick.labelsize': 14,  # y轴刻度标签加大
    'legend.fontsize': 12,  # 图例字体加大
    'lines.linewidth': 2,  # 加粗曲线
    'grid.linewidth': 0.7,  # 网格线加粗
    'lines.markersize': 6,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.spines.top': True,  # 显示所有边框
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
})

# ======================
# 精美科研配色方案
# ======================
COLOR_PALETTE = {
    'main': '#3b5b92',  # 主色 - 优雅深蓝色
    'ci': '#8395b1',  # 置信区间 - 柔和蓝灰色
    'positive': '#36a168',  # 正SHAP - 翡翠绿色
    'negative': '#e05263',  # 负SHAP - 玫瑰红色
    'data': '#e0e0e0',  # 数据点 - 淡灰色
    'zero_line': '#666666',  # 零线 - 中灰色
    'tipping_point': '#f0746e',  # 临界点 - 珊瑚色
    'background': '#f9f9f9'  # 背景色 - 几乎白色背景
}


# ======================
# 增强的GAM拟合函数
# ======================
def fit_enhanced_gam(X, y):
    gam = LinearGAM(s(0, n_splines=20, spline_order=3, lam=3)).gridsearch(X, y)
    return gam


# ======================
# 科研风格绘图函数
# ======================
def plot_scientific_style(ax, XX, y_pred, ci, feature_name, tipping_points=None, p_value=None):
    """全边框科研风格绘图"""

    # 设置背景色
    ax.set_facecolor(COLOR_PALETTE['background'])
    fig = plt.gcf()
    fig.patch.set_facecolor('white')

    # 1. 绘制零线
    ax.axhline(0, color=COLOR_PALETTE['zero_line'],
               linestyle=(0, (5, 2)), linewidth=1.2,
               alpha=0.9, zorder=2)

    # 2. 绘制置信区间（渐变填充）- 移除标签
    ax.fill_between(XX.flatten(), ci[:, 0], ci[:, 1],
                    color=COLOR_PALETTE['ci'],
                    alpha=0.4, zorder=3,
                    edgecolor='none')

    # 3. 主趋势线 - 移除标签
    ax.plot(XX, y_pred, color=COLOR_PALETTE['main'],
            linewidth=2.5, zorder=5)

    # 4. 临界点标记和坐标 - 只显示数值
    if tipping_points:
        for i, (x, y) in enumerate(tipping_points):
            ax.axvline(x, color=COLOR_PALETTE['tipping_point'],
                       linestyle='--', linewidth=1.5, alpha=0.8)
            ax.scatter(x, y, color=COLOR_PALETTE['tipping_point'],
                       s=70, zorder=6, edgecolor='white',
                       linewidth=1)

            # 添加坐标文本注释 - 只显示数值
            ax.annotate(f'{x:.2f}',
                        xy=(x, y),
                        xytext=(10, 10),  # 文本偏移，避免重叠
                        textcoords='offset points',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor='white',
                                  alpha=0.8,
                                  edgecolor=COLOR_PALETTE['tipping_point']))

    # 5. 正负区域半透明渐变 - 保留标签
    pos_mask = y_pred > 0
    if any(pos_mask):
        ax.fill_between(XX.flatten(), 0, y_pred,
                        where=pos_mask,
                        color=COLOR_PALETTE['positive'],
                        alpha=0.15, zorder=0,
                        label='Positive')  # 保留正区域的图例标签

    neg_mask = y_pred <= 0
    if any(neg_mask):
        ax.fill_between(XX.flatten(), 0, y_pred,
                        where=neg_mask,
                        color=COLOR_PALETTE['negative'],
                        alpha=0.15, zorder=0,
                        label='Negative')  # 保留负区域的图例标签

    # 6. 删除次刻度线
    ax.tick_params(which='minor', size=0)

    # 7. 添加坐标轴标签
    ax.set_xlabel(feature_name, labelpad=5, fontweight='bold')
    ax.set_ylabel('SHAP Value', labelpad=5, fontweight='bold')

    # 8. 设置坐标轴边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 9. 添加p值
    if p_value is not None:
        p_text = f"p < 0.01" if p_value < 0.01 else f"p = {p_value:.3e}"
        ax.text(0.95, 0.95, p_text,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

    # 10. 显示图例，只显示Positive和Negative
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='none')
# 使用


# 遍历所有特征
# for n in range(0,len(df_test_list_X.columns)):
feature_num=12     #遍历4个特征
if feature_num>len(df_test_list_X.columns):
    feature_num=len(df_test_list_X.columns)
for n in range(0,feature_num):
    feature=df_test_list_X.columns[n]
    print(f"Processing feature: {feature}")

    # 准备数据
    X_feature = df_test_list_X[feature].values.reshape(-1, 1)
    y_shap = shap_values.values[:,n].reshape(-1, 1)

    # 增强GAM拟合
    gam = fit_enhanced_gam(X_feature, y_shap)
    R_squared = gam.statistics_['pseudo_r2']['explained_deviance'] * 100

    # 提取p值
    p_value = gam.statistics_['p_values'][0]  # 假设该特征是模型中的第一个

    # 生成预测网格
    XX = gam.generate_X_grid(term=0, n=300)
    y_pred = gam.predict(XX)
    ci = gam.prediction_intervals(XX, width=0.95)

    # 查找临界点
    zero_crossings = np.where(np.diff(np.sign(y_pred)))[0]
    tipping_points = [(XX[i][0], y_pred[i]) for i in zero_crossings if i < len(XX)]


    # 创建图形（正方形比例）
    fig, ax = plt.subplots(figsize=(5, 4))

    # 专业绘图
    plot_scientific_style(ax, XX, y_pred, ci, feature,
                            tipping_points=tipping_points,
                            p_value=p_value)

    # 添加统计信息
    ax.text(0.05, 0.95,
                f"$R^2$ = {R_squared:.1f}%",
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9,
                            edgecolor='none', pad=3))

    # 设置坐标轴范围
    y_padding = 0.15 * np.ptp(y_shap)
    y_min = min(ci[:, 0].min(), y_shap.min()) - y_padding
    y_max = max(ci[:, 1].max(), y_shap.max()) + y_padding
    ax.set_ylim([y_min, y_max])

    # 保存图形
    plt.tight_layout()
    plt.show()
    # fig.savefig(f"{output_dir}/{feature}_shap_plot.pdf")
    # plt.close(fig)

