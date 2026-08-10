import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 基于之前的定义：
# df_train_list_X=X_train_nor_df

# df_test_list_X=X_test_nor_df

# explainer=shap.Explainer(model_get,df_train_list_X)

# shap_values=explainer(df_test_list_X)

# shap_values_list.append(shap_values)

# mean_shap_value=abs(shap_values.values)

# mean_shap_value1=mean_shap_value.mean(0)

# mean_shap_value_m.append(mean_shap_value1)
# labels = df_top.columns

# sizes = mean_shap_value1[top_indices]
# top_indices = np.argsort(mean_shap_value1)[::-1][:max_display]
# df_top = df_test_list_X.iloc[:, top_indices]
# shap_values_top = shap_values[:, top_indices]
# 假设 sizes、labels、colors_corrected 已经准备好
fig, ax = plt.subplots(figsize=(7, 6))
# 生成颜色映射
cmap_name = "RdGy"  # 可以换成 viridis, coolwarm 等
cmap = plt.get_cmap(cmap_name)
colors = cmap(np.linspace(0, 1, len(ax.patches)))
colors_corrected = colors[::-1]
sizes_rev = sizes[::-1]
labels_rev = labels[::-1]
colors_rev = colors_corrected[::-1]
# ------------------ 横向柱状图 ------------------
bars = ax.barh(range(len(sizes)), sizes_rev, color=colors_rev, alpha=0.7)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels_rev, fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=12, fontweight="bold")
ax.set_title("Top Features - SHAP Horizontal Bar Chart with Embedded Donut", fontsize=12, fontweight="bold")

# ------------------ 嵌入环形饼图 ------------------
# inset_axes 参数：[x0, y0, 宽, 高]，坐标范围 0~1，相对于 ax
ax_inset = inset_axes(ax, width="60%", height="60%", loc='center right')

wedges, texts, autotexts = ax_inset.pie(
    sizes,
    labels=None,               # 嵌入图不显示标签，避免重叠
    autopct='%1.1f%%',
    startangle=45,
    colors=colors_corrected,
    wedgeprops=dict(width=0.4, alpha=0.7),
    textprops={'fontsize': 10},
    pctdistance=1.1
)

# 调整百分比字体
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_weight("bold")

ax_inset.set_title("SHAP %", fontsize=10, fontweight="bold")
ax_inset.axis('equal')  # 保证饼图圆形

plt.tight_layout()
plt.show()