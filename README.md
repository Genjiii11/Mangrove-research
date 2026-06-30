# Mangrove Research Data and Code

本仓库保存红树林研究相关的原始代码。大体量数据文件不直接放入 GitHub，避免仓库过大；数据文件清单见 `DATA_MANIFEST.csv`，完整数据归档在 Zenodo 草稿中。

## 内容

1. Python 脚本和 Jupyter 笔记本。
2. 研究数据清单。
3. 运行说明和数据来源说明，后续可继续补充。

## 数据

原始数据和中间表格包括土地利用、红树林掩膜、人口、夜间灯光、植被指数、保护地和生态系统服务价值相关表格。海岸线 shapefile 不随本归档发布。

Zenodo 链接：https://zenodo.org/records/21064003

## 复现

本目录没有统一的构建或测试流程。常用方式是直接运行对应脚本，例如：

```bash
python calculate_yearly_esv.py --output-dir .
python calculate_transition_esv.py --output-dir .
python plot_esv_time_series.py --input yearly_class_area_esv.csv --output mangrove_esv_time_series.png
```

运行前请先从 Zenodo 下载数据，并保持原目录结构。
