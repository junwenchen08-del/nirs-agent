---
name: nir-gallery
description: >-
  Generate visualization plots for NIR analysis results (spectra, predicted vs
  reference, residuals, drift heatmap, comparison gallery). Activates via /nir-gallery.
allowed-tools:
  - read_file
  - write_file
  - ls
  - nir_inspect
  - nir_analyze
  - nir_compare
  - nir_preprocess
  - nir_train_model
  - present_files
---

# NIR 可视化技能

## 用途
为 NIR 分析结果生成可视化图表。

## 核心原则

**❌ 禁止用 `bash` + `python -c` 写脚本生成图表**
**✅ 使用 `nir_*` 工具的内置绘图功能——它们自动生成 PNG/HTML 并保存到输出目录**

## 可视化类型与对应工具

### 1. 原始光谱图
调用 `nir_inspect` 查看数据结构，然后调用 `nir_preprocess` 对原始光谱做预处理——工具返回中包含光谱摘要统计信息。
```
nir_inspect(file_path="/mnt/user-data/uploads/data.csv")
nir_preprocess(input_path="...", method="snv", output_path="...")
```

### 2. 预测vs参考图 / 残差图
`nir_analyze` 和 `nir_train_model` **已自动生成** `predicted_vs_reference.png`，无需额外画图。
```
nir_train_model(
  input_path="/mnt/user-data/outputs/data.npz",
  pipeline_steps='["snv","sg_smooth","mean_center"]',
  method="pls",
  model_output="/mnt/user-data/outputs/model.pkl",
  metrics_output="/mnt/user-data/outputs/metrics.json"
)
# → 自动生成 predicted_vs_reference.png
```
用 `read_file` 读取 `metrics.json` 获取 R²/RMSE/RPD 等指标，在报告中引用。

### 3. 多模型对比画廊
`nir_compare` **已自动生成** HTML 对比画廊和 Markdown 摘要表。
```
nir_compare(
  input_path="/mnt/user-data/outputs/data.npz",
  pipelines='[["snv","sg_smooth","mean_center"],["msc","derivative1","autoscale"]]',
  methods='["pls","svr"]',
  output_dir="/mnt/user-data/outputs/comparison"
)
# → 自动生成 gallery.html + comparison_table.md
```
用 `present_files` 展示生成的 HTML 文件。

### 4. 质量评估图
`nir_reflect` 返回残差诊断（`diagnostics` 字段），包含 `residual_trend`、`residual_variance`、`outlier_count`，可直接在报告中描述。

## 工作流程

1. **确认数据已加载**：若未加载，先 `nir_load_data`
2. **选择可视化方式**：
   - 单模型分析 → `nir_analyze`（自动生成预测图）
   - 多模型对比 → `nir_compare`（自动生成 HTML 画廊）
   - 分步+反思 → `nir_train_model` + `nir_reflect`（自动生成预测图+诊断）
3. **用 `read_file` 读取** metrics.json / comparison_table.md 获取数值结果
4. **用 `present_files` 展示**生成的 PNG/HTML 文件

## 注意
- 所有 `nir_*` 工具的绘图使用 Agg backend，无需手动设置
- 图表文件保存在 `/mnt/user-data/outputs/` 目录下
- 如需查看已有图表文件，用 `ls` 列出目录，再用 `read_file` 或 `present_files` 展示
