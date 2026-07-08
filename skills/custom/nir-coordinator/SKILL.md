---
name: nir-coordinator
description: >-
  完整的近红外光谱分析工作流。协调数据加载、预处理优化、模型建立和质量评估。
  包含反思闭环机制——当模型质量不达标时自动尝试不同的预处理组合。
  通过 /nir-coordinator 命令激活。
allowed-tools:
  - read_file
  - write_file
  - ls
  - task
  - present_files
  - nir_load_data
  - nir_inspect
  - nir_preprocess
  - nir_train_model
  - nir_predict
  - nir_analyze
  - nir_reflect
  - nir_compare
  - nir_register_model
---

# NIR 光谱分析协调器

## ⛔ 绝对禁止（违反会得到错误结果，必须严格执行）

**❌ 禁止调用 `bash` 工具执行任何 Python/scipy/sklearn 代码**——本 Skill 不在 allowed-tools 中包含 bash。
**❌ 禁止写 `python -c "import ..."`、`python << EOF`、`.py` 脚本**——所有数据处理必须通过 `nir_*` 工具。
**❌ 禁止用 `write_file` 创建任何分析脚本**（如 `analyze.py`、`train.py`）——工具会自动完成。
**❌ 禁止手动 import `scipy.io.loadmat`、`sklearn.cross_decomposition.PLSRegression` 等**——由 `nir_*` 工具内部处理。
**❌ 禁止自己用 matplotlib 画图**——`nir_analyze` / `nir_train_model` 自动生成 PNG。
**❌ 禁止自己编造预处理流水线组合**——必须调用 `nir_reflect` 获得确定性的下一步。
**❌ 禁止自己实现 snv/sg_smooth/derivative1 等算法**——`nir_preprocess` 已提供。

> ⚠️ **反例警告**：曾经在 LLM 自己写完整 Python 脚本（手写 SNV/SG/MSC 算法 + 三集分离 + 嵌套CV + PLS建模 + 反射闭环）时失败，因为：
> 1. 沙箱中 `nir_core` 未安装 → `import nir_core` 失败
> 2. 自己实现的算法与标准化学计量学实践不一致
> 3. 错误地全量预处理后再划分 → 数据泄露
> 4. 浪费 token 写长代码而不是专注于决策
>
> **任何 NIR 任务的第一选择都是调用 `nir_analyze` 或 `nir_train_model` 工具**。

## 你能做什么

- 调用 `nir_analyze`：一键完成完整分析（推荐 90% 场景）
- 调用 `nir_load_data` / `nir_inspect`：加载或预览数据
- 调用 `nir_preprocess` / `nir_train_model`：分步执行（用于反思闭环）
- 调用 `nir_reflect`：获取下一步重试策略（防幻觉）
- 调用 `nir_register_model`：注册最佳模型到版本库（反思闭环后串行调用）
- 调用 `read_file` 读取工具生成的 `report.md`、`metrics.json` 或 PNG 图片
- 调用 `ls` 查看目录
- 调用 `task` 委托 `nir-preprocessor` / `nir-modeler` 并行处理

## 核心原则

1. **数据永不进入你的上下文**——你只看到摘要、指标和路径
2. **强制三集分离**——训练集、验证集、测试集在建模时自动分离
3. **验证集绝不参与任何参数选择**——只有训练集用于 CV 和搜索
4. **测试集只在最终评估时使用一次**
5. ★ v3 **防泄露预处理**——使用 `nir_train_model(pipeline_steps=...)` 时预处理参数仅在训练集拟合

## ★ v3 按需知识加载

不要在开始时读取全部知识库文档。按需读取：

- 如果用户提到 **土壤** / **soil** / **有机碳** / **SOC**：读取 `/mnt/skills/custom/nir-knowledge/docs/soil-tips.md`
- 如果用户提到 **食品** / **玉米** / **小麦** / **谷物** / **food**：读取 `/mnt/skills/custom/nir-knowledge/docs/food-tips.md`
- 如果模型出现 **过拟合** (RMSEP > 2×RMSECV)：读取 `/mnt/skills/custom/nir-knowledge/docs/troubleshooting.md`
- 如果需要 **指标解读** 参考：读取 `/mnt/skills/custom/nir-knowledge/docs/metrics-interpretation.md`
- 化学计量学硬规则始终生效（已内联在本 SKILL 中），无需额外读取

## 开始分析前

在处理任何 NIR 分析请求前，请先阅读知识库文档：
1. 读取 `/mnt/skills/custom/nir-knowledge/docs/chemometrics-rules.md`（必读）
2. 根据任务类型选择性阅读其他文档

## 模式 A：快速模式（强烈推荐优先使用）

对绝大多数请求，直接调用 `nir_analyze` 一次即可。它会自动完成：
加载 → 三集分离 → 嵌套 CV 预处理选择 → 建模 → 评估 → 生成报告和图。

调用示例：
```
nir_analyze(
  data_path="/mnt/user-data/uploads/synthetic_protein.mat",
  auto_preprocess=true,
  method="pls",
  domain="food_protein",
  output_dir="/mnt/user-data/outputs/nir_analysis"
)
```

**返回的 JSON 包含**：
- `status`, `method`, `n_components`, `preprocessing`
- `R2_val`, `RPD`, `RMSEP`
- `grade`, `passed`, `action`, `thresholds_used`
- `report`: Markdown 报告路径
- `model`: 序列化模型路径
- `metrics`: 完整指标 JSON 路径

**调用后你必须**：
1. 用 `read_file` 读取 `report` 路径的 Markdown 报告
2. 把报告内容展示给用户（关键指标和结论）
3. **调用 `present_files` 把所有输出文件（report.md、metrics.json、*.png、model.pkl）展示给用户**，路径全部使用工具返回值中的路径

## 模式 B：分步模式（仅当快速模式结果不达标或用户明确要求优化时使用）

### 反思闭环流程（使用 `nir_reflect` 确定性驱动）

每次 `nir_train_model` 完成后，**必须调用 `nir_reflect`** 来获得确定性的重试决策。不要自己解读指标或决定下一个预处理组合——`nir_reflect` 会基于 `should_retry` + `get_next_pipeline` + `suggest_lv_adjustment` 给出明确指令。

```
步骤1: nir_load_data(file_path, output_path) → 生成 data.npz
步骤2: nir_preprocess(input_path, method, output_path) → 预处理
步骤3: nir_train_model(input_path, method, domain, model_output, metrics_output) → 建模
步骤4: nir_reflect(metrics_path, history, domain, attempt) → 获得重试决策
       ↓
       should_retry == true?
       ├─ 是: 按 next_pipeline 调用 nir_preprocess → 回到步骤3，attempt+1
       └─ 否: 闭环结束，进入报告生成
步骤5: 调用 present_files 展示所有输出文件
```

### `nir_reflect` 调用示例

第1次尝试（原始光谱建模后）：
```
nir_reflect(
  metrics_path="/mnt/user-data/outputs/metrics.json",
  history="[]",
  domain="food_protein",
  attempt=1
)
```

第2次尝试（SNV预处理后，history 带上第1次记录）：
```
nir_reflect(
  metrics_path="/mnt/user-data/outputs/metrics.json",
  history='[{"pipeline":[{"method":"raw"}],"metrics":{"R2_val":0.72,"RPD":2.1}}]',
  domain="food_protein",
  attempt=2
)
```

`nir_reflect` 返回的关键字段：
- `should_retry`: 是否应继续重试
- `next_pipeline`: 下一个应尝试的预处理方法列表（如 `["snv","sg_smooth","mean_center"]`）
- `lv_adjustment.suggested_n`: 建议的成分数（过拟合时减少，欠拟合时增加）
- `reason`: 中文决策理由

### 反思重试约束

- 最多 3 次重试（由 `nir_reflect` 的 `max_retries` 控制）
- 连续两次 R² 改善 < 0.02 时自动停止（plateau 检测）
- 候选流水线耗尽时停止
- 你**不需要**自己判断是否达标——`nir_reflect` 会告诉你

## 质量门禁（按领域分级）

阈值由 `nir_train_model` / `nir_analyze` 自动处理，你只需读取返回的 `grade`、`passed`、`action`、`thresholds_used`。

| 领域 | R²_min | RPD_min | 说明 |
|------|--------|---------|------|
| food_moisture | 0.90 | 4.0 | 水分（通常容易） |
| food_protein | 0.85 | 3.5 | 蛋白质 |
| pharma | 0.85 | 3.0 | 制药 |
| feed | 0.75 | 2.5 | 饲料 |
| soil | 0.70 | 2.0 | 土壤（难度大） |
| default | 0.80 | 3.0 | 默认 |

注：样本量 < 100 时自动放宽。

## 并行预处理比较

### 方式 1：使用 `nir_compare` 一键比较（推荐）

当用户要求比较多种预处理方法时，直接调用 `nir_compare`，它会自动完成所有流水线的加载→预处理→建模→评估→对比画廊生成：

```
nir_compare(
  data_path="/mnt/user-data/uploads/synthetic_protein.mat",
  pipelines='[["snv","sg_smooth","mean_center"],["msc","derivative1","autoscale"],["airpls","sg_smooth","snv","mean_center"]]',
  method="pls",
  domain="food_protein",
  output_dir="/mnt/user-data/outputs/nir_comparison"
)
```

返回最佳流水线 + 对比画廊 HTML + Markdown 摘要。

### 方式 2：使用 `task` 委托子代理并行执行

当需要更灵活的编排（如不同建模方法、不同数据子集）时：
```
task(
  description="SNV预处理",
  prompt="对 /mnt/user-data/outputs/data.npz 应用 SNV，输出到 /mnt/user-data/outputs/out_snv.npz",
  subagent_type="nir-preprocessor"
)
```

并行启动多个子代理后，对每个结果调用 `nir_train_model`，再汇总比较。

## 输出规范

分析完成后，**必须执行以下步骤**：

1. 读取 `nir_analyze` 或 `nir_train_model` 返回的 `report` 路径的 Markdown 报告
2. 向用户展示：数据概览、预处理方法、模型指标、质量结论、部署建议
3. **必须调用 `present_files` 把所有输出文件展示给用户**，让用户可以直接在界面中查看和下载。

示例：
```
present_files(
  filepaths=[
    "/mnt/user-data/outputs/nir_analysis/report.md",
    "/mnt/user-data/outputs/nir_analysis/metrics.json",
    "/mnt/user-data/outputs/nir_analysis/predicted_vs_reference.png",
    "/mnt/user-data/outputs/nir_analysis/raw_spectra.png",
    "/mnt/user-data/outputs/nir_analysis/residuals.png",
    "/mnt/user-data/outputs/nir_analysis/cv_curve.png",
    "/mnt/user-data/outputs/nir_analysis/model.pkl",
  ]
)
```

> **关键**：`present_files` 只接受 `/mnt/user-data/outputs/` 下的文件路径。NIR 工具的默认输出目录已改为 `/mnt/user-data/outputs/nir_analysis`，直接使用返回值中的路径即可。

4. 不要自己重新组织或计算指标——直接使用工具返回的数据
