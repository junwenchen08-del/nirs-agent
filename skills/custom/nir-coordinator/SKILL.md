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
  - nir_search_knowledge
---

# NIR 光谱分析协调器

## ⛔ 绝对禁止（违反会得到错误结果，必须严格执行）

**❌ 禁止调用 `bash` 工具执行任何 Python/scipy/sklearn 代码**——本 Skill 不在 allowed-tools 中包含 bash。
**❌ 禁止写 `python -c "import ..."`、`python << EOF`、`.py` 脚本**——所有数据处理必须通过 `nir_*` 工具。
**❌ 禁止用 `write_file` 创建任何分析脚本**（如 `analyze.py`、`train.py`）——工具会自动完成。
**❌ 禁止手动 import `scipy.io.loadmat`、`sklearn.cross_decomposition.PLSRegression` 等**——由 `nir_*` 工具内部处理。
**❌ 禁止自己用 matplotlib 画图**——`nir_analyze` / `nir_train_model` 自动生成 PNG。
**✅ V3.6: 允许根据残差诊断自主构造 `pipeline_steps`（含超参数）**——`nir_train_model` 内置 `validate_pipeline` 守护，非法组合会被拦截并返回原因。
**❌ 禁止自己实现 snv/sg_smooth/derivative1 等算法**——`nir_preprocess` 已提供。

### 🚨 CSV / .mat / .txt 文件布局：禁止自己解析

`nir_load_data` **自动检测**典型 NIR 光谱文件布局：
- **CSV 第一行是波长、第一列是 y 值**（NIR 行业惯例）—— `nir_load_data` 看到 `(0,0)` 角是 NaN、其余第一行是 NIR 波长范围数字时，**自动分离** y 和 wv
- **MAT** —— 自动识别 X/y/wv 变量
- 解析成功后，工具 summary 里会包含 `y_separated: true`、`y_first_values: [5.0, 5.25, ...]` 等强证据

**❌ 看到 "structure: samples_in_columns" 也不要自己写 Python 解析**——这只是形状启发式，不是布局类型。
**✅ 看到 `nir_inspect` 输出 `corner_is_nan: true` 时，直接调用 `nir_load_data` 信任工具结果**。

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

## ★ 语义知识检索（论文库）—— 事件驱动触发规则

知识库通过 `nir_search_knowledge` 工具进行语义检索。**触发条件必须是可观测信号**，不是模糊的"当需要时"。

### 必须调用的情况（不是可选）

| 触发信号（来自工具输出） | 查询模板 |
|------------------------|---------|
| `nir_train_model` / `nir_analyze` / `nir_reflect` 返回的 `knowledge_hint != null` | **直接使用 `knowledge_hint.query` 字段**，不要自己构造 |
| `domain` 不在 {food_moisture, food_protein, pharma, feed, soil, default} 内 | `query="<domain> NIR calibration PLS preprocessing"` |
| `nir_reflect` 返回 `grade in {C, D, F}` 且 `attempt >= 2` | 参考 `knowledge_hint.query`（已按 diagnostics 生成） |
| `diagnostics.residual_trend == "upward"` 且重试1次后仍无改善 | `query="airpls baseline correction <domain> NIR"` |
| `diagnostics.residual_trend == "downward"` 且重试1次后仍无改善 | `query="snv vs msc scatter correction <domain> NIR"` |
| `diagnostics.residual_variance == "high"` 且重试1次后仍无改善 | `query="sg_smooth window selection <domain> NIR noise"` |
| `R²_val < 0.7` | `query="typical R2 RPD PLS <domain> NIR"` |
| 用户问"为什么 R² 这么低" / "正常吗" / "合理范围" | `query="typical R2 RPD PLS <domain> NIR"` |
| 用户在 SNV vs MSC / airPLS vs asLS / derivative1 vs derivative2 之间二选一 | `query="SNV vs MSC <domain> NIR"` 等 |

### 禁止调用的情况

- **同一对话里已经检索过相同或相似 query**（避免重复 token 浪费）
- `grade` 已为 A/B 且用户没有要求优化
- 用户明确要求"快速分析"且结果已通过门禁
- 知识库为空（首次返回 `count: 0` 时不要再检索）

### 调用方式

```
nir_search_knowledge(query="SNV vs MSC soil data scatter correction", top_k=3)
```

**返回内容**：论文段落（截断 1000 字）+ 来源文件 + 相似度分数 + 实体（methods/models/datasets/metrics）+ `available_documents` 列表。

**注意**：知识库为空时返回空结果（非报错）。论文导入方式：`python -m nir_core.knowledge.cli import-dir <论文目录>`

### knowledge_hint 字段说明

V3.7 起，`nir_train_model` / `nir_analyze` / `nir_reflect` 在以下情况下会在返回 JSON 里附带 `knowledge_hint` 字段：

```json
{
  "knowledge_hint": {
    "should_search": true,
    "query": "soil NIR PLS improve RPD preprocessing",
    "reason": "重试 2 次仍未通过门禁 (grade=C)，建议检索论文库寻找该领域提升 RPD 的预处理组合经验。"
  }
}
```

- `should_search == true` 时，**必须**在下一步调用 `nir_search_knowledge(query=hint['query'])`
- `query` 字段已构造好，直接使用，不要自己重写
- `reason` 字段解释为什么建议检索，可以引用到给用户的回复里
- `knowledge_hint == null` 时不强制检索，但可根据上表其他信号主动检索

### 调用示例

**示例 1**：用户上传 soil 数据，初始 grade=C，knowledge_hint 已给出 query
```
步骤1: nir_train_model(pipeline_steps='["snv","mean_center"]', domain="soil")
  → 返回 grade="C", R²_val=0.65, knowledge_hint={should_search:true, query:"soil NIR PLS improve RPD preprocessing"}
步骤2: nir_search_knowledge(query="soil NIR PLS improve RPD preprocessing", top_k=3)
  → 返回论文建议 soil 数据常用 SNV+SG+derivative1
步骤3: 根据论文构造 pipeline_steps='["snv","sg_smooth","derivative1","mean_center"]'
步骤4: nir_train_model(pipeline_steps='["snv","sg_smooth","derivative1","mean_center"]', domain="soil", attempt=2)
步骤5: nir_reflect(metrics_path, history, domain="soil", attempt=2)
  → 若 knowledge_hint 仍非空且 query 不同，再次检索；若 query 相同则跳过
```

**示例 2**：用户问"我的 R²=0.6 正常吗"
```
步骤1: nir_search_knowledge(query="typical R2 RPD PLS soil NIR", top_k=3)
  → 返回 soil 领域 R² 通常 0.7-0.85
步骤2: 基于论文数据回答用户：当前 R² 略低于领域典型范围，建议尝试 SNV+SG 组合
```

**示例 3**：用户上传未知领域数据（domain="textile"）
```
步骤1: nir_analyze(data_path, domain="textile")
  → 返回 knowledge_hint={should_search:true, query:"textile NIR calibration PLS preprocessing"}
步骤2: nir_search_knowledge(query="textile NIR calibration PLS preprocessing", top_k=3)
  → 返回该领域常用预处理和指标范围
步骤3: 基于论文知识判断结果是否合理，必要时进入反思闭环
```

## 开始分析前

在处理任何 NIR 分析请求前，请先阅读知识库文档：
1. 读取 `/mnt/skills/custom/nir-knowledge/docs/chemometrics-rules.md`（必读）
2. 根据任务类型选择性阅读其他文档

## 模式 A：快速模式（强烈推荐优先使用）

对绝大多数请求，直接调用 `nir_analyze` 一次即可。它会自动完成：
加载 → 三集分离 → 嵌套 CV 预处理选择 → 建模 → 评估 → 生成报告和图。

**CSV 文件直接调用（无需先调用 nir_load_data）**：
```
nir_analyze(
  data_path="/mnt/user-data/uploads/corn_moisture.csv",
  auto_preprocess=true,
  method="pls",
  domain="food_moisture",
  output_dir="/mnt/user-data/outputs/nir_analysis"
)
```
`nir_analyze` 内部自动处理 CSV 布局检测（角 NaN 信号识别）→ 分离 y 和 wv → 训练。

如果用 `nir_load_data` + `nir_train_model` 分步模式：
1. 调用 `nir_load_data(file_path, output_path=".../data.npz")` —— 看返回 summary 的 `y_separated: true` 和 `y_first_values: [...]` 确认 y 已分离
2. 调用 `nir_train_model(input_path=".../data.npz", ...)` —— 训练模型
3. **不要自己解析 CSV** —— 工具已自动处理

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

### 反思闭环流程（V3.6: 诊断驱动自主决策）

每次 `nir_train_model` 完成后，**必须调用 `nir_reflect`** 获取残差诊断和重试预算。
`nir_reflect` 返回 `diagnostics`（残差趋势/方差/异常点）和 `should_retry`（是否还能重试），
你根据诊断信息**自主构造**下一步 `pipeline_steps`（含超参数），`nir_train_model` 内置 `validate_pipeline` 守护。

```
步骤1: nir_load_data(file_path, output_path) → 生成 data.npz
步骤2: nir_train_model(input_path, pipeline_steps, method, domain, ...) → 建模
步骤3: nir_reflect(metrics_path, history, domain, attempt) → 获取诊断+重试决策
       ↓
       should_retry == true?
       ├─ 是: <thought> 分析 diagnostics → 自主构造 pipeline_steps → 回到步骤2，attempt+1
       │      仅当连续2次自构造效果不如 best_so_far 时，使用 fallback_suggestion 兜底
       └─ 否: 闭环结束，进入报告生成
步骤4: 调用 present_files 展示所有输出文件
```

### 残差模式-解决方案映射（推理跳板）

根据 `nir_reflect` 返回的 `diagnostics` 选择下一步策略：

| 诊断信号 | 含义 | 建议操作 |
|----------|------|----------|
| `residual_trend: upward` | 高端预测偏低（基线漂移） | 加入 `airpls` 基线校正，`lambda_` 默认 1e6，漂移严重时减小 |
| `residual_trend: downward` | 高端预测偏高（乘性散射） | 加入 `snv` 或 `msc` 散射校正 |
| `residual_variance: high` | 残差离散（噪声大） | 加入 `sg_smooth`，`window` 试 11→15 |
| `outlier_count` 高 | 异常样本干扰 | 尝试 `remove_outliers=True`，或简化流水线 |
| R²提升但RMSEP仍高 | 过拟合 | 减小 `max_components`，或增大 SG 窗口 |

### 超参数调整模板

构造 `pipeline_steps` 时支持 `params` 字段：
```json
[
  {"method": "airpls", "params": {"lambda_": 1000000}},
  {"method": "sg_smooth", "params": {"window": 15, "order": 2}},
  {"method": "snv"},
  {"method": "mean_center"}
]
```

约束（`validate_pipeline` 会拦截违规）：
- SG 窗口：5-15 的奇数
- airPLS lambda：1e3-1e8
- `snv` 和 `msc` 互斥，`derivative1` 和 `derivative2` 互斥
- 方法顺序：基线校正 → 散射校正/平滑 → 缩放

### `nir_reflect` 调用示例

第1次尝试（SNV建模后）：
```
nir_reflect(
  metrics_path="/mnt/user-data/outputs/metrics.json",
  history="[]",
  domain="food_protein",
  attempt=1
)
```

第2次尝试（history 带上第1次记录）：
```
nir_reflect(
  metrics_path="/mnt/user-data/outputs/metrics.json",
  history='[{"pipeline":[{"method":"snv"}],"metrics":{"R2_val":0.72,"RPD":2.1}}]',
  domain="food_protein",
  attempt=2
)
```

`nir_reflect` 返回的关键字段：
- `should_retry`: 是否应继续重试
- `diagnostics`: 残差诊断（`residual_trend`/`residual_variance`/`outlier_count`）
- `fallback_suggestion`: 兜底候选流水线（仅当自构造效果不佳时使用）
- `fallback_suggestion_steps`: 兜底候选的步骤详情
- `lv_adjustment.suggested_n`: 建议的成分数
- `best_so_far`: 历史最优记录
- `reason`: 中文决策理由（含诊断摘要）

### 反思重试约束

- 最多 3 次重试（由 `nir_reflect` 的 `max_retries` 控制）
- 连续两次 R² 改善 < 0.02 时自动停止（plateau 检测）
- 候选流水线耗尽时停止
- `should_retry` 由确定性规则判断——你只需要决定**试什么**

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
