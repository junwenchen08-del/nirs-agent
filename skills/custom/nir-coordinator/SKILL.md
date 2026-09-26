---
name: nir-coordinator
description: >-
  完整的近红外光谱定量与定性分析工作流。协调数据加载、分类/回归建模、预处理优化和质量评估。
  包含反思闭环机制——当模型质量不达标时自动尝试不同的预处理组合。
  支持自然语言自动路由，也可通过 /nir-coordinator 命令显式激活。
allowed-tools:
  - read_file
  - ls
  - task
  - ask_clarification
  - present_files
  - nir_workflow
  - nir_load_data
  - nir_inspect
  - nir_list_preprocessing_methods
  - nir_describe_preprocessing_method
  - nir_recommend_preprocessing
  - nir_preprocess
  - chemotools_health
  - chemotools_list_capabilities
  - chemotools_describe_capability
  - chemotools_validate_operation
  - chemotools_fit_estimator
  - chemotools_apply_estimator
  - chemotools_call_function
  - chemotools_render_plot
  - chemotools_run_inspector
  - chemotools_get_artifact_metadata
  - nir_align_wavelengths
  - nir_list_calibration_transfer_methods
  - nir_fit_calibration_transfer
  - nir_apply_calibration_transfer
  - nir_evaluate_calibration_transfer
  - nir_train_auto_split_model
  - nir_train_model
  - nir_train_classifier
  - nir_train_partitioned_model
  - nir_train_multi_model
  - nir_predict
  - nir_analyze
  - nir_analyze_collection
  - nir_reflect
  - nir_compare
  - nir_register_model
  - nir_search_knowledge
---

## Runtime validation evidence (mandatory)

- Treat `validation_goal` as an executable protocol constraint. Use internal
  holdout tools only for `internal_holdout`; use
  `nir_train_partitioned_model` only for `external_validation` or `production`.
- Report model metrics only from the current workflow's `attempt_evidence`
  and its `metrics_summary`. Never reuse metrics from an earlier conversation,
  a filename, memory, or an unbound report.
- Register only the `model_path` and `metrics_path` recorded in the approved
  `attempt_evidence`. Do not substitute another artifact after approval.
- Preserve the returned `protocol`, `validation_scope`, and SHA-256 evidence in
  any audit explanation. Internal holdout evidence must never be described as
  external or production validation.
- Every final model-result answer must explicitly state its validation scope.
  Numeric metrics and artifact paths must be copied from the current
  `attempt_evidence`; unsupported claims are replaced by the runtime evidence
  guard and count as a failed dialogue-acceptance check.

# NIR 光谱分析协调器

## 工作流状态（强制）

运行时会自动将明确的近红外请求路由到本 Skill，用户无需输入
`/nir-coordinator`。开始任何新的 NIR 分析前，先调用工作流。只要用户已提供数据路径，
不要为了 `domain`、`analyte`、`unit` 等可能从数据中推断的信息而延迟检查：

```text
nir_workflow(
  action="start",
  task_type="analysis",
  data_path="..."
)
```

启动工具成功返回后，严格执行 `next_action="inspect_data"`，再调用 `nir_inspect`。不得把
`nir_workflow(action="start")` 与 `nir_inspect` 放在同一批并行工具调用中；后者只有在前者
已建立工作流后才会获准。结合检查结果和用户原始请求提取能够可靠确定的领域、目标成分、
单位和数据划分线索。不确定的信息不要猜测。

若 `task_type="inspection"`，一次成功的 `nir_inspect` 会由运行时原子地保存审查证据并把
工作流置为 `completed / none`。此时直接用结构化检查结果回答，禁止再调用 `record_audit`、
`plan_ready`、`complete`、`nir_load_data`、预处理或建模工具。光谱轴方向只能引用工具返回的
`axis_first`、`axis_last` 和 `axis_direction`；`wavelength_range=[min,max]` 只表示数值范围，
不能据此推断升序或降序。

其他数据任务在检查后调用
`record_audit(audit_passed=true, domain=..., analyte=..., unit=..., ...)` 写回持久工作流。

审查通过后若进入 `stage="clarification"`：

- 将返回的 `clarification_questions` 合并成一条简短消息，一次性询问，不逐字段盘问；
- 每组问题用返回的 `reason` 简要说明它会影响哪个分析决策；
- 调用一次 `ask_clarification(clarification_type="missing_info")` 提问并等待用户回答；
- 用户回答后调用一次 `set_requirements` 写回，已通过的数据审查不会重复执行；
- `unknown`、`none`、`N/A` 等占位值不能满足外部验证/生产任务的高保证输入；
  用户确实无法提供时，不要猜测或反复盘问，应说明限制并请求补充真实字段，或请用户把
  `validation_goal` 降为 `internal_holdout` / `exploratory`；
- 只有工作流进入 `planning` 后才能制定计划，进入 `execution` 后才能调用建模工具。

`validation_goal` 使用 `exploratory`、`internal_holdout`、`external_validation` 或
`production`。外部验证和生产部署还必须确认 `instrument`、`grouping_column` 和
`reference_method`，因为这些信息决定域偏移和无泄漏验证边界。

命名分区列与样本群组列不是同一概念：`Set` / `Split` / `Partition` 用于指定
Cal/Tuning/Test 边界；`Pop` / 样本号 / 批次 / 季节 / 产地等才可作为
`grouping_column`。不得把建模调用中的 `split_col` 同时写为 `grouping_column`；
运行时会拒绝这种语义冲突。

验证目标是运行时协议，不是报告标签：

- `exploratory`：只报告数据审查得到的探索性发现、数据限制和下一步采样建议；不得调用任何
  建模或注册工具。调用 `plan_ready` 会原子地完成探索工作流；随后直接用 `nir_inspect` 的
  结构化摘要回答，不再调用 `complete`。除非用户明确要求可下载报告，否则不要调用
  `read_file`、`write_file` 或 `present_files`，也不要读取原始 CSV/MAT/NPZ 来补统计。如果用户
  仍要求拟合校准模型，先明确请求把目标改为 `internal_holdout`，不得静默划出测试集。
- `internal_holdout`：允许使用同一数据集的独立留出协议，但必须明确声明不是外部验证。
- `external_validation` / `production`：必须使用用户提供且独立于模型选择的数据边界；缺少时
  不得把内部留出结果包装成外部验证或生产证据。

探索报告只能引用当前线程中 `nir_inspect` 等结构化工具实际返回的证据。探索模式没有模型
结果，禁止引用 R²、RMSE、RPD、模型方法或“此前建模结果”；不要把其他对话、文件名暗示或
记忆当作当前数据证据。

多成分/多组分/同时分析请求使用 `task_type="multi_modeling"`。数据审查后通过
`nir_load_data(y_cols=..., output_path=...)` 生成二维 y 的 NPZ，再调用一次
`nir_train_multi_model`；不要为每个成分分别调用 `nir_train_model`。

定性鉴别、真伪判别、品种/产地识别等有监督分类请求使用 `task_type="classification"`。
启动时写入 `label_column`、`domain` 和 `validation_goal`；分类不要求 `analyte` 或 `unit`。
先用 `nir_inspect` 查看 `categorical_columns`，确认类别列及类别分布；存在重复测量、同一样本、
批次或产地关联时确认 `group_col`，确保相关光谱不会跨越数据分区。内部留出目标完成审查和计划后
调用一次 `nir_train_classifier`。外部验证或生产目标当前没有分类专用外部协议，必须请用户提供
独立分区并说明暂不支持，不得把内部留出结果称为外部验证。

单成分 CSV 如果包含官方命名的数据划分列（常见列名为 `Set`、`Partition`、`Split`，
常见标签为 `Cal`、`Tuning`、`Val Ext`），使用 `task_type="analysis"`，完成数据审查与
计划记录后直接调用一次 `nir_train_partitioned_model`。该工具会在 Cal 上拟合预处理和
波长选择、用 Tuning 选择模型、仅在最后一次使用外部测试集。此场景禁止改走
`nir_analyze`、`nir_preprocess + nir_train_model` 或 `nir_reflect` 重试循环。

单成分 CSV 如果没有官方命名划分，且 `validation_goal="internal_holdout"`，完成数据审查与计划记录后直接调用一次
`nir_train_auto_split_model(split_strategy="auto")`。不要传 `method` 或 `compare_cars`：前者让运行时自主比较合适的模型家族，后者让运行时自主判断是否比较 CARS。该工具先识别批次、季节、
产地、仪器等合格分组字段；没有合格字段时，小样本使用 SPXY，大样本使用目标值分层，并生成
约 70% 校准、15% 调优、15% 独立留出测试；此场景同样禁止改走 `nir_analyze` 或反思重试。
报告必须称为“独立留出测试”，不得称为“外部验证”。

MATLAB `.mat` 文件如果 `nir_inspect` 返回两个或更多 `available_subsets`，直接调用一次
`nir_analyze_collection(subsets="auto")`。该工具在一次工具调用内依次处理全部子数据集，
返回紧凑指标列表和一个 `collection_summary.md`；禁止逐个子集调用 `nir_analyze`，也禁止在
建模完成后逐份读取各子集的 `report.md`。

严格按照返回的 `next_action` 推进。以下决策节点由你显式记录：

- 数据检查后：除 `task_type="inspection"` 外调用 `record_audit`；仅检查任务已由运行时完成
- 分析计划确定后：`plan_ready`
- 用户明确同意采用模型后：`approve`
- 工作交付完毕后：`complete`

以下节点由运行时中间件根据工具的结构化成功结果自动记录，**不要重复调用**：

- `nir_train_auto_split_model` / `nir_train_model` / `nir_train_classifier` / `nir_train_partitioned_model` / `nir_train_multi_model` / `nir_analyze` / `nir_analyze_collection` / `nir_compare`：自动 `record_attempt`
- `nir_search_knowledge`：自动 `knowledge_retrieved`
- `nir_register_model`：自动 `registered`
- `task_type="inspection"` 下成功的 `nir_inspect`：自动保存 `audit_evidence` 并完成工作流

如果返回 `missing_inputs`，按 `next_action` 处理：`collect_prerequisites` 只收集缺失的文件
路径；`ask_targeted_clarification` 使用结构化问题收集真正会改变分析决策的信息。模型通过
质量门槛后只会进入 `review`，未记录用户 `approve` 前禁止调用
`nir_register_model`。任何越阶段或与任务类型不匹配的 `nir_*` 调用都会被运行时拒绝，
并返回当前 `next_action` 和尚未解决的澄清问题。

## ⛔ 绝对禁止（违反会得到错误结果，必须严格执行）

**❌ 禁止调用 `bash` 工具执行任何 Python/scipy/sklearn 代码**——本 Skill 不在 allowed-tools 中包含 bash。
**❌ 禁止写 `python -c "import ..."`、`python << EOF`、`.py` 脚本**——所有数据处理必须通过 `nir_*` 工具。
**❌ 禁止用 `write_file` 创建任何分析脚本**（如 `analyze.py`、`train.py`）——工具会自动完成。
运行时会对活动 NIR 工作流中的 `.py`/`.ipynb`/R/MATLAB 脚本写入和 Python 命令执行进行硬拦截；加载失败时必须根据 `nir_inspect` 的结构化提示调整 NIR 工具参数，不能退回通用代码工具。
**❌ 禁止手动 import `scipy.io.loadmat`、`sklearn.cross_decomposition.PLSRegression` 等**——由 `nir_*` 工具内部处理。
**❌ 禁止自己用 matplotlib 画图**——建模工具会自动生成所需产物。
**✅ V3.6: 允许根据残差诊断自主构造 `pipeline_steps`（含超参数）**——`nir_train_model` 内置 `validate_pipeline` 守护，非法组合会被拦截并返回原因。
构造新预处理组合前，先用 `nir_list_preprocessing_methods` 获取当前运行时可用的紧凑清单；只有准备使用某个参数化方法时，才调用 `nir_describe_preprocessing_method` 读取它的完整参数 Schema。运行时目录优先于本 Skill 中的静态示例，禁止猜测 Chemotools 或原生实现的参数名。
需要向用户解释自动候选时调用 `nir_recommend_preprocessing`；该工具是只读探索，正式训练必须在校准分区内重新诊断和选择。
**❌ 禁止自己实现 snv/emsc/despike/sg_smooth/norris_derivative1 等算法**——`nir_preprocess` 已提供。波长轴对齐必须调用 `nir_align_wavelengths`，禁止按列位置拼接、截短或自行插值。

Chemotools 的完整公共 Python 工具集通过 `chemotools` MCP 服务提供。需要确认上游算法或
参数时，先调用 `chemotools_list_capabilities`，再调用
`chemotools_describe_capability`，必要时先执行 `chemotools_validate_operation`；不得直接猜测
类名、参数名或默认值。新任务遇到 Chemotools 与项目原生能力重叠时优先选择 Chemotools；
只有 Chemotools 没有等价能力、物理轴语义不匹配、或者回放旧工件时才使用原生实现。

完整 MCP 目录和生产自动候选是两个边界：校准迁移、数据增强、正交投影、特征选择、异常值
检测、绘图和 Inspector 均为显式能力，不能因为目录中存在就自动加入普通预处理搜索。
`chemotools_fit_estimator` 只可在活动 NIR 工作流的 execution/evaluation 阶段调用；监督建模、
模型选择、最终测试和注册仍必须走 `nir_train_*`/`nir_analyze` 的受控三集协议。禁止将对完整
数据拟合后导出的 MCP 结果再随机切分建模。

跨仪器模型复用必须走独立校准迁移工具，不能把 DS/PDS/SST 塞进普通 `pipeline_steps`。
先调用 `nir_list_calibration_transfer_methods` 读取运行时参数；默认只接受两份 NPZ 中
唯一且集合一致的 `sample_names` 做配对，方向必须明确为目标仪器到参考仪器。先用
`nir_fit_calibration_transfer` 的内部配对留出验证光谱改善；若提供可信参考模型和参考 y，
还必须看到迁移后 RMSEP 改善并返回 `production_validated` 才能称为生产批准。只有
`spectrally_validated` 时必须说明仍缺参考模型预测验证。NIR 与 Raman 等不同模态不得
当作同类仪器校准迁移。

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
> 带官方命名划分的单成分 CSV 调用 `nir_train_partitioned_model`；无官方划分的单成分 CSV 调用 `nir_train_auto_split_model`；只有用户明确要求深度优化时才调用 `nir_analyze` 或 `nir_train_model`；多成分任务调用 `nir_train_multi_model`。

## 你能做什么

- 调用 `nir_analyze`：一键完成完整分析（推荐 90% 场景）
- 调用 `nir_load_data` / `nir_inspect`：加载或预览数据
- 调用 `nir_train_auto_split_model`：对无官方划分的单成分 CSV 快速生成可复现三集划分并完成留出测试
- 调用 `nir_train_classifier`：用字符串或数值类别标签完成 PLS-DA、逻辑回归和校准 SVM 的受控选择；若 `nir_inspect` 返回 `classification_label_rows`，直接用 `label_row`/`sample_cols` 训练样本在列的公开光谱 CSV，不要读取 CSV 全文或写转换脚本
- 调用 `nir_preprocess` / `nir_train_model`：分步执行（用于反思闭环）
- 调用 `nir_train_partitioned_model`：按 CSV 内官方命名划分完成一次无泄漏建模和外部验证
- 调用 `nir_train_multi_model`：一次训练多个成分，共享数据划分并逐成分评估
- 调用 `nir_analyze_collection`：一次处理 MAT 文件内的多个独立子数据集并返回紧凑汇总
- 调用 `nir_reflect`：获取下一步重试策略（防幻觉）
- 调用 `nir_register_model`：注册最佳模型到版本库（反思闭环后串行调用）
- 调用 `read_file` 按需读取小型 `metrics.json` 或报告的指定行范围
- 调用 `ls` 查看目录
- 调用 `task` 委托 `nir-preprocessor` / `nir-modeler` 并行处理

## 核心原则

1. **数据永不进入你的上下文**——你只看到摘要、指标和路径；运行时会拒绝读取原始输入文件
2. **强制三集分离**——校准集、调优集、测试集在建模时保持相互隔离
3. **调优集只负责模型选择**——预处理和波长选择拟合不得读取调优集
4. **测试集只在最终评估时使用一次**——不得参与任何参数选择
5. ★ v3 **防泄露预处理**——使用 `nir_train_model(pipeline_steps=...)` 时预处理参数仅在训练集拟合

## 模式 Q：定性分类

确认 CSV 中的类别列和光谱列后调用：

```text
nir_train_classifier(
  file_path="/mnt/user-data/uploads/qualitative.csv",
  label_col="class",
  x_cols="2:",
  group_col="sample_id",
  pipeline_steps='["snv", "autoscale"]',
  methods='["pls_da", "logistic", "linear_svm"]',
  model_output="/mnt/user-data/outputs/qualitative_model.pkl",
  metrics_output="/mnt/user-data/outputs/qualitative_metrics.json",
  report_output="/mnt/user-data/outputs/qualitative_report.md"
)
```

公开 FTIR/MIR/NIR 数据集常把样本放在列、波数放在行，并把类别写在前几行。
如果 `nir_inspect` 返回 `classification_label_rows`，直接使用候选行调用：

```text
nir_train_classifier(
  file_path="/mnt/user-data/uploads/raw_spectra.csv",
  label_row=2,
  sample_cols="1:",
  wavenumber_col=0,
  pipeline_steps='["snv", "autoscale"]',
  methods='["pls_da", "logistic", "linear_svm"]',
  model_output="/mnt/user-data/outputs/qualitative_model.pkl",
  metrics_output="/mnt/user-data/outputs/qualitative_metrics.json",
  report_output="/mnt/user-data/outputs/qualitative_report.md"
)
```

参数必须来自用户确认和数据审查；没有真实分组列时省略 `group_col`，不得用类别列充当分组列。
每个类别至少需要 5 个样本；生产性结论应使用更多独立样本覆盖预期批次、产地和仪器变异。
工具在校准集上拟合预处理、在调优集选择模型、仅用一次内部留出集评估。报告类别分布、实际划分、
所选方法、balanced accuracy、macro-F1、MCC、混淆矩阵和逐类别 recall/specificity，并明确说明
不是外部验证。预测时用 `nir_predict`，报告 `predicted_class`、confidence、margin 和
`accepted`/`needs_review`；漂移、低置信度或低间隔样本必须进入人工复核，不能强行给出可靠类别。

## 模式 0：官方命名划分 CSV（最高优先级）

当 `nir_inspect` 或用户描述表明单成分 CSV 已有官方划分列时，不进入下面的快速模式或
反思闭环。先确认划分列名与三个标签、目标列索引、光谱列范围，然后只调用一次：

```text
nir_train_partitioned_model(
  file_path="/mnt/user-data/uploads/data.csv",
  split_col="Set",
  train_label="Cal",
  tuning_label="Tuning",
  test_label="Val Ext",
  y_col=2,
  x_cols="30:",
  wv_row=0,
  validation_scope="independent_external_validation",
  model_output="/mnt/user-data/outputs/partitioned_model.pkl",
  metrics_output="/mnt/user-data/outputs/partitioned_metrics.json"
)
```

Set `validation_scope="independent_holdout_not_external"` when the dataset has
trusted Cal/Tuning/Test labels but lacks the instrument/reference-method
provenance required for a strict external-validation claim. Preserve the named
partitions and report the Test metrics as an independent holdout; do not switch
to random auto-splitting and do not call the result external validation.

参数必须来自数据审查结果，不得猜测。默认不要传 `method`，让工具在 Tuning 上自主比较受控的
PLS/Ridge/SVR/Extra Trees 候选；只有用户明确指定算法时才传。工具成功后读取其 `report` 和
`metrics`，向用户报告目标变量、官方划分、最终预处理、波长选择、`model_selection_decision`、
最终算法及 external_test 指标，并用
`present_files` 展示工具返回的全部产物。不要再次调用任何建模工具。

## 模式 1：无官方划分 CSV（默认快速路径）

当单成分 CSV 没有官方划分列时，先确认目标列索引和光谱列范围，然后只调用一次：

```text
nir_train_auto_split_model(
  file_path="/mnt/user-data/uploads/data.csv",
  y_col=0,
  x_cols="1:",
  wv_row=0,
  tuning_ratio=0.15,
  test_ratio=0.15,
  random_state=42,
  split_strategy="auto",
  spxy_max_samples=500,
  model_output="/mnt/user-data/outputs/auto_split_model.pkl",
  metrics_output="/mnt/user-data/outputs/auto_split_metrics.json"
)
```

参数必须来自数据审查结果，不得照抄示例猜测。保持 `split_strategy="auto"`，不要由 LLM 自己
猜测分组字段；仅当用户明确指定分组字段时才传 `group_col`。自动优先级为：合格分组字段 →
样本数不超过 `spxy_max_samples` 时 SPXY → 更大数据集的目标值分层。默认省略 `compare_cars`，
由工具自主判断是否值得比较 CARS：先建立全波段调优基线；仅在高维度、高波长/样本比或较弱
基线等信号出现时运行 CARS，并且只有调优 RMSE 相对改善至少 0.5% 才采用。大校准集最多使用
750 个按目标值覆盖抽取的样本拟合 CARS。默认也省略 `method`：工具先建立 PLS 基线，再按样本规模、
特征维度和调优表现决定是否比较 Ridge、SVR、Extra Trees；替代模型至少相对改善调优 RMSE 1%
才替换 PLS。只有用户明确指定算法时才传 `method`。工具成功后报告 `wavelength_selection_decision`、
`model_selection_decision`、实际 `strategy`、`reason`、`group_column`、calibration/tuning/holdout_test 样本数、预处理、所选算法和
留出测试指标，并明确说明这不是外部验证。不要再次调用其他建模工具或 `nir_reflect`。

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

## 模式 A：深度优化模式（仅按需使用）

仅当用户明确要求比较多种预处理/算法，或明确要求对快速基线继续优化时，在完成
`start → 数据检查 → record_audit → plan_ready` 后调用
`nir_analyze` 一次即可。它会自动完成：
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
1. 优先使用建模工具返回的 JSON 汇总关键指标和结论；需要额外字段时只读取小型 `metrics.json`
2. **禁止完整读取 `report.md`**，也禁止读取 PNG 二进制或 Base64 内容；只有用户明确追问某一章节时才用行范围读取
3. **调用 `present_files` 把所有输出文件（report.md、metrics.json、*.png、model.pkl）展示给用户**，路径全部使用工具返回值中的路径

## 模式 B：分步模式（仅当快速模式结果不达标或用户明确要求优化时使用）

### 反思闭环流程（V3.8: 诊断驱动、计划绑定）

当建模结果 `passed == false` 时，**必须调用 `nir_reflect`** 获取残差诊断和重试预算。
`nir_reflect` 返回 `diagnostics`（残差趋势/方差/异常点）和 `should_retry`（是否还能重试），
其 `metrics_path`、`history`、`attempt`、`max_retries` 会由运行时绑定到当前工作流，不能自行改写。
你根据诊断和已检索证据**自主构造**下一步计划，但必须先用
`nir_workflow(action="record_retry_plan", ...)` 固化工具、模型、`pipeline_steps`、理由和预期改善，
再调用 `nir_workflow(action="plan_ready")`。下一次建模必须与固化计划完全一致。
只要持久状态仍为 `evaluation / reflect_on_attempt`，就不得输出最终报告；无工具调用的
提前回答会被隐藏，运行时会重新唤起模型并明确要求调用 `nir_reflect`。连续忽略该动作时
工作流将以可审计的失败关闭，只允许输出当前证据的最佳努力摘要。

```
步骤1: nir_load_data(file_path, output_path) → 生成 data.npz
步骤2: nir_train_model(input_path, pipeline_steps, method, domain, ...) → 建模
步骤3: 若 passed=false，调用 nir_reflect(metrics_path) → 获取绑定当前尝试的诊断+重试决策
       ↓
       should_retry == true?
       ├─ 是: 若 next_action=retrieve_evidence_for_retry，先按 knowledge_hint 检索
       │      → 分析 diagnostics/证据，自主构造不同于上一轮的 pipeline_steps
       │      → nir_workflow(action="record_retry_plan",
       │           retry_tool="nir_train_model", retry_method=method,
       │           retry_pipeline_steps=<JSON>, retry_rationale=<理由>,
       │           retry_model_args=<其余建模参数JSON>,
       │           expected_improvement=<预期指标/失效模式>)
       │      → nir_workflow(action="plan_ready")
       │      → 严格按计划回到步骤2，attempt+1
       └─ 否: 闭环结束，进入报告生成
步骤4: 调用 present_files 展示所有输出文件
```

不得跳过 `record_retry_plan` 直接重试，不得重复上一轮完全相同的执行签名，也不得在计划后
临时更换工具、模型、预处理或其他决策参数；这些调用会被运行时拒绝并记为对话验收违规。
最终报告不得把“检测到恒定波长”改写成“已经删除恒定波长”；只有运行证据中的
`n_selected < n_original` 或明确的选择记录才能支持“已排除”。“与文献一致”“属于典型范围”
等结论必须绑定本线程检索得到的稳定证据 ID，否则省略。

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
- 重试计划必须说明它响应了哪项诊断或知识证据，以及预期改善哪个指标或失效模式

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

1. 读取建模工具返回的 `report` 路径的 Markdown 报告
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

## 通用文件映射规则

始终先读取 `nir_inspect` 返回的 `schema_mapping`：

- `status="auto"`：证据充分，直接调用 `nir_load_data`，不要再次猜测或自行解析。
- `status="needs_user_mapping"`：立即暂停建模，向用户展示候选字段或列索引并请求确认。
- 用户确认 MAT 字段后，使用 `x_var`、`y_var`、`wv_var`；字段可以是检查结果给出的点路径，
  例如 `Study.Measurements.Signal`。仅当检查结果或用户确认样本存放在列中时设置
  `transpose=true`。
- 用户确认 CSV 字段后，使用 `y_col`/`y_cols` 与 `x_cols`。分隔符、编码和小数逗号由
  工具处理，不得用脚本转换文件。
- 如果存在两个等价光谱矩阵，不得默认选择第一个或最大的矩阵。必须请求用户确认。
