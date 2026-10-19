# ChemAgent 第一阶段实施报告

日期：2026-10-08。工作目录：`D:\ChemAgent\nirs-agent`。

本轮完成用户限定的第一阶段最小安全搜索路径：单目标回归的 Cal/Val 执行、Test 封存、显式冻结和单次最终评价。PLS、Ridge、SVR、Extra Trees 已接入这一执行路径。尚未开发第二阶段的完整 LLM 决策合同、自主选择与优化循环。

## 1. 实施范围与结论

参考 `chemagent-current-decision-audit.md`、`chemagent-llm-decision-design.md` 和 `chemagent-implementation-roadmap.md`，按本轮要求采用兼容的新增协议，而不是一次性替换全部业务入口：

- 新协议为 `search_protocol="sealed_holdout"`；仅支持 `analysis` / `calibration` 单目标回归及内部留出验证。
- 旧协议默认仍为 `legacy`。已有分类、多目标、MAT 集合及其他算法入口保留；没有改造前端、开展论文对比实验或开发跨任务历史记忆。
- 高层模型族、预处理和冻结对象必须由调用方显式指定。执行器不自动扩展模型族、不根据质量规则决定停止，不在 Test 评价后重新搜索。现阶段只提供第二阶段需要的安全执行基础，不能据此宣称已经实现完整 LLM 自主决策。
- 旧建模入口仍保留每次调用的 Test 评价，以兼容既有业务；新协议通过服务边界和工具权限阻止使用这些入口。**全流程 Test 封存保证适用于新协议，不能推广到所有旧入口。**

六项验收的证据见第 5 节。相关自动化测试通过；全仓库存在的非 NIR 失败和环境限制见第 6 节。

## 2. 拆分任务与实现

### 2.1 指标、质量门禁和反思范围

新增 `search_metrics()`，仅接受明确的 Val 指标，不从 Test、顶层旧 `RPD` / `RMSEP` 或缓存质量结论推断验证性能。`evaluate_quality()` 和 `suggest_lv_adjustment()` 增加可选 `evaluation_scope`：

- `validation`：仅使用 Val 性能、偏差和 Cal CV 证据。
- `final_test`：必须提供 Test 指标，专用于冻结后的评价。
- `legacy`：保留原公共函数默认行为，避免改变所有非适配调用。

回归反思先投影当前与历史记录，忽略没有明确 Val 证据的旧 Test 历史；比较报告的数组合同也可读取。当前与历史最佳方案依据 Val RMSE，潜变量建议使用 Val 偏差与误差。已消费 Test 的结果直接返回搜索关闭，不能再次反思启动搜索。分类反思逻辑保留。

原回归训练入口的搜索质量调用改为明确的 `validation`，并使用 Cal 样本数。`nir_compare` 不再对 Test 做 transform/predict/指标计算；比较排序、图表和报告改用 Val，标为 `test_status="not_evaluated"`。该旧工具仍会加载原数据并沿用旧训练内核，故不属于封存执行路径，在新协议内被阻止。

注册门禁单独核验最终 Test 质量：验证质量通过不能替代最终测试质量通过。原有科学证据、工件完整性和审批等检查保留。

这一小任务先验证质量、建议和历史投影，再运行原质量/重试相关测试；后续增加比较、最终关闭和注册边界测试。

### 2.2 不接收 Test 的四模型执行内核

新增 `fit_search_model(X_cal, y_cal, X_val, y_val, ...)`，**函数签名没有 Test 参数**。实际执行过程：

1. 核验单目标数值数据、Cal/Val 光谱轴与分区分离，以及预处理合法性。
2. 在固定 Cal CV 折内，对每个有限数值参数候选重新拟合整条预处理/模型管线；状态型预处理和模型内部 scaler 只见折训练行。
3. 依据 Cal CV 的平均折 RMSE 选定具体参数。
4. 固定同一组参数，在完整 Cal 上拟合一次，对 Val 评价一次；不调用旧自动选择/重新搜索入口。

PLS 使用受限成分数；Ridge、SVR、ET 复用现有有限网格。ET 固定随机种子并使用单线程。KFold 固定 seed=42；显式分组采用 GroupKFold。Cal 内部跨 CV 折的重复光谱会被拒绝，防止重复样本泄露。

输出范围明确：`train.scope=calibration`、`train_cv.scope=calibration_cv`、`val.scope=validation`、`diagnostics.scope=validation`、`evaluation_scope=validation`、`test_status=sealed`。搜索结果不存在 Test 指标或预测。

`RMSECV` 记录候选的平均折 RMSE；`train_cv.RMSE` 是合并 OOF 预测的 RMSE，两者在折大小不等时可能不同，均来自 Cal CV。质量投影优先使用 OOF 的 RMSE。该执行器没有进行 Cal+Val 重新拟合；冻结的就是已验证的 Cal 拟合模型。

此任务独立测试四个真实 sklearn 模型、每折预处理拟合范围、分组 CV、非法模型、分区重叠和重复光谱边界。

### 2.3 私有分区会话、冻结和 Test 消费

新增 `SearchSessionStore`，复用现有 owner/thread 私有 `ProcessStore`。可信摄取核验三分区的数值形状、光谱重复/重叠和可选组边界，之后写入两个不同的私有文件：

| 文件 | 内容 | 可访问阶段 |
|---|---|---|
| `search.npz` | Cal、Val、波长、可选 Cal/Val 组 | 搜索、最终工件导出 |
| `holdout.npz` | Test X/y | 冻结后的最终评价 |

文件位于线程的 `nir-process/sessions/search_<uuid>/`，在 sandbox 可见的 `user-data` 之外。摄取会接触 Test 以完成合法性、分区和完整性检查，但不会计算 Test 性能、诊断或提供 Test 摘要。返回的 Profile 只基于 Cal，数量只标明 Cal/Val。包含 Test 的原始绑定哈希和 Test 文件哈希保留在服务内部，不进入搜索反馈。

数据库增加 `search_sessions` 和 `search_experiments`，schema 版本从 2 升至 3；已有 v1/v2 的 run、attempt、step、chart 和 event 记录保留。输入、分区、CV 设置、资源预算及 domain 的绑定不可更换。每个 tool call 绑定不可变实验规格，成功重放读取旧结果；失败或运行中的同一请求不能重放拟合。

显式 `freeze(experiment_id)` 只接受当前会话已成功的实验，存在未完成实验时拒绝冻结。冻结后所有新的 search 请求被拒绝。最终评价先在 SQLite 中持久化唯一 `testing` lease，再检查冻结模型与 Test 哈希、加载 Test 并预测。成功结果缓存；再次 finalize 不重新读取 Test。并发最多一个最终评价读取 Test。最终评价失败保持 `final_test_failed`，不能重新发放 Test lease，也不能回到 searching。

每次拟合开始前扣减持久化 fit 预算并调用现有 ResourceBudget checkpoint。上限保留为 200 次 fit；实验数量受工作流 `max_attempts` 约束。没有“失败后自动换算法”的规则。

此任务独立测试真实文件读取边界、Test 目标不变性、冻结前拒绝、消费/并发/失败状态、预算和历史重放。

### 2.4 最小工作流接入和工具绕行阻止

新增四个工具：`nir_prepare_search`、`nir_search_model`、`nir_freeze_search`、`nir_finalize_search`。

- 摄取源只取工作流绑定的 `data_path`，不接受每轮换数据路径。
- 支持数值 NPZ（X/y 或显式命名三分区）及明确指定目标列的 CSV/TXT/TSV。原始未划分数据固定随机划分；现有 `split_dataset` 的 `test_ratio=0.2`、`val_ratio=0.2` 都相对于总量，Cal/Val/Test 实际约为 60%/20%/20%，整数样本数以 sklearn 划分为准。此比例不由后续搜索修改。
- 分组任务仅接受已分离的命名 NPZ 三分区及三组数组，不做原始分组数据的自动划分。
- 新协议中只允许四个执行工具和无数据的方法/能力目录读取工具；旧训练、比较、检查、文件读取、bash/python、task、MCP fit 和未知工具均被中间件阻止。新协议开始后 `nir_workflow` 仅允许 status，防止重启或切换协议绕过封存。
- 光谱上传及配对转换 Markdown 不自动注入内容预览，包括工作流开始前；协议激活后所有上传仅注入元数据，并提示使用可信摄取路径。
- 最终导出复用原模型完整性 manifest、metrics 绑定和报告生成工具，写到新的 `outputs/search_<uuid>/`。导出模型与被冻结模型的预测一致；重复完成的导出复用记录。

增加可选工作流字段和 `frozen` 阶段，不调整已有工具必填参数。仅更新 `config.example.yaml` 的四个工具条目；**没有覆盖现有本地 config.yaml、启动/重启服务或部署。**现有实例启用时需要合并配置并重新加载。

此任务独立测试工具拒绝范围、非适配任务拒绝、NPZ/CSV 的实际工具全链路及预测、旧工件保留和上传中间件兼容性。

## 3. 使用方式与实际能力边界

在尚未接触 Test 的新会话中开始，需一次提供完整任务输入。示例工具参数：

```json
{
  "action": "start",
  "task_type": "calibration",
  "data_path": "/mnt/user-data/uploads/data.npz",
  "domain": "default",
  "analyte": "protein",
  "unit": "%",
  "validation_goal": "internal_holdout",
  "search_protocol": "sealed_holdout",
  "max_attempts": 3
}
```

随后执行 prepare，读取既有 Chemotools 能力目录（保留原目录尝试约束），显式 search，显式选择成功的 `experiment_id` 冻结，再 finalize。PLS 的 method 为 `pls`，其他为 `ridge`、`svr`、`et`；`pipeline_steps` 是显式预处理数组的 JSON 字符串。

本轮验证的是工具、训练内核、隔离服务及实际导出预测的功能链路。没有执行连接真实 LLM/API 的完整对话验收或启动 Web 服务。第二阶段还需将新实验记录接入完整 DecisionContext、LLM 授权及响应证据合同。

## 4. 实际修改文件

以下路径相对于 `D:\ChemAgent\nirs-agent`，新增文件标为“新增”。

| 文件 | 实现内容 |
|---|---|
| `nir_core/model/safe_search.py`（新增） | 无 Test 参数的四模型、每折完整管线执行与 Cal/Val 指标 |
| `nir_core/utils/metrics.py` | 搜索证据投影和明确质量评价范围 |
| `nir_core/utils/validation.py` | Val 范围的潜变量建议，旧默认接口兼容 |
| `nir_core/plotting/gallery.py` | Val 比较的指标与图表范围标识 |
| `backend/packages/harness/deerflow/community/nir/dataset_session.py`（新增） | 私有分区、实验预算/重放、冻结和单次 Test 消费 |
| `backend/packages/harness/deerflow/community/nir/search_tools.py`（新增） | 四个最小安全工具和独立工件导出 |
| `backend/packages/harness/deerflow/community/nir/process.py` | v3 增量数据库迁移 |
| `backend/packages/harness/deerflow/community/nir/workflow.py` | 可选协议字段、限定任务范围、状态显示 |
| `backend/packages/harness/deerflow/agents/thread_state.py` | 协议/会话字段和 frozen 状态类型 |
| `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py` | 工具白名单、阶段约束及完成提示 |
| `backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py` | 光谱预览隔离及安全协议元数据提示 |
| `backend/packages/harness/deerflow/community/nir/response_grounding.py` | 新协议未完成时的必需动作映射 |
| `backend/packages/harness/deerflow/community/nir/modeling.py` | 原回归质量改用 Val；比较不再预测 Test |
| `backend/packages/harness/deerflow/community/nir/single_target.py` | 原单目标质量调用标为 validation |
| `backend/packages/harness/deerflow/community/nir/reflect.py` | Val 历史、比较数组、Val 最佳及最终关闭 |
| `backend/packages/harness/deerflow/community/nir/registration.py` | Val 质量通过时仍要求最终 Test 质量门禁 |
| `backend/tests/test_nir_search_metrics.py`（新增） | 质量/建议/反思/注册的 Test 不变性及范围 |
| `backend/tests/test_nir_safe_search_core.py`（新增） | 四模型、每折 fit、分组和科学边界 |
| `backend/tests/test_nir_test_sealing.py`（新增） | 私有文件访问、目标扰动、消费/并发/失败/预算/重放 |
| `backend/tests/test_nir_sealed_search_tools.py`（新增） | 绕行拒绝、NPZ/CSV 全链路、预测和工件保留 |
| `backend/tests/test_nir_compare_validation.py`（新增） | 真实 compare 执行不预测 Test |
| `backend/tests/test_nir_upload_preview_isolation.py`（新增） | 上传及转换预览不泄露光谱行；普通文档兼容 |
| `backend/tests/test_nir_process.py` | 旧数据库迁移测试预期升级为 v3 |
| `config.example.yaml` | 四个可选新工具注册示例 |
| `skills/custom/nir-coordinator/SKILL.md` | 安全协议工具顺序、范围及新会话要求 |
| `README.md`、`AGENTS.md`、`backend/AGENTS.md` | 同步新增架构、使用与限制 |
| `docs/chemagent-phase1-implementation-report.md`（新增） | 本报告 |

原三份分析/设计/路线文档未改写，其原审计引用仍指向当时的固定代码版本。没有删除其他算法或任务代码，也没有修改历史实验目录、模型或报告。

## 5. 验收证据

| 要求 | 实际验证与结果 |
|---|---|
| 现有单目标回归正常运行 | 原 single-target/auto-split/partitioned/analysis、process training、预测与注册等相关 NIR 回归测试通过；新 NPZ/CSV 工具路径真实执行并导出可预测模型 |
| 搜索不依赖 Test | 四模型搜索时 spy `np.load`，任何 holdout 读取即失败；实际只打开 `search.npz`。内核 API 不接收 Test |
| 改变锁定 Test 目标不改变搜索 | 四模型固定 Cal/Val 后将另一个会话的 Test y 加 10000，参数和全部搜索 metrics 完全相同；另有测试在同一已锁定会话中直接修改 holdout y，之后搜索参数/metrics 仍相同，最终评价因哈希不符拒绝 |
| 范围正确 | Cal、Cal CV、Val 和 final Test 指标分别标注；没有 Val 的 Test-only 历史不能参与质量/建议/反思；比较图表使用 Val |
| 历史工件不覆盖 | 测试预置旧 model.joblib、metrics.json、report.md，完成新 NPZ/CSV 导出后逐字节相同；新工件唯一目录且禁止覆盖；ProcessStore 旧记录迁移与重放测试通过 |
| 自动化测试通过 | 定向 NIR、nir_core 快速、上传兼容测试全部通过，Ruff 和 diff 检查通过；全仓库限制单列下节 |

Test y 的人为修改测试只修改 pytest 临时数据，用于证明搜索不受影响；这不是支持修改已锁定 Test 的业务能力。真实最终评价会核验原 Test 完整性并拒绝篡改。

## 6. 测试环境、命令与结果

使用 Python 3.12.0 和 scikit-learn 1.9.0。当前仓库没有可运行的独立虚拟环境，Docker daemon 未就绪；复用 `D:\NIR-Agent\deer-flow\backend\.venv\Scripts\python.exe` 作为解释器，但 `PYTHONPATH` 明确指向本仓库源代码。该解释器缺少 Chemotools，经授权仅把 `chemotools==0.4.4` 装到本仓库忽略目录 `.tmp/phase1-deps`，未修改外部虚拟环境。

所有测试使用隔离的 `.tmp/phase1-*-home` / basetemp、关闭 pytest cache 和字节码写入；没有连接生产模型库。完整原始日志保存在 `.tmp`，没有作为实验成果写入历史结果目录。

PowerShell 重现环境（路径按本机实际解释器替换）：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'D:\ChemAgent\nirs-agent\.tmp\phase1-deps;D:\ChemAgent\nirs-agent\backend\packages\harness;D:\ChemAgent\nirs-agent\backend;D:\ChemAgent\nirs-agent'
$env:DEER_FLOW_HOME = 'D:\ChemAgent\nirs-agent\.tmp\phase1-recheck-home'
```

每次运行使用新的 basetemp 名称，避免 pytest 清理已有测试目录。主验证命令与结果：

| 检查 | 命令（在注明目录执行） | 结果 | 日志 |
|---|---|---|---|
| 后端 NIR 最终回归 | backend：`python -m pytest tests -k nir -q -p no:cacheprovider --basetemp=<新临时目录>` | **556 passed, 3 skipped, 5786 deselected**，53.87s | `.tmp/phase1-nir-release.log` |
| nir_core 快速回归 | repo：`python -m pytest nir_core/tests -m 'not slow' -q -p no:cacheprovider --basetemp=<新临时目录>` | **670 passed, 2 skipped, 26 deselected**，33.53s | `.tmp/phase1-core-release.log` |
| 上传预览与原中间件/阻塞 I/O 兼容 | backend：`python -m pytest tests/test_nir_upload_preview_isolation.py tests/test_uploads_middleware_core_logic.py tests/blocking_io/test_uploads_middleware.py -q -p no:cacheprovider --basetemp=<新临时目录>` | **58 passed**，4.82s | `.tmp/phase1-preview-release.log` |
| 新隔离/科学边界专项（最后一轮全量前） | metrics、core、sealing、compare 四测试文件 | **40 passed**，9.44s；之后均已包含在 556 项回归中 | `.tmp/phase1-boundary-final.log` |
| 文件快照额外失败独立复查 | backend：`tests/test_mcp_file_migration.py::TestWorkspaceSnapshots::test_changed_workspace_files_detects_created_and_modified_files` | **1 passed**，4.43s | `.tmp/phase1-snapshot-check.log` |
| 后端全仓库最终回归 | backend：`python -m pytest tests -q -p no:cacheprovider --basetemp=<新临时目录>` | **6213 passed, 87 failed, 45 skipped**，356.59s；失败集合与可运行基线一致 | `.tmp/phase1-full-release.log` |
| 格式/静态检查 | 修改/新增的 23 个 Python 文件 `ruff format`、`ruff check`；`git diff --check` | 通过 | 终端输出 |

各组存在覆盖重叠，不将通过数相加作为唯一测试数量。nir_core 的 slow 测试明确未执行。警告主要来自 Chemotools、LangGraph 的弃用/实验接口及已有绘图字体，不影响上述隔离断言。

全仓库测试记录：

- 最初因复用环境缺少 Chemotools 在收集阶段失败；补齐隔离依赖后得到可运行基线。
- 可运行基线：`87 failed, 6168 passed, 45 skipped`，434.28s，日志 `.tmp/phase1-baseline-full.log`。此时早期新增指标接口/测试已在工作树中，不能把该记录称为纯净 HEAD 基线；主要执行入口和会话隔离尚未改造。
- 主实现后的全量：`88 failed, 6202 passed, 45 skipped`，364.28s，日志 `.tmp/phase1-full-final.log`。与基线相比，原 87 个失败均保留；额外一个是上述 workspace snapshot 用例，独立复查通过。其实现和测试文件均未被本轮修改，签名仅检查 mtime/文件长度，对等长快速改写敏感；这是可能的时间分辨率波动，不能把重跑通过等同于已修复该问题。
- 最后所有代码修改后的全量：`87 failed, 6213 passed, 45 skipped`，356.59s，日志 `.tmp/phase1-full-release.log`。将两个日志的 `FAILED` 测试身份集合逐项比较，差异为空；NIR 失败为 0，上述 workspace snapshot 本轮也通过。最终结果相对于可运行基线没有新增失败。

既有失败包括 Windows 下 POSIX shell 路径/命令、symlink/chmod 语义、sandbox/MCP 虚拟路径转换、未运行的 live client 服务及少量顺序/环境测试。未为通过本阶段而修改这些无关模块或降低断言。**不能声称整个仓库测试全部通过，也没有实际服务/LLM 对话的验证记录。**

## 7. 保留问题及第二阶段前置条件

1. **启用条件**：将四个新增工具条目合并到运行环境配置，使用尚未读取 Test 的新会话；完成部署环境中的小规模工具/预测验证。本轮没有操作现有服务配置或生产存储。
2. **严格限定安全入口**：全部后续自主搜索必须使用封存协议；旧训练/检查工具不能被包装成新模式的搜索动作。已经含 Test 信息的会话历史不会被抹除，不可复用为安全搜索上下文。
3. **协议完整性**：本阶段固定成员、CV 折数及确定性拆分策略；尚未单独持久化完整逐行 fold_manifest、版本化 split_spec 和完整科学复现 manifest。进入更复杂决策/实验接口前需补足这些审计合同；不能从现有简单 scientific_validation 标记推断注册证据齐全。
4. **LLM 合同**：下一阶段需定义统一 DecisionContext/DecisionProposal、理由与候选备选、授权空间、动作和停止决定的持久化记录。LLM 负责高层选择与停止；规则只做合法性/科学/资源限制，数值执行器只搜索授权空间。本阶段有限网格是兼容执行基础，不替代这些合同。
5. **结果证据接入**：安全实验目前存于新增表，尚未完整适配原 attempts/step/响应 grounding 的全部结论合同；第二阶段需要接通新实验身份和最终证据，补实际 LLM 全对话验证。新导出模型可以使用现有可信工件加载与预测，但尚未开启新协议的注册/审批流程。
6. **任务范围**：MAT 自动映射、多目标、分类、其他模型和跨仪器外部验证不在新协议范围；这些原业务入口保留。原始分组 CSV 自动划分、波长优化、目标变换、自定义搜索空间等需后续明确适配。
7. **失败恢复**：拟合/Test 不确定失败保守关闭；自动恢复、取消和事务式工件发布未完成。若最终评价已成功而导出在中途崩溃，独立目录会留下部分文件，重放会拒绝覆盖；当前应由运维检查，不能重新消费 Test。需在恢复/资源治理阶段完善。
8. **资源治理**：持久化实验/fit 上限及现有 checkpoint 已接入；没有新增进程级强制墙钟终止、并行 optimizer 或完整多维成本账本。
9. **回归环境**：整理全仓库既有 Windows/服务依赖失败，在项目标准运行环境中建立可用基线；不能把本阶段定向通过替代发布前的完整环境验证。

本轮不继续第二阶段，不改写历史模型/报告，也未提交、推送或部署。等待用户下一步指令。
