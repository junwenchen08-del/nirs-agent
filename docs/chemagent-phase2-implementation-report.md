# ChemAgent 第二阶段实施报告

日期：2026-10-09。工作目录：`D:\ChemAgent\nirs-agent`。

**已实现 LLM 自主决定首轮光谱预处理、模型家族和固定参数／有界网格策略，并使工具按已接受的决策严格执行。** 适用范围是新对话中的单目标回归，支持 PLS、Ridge、SVR、Extra Trees。完成一次初始执行后返回真实 Cal CV/Val 反馈，进入 `initial_result`，等待下一步指令。本轮没有实现第三阶段的多轮动作、保留模型或自主停止策略。

该结论有两类不同证据：Fake LLM 驱动真实 Lead Agent 工具链覆盖四种家族和错误修正；使用本地已配置的真实 DeepSeek，通过相同决策／搜索服务完成小预算调用。真实调用不使用预设候选管线，也没有强制 PLS 基线。真实测试范围及限制见第 9 节。

## 1. 第一阶段前置检查

逐项检查了第一阶段报告、决策审计、LLM 设计和路线图，以及报告所列的真实实现。检查结论如下。

| 前置能力 | 实际代码与检查结论 |
|---|---|
| Cal/Val 搜索内核 | `nir_core/model/safe_search.py` 的 API 不接收 Test；每个 Cal CV 折重新拟合完整预处理／模型／内部 scaler，然后仅在 Cal 重拟合、评价 Val |
| 私有封存 | `dataset_session.py` 复用 owner/thread 的 `ProcessStore`，`search.npz` 与 `holdout.npz` 分开，位于 sandbox 可见目录之外；Cal Profile 不包含 Test 摘要 |
| 四个安全工具 | `nir_prepare_search`、`nir_search_model`、`nir_freeze_search`、`nir_finalize_search` 已实现；显式冻结、持久化单次 Test lease、失败关闭与重放机制真实存在 |
| 旧入口隔离 | `nir_workflow_middleware.py` 已对 sealed 工作流限制工具；旧训练、文件读取、脚本、子任务、MCP fit 等不能绕行 |
| 评价范围 | 搜索质量／反思／比较使用显式 Val 证据；最终 Test 门禁独立；旧模式仍保留原业务接口，不能对所有 legacy 入口宣称全流程封存 |
| 原有配置 | `config.example.yaml` 有四个安全工具；本地 `config.yaml` 在本轮开始时尚未合并，报告这一限制准确 |
| 前置缺口 | 当时没有结构化 LLM 决策合同、可信模型调用来源、决策与实验身份关联，也未持久化精确 CV 折成员。本轮补充这些最小审计能力 |

工作分支保持 `dev/Duan`，起始 HEAD 为 `ef288ea3ec241507c63164828a012a2fd519a440`。第一阶段代码与四份文档原本就未提交；没有 reset、checkout、提交或推送。原三份设计文档及第一阶段报告没有回写为第二阶段状态。本轮是在现有未提交实现上继续开发，不能用相对 HEAD 的整个 diff 当成第二阶段新增量。

## 2. 实际代码文件

下列路径相对仓库根目录。`H` 指 `backend/packages/harness/deerflow`，`N` 指 `H/community/nir`。

| 文件 | 本轮实现 |
|---|---|
| `nir_core/model/parameter_control.py`（新增） | 四模型实际参数 schema、固定参数／显式 Cartesian 网格、数学／类型／预算检查、运行时能力说明 |
| `nir_core/model/safe_search.py`（扩展第一阶段文件） | 可选授权参数计划、精确 CV manifest、完整实际模型参数、每折误差；无计划时保留旧有限网格 |
| `N/decision_contracts.py`（新增） | 严格的 `DecisionContext`、`DecisionProposal`、预处理、模型、搜索、观察和候选合同；仅开放 initial EXPLORE |
| `N/decision_context.py`（新增） | 有来源的初始上下文、接受／预算预留、不可变决策、运行时绑定、单次受控执行、独立成功／失败反馈 |
| `N/decision_origin.py`（新增） | Lead Agent 模型响应边界的可信调用回执与精确参数哈希核验；区分 Fake 与真实 provider |
| `N/decision_tools.py`（新增） | 上下文读取、结构化提交、仅按 decision_id 执行三个工具；字段错误与真实反馈 |
| `N/dataset_session.py`（扩展第一阶段文件） | 决策策略绑定、持久 CV manifest、自主模式只接受已提交且正在执行的完整规格／身份 |
| `N/search_tools.py`（扩展第一阶段文件） | 准备时绑定策略；自主模式在工具函数层拒绝直接搜索 |
| `N/process.py` | 私有 SQLite v4 增量迁移，新增 context／decision／model-call 表；原有表和记录保留 |
| `N/workflow.py` | `decision_strategy` 与验证协议区分、强制 sealed、用户硬约束及决策／执行状态显示 |
| `N/response_grounding.py` | 新模式必需动作映射、已执行证据关联、封存期间 Test 性能宣称拒绝 |
| `H/agents/thread_state.py` | 兼容可选策略／约束／context／decision／initial_execution 字段及 initial_result 状态 |
| `H/agents/middlewares/nir_workflow_middleware.py` | 同步／异步真实模型来源捕获、工具白名单与阶段限制、记忆消息过滤、后续动作提示 |
| `H/agents/middlewares/dynamic_context_middleware.py`、`memory_middleware.py` | 自主模式不注入或写回跨任务全局记忆，其他模式保持兼容 |
| `H/agents/lead_agent/prompt.py` | 指定真实上下文→自主比较→提交→执行链；legacy 固定流程不支配新策略 |
| `skills/custom/nir-coordinator/SKILL.md` | 三个工具权限及首轮自主流程，覆盖旧自动建模配方 |
| `config.example.yaml`、本地 `config.yaml` | 注册三个决策工具；本地配置同时补齐第一阶段缺少的四个工具 |
| `backend/scripts/verify_nir_initial_llm.py`（新增） | 必须显式 opt-in 的真实 provider 合成数据验证，独立线程、最多 7 fits、独立 JSON 证据 |
| `backend/tests/test_nir_authorized_parameters.py`（新增） | 固定参数、授权空间、无隐藏 PLS、非法参数和预算 |
| `backend/tests/test_nir_initial_decisions.py`（新增） | 上下文、可信来源、不可变记录、身份、并发／重放、Test 扰动、权限、CV 篡改、约束和 provider 变化 |
| `backend/tests/test_nir_autonomous_initial_agent.py`（新增） | 实际 Lead 工厂／Prompt／NIR 工具链的 Fake LLM 集成，四家族、组合、Raw、修正和异步 |
| `backend/tests/test_nir_process.py` | 迁移版本预期更新为 v4，保留原记录的回归测试 |
| `README.md`、`AGENTS.md`、`backend/AGENTS.md` | 用法、控制权、持久化、测试和边界说明 |
| 本报告、`docs/chemagent-phase2-real-llm-evidence.json` | 交付结论及最终真实运行证据；JSON 仅含合成任务、决策与反馈，无密钥／原始光谱 |

没有改造前端、删除其他模型或任务入口，也没有增加一套独立分区、训练、实验资产或跨任务经验系统。公共 API 扩展主要是可选 keyword 参数和 JSON 可选字段；`legacy` 仍为默认值。

## 3. 真实 Agent 调用链

```text
DeerFlow Lead Agent（现有工厂、工具注册、更新后的 Prompt/Skill）
  → nir_workflow(start, decision_strategy="llm_autonomous",
                       search_protocol="sealed_holdout")
  → nir_prepare_search（复用第一阶段可信摄取／私有封存）
  → nir_decision_context（真实 Cal / 方法知识 / runtime / 固定 CV / 预算）
  → LLM 比较候选、自主选择预处理、模型、参数策略与理由
  → Lead 模型调用返回 AIMessage.tool_calls
  → NIR middleware 先保存该响应所签发的提交调用回执
  → nir_submit_decision：核验、保存不可变决策、预留预算
  → LLM 用已接受 decision_id 调用 nir_execute_decision
  → middleware 保存执行调用回执
  → InitialDecisionService 获得持久执行 lease
  → SearchSessionStore.search → 原 safe_search 执行授权规格
  → 分别保存实际参数、trials、Cal CV/Val、成本及结果身份
  → initial_result / await_next_phase
```

没有新建旁路模型选择器。现有 Lead 工厂已经安装 NIR middleware，因此在该 middleware 加入模型响应捕获即可进入实际同步和异步工具调用链，无须复制 Agent 工厂。

本地配置采用原条目格式定点插入 `nir` group 工具。写入前后校验所有非 tools 配置对象及原有工具对象完全相同；保留用户模型设置、密钥和其他选项。没有复制整个示例配置覆盖本地配置。没有重启／部署 Gateway 或前端；已运行实例是否已热加载本次修改不属于本轮已验证范围。

## 4. DecisionContext 与决策合同

首轮上下文有版本、内容 hash、context_id 与 session_id，并保存到现有 ProcessStore。数据来源为：

| 内容 | 来源与范围 |
|---|---|
| 任务和约束 | 工作流用户字段及原始 HumanMessage，标注 `original_human_message`；保留明确模型／有序管线硬约束 |
| 光谱特征 | 封存服务保存的 Cal-only `profile_preprocessing_data`，包括样本数、维度、轴规则性、噪声／散射等描述性代理及标签 |
| 目标摘要 | 校验过哈希的 `search.npz` 内 `y_cal`，仅 Cal 数量、均值、标准差及范围 |
| 诊断不确定性 | 明确说明分数是描述性代理，不能据此确立化学原因或强制某种预处理 |
| 方法知识 | 现有托管方法库 published 快照及 `MethodKnowledgeBase.search`；保留证据 ID、方法用途、条件、限制、参数说明和官方来源 |
| 预处理能力 | 现有 registry 的方法／参数／provider／轴等能力；首轮执行可用标志与实际 provider 构造校验 |
| 模型能力 | 四个实际 sklearn adapter 的参数边界、公开默认值、版本、操作参数及内部 scaler 说明 |
| 验证和预算 | 已锁定 session 的 Cal/Val 数量、Cal 数据 hash、精确 CV 折、剩余 fits／实验，以及候选×CV+重拟合成本公式 |

检索过程不从方法卡片 priority 自动生成最终执行方案，也不使用 profile 自动屏蔽某个模型家族。知识排序表示相关性，`ranking_is_execution_order=false`。方法卡片仍是有出处但不可信的解释性证据，不能覆盖执行器 schema 或系统约束。模型能力以本地真实 adapter 为准，不能用文档卡片冒充运行时支持；数据无关的 MCP 目录读取仍可调用，MCP fit 不进入自主执行链。

没有历史实验列表、Test 指标、Test 数量、Test 目标分布、Test Profile 或含 Test 的源绑定 hash。首轮 `history=[]`；已有全局记忆的注入／更新在新模式被抑制。使用新对话是协议前提，系统不会删除旧对话中用户已见的 Test 内容。

`DecisionProposal` 使用 strict／extra-forbid 合同，包含 context_id、revision/hash、观察与证据 ID、2–4 个候选及比较、最终候选 ID、理由、预期效果、完整管线和参数计划。初期只允许 `phase=initial/action=EXPLORE`。Raw 是空步骤数组；`auto` 模型／自动预处理／波长选择以及后来阶段动作不能提交。

提交检查合法性、版本／hash、证据存在性、明确用户约束、实际能力、数学限制与预算。非法计划返回字段路径及错误；预算错误给出候选数、冻结 CV 折数、所需 fits 与授权额度，便于 LLM 显式修正。程序不会替换为默认模型／默认预处理，也不会自动缩小搜索范围。

## 5. 实际控制权与参数能力

LLM 控制外部光谱预处理步骤、顺序、参数和组合，四模型家族，以及固定／grid 策略。诊断标签、R² 阈值、卡片排序和质量评价都不能直接选管线。未选择的候选只用于比较，不进行训练。

| 模型 | 本轮可固定或在显式离散列表内优化的参数 |
|---|---|
| PLS | `n_components`、`scale`、`max_iter`、`tol`；潜变量数必须明确，超出 Cal 折／轴数学上限直接报错，不截断 |
| Ridge | `alpha`、`fit_intercept`、`tol`、`max_iter` |
| SVR | `C`、`gamma`（scale／auto／正数）、`epsilon`、`kernel`、`degree`、`coef0`、`tol`、`shrinking`、`max_iter` |
| Extra Trees | `n_estimators`、`max_depth`、`min_samples_split`、`min_samples_leaf`、`max_features`、`bootstrap`、`criterion`；随机种子 42、单线程是公开的操作约束 |

参数边界以返回的 runtime schema 为准，不宣称暴露了 sklearn 的全部调参接口。未知参数、非有限数值、非法类型、fixed/space 同名、无明确 PLS 成分数或超预算均拒绝。其他 sklearn 操作默认值通过实际 estimator 的 `get_params()` 公开并返回。模型和预处理合法默认值先在上下文声明，接受时写入执行规格；用户已给的合法值不会被覆盖。

方式 A 对唯一参数配置做冻结 Cal CV 评价，再在完整 Cal 拟合一次；这不是重新搜索。方式 B 只穷举 LLM 授权的离散 Cartesian product，并按 Cal 平均折 RMSE 选内部参数。预处理保持固定，Val 只评价选定参数，没有另一次 Val 调参、隐藏家族比较或额外 PLS baseline。计划最多 40 trials／200 fits，并受更小的 session 与 proposal 限额约束。

Ridge/SVR 沿用第一阶段 adapter 的固定内部 StandardScaler，能力中明确公开，每个 Cal 折重新拟合。Raw 表示没有外部光谱预处理步骤；本轮没有增加内部 scaler 开关。外部预处理不应用旧的 `auto_eligible`、`priority` 或 `explicit_only` 策略来替 LLM 选择，只核验可执行性及科学／轴约束。

## 6. 决策、身份和执行记录

SQLite v4 增量新增三个表：`decision_contexts`、`initial_decisions`、`model_issued_calls`，保留 v1/v2/v3 记录和第一阶段会话／实验表。每个 session 只接受一个初始决策。

执行前不可变 payload 保存 decision_id、context_id/hash、session_id、预定 experiment_id、真实模型调用回执、观察／证据／候选／最终选择／理由／预期、原始 proposal 及 hash、规范化完整执行规格、精确 CV manifest ID、provider/runtime 绑定、预留 fits 和时间。提交事务在返回给 LLM 前完成持久化。

来源在 **实际模型 handler 返回后、任何工具执行前** 记录，包含模型类／类型／名字、AIMessage ID、tool_call_id、可用的 Agent run_id、响应 hash、工具参数 hash、时间及 token usage。提交与执行均要求精确匹配这一 owner/thread 的调用回执。伪造 `author="llm"`、改动工具参数或任意 SDK 直接调用不能产生可信决策来源。Fake 类型明确为 `fake_model`，真实 DeepSeek 为 `provider_model`；这不是用 caller 标签认定来源。

执行工具只接受 `decision_id`，从数据库读取已接受规格，不接受 method／数据路径／参数覆盖。接受状态变为持久 `executing` 后，现有 SearchSessionStore 再检查决策状态、身份与完整规格完全一致，方能进入训练。provider 或 adapter 变化在拟合前拒绝。结果另存 execution/status，不能回写执行前理由。

执行后反馈包括相同 decision/context/experiment ID、真实配置、完整实际 sklearn 参数、每个 trial／折误差、Cal CV／Val／校准性能、精确 CV manifest、fits、耗时和成功／失败原因。fit 预算计数代表拟合尝试开始；失败仍计已发生的成本。失败不自动换模型。并发仅一个 lease 可训练，终态重放读取原反馈、不重拟合。

没有为了关联身份增加新的完整实验平台；实验仍保存在第一阶段 `search_experiments`，模型仍位于相同私有 session 目录。新工作流只保存决策／实验引用和有界初始结果，并向既有 response grounding 提供 Cal CV/Val 证据。

## 7. CV 与 Test 封存审计

本轮补充 `make_cv_manifest`：KFold 使用 `shuffle=True, random_state=42`；分组模式使用不打乱的 GroupKFold。保存真实 splitter、n_splits、seed、每折 Cal 局部训练／验证行号与 SHA-256，执行前重新核验实际分折与 manifest 相同。折成员列表比仅记录 seed 更直接；这些行号不是原始 Test 成员。每折预处理仍只拟合折训练行。

自主模式沿用第一阶段两文件隔离及最终 Test lease，没有新增分区或 Test 管理系统。执行内核、初始 context 和模型优化期间只加载 `search.npz`。可信摄取仍允许接触 Test 做合法性、分区分离和完整性检查，不计算其选择指标。

中间件及工具函数共同阻止直接 `nir_search_model`、旧训练、比较、原始文件、脚本、子任务、MCP fit、未知工具和工作流重启。自主初始阶段也不能直接 freeze/finalize，以免越过后续 LLM 保留／停止决策。本轮 Test 始终封存；第一阶段手动策略的冻结与最终评价功能继续可用。

测试对锁定 Test y 加 100000，初始 context、授权配置、参数优化 trial、CV manifest、全部 Cal CV/Val 指标与成本均不变；spy 在真正文件加载处拒绝任何 holdout IO。第一阶段原测试还证明最终评价因 Test hash 被篡改而拒绝。不是仅删除输出 Test 指标。

指标继续标记 `calibration`、`calibration_cv`、`validation`；最终 Test 成绩不能用这些值冒充。质量门禁只产生科学反馈，新模式不会因此自动通过、换模型、停止研究搜索或追加一次执行。

## 8. 自动化验证

按小任务先完成测试，再实现参数接口、提交／执行、Agent 接入及边界修正；固定参数、记忆隔离、硬约束规范化和 provider 变化均经历对应的失败→通过验证。第一阶段核心封存／工具生命周期与 legacy 回归同时保留。

| 用户要求 | 证据 |
|---|---|
| 首轮 PLS | Fake Lead 选择 PLS，固定 `n_components=2`，仅一个候选 |
| 首轮非 PLS | Fake Lead 分别首次选择 Ridge、SVR、ET；最终真实 DeepSeek 首选 Ridge |
| 合法组合、Raw | Fake SVR 执行 SNV→mean_center；其他 Fake 用例为空步骤；真实 Ridge 为 SG→SNV |
| 无隐藏 PLS | 非 PLS Fake Agent 将 `PLSRegression.fit` 设为即失败；SVR 网格核心用例还禁止 PLS 构造；全部通过 |
| 固定参数准确执行 | 四模型逐值断言实际 `get_params()` 与指定参数，只有一个候选，3-fold + Cal refit = 4 fits |
| 网格不越界 | SVR 所有 trial 都属于授权 C/gamma 列表且固定 epsilon 不变；真实 Ridge 只试 alpha 1、10 |
| 非法方案可修正 | 负 alpha 等在拟合前拒绝；Fake Lead 根据工具错误重新提交合法方案后完成执行，没有替换方案 |
| 实际执行与接受一致 | 真实／Fake feedback.configuration 等于 immutable accepted_spec，结果参数属于授权计划；provider 变更在零 fits 时失败 |
| 先保存再训练 | search spy 在拟合前读到 durable decision 的 executing 状态；无 accepted ID、错误调用 hash 或改写规格被拒绝 |
| 无 Test 读取 | context／真实 Agent／搜索服务文件加载 spy；锁定 Test 目标扰动不变性 |
| 无旧工具绕行 | 新模式明确拒绝直接安全搜索、legacy、freeze/finalize、restart、bash、read_file、task；无需依赖 Prompt 自觉 |
| 历史和 legacy 兼容 | 原 NPZ/CSV 搜索→冻结→报告→可预测工件测试通过，旧工件逐字节不变；原工作流／ProcessStore 迁移测试保留 |
| CV／身份／并发 | 精确折篡改在拟合前失败，预定 experiment_id 与实际一致，并发只创建一个实验，重放无 IO／refit |

| 验证批次 | 实际结果与日志 |
|---|---|
| 改动前完整 backend 基线 | 6213 passed、87 failed、45 skipped；`.tmp/phase2-baseline.log` |
| 参数／核心／封存独立任务 | 30 passed；`.tmp/phase2-params.log` |
| 决策／迁移／封存独立任务 | 28 passed；`.tmp/phase2-decision.log` |
| 主要安全边界和 Agent 链 | 74 passed；`.tmp/phase2-closure-green.log` |
| 最终定向回归（含预算错误完善后的代码） | 312 passed；`.tmp/phase2-focused-final2.log` 和对应 JUnit XML |
| 最后失败身份补充检查 | 24 passed；`.tmp/phase2-final-identity.log`，确认失败反馈也关联预定实验／CV ID |
| 主要实现后的完整 backend 回归 | 6254 passed、87 failed、45 skipped；`.tmp/phase2-full-final.log` 和 JUnit XML |
| nir_core 快速门禁 | 670 passed、2 skipped、26 deselected；`.tmp/phase2-core-final.log` |
| 真实服务最终运行 | passed，7 fits；第 9 节 JSON 证据及 `.tmp/phase2-final-real2.log` |
| Ruff／diff | 本轮代码／测试／脚本的 Ruff 通过；`git diff --check` 通过 |

完整 backend 并非全绿。逐项规范化比较前后 87 个失败 node ID，集合完全相同，新增失败为零（`.tmp/phase2-failure-comparison.json`）。失败集中在既有 Windows shell／进程／chmod／symlink／权限／旧测试服务或客户端环境，以及之前已有的 sandbox、MCP、技能、渠道、部署检查；没有把它们当成第二阶段测试通过，也没有为本任务修改这些无关功能。JUnit 中 `test_nir` 为 580 passed、2 skipped、0 failed。完整回归后，本轮最后的预算成本提示与 schema 说明另经 312 项定向回归验证。

测试日志及 JUnit 位于仓库忽略目录 `.tmp`；真实运行证据单独保存在 docs，不混入 Fake 测试结果。

## 9. 真实 LLM 验证

已执行真实 provider 调用，**不是 Fake LLM 的结果**。最终运行通过本地 `config.yaml` 的模型工厂创建 `PatchedChatDeepSeek`，模型名称为 `deepseek-flash`。调用前只在内存副本限制输出／超时／重试，没有写改用户模型配置。

验证脚本复用第一阶段 prepare 工具摄取独立合成数据；沿用真实 Lead 工厂／Prompt、真实 NIR 与 DynamicContext middleware，以及从实际配置 use 路径加载的三个决策工具。为在没有部署 Gateway 的本机完成验证，虚拟上传路径解析和工具集范围被隔离，标题、sandbox、MCP、tracing 等无关服务 middleware 未启动。**这是实际模型服务的进程内 Agent／科学执行完整链验证，不是已部署 Gateway／网页的端到端验证。**

最终证据：[chemagent-phase2-real-llm-evidence.json](chemagent-phase2-real-llm-evidence.json)。复现脚本：[verify_nir_initial_llm.py](../backend/scripts/verify_nir_initial_llm.py)。

| 项目 | 最终运行 |
|---|---|
| 数据 | seed=3907，48×12 合成数据；复用固定随机分区，Cal=28、Val=10，Test 数量不进入 context |
| 自主选择 | `SG(window=7, order=2) → SNV → Ridge`，自主比较其他 Raw／Whittaker 等候选，未训练这些候选 |
| 参数授权 | alpha ∈ {1, 10}，2 trials、3 折、最多 7 fits；其他公开合法默认值写入接受规格 |
| 实际参数 | alpha=1；完整 solver、tol、fit_intercept、copy_X 等在证据中返回 |
| CV | KFold，shuffle=True，seed=42；完整 Cal 折成员及 `cv_3d9f342df3b6ee79e7b8de2de2dcd7ff03e6403dcf19caf3f4b4f6db5906aded` |
| 实际成本 | 7 fits；科学执行 0.25 s；测试流程 21.59 s；4 次模型响应累计 input/output=111018/2562 tokens |
| 性能 | OOF Cal CV RMSE=1.712982；Val RMSE=1.248008、R²=0.024930；Val 门禁 poor / passed=false |
| Test | 搜索只打开 `search.npz` 两次；未打开 holdout，未冻结／最终评价 |
| 结果状态 | succeeded 表示合法科学执行成功，**不表示预测质量达标**；工作流停在 initial_result |
| 决策 | `decision_d8d78504618d6e54385c4fc73d8a1091` |
| Context | `ctx_71c827816ba0836064097d15f6c4d15c` |
| 实验 | `exp_2455844ea7aafc57aece2443cf537d5d` |
| 调用来源 | provider_model / lead_agent_model_call；AI message、Agent run 和 tool call ID 均在 JSON |

真实验证过程也保留失败事实：最初脚本重复传入 timeout，修正本地测试构造后才进入服务；沙箱 APIConnectionError 后按权限联网重试。一次完成的初步运行选择 SNV→SG→PLS，共 7 fits。后续对最终接口复测时，LLM 反复提交 fits 不足或 fixed/space 冲突，工具拒绝且未训练，最终达到图调用上限；依据这一实际反馈补充明确成本公式和字段错误，最后由上表 Ridge 运行通过。没有把失败尝试改标成功或使用代码替 LLM 缩小候选空间。

## 10. 重现与检查环境

Python 3.12.0、scikit-learn 1.9.0，复用现有 `D:\NIR-Agent\deer-flow\backend\.venv\Scripts\python.exe`；`PYTHONPATH` 指向本仓库和已授权安装的 `.tmp/phase1-deps` Chemotools 0.4.4。没有更改外部环境或安装新的优化框架。测试关闭字节码和 pytest cache，采用独立 DEER_FLOW_HOME 与 basetemp；未运行真实 Wheat／Meat／Corn 实验或接触历史实验数据。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'D:\ChemAgent\nirs-agent\.tmp\phase1-deps;D:\ChemAgent\nirs-agent\backend\packages\harness;D:\ChemAgent\nirs-agent\backend;D:\ChemAgent\nirs-agent'
$env:DEER_FLOW_HOME = 'D:\ChemAgent\nirs-agent\.tmp\phase2-recheck-home'
$py = 'D:\NIR-Agent\deer-flow\backend\.venv\Scripts\python.exe'
& $py -m pytest backend/tests/test_nir_authorized_parameters.py backend/tests/test_nir_initial_decisions.py backend/tests/test_nir_autonomous_initial_agent.py -q -p no:cacheprovider --basetemp=.tmp/phase2-recheck-new
& $py -m pytest backend/tests -q -p no:cacheprovider --basetemp=.tmp/phase2-full-recheck-new
& $py -m pytest nir_core/tests -m 'not slow' -q -p no:cacheprovider --basetemp=.tmp/phase2-core-recheck-new
# 需要可用的真实服务／凭据／网络，会产生模型 API 成本：
& $py backend/scripts/verify_nir_initial_llm.py --allow-real-provider
```

真实脚本每次使用唯一 QA 线程和唯一证据文件，仅允许 `.tmp` 下隔离 home。新增正常执行模型位于唯一私有 session；已有模型、报告、历史 metrics 不会由该脚本覆盖。正常协议执行和重放同样禁止覆盖已有实验模型。

## 11. 限制及后续阶段条件

- 本轮仅 initial EXPLORE 和一次执行。之后不接受第二初始决策，不实现 TUNE／CHANGE_PREPROCESS／SWITCH_MODEL／STOP／保留模型；不能对外宣称已实现完整自主优化与停止。
- 初始工作流支持内部留出单目标回归及明确映射 CSV/TXT/TSV／数值 NPZ；其他算法、分类、多目标、MAT 集合和生产注册继续走 legacy。程序化已隔离分组可用，但新模式没有分组原始文件自动摄取。
- 离散 Cartesian 网格是唯一数值优化框架。完整 runtime schema 外的模型参数明确拒绝；状态型预处理每折 fit，监督或改变光谱轴的预处理未适配。
- 已接受决策和 lease 采用失败关闭。进程若在 lease 后崩溃，不会自动发放第二执行；需后续设计可审计恢复，不能靠重放重训。本轮没有完整 crash-recovery 平台。
- 原始用户任务文本有 4000 字符上限，截断长度明确显示；特别长的硬要求应进入工作流结构化约束。既有会话中的旧 Test 结果不删除，必须新对话启动。
- 方法知识是参考而非真值。LLM 选择可能质量差或反复产生不合法计划；工具阻止非法执行，但不保证选择最佳方案。真实合成验证质量差如实返回，不能作为论文效果实验。
- 真实验证仅覆盖本机配置 provider 的进程内调用链；未部署、未进行 Gateway／前端联调。全量测试的既有环境失败在最终验证结果中独立列出。
- 第一阶段的最终工件注册／导出中断恢复等限制仍存在；本阶段保留第一阶段手动 finalize 功能，没有以初始结果自动开启最终 Test 或注册。

进入第三阶段前需用户明确授权，并在现有 contract/service 上逐项增加后续动作、反馈上下文和 LLM 的保留／停止决策，再贯穿到显式 freeze/finalize；同时保留 immutable lineage、参数范围、预算和 Test lease。运行服务联调及上下文 token 体积治理可分别安排。本轮在第二阶段结束，不自动开展后续开发。
