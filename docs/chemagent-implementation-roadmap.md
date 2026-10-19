# ChemAgent LLM 自主决策模块分阶段实施方案

日期：2026-10-08；基线：`ef288ea3ec241507c63164828a012a2fd519a440`。

本轮只生成文档，**下面所有改动、接口和测试均是下一阶段计划，尚未执行**。现状依据见[源码审查报告](chemagent-current-decision-audit.md)，合同和流程见[机制设计](chemagent-llm-decision-design.md)。

## 1. 实施边界及开发起点

在 `D:\ChemAgent\nirs-agent` 实施，保留当前分支和既有记录。本轮不得修改业务代码、启动建模或建立论文比较实验。未来代码修改遵守根及backend AGENTS.md：功能／修复配套测试，并同步相关README和AGENTS；本轮文档-only不触发修改这些文件的要求。

**首先修正验证协议，再交付LLM选择接口。** 如果仅增加自主Prompt，现有Test RPD仍会决定反思及最佳方案，LLM也无法指定SVR网格或可靠停止，因此不能作为新模式的完成标准。

先交付单目标回归的完整纵向流程，首批参数adapter为PLS、Ridge、SVR、ET。现有12种family不删除；新模式支持状态逐项声明，未适配任务／family返回明确错误，不偷偷执行legacy。每阶段必须在上一阶段安全合同上独立验收。

文中缩写：`H` = `backend/packages/harness/deerflow`；`N` = `H/community/nir`；`C` = `nir_core`。表内使用这些前缀时按此展开。

## 2. 建议保留和复用的现有模块

| 文件／组件 | 保留的能力 | 复用时的边界 |
|---|---|---|
| `C/io/loaders.py`、schema、validation | CSV/TXT/MAT/NPZ解析及高置信度映射 | trusted摄取执行；给LLM结构化字段，不给混合Test统计 |
| `C/utils/scientific_validation.py`、`N/_science_gate.py` | 轴、重复、分区污染检查及重现证据 | 不因新策略降低科学门禁 |
| `C/preprocess/recommendation.py::profile_preprocessing_data` | Cal诊断数值 | 新策略仅取Profile；不用recommend的候选顺序选方案 |
| `C/knowledge/methods.py`、method_store、method_inventory | 卡片检索、来源／版本、管理快照 | 卡片可用于知识依据；不能直接替代执行schema |
| `N/method_catalog.py`、knowledge.py | lazy管理快照、方法和论文检索 | 记录decision_id与snapshot；两种证据门槛分开 |
| `C/preprocess/registry.py`、pipeline.py、providers | 已验证算法、参数合法性、state及provider回放 | 每折fit完整管线；不自动替换不合法步骤 |
| `C/model/pls.py`及其他训练文件 | 模型构造、固定拟合、现有CV／网格原语 | 增加参数透传和计数，禁止隐藏家族或全失败假成功 |
| `C/utils/metrics.py`、diagnostics.py | RMSE/R²/RPD/bias和残差数值 | 明确scope与残差符号；不由诊断标签选动作 |
| `N/workflow.py`、ThreadState、middleware | task状态、约束、计划签名、结果自动写回 | 分策略执行；修正Test及错误来源语义 |
| `C/io/resources.py`、`N/_resources.py` | 文件／维度／内存估算、取消及阶段deadline | 增加任务累计fit/trial；抢占能力诚实声明 |
| `N/process.py`、Gateway nir_process.py | owner/thread记录、分页、图表哈希及生命周期 | 扩展不可变decision/feedback，搜索图表只有Cal/Val |
| artifacts.py、_report.py、delivery.py、registration.py | 可信工件、报告／ZIP、注册哈希与用户批准 | 最终冻结和Test完成后生成可注册工件 |
| `N/io_tools.py`、预测和漂移监测 | 工件回放、预测、审计、监测 | 不重写预测算法；验证新工件仍兼容 |

保留原始卡片、模型、metrics和ProcessStore记录；不回填“LLM选择”字段把旧规则决策改名。旧记录缺字段就标记legacy／unrecorded。

## 3. 需要修改的现有模块及接口

| 文件 | 主要改动 | 输入 → 输出／保护 |
|---|---|---|
| `H/config/app_config.py` + `config.example.yaml` | 增加typed nir_decision配置，启动时冻结到workflow | 配置 → policy快照、支持任务、预算；活跃会话不热改策略 |
| `H/agents/thread_state.py` | 可选strategy、session、decision、retained、freeze、budget字段；增加状态字面量 | checkpoint合并保持旧字段；决策体不全塞checkpoint |
| `N/workflow.py` | 分策略transition；首轮／后续submit、STOP、freeze、test完成；Raw []合法 | 原控制平面 → 新策略指针／版本；legacy action保持兼容 |
| `H/agents/middlewares/nir_workflow_middleware.py` | 能力授权、签名、用户强制family来源、上下文绑定、结果scope及消费幂等 | tool request → 接受／拒绝；新模式禁止旧auto／Test绕行 |
| `H/agents/lead_agent/agent.py`、prompt.py | 条件策略Prompt和工具装配；不改变通用Agent循环 | 冻结policy → 该模式工具及system约束 |
| `H/tools/tools.py` | 配置注册新工具；与skill allowed-tools、deferred MCP过滤保持一致 | 实际工具集 → 新策略最小允许集；middleware再硬守护 |
| `H/agents/middlewares/durable_context_middleware.py` | 注入有界DecisionContext摘要，history分页提示 | 可信记录 → data-role模型视图，完全排除Test派生字段 |
| uploads／sandbox工具权限边界 | 新模式不给LLM原始输入读能力；所有别名／MCP／task可绕行路径封闭 | 可信摄取保留上传；通用模型工具不可读完整dataset或私有分区 |
| `N/modeling.py`、single_target.py | 抽出Cal/Val执行与final测试；透传固定params/search；移除final refit再搜索 | 原入口兼容；新实验内核不接受X_test/y_test |
| `N/candidate_selection.py`、`N/_preprocessing_plan.py` | 规则auto选择仅legacy调用 | autonomous不可调用choose_model、choose_wavelength或固定预处理winner |
| `N/reflect.py` | legacy规则反思改用search指标；新策略仅复用诊断计算而不消费should_retry | 验证证据 → 描述性反馈；不自动生成新动作 |
| `C/utils/validation.py` | 反思安全输入、scope显式；旧阈值策略保留legacy | 不读取test.bias/RMSEP做下一步建议 |
| `C/utils/metrics.py` | 加scope明确评估接口；搜索质量与final质量分离 | train_cv/validation或final_test → 相应指标／门禁，禁止混用 |
| `C/model/*.py` | 暴露fixed params／space；全管线CV adapter；所有fit计数 | 参数合法后精确执行；不能clamp、丢参数或重选家族 |
| `N/process.py` | 新record表和终态事件；支持新实验工具；搜索和最终图表scope分开 | decision/experiment/feedback → 不可改写owner记录及旧steps兼容投影 |
| `N/decision_facts.py`、response_grounding.py | origin、scope和状态；新指标及停止来源有据输出 | 不把candidate提案显示为已执行，STOP与预算终止分开 |
| `N/artifacts.py`、_report.py、delivery.py | 搜索摘要及final交付；新manifest字段兼容旧预测 | 只在final绑定最终metrics；不能改旧manifest字节 |
| `N/registration.py`、models/service.py | 校验freeze/test及model/metrics哈希；只对final工件推广 | 搜索工件不可注册；用户批准及validation_goal规则保留 |
| `backend/app/gateway/routers/nir_process.py`、`contracts/nir-process-v1.*` | 添加可选decision/feedback查询字段和scope | owner权限及分页；不暴露私有holdout |
| `frontend/src/core/nir/*.ts` | 后续仅最小合同解码／投影适配 | 可选字段、大小边界、旧记录兼容；本轮不实施UI |
| `skills/custom/nir-coordinator/SKILL.md` | 分策略指南；新模式动作、submit/execute/finalize；修正冲突语句 | Skill为操作指导，真实策略及授权来自runtime |

uploads具体改点以源码中的 `H/agents/middlewares/uploads_middleware.py` 和 `H/sandbox/tools.py` 为起点，需覆盖初始消息、文件列表和所有通用工具路由。现有脚本黑名单不足以隔离Test；不以“Prompt说不要读”作为验收。

## 4. 建议新增的最小组件

不新增独立Agent、跨任务记忆服务或第二套实验数据库。

| 新文件（计划） | 主要接口 | 责任 |
|---|---|---|
| `H/config/nir_decision_config.py` | `NIRDecisionConfig` | 策略、支持任务、硬预算、最终拟合策略；Pydantic验证 |
| `N/decision_contracts.py` | `DecisionProposal`、`PipelineSpec`、`SearchSpec`、`ExecutionResult`、`FeedbackRecord` | 严格数据合同、动作判别、实际执行比较及规范化签名 |
| `N/decision_context.py` | `build_decision_context(session_id)`、`project_context_for_llm()` | 从真实数据／记录／预算构造有界上下文及分页历史 |
| `N/dataset_session.py` | `prepare_search_dataset()`、`get_search_partitions()`、`open_final_test()` | 一次分区、私有holdout、权限及scope句柄 |
| `N/experiments.py` | `nir_prepare_search`、`nir_get_decision_context`、`nir_submit_decision`、`nir_execute_decision`、`nir_finalize_model` | 工具薄层：调用已有工作流、训练内核与ProcessStore，不选高层动作 |
| `N/decision_budget.py` | `reserve()`、`before_fit()`、`settle()`、`recover_reservations()` | 任务累计trial／fit／墙钟／decision／检索预算及取消 |
| `C/model/adapters.py` | `list_model_capabilities()`、`build_estimator()`、`fit_fixed()` | 从既有family分发提供真实执行schema；不采用模型知识卡片schema代替 |
| `C/model/search.py` | `run_bounded_search()` | 现有GridSearchCV/ParameterGrid/CV原语薄封装、完整管线每折fit及计数 |

可将函数数较少的decision_budget并入experiments，但预算账本仍须独立持久状态和单元测试。dataset_session的私有数据与决策表放同一owner/thread目录治理，不依赖跨任务资产库开启。

## 5. 配置、状态和控制权

### 5.1 配置草案

```yaml
# 拟新增；当前配置没有这份合同
nir_decision:
  default_strategy: legacy
  allowed_strategies: [legacy, llm_autonomous]
  autonomous_tasks: [single_target_regression]
  final_fit_policy: keep_calibration_fit
  max_experiments: 12
  max_parameter_trials: 30
  max_model_fits: 200
  max_wall_seconds: 900
  max_decision_calls: 40
  max_evidence_requests: 8
  max_invalid_decision_repairs: 2
```

调用 `nir_workflow(start, decision_strategy=...)` 时由服务器验证用户／配置许可，将resolved_policy、schema版本、能力版本和预算保存到本会话；无参数仍legacy。用户可选更小资源额度，不能扩大服务器硬上限。旧会话恢复缺少strategy按legacy读取，**不能为旧记录补上autonomous身份**。

`single_target_regression` 是新策略能力类型，需由已有analysis/calibration任务及可信target_count=1映射，不直接替换当前workflow的task_type白名单。分类、多目标和collection在映射时检查适配状态。

保留legacy的规则选方案能力，同时统一修正它的Test指导搜索。旧数字结果和工件不改写；新运行指标语义调整明确记录协议版本。若研究复现需要旧行为，另行申请明确的历史复现模式，不把其结果宣称独立最终测试。此点是保留兼容与科学修复之间的默认取舍。

### 5.2 新状态字段

```text
decision_strategy, resolved_policy, search_session_id,
split_manifest_id, cv_manifest_id, capabilities_sha256,
decision_context_revision, active_decision_id, active_experiment_id,
retained_experiment_id, budget_snapshot,
freeze_id, frozen_model_sha256, test_status,
stop_source, stop_code, final_evidence
```

完整不可变decision/feedback存在ProcessStore，新checkpoint只引用ID及有界摘要。已有attempts/reflections/retry_plans继续用于旧模式；新模式投影可给attempts兼容摘要，但author、selection_origin和metric_scope必须明确。

状态新增searching/frozen/testing；searching内next_action区分need_decision/execute_decision/query_evidence/consume_feedback。新模式任何质量达标不直接review；只有STOP或硬预算终结后、模型冻结和final_test完成才review／交付。

### 5.3 执行前提交、执行后记录

1. 验证context revision、证据scope、用户约束、运行能力、计划参数及历史重复。
2. 同一事务接受决策、保存不可变快照、预留预算和创建执行lease。
3. `nir_execute_decision(decision_id)`服务器读取已接受计划，禁止追加参数。
4. 专业工具记录开始、每次fit和终态。消费原decision仅一次；重放返回旧结果。
5. 比较actual_spec和授权spec；optimizer只可选已授权空间内参数。
6. 原子保存execution/feedback及新context revision；失败同样有配置、成本和错误。

不要求LLM自报“已经执行”，也不允许它修改rationale或expected_outcome。修正计划产生新ID；执行后的解释写后验记录，不覆盖原预期。

## 6. 阶段一：修正数据隔离，冻结验证和决策接口

**依赖：无；先于LLM功能。**

主要改动：dataset_session、metrics scope、modeling/single_target训练内核拆分；workflow策略字段；中间件工具权限；ProcessStore分区及Test消费账本。先让原规则控制可以使用安全搜索内核，再做新模式。

任务：

- 把每轮Test预测／质量从训练入口移到finalization；Cal/Val训练函数签名不接收Test。
- `evaluate_quality` 增加search与final的明确调用，reflect／LV建议／best只用Val；`nir_compare`改为Val比较。去掉Test派生grade、action、hint和历史字段。
- 修复compare的all_metrics数组与reflect单对象合同不匹配；比较展示不冒充带model_path的可注册实验结果。
- 固定split与CV fold_manifest；禁止跨入口、比例或seed改变成员；group任务内部CV也隔离。
- 自动预处理CalCV正确逻辑复用，补后续模型CV外层状态型预处理的每折fit。
- CSV最终Cal+Val阶段只固定参数refit，禁止调用再次搜索的 `_train_one_model()`；顶层RMSECV不再冒称Tuning误差。
- 可信摄取私有存储／句柄及允许工具集；新模式关闭raw read、通用执行、MCP fit及旧建模绕行；搜索报告／图表仅Cal/Val。
- 单次FinalTest lease与hash绑定；测试失败不回到searching。原数据／工件保持原始记录。

输入→输出：上传绑定与冻结split_spec→session_id/manifest/Profile；授权搜索计划→train_cv/validation结果；freeze_id→final_test结果与兼容metrics。

验收：

- 固定分区后改变Test X/y，搜索期科学观察和所有指标不变；只有最终结果变化，数据摘要哈希变化不被用于选择动作。
- 对工具结果、workflow注入、summary、错误、reports/charts和文件访问做scope检查，没有Test值或派生判断。
- 状态型预处理每折fit只见折训练行；分组成员不跨CV折。
- 同一会话Test只能发放一个lease；再次finalize不重新计算；失败测试不启动新模型。
- 原报告和预测工件仍可读取；没有重写历史metrics。

建议测试：新增 `backend/tests/test_nir_search_dataset.py`、`test_nir_test_sealing.py`；扩展 `test_nir_workflow_middleware.py`、`test_nir_preprocessing_selection.py`；core anti-leakage及groupCV测试使用spy训练边界，不能只断言JSON不含test键。

## 7. 阶段二：LLM初始模型管线自主选择

**依赖：阶段一；交付首轮显式fixed/grid基本能力。**

主要改动：decision_contracts、decision_context、experiments submit/execute、model adapters；Agent Prompt、Skill及config注册；ProcessStore不可变decisions表。

任务：

- 建立Cal Profile、任务限制、方法卡片／runtime能力和预算的统一context；初始history为空。
- 接通Live Chemotools目录尝试与方法检索，不再用card排序生成自动执行顺序。
- LLM首轮EXPLORE提交alternatives、理由、证据、具体pipeline/model和预期，执行前保存。
- 首批四family公开完整fixed参数和受限grid；参数schema来自实际adapter，映射model_*卡片ID。
- 新模式拒绝method/pipeline/wavelength_selection的auto；合法Raw []可接受。
- 不把旧auto_eligible、priority、低SNR诊断或explicit_only当成新候选的自动选模规则；硬权限和数学合法性单列，其余作为LLM证据。
- 不调用旧 `_select_model_family()` 的额外PLS baseline，也不调用CalRMSECV winner替LLM选预处理。

输入→输出：DecisionContext→LLM DecisionProposal→DecisionRecord→实际固定管线及Val结果。

验收：LLM可首选Ridge或SVR；执行日志没有隐藏PLS拟合。用户明确指定SNV/PLS等仍为硬约束；错误参数返回字段路径而非自动改正。改变卡片priority不会改变已接受计划；首轮已有决策快照而不只有事后报告。

建议测试：新增 `test_nir_decision_contracts.py`、`test_nir_decision_context.py`、`test_nir_autonomous_initial.py`；用真实小数据和spy/fake LLM验证至少Raw+PLS与自构组合+非PLS，不固定这些方案为系统候选清单。

## 8. 阶段三：多层级动作与LLM保留／停止选择

**依赖：阶段二；此时每轮都可记录完整决策。**

主要改动：workflow新状态、submit动作判别、context历史、知识工具阶段权限、middleware选择来源及response_grounding。

任务：

- TUNE、CHANGE_PREPROCESS、SWITCH_MODEL、EXPLORE、STOP及REQUEST_KNOWLEDGE/DIAGNOSTIC逐项验证。
- 检查base_experiment_id，动作所变配置与声明一致；新增signature按完整历史去重。
- LLM每轮显式retained_experiment_id；统计best_validation_rmse只供参考，不自动冻结最低误差模型。
- 新模式移除quality passed／R² plateau／候选耗尽驱动的强制优化停止。
- 用户强制family与LLM自选family分开；LLM自选失败后可自主换家族，用户硬要求不可违反。
- 知识请求先记录再执行，新证据归decision_id；无需伪造should_retry才能进入knowledge。
- 修正hint的C/D/F与实际等级不一致，legacy保留清晰触发；新模式LLM决定何时检索。

验收：两次验证改善<0.02仍能由LLM合法EXPLORE；达验证门槛后能继续TUNE；未达门槛能主动STOP并诚实报告。已尝试的参数可在理由明确的replicate路径重新评估但须记录预算。历史失败重新进入context，不用LLM自己编写history。

建议测试：新增 `test_nir_autonomous_actions.py`；扩展workflow、middleware、知识阶段及grounding测试。Fake LLM分别输出六类后续动作，断言实际执行及保留选择，不以精度升高作为动作控制权通过条件。

## 9. 阶段四：局部参数优化与实验反馈闭环

**依赖：阶段二的基本搜索；阶段三的动作可调用它。**

主要改动：core search、各model固定拟合adapter、task fit账本、feedback和搜索trial持久表。阶段二已有最小网格执行，本阶段完善区域变更、失败、成本和反馈。

任务：

- LLM指定离散space、参数、局部max_trials/max_fits；复用GridSearchCV／ParameterGrid与成分枚举。
- 完整raw-Cal管线每折fit；固定fold及groups；严格范围和非有限值检查。
- 数值程序按train_cv_rmse选当前授权空间内参数；Val只评价局部结果，不让optimizer跨家族。
- LLM下一轮可扩大或缩小search_space；同一族的未改善、边界命中、失败trial都有真实反馈。
- 扩展trial结果：params、每折分数、真实fit、失败码、selected_by=optimizer；不要只返回当前best。
- DecisionRecord预期不改写；FeedbackRecord计算与前轮／retained的delta、预期达成和成本，LLM后验说明单独保存。
- 对其他现有family逐项公开支持能力；未适配能力维持明确错误。

验收：SVR可指定不同于内置网格的合法C/gamma；PLS能真正固定n_components；fixed模式没有第二次搜索；optimizer输出参数全部属于LLM授权space。全trial失败返回失败而非强行1成分成功。预算精确计候选×折+refit，失败fit也扣减。

建议测试：`nir_core/tests/test_model/test_bounded_search.py`、`test_model_adapters.py`；backend `test_nir_autonomous_feedback.py`、`test_nir_search_budget.py`。固定4trial×5fold+1refit应为21fit；被提前取消时按实际已开始fit计，不能总填21。

## 10. 阶段五：自主停止、异常恢复和过程记录

**依赖：前四阶段；补齐可部署的生命周期。**

主要改动：decision_budget、ProcessStore事务迁移、run恢复、finalization、注册及最终交付。

任务：

- 停止来源llm/user/budget/safety/tool_failure/decision_invalid分别保存；当前STOP与强制预算终止不能混称。
- 保留验证质量作为证据；LLM STOP后冻结此前明确选定工件，默认不再拟合。
- 可选Cal+Val refit仅使用已冻结参数／步骤，完成hash后才发Test lease。
- 用户取消中断执行且不自动运行Test；调用重放幂等，服务重启识别未完成attempt和预算预留。
- 墙钟严格限制需要worker或子进程终止；若仅checkpoint，明确cooperative能力限制。
- 启动时为finalization预留固定refit／评价／持久化资源，与search budget分开；搜索耗尽仍能可靠结束，但不得因此重新开放搜索。
- 无效决策最多两次修正；超限没有legacy fallback；错误及未完成动作可追溯。
- ProcessStore user_version从当前2进行事务迁移，决策同ID不同hash写入拒绝；原表及chart文件保留。
- 最终model/metrics/science/freeze/test绑定；报告、ZIP、注册和资产库只使用最终工件。
- API扩展decision/feedback可选投影，已存在前端选择器可以接入；本阶段不要求重做页面。

验收：重启后预算不会回零；并发重复execute不重复fit；已有决策理由无法被结果覆盖；任何actual-spec偏离不可注册；错误重放不能覆盖成功记录。Test结果不好只能影响final_quality／注册，不会新增search实验。

建议测试：新增 `test_nir_autonomous_recovery.py`、`test_nir_decision_store.py`、`test_nir_autonomous_finalization.py`；扩展process router、registration、delivery及实际process-training测试。

## 11. 阶段六：系统功能验证与发布闸门

**依赖：完整安全闭环；不是论文比较／消融实验。**

| 验证层级 | 场景 | 通过条件 |
|---|---|---|
| 合同单元 | schema、证据ID、非有限参数、互斥步骤、Raw、过期context、重复签名 | 明确错误码／字段；不替换参数 |
| 数据隔离 | 固定split的Test扰动、文件/MCP绕行、历史消息及图表读取 | 搜索证据无Test依赖；可信权限阻断 |
| 数值边界 | MSC/EMSC/scaler每折fit；group CV；supervised selector | fit只接收折训练样本；没有跨组泄露 |
| 选择控制权 | Fake LLM首选非PLS、切换／TUNE／改预处理／EXPLORE／STOP | 实际动作及参数符合提交；没有隐藏auto |
| 参数搜索 | 合法自定义grid、fixed及改区域、全部失败 | params在授权space；真实trial/fit和failure |
| 反馈学习 | 假设不兑现、错误模型、重复无效空间 | 后续context包含旧预期及失败；原决策不变 |
| 停止 | LLM、用户、fit耗尽、wall耗尽、安全、反复技术失败 | stop_source准确，恢复不冒称自主停止 |
| 幂等和恢复 | 同ID双调用、checkpoint/DB提交中断、重启、最终测试中断 | 不重复执行或测试；账本和记录一致 |
| 生命周期 | 单目标CSV/NPZ→搜索→冻结→Test→报告→批准→注册→预测 | 工件hash绑定、scope正确、旧预测回放兼容 |
| 兼容 | legacy规则选择、旧metrics/ProcessStore、资产库关闭 | 旧记录可读；没有新模式隐式回退 |

建议仅用仓库已有合成数据生成器和少量固定分区小样本；不跑Wheat/Meat/Corn大规模搜索。少量真实LLM端到端功能验证使用人工查看决策／实际执行对应关系；同一个已曝光Test不能反复用于模型改进验证。真实LLM测试须有明确调用成本、固定小预算和可重放记录。

现有回归锚点：`backend/tests/test_nir_workflow.py`、`test_nir_workflow_middleware.py`、`test_nir_method_knowledge.py`、`test_nir_method_catalog.py`、`test_nir_preprocessing_selection.py`、`test_nir_process_training.py`、`test_nir_process.py`、`test_nir_process_router.py`、`test_nir_resource_budget.py`、`test_nir_end_to_end_regression.py`、`test_nir_delivery.py`；core preprocessing、CV、validation及anti-leakage测试。

实施代码后按仓库约定执行backend完整测试及NIR fast/slow适用门禁；前端合同若实际修改，再执行对应unit和pnpm check／桌面Chromium流程。本轮没有业务变更，**未运行这些训练／测试**；本轮验证仅文档内容、引用、JSON示例、交付文件及Git变更范围。

上线建议先feature flag关闭新模式，在小功能验证通过后按会话显式开启；观察invalid率、actual-spec一致率、预算计数、无Test搜索和记录完整性，而不是只看模型R²。

## 12. 兼容性影响与回退

| 风险／影响 | 应对 | 回退边界 |
|---|---|---|
| 旧工具一次调用立即返回Test及报告的行为改变 | 协调器用相同用户流程自动完成search/finalize；公开协议版本及输出scope | 可回到旧规则选方案；不可恢复Test指导下一轮作为普通默认行为 |
| 注册要求当前attempt，LLM保留较早实验 | freeze绑定retained experiment及其实际artifact，不直接改当前attempt路径 | 注册只认冻结链；未适配时明确拒绝注册 |
| CSV合并Cal+Val后原先重新搜索 | 保留固定参数refit或默认Cal工件；报告分清两种fit_scope | 旧工件可预测，不改写其历史指标 |
| 搜索指标顶层RPD/RMSEP兼容 | 搜索合同scope显式；final输出旧字段 | 老消费者从final适配器读取，不对搜索写伪值 |
| 方法库管理修改 | decision固定snapshot hash及provider；执行前核查能力版本 | 旧决策以旧证据回放，版本不符返回需重新决策 |
| 无效LLM输出／理由薄弱 | strict合同、证据引用、有限修正及人工审查 | 终止或新会话明确选择legacy，不能同会话秘密fallback |
| 长训练／GridSearch内部循环 | fit钩子、任务累计预算、worker超时及取消 | 能力不足就限制模型／空间并告知；不可虚报硬截止 |
| DB/checkpoint一致性 | decision事务、唯一ID、lease、append事件及恢复对账 | 安全阻断；不删除记录后重跑 |
| 新图表／前端旧字段 | 可选schema，旧selector不读取新增字段时仍显示原流程 | 关闭新投影，保留后台记录，不重做前端 |
| 用户指定family失败 | 根据user constraint保护；LLM自选family自由重新决策 | 只有违反用户硬要求才需用户更改要求 |
| 旧Test已曝光 | 会话标记contaminated，另用未参与决策的独立Test | 保留旧记录为legacy；无法通过换名字恢复独立性 |

回退以新会话显式策略选择和feature flag为单位；活跃autonomous失败记录保留，不能改写strategy=legacy。数据库迁移尽量追加表和可选字段，旧版本应用可继续读旧结构；降级不删除新决策审计表。

## 13. 阶段依赖及完成定义

```mermaid
flowchart LR
    P1[1 数据隔离及协议] --> P2[2 首轮LLM决策]
    P2 --> P3[3 多层动作]
    P2 --> P4[4 可控优化与反馈]
    P3 --> P5[5 停止恢复及交付]
    P4 --> P5
    P5 --> P6[6 系统功能验证]
```

阶段四的数值adapter可在阶段三期间开发，但不能提前绕过阶段一的数据边界；发布必须合并验收完整纵向闭环。每阶段提交应围绕一个可验收能力，包含迁移／兼容测试及对应文档，不要求一次性改完全部family。

新模式完成定义：一次任务中，LLM依据Cal/Val真实证据选择首轮管线，能按反馈改变优化层级及参数区域，能明确保留历史工件和STOP；每项实际执行匹配已接受计划，所有试验／失败／成本可追溯；Test冻结后只评价一次并禁止继续优化；旧规则模式和原始工件仍可用。

## 14. 实施前需要确认的关键分歧与默认建议

1. **legacy是否保留旧Test反馈行为？** 推荐保留规则选择、修正泄露及指标命名；不把已知错误视为兼容需求。若需要完全历史复现，另设明确用途和污染标记。
2. **最终是否Cal+Val重拟合？** 推荐默认保留Cal工件；可选启动时声明的固定参数refit。不能最终重训时又重新优化。
3. **第一批是否必须全12种family／分类／多目标？** 推荐先单目标四family贯通；运行时明确支持清单，其他原功能保持legacy，避免扩大方法研究范围。
4. **“最佳”模型谁决定？** 推荐LLM保留选择，验证RMSE榜单为证据；数值程序只在LLM指定局部空间选参数。
5. **数值搜索是否引入新框架？** 推荐不引入；先grid/fixed及完整管线CV，预算靠现有原语薄封装实现。
6. **是否重做过程网页？** 推荐不重做；先完善数据记录及owner API，之后仅最小合同和字段显示适配。

以上分歧不妨碍本轮交付。下一步实施应从阶段一开始，按用户后续指令进行；本轮到此只完成审查和设计。
