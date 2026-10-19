# ChemAgent 第四阶段实施报告

日期：2026-10-09。工作区：`D:\ChemAgent\nirs-agent`。分支：`dev/Duan`；HEAD：`ef288ea3ec241507c63164828a012a2fd519a440`。

**原实施轮结论：第四阶段执行代码和自动化完整链路已实现并验证；当时真实 DeepSeek 完整交付尚未通过验收。** 真实服务可用，已经观察到自然语言启动、自主初始选择、数值执行和基于反馈的模型切换；当时尚无一次真实调用完成保留、冻结、Test 和报告。下面严格区分原实施轮的实现、Fake LLM 验证和真实运行证据。原实施轮未进入第五阶段，未提交、推送、切换分支或重置工作区。

## 1. 第四阶段实际实现

补充验收更新（2026-10-09）：本文件上方结论记录原五次真实尝试的状态。后续独立修复已完成一次真实 DeepSeek 的预算 RETAIN → Freeze → 唯一 Test → Report/模型交付，详见 [第四阶段最终验收报告](chemagent-phase4-final-acceptance-report.md)。原失败证据保留；没有追写或伪造旧运行结果。

### 前置审查与复用

已检查 README、根及 backend AGENTS、三个设计文档和前三阶段实施报告。开始时有大量前三阶段未提交修改和未跟踪文件，均作为现有基础保留，没有重新建立 Agent、数据分区、优化器或实验数据库。

实际源码具备 `sealed_holdout`、四模型固定参数/有界网格、真实模型调用收据、不可变决策与多轮反馈。第三阶段 STOP 只到 `awaiting_final_confirmation`，自主模式中间件禁止 freeze/finalize；实验、fits、动作及 LLM 成本也有分散的旧范围限制。这些差异由本阶段显式新开关解决，不改变旧模式的默认语义。

实际忽略文件 `config.yaml` 已注册本阶段需要的八个工具：workflow、prepare、context、submit、execute、history、freeze、finalize。`--check-local` 从该配置解析实际工具对象成功，provider 调用数为零。本轮没有改写该文件、密钥或其他运行设置。仅更新示例配置说明。

### 本阶段新增/修改文件

以下路径相对仓库根；不把此前已有的未提交修改冒充本阶段成果。

| 文件 | 本阶段内容 |
| --- | --- |
| `backend/packages/harness/deerflow/community/nir/decision_budget.py`（新增） | 有限预算配置、服务器校验、调用前租约、真实 usage、intake 转移、终结额度、失败关闭 |
| 同目录 `decision_delivery.py`（新增） | 复用工件/报告/ZIP 的安全发布适配器与导出幂等性 |
| 同目录 `workflow.py` | `autonomous_delivery`、`budget_profile`、`task_budget`；记录真实 start 调用身份，澄清启动参数 |
| 同目录 `dataset_session.py` | 新模式预算持久化、STOP 绑定冻结审计、原 Test lease、真实 intake 成本转移 |
| 同目录 `decision_contracts.py` | RETAIN、final_selection、STOP 证据；新 context 版本，旧字段兼容 |
| 同目录 `decision_context.py` | 动态停止证据、剩余预算、只读最终选择、预算错误详情、保留选择与源区分 |
| 同目录 `decision_origin.py` | intake/真实 response 关联与 usage 来源，不以作者字段认定 LLM |
| 同目录 `decision_tools.py` | STOP/RETAIN 状态推进、明确合同修正错误、无效提案上限、安全失败源同步 |
| 同目录 `search_tools.py` | 原 freeze/finalize 接入新生命周期、独立审批证据、交付适配 |
| 同目录 `_report.py` | 在已有报告中增加范围明确的评价行、决策/停止/成本/实际配置 |
| `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py` | 真正 Lead 调用额度、当前状态提示、受控工具权限、明确请求不得降级 legacy、上下文压缩修正 |
| `backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 新模式启动、自主停止、精确提案和终结流程说明 |
| `backend/packages/harness/deerflow/agents/thread_state.py` | 可选新策略/预算/start 身份字段 |
| `backend/tests/test_nir_delivery_{budget,decisions,freeze,export,agent,recovery}.py`（六个新增文件） | 第四阶段预算、动作、科学协议、工件、真实 Agent Fake 模型链路、恢复/门禁测试 |
| `backend/tests/test_nir_multiround_context_cost.py` | 压缩保留准备证据、当前状态与关闭后的过期上下文删除回归 |
| `backend/scripts/verify_nir_delivery_llm.py`（新增） | 显式授权、有限调用/fit 的真实 provider 自然语言任务验证；失败也保存证据 |
| `README.md`、`AGENTS.md`、`backend/AGENTS.md`、`config.example.yaml`、`skills/custom/nir-coordinator/SKILL.md` | 使用约定与能力/限制同步 |
| 本报告及 `docs/chemagent-phase4-real-llm-evidence.json`（新增） | 实际交付状态、全部真实尝试及可追溯审计 |

### 真正调用链

```text
HumanMessage 自然语言任务
  → 原 DeerFlow Lead Agent / Prompt / nir-coordinator 工具约定
  → nir_workflow(start, llm_autonomous, sealed_holdout,
                 autonomous_rounds=true, autonomous_delivery=true)
  → nir_prepare_search（原单次数据绑定/封存）
  → nir_decision_context（真实 Cal 特征、知识、能力、历史、预算）
  → 真实模型调用收据 → nir_submit_decision（先持久化）
  → nir_execute_decision（只取已接受 decision_id）
  → 原 safe_search → Experiment → Feedback → 新 Context → LLM
  → LLM STOP，或预算终止后的 LLM RETAIN
  → nir_freeze_search（只冻结明确保留的原 Cal 工件）
  → nir_finalize_search（唯一 Test lease）
  → 原可信工件/报告/ZIP → review，等待真实用户审批
  → 仅在原科学/审批门禁允许时注册
```

生产调用链沿用原 Lead；测试在进程内使用它的 factory、工具与 NIR 中间件。没有新建另一种 Agent。新执行工具没有临时模型/参数覆盖入口。LLM 决策与数值优化器的控制界限保留：工具只执行选中候选的管线和授权参数区域，不隐式训练其他候选。

## 2. 预算改造

### 范围与默认值

新增能力仅在四个开关明确结合时生效。旧 workflow 的 `max_attempts=1..10`、旧多轮 `max_fits=1..200` / `max_decisions=1..24` 及直接安全服务的旧范围仍保留。

| 任务字段 | development | autonomous | 默认服务器上限 |
| --- | ---: | ---: | ---: |
| max_experiments | 5 | 40 | 40 |
| max_fits | 50 | 1000 | 1000 |
| max_decisions | 15 | 100 | 100 |
| max_wall_seconds | 600 | 7200 | 7200 |
| max_llm_calls（搜索含 intake） | 30 | 150 | 150 |
| max_llm_tokens（保守准入量） | 1200000 | 3000000 | 3000000 |
| max_output_tokens | 4096 | 4096 | 4096 |
| max_invalid_decision_repairs | 2 | 2 | 2 |
| final_llm_calls（独立预留） | 6 | 6 | 6 |
| final_llm_tokens（独立预留） | 500000 | 500000 | 500000 |
| final_wall_seconds | 300 | 300 | 300 |

字段对应 `NIR_AUTONOMOUS_<字段大写>` 环境硬上限。`task_budget` 是 JSON 字符串，可明确请求上限内不同额度；未知字段、非正整数、越界及 unlimited 都拒绝。profile 的默认值如果高于部署上限，也要求明确重新请求更小值，不静默截断。final_llm_calls 至少四次以容纳工具式终结，但任意上下文规模仍不保证这些资源足够。

启动后 policy 固定在现有会话 JSON。重复 prepare、旧工具、改策略或 restart 不能刷新。新模式不能同时使用非默认的旧三个预算参数覆盖 policy。单轮仍受 40 个 Cartesian 候选、200 fits、预处理/参数合法性及原 ResourceBudget 约束。原默认文件 512 MiB、矩阵 5000 万元素、估算峰值 4 GiB、单操作 900 秒等门禁保留，部署环境可以收紧。

### 统计与执行

高层接受动作、管线实验、参数候选和拟合调用分别计数。预计 fits 为选中候选网格数 × 冻结 CV 折数 + 1 次完整 Cal 拟合；固定参数仍做 CV，不隐式调参。

剩余 50 fits、3 折、20 候选的 61 fits 计划实际被拒绝，错误返回 remaining=50、cv_folds=3、candidates=20、required=61、max_executable_candidates=16；没有自动缩小到 16 个候选。LLM 决定如何修正。

显式交付请求在准备前就从既有 `model_issued_calls` 表持久租用 intake 调用额度；准备后转入会话累计账本。真实 start 收据补计首次调用，lease 标识避免双重计数。准备前/后并发或未知调用不退款、不重放。真正 Lead 同步及异步请求都先租用调用和估算 token，随后登记 provider input/output usage、耗时与 unknown usage。

准入估算包含系统提示、消息和工具结构的 UTF-8 字节，加有界输出；不是精确 tokenizer。actual usage 独立保留，超出估算部分继续扣记。最终叙述/后续审批在交付完成后用原 runtime 限制，不计入不可变交付成本快照。没有准确金额限制；provider 未给 usage 时不声称准确计费。调用硬上限可执行，token admission 保守且可能提前拒绝，也不能追回已发生的 provider 开销。

训练/终结墙钟限制是 checkpoint 的合作式检查，第三方 `.fit()` 不保证在截止瞬间被强制中断。模型 fits 指回归拟合调用/尝试，不计诊断 PCA 或预处理内部拟合为额外回归模型。失败的部分 trial 详情尚未完整持久化，报告把其候选数量标为 unavailable，不能用计划候选冒充完成数量。

## 3. LLM 自主停止机制

新模式不执行 legacy 的 record_attempt→阈值触发 review 或反思固定重试结束；权限层拒绝这些绕行。质量门禁和诊断仍计算，作为证据供 LLM 判断以及最终注册审查，不决定正常 STOP 或下一模型。

动态上下文从可信记录读取当前/最近/最佳/明确保留实验、已试区域、失败原因、上一轮假设及预期实现情况、真实预算。排名只是信息。当前用户硬模型/管线约束不可违反；LLM 自己先前选择的模型不会变成用户强制约束。

STOP 除既有观察、候选比较、理由、预期、拒绝其他候选原因外，要求零 fits、空执行方案、同会话实际成功的 `retained_experiment_id`、对应 `experiment:<id>` 证据和 stop_evidence（优势、局限、覆盖、未继续验证方向/原因）。接受时把真实预算快照固化到执行前记录。新反馈只另存，不能回写原理由。

实验/fit/动作/搜索调用/token/合作式时间终止记录 `stop_source=budget`。存在成功实验时提供只含现有 Cal/Val 的 `final_selection`，允许 LLM RETAIN 而不训练；`retained_selection_source=llm`，保持预算来源。不允许程序默认保留最小 RMSE。没有成功实验或终结额度不足时明确未冻结、无交付，不借旧模型或 Test 制造成功。

预算关闭后的无效终结决策或 provider 失败另外记录 `terminal_failure_source/reason`，不覆盖原 `stop_source=budget` 和原预算原因；这项修正有两个新增回归用例。用户取消、安全故障与 LLM 正常 STOP 有不同终态。连续无效结构提案限制是工程保护，与“连续建模效果不改善”完全不同。

自动化验证包括：好质量仍继续/提前 STOP；连续三轮无改善后仍可 TUNE、CHANGE_PREPROCESS、SWITCH_MODEL 或 STOP；保留更早成功实验且不自动改为最小 RMSE；11 轮实验共享 44 fits 并超过原 10 轮入口；非法参数、硬约束、重复区域均拒绝。

## 4. 冻结与最终测试

接受 STOP/RETAIN 后，workflow 从 `awaiting_final_confirmation` 要求 freeze；会话已 stopped 并绑定保留实验。更早模式仍停待确认，不自动解锁 freeze。

原 freeze 服务校验 owner/thread/session、真实 terminal 决策来源、成功执行身份、无活动决策/实验/模型调用、接受 spec 与实际参数、原 Cal/Val SHA、冻结 CV manifest、runtime adapter、用户硬约束、工件 SHA 和累计 fits/实验账本。只能冻结 LLM 明确选择的实验，不能换模型。

冻结直接复用已有 Cal 拟合 joblib，不重训 Cal+Val、不重新选预处理/波长/模型/参数。绑定两个决策身份、实验身份、完整参数、数据/CV 身份及哈希。`preprocessing_state_sha256` 覆盖序列化完整管线，`preprocessing_state_hash_scope=serialized_full_pipeline_artifact` 明确它不是独立预处理文件的哈希；工件实际保存拟合状态。

原 finalize 在 Test IO 前持久取得唯一 `testing` lease，检查工件及私有 holdout 哈希；成功结果缓存，异常进入 `final_test_failed`，不再访问 Test。重复 finalize 读取缓存。换模型或改哈希不产生新的许可。最终指标差或质量失败只改变报告/注册门禁，不重新打开搜索。

搜索上下文、模型/参数建议、反馈和只读最终选择均无 Test 指标、预测或统计。私有 Test 仅在数据接入验证/封存边界处理，后续搜索读取 `search.npz`；最终预测才打开 `holdout.npz`。Fake 完整链路用实际数值工具确认只打开一次 holdout；前置泄露/目标扰动回归和绕行测试通过。真实五次尝试均未取得 Test lease。

## 5. 模型交付

`decision_delivery.py` 只是原 `_build_model_artifact`、可信 manifest、`_report` 和 `build_delivery` 的适配层。按独立 generation 暂存、校验后发布，不覆盖历史路径。交付：model.joblib、manifest、metrics.json、decisions.json、Markdown/HTML、delivery.zip、publication.json。模型兼容原 `nir_predict` 的加载、预处理回放、监测/审计接口。

报告分别列 Cal、Cal CV、Val、Test；不把 fold-mean RMSECV 与 OOF train_cv RMSE 或最终 RMSEP 混作同一量。记录真实协议/折配置/seed/hash、完整实际管线/参数、每轮计划和反馈、原预期实现状态、最终选择/停止来源、成功/失败实验与资源成本。备选提案与实际数值 trial 单独保留，未选中管线不当成已训练模型。

交付构建 Cal 监测参考，未重新拟合回归模型或预处理。成本快照范围明确为任务 intake 到 finalize 工具，不虚报后续叙述/审批开销。科学诊断和最终评价工具保留。

搜索停止、冻结、Test 完成、quality_passed、用户批准、注册成功各自独立。交付初始审批/注册均为 false，review 等待真正最新 HumanMessage 的明确授权。AI 文本不能审批。注册只允许绑定的准确工件/指标，质量或科学校验失败不能强行通过。Fake 测试用现有真实预测和注册工具验证这些门禁；没有把任何真实验证模型自动注册。

## 6. 异常恢复和兼容性

- STOP/RETAIN 重放读取原决策；后续已冻结/交付时不把 workflow 回退到等待冻结，不产生新 fits。
- freeze 串行持久绑定同一工件，重复/并发请求不复制模型。账本、CV、参数、哈希不一致拒绝。
- Test lease 先持久后预测；并发只允许一个。中断/未知 lease 失败关闭，服务重启不自动重新测试。
- export 已完成时核验发布文件哈希并读取结果；捕获到的失败可新 generation 只恢复报告/发布，使用已保存 Test 指标。building 状态未知时需要人工审计，不冒然重放。
- 已完成 Test/交付之后的无效决策重放不能覆盖 finalized 状态；Test 完成但导出未完成时，终结调用额度或 provider 失败也不销毁缓存，有专门恢复回归。
- fit/最终 Test 前取消关闭状态；完成 Test 后取消不能撤销其消费。模型调用 lease 未知不重新请求。硬预算后未执行的已接受计划以独立失败反馈关闭，原计划不改写。
- 保留旧字段默认，ProcessStore 仍为 v5，未做破坏性迁移。阶段 1 手动封存、阶段 2 首轮、阶段 3 多轮及 legacy/分类/多目标业务入口继续回归；旧工件/报告测试前后字节一致，新路径不同。
- 没有改造前端源码。扩展回归时 pnpm 触发本地依赖重建，已中止；离线安装缺缓存而未下载。利用原 `.package-map.json` 和完整缓存恢复缺失的本地依赖链接/命令后，直接运行现有 Rstest，沙箱外通过；没有更改 package.json、锁文件或其他前端受管文件。

没有通用分布式事务平台或自动人工恢复；不确定租约仍要求操作者核查。

## 7. 测试结果

### 分批开发与最终回归

新增行为先写失败测试，再实施并跑对应回归。阶段记录位于 `.tmp/phase4-*.log`：

| 检查 | 实际结果 |
| --- | --- |
| 4.1 预算/工作流/封存/兼容 | 149 passed |
| 4.2 STOP/RETAIN/继续/失败 | 68 passed |
| 4.3 冻结/单次 Test/完整性 | 34 passed |
| 4.4 报告/加载/预测/门禁 | 22 passed，1 skipped |
| 最终定向，含修正后的来源/上下文/阶段 1–3/原部署回归 | **193 passed**，30.79 秒 |
| 完整 backend 修改前基线 | 6312 passed，88 failed，45 skipped |
| 完整 backend 最终复跑 | **6392 passed，87 failed，45 skipped**，376.21 秒；新增失败为零 |
| 全部 backend NIR 子集 | **718 passed、2 skipped、0 failed**（从同次完整 backend XML 汇总） |
| nir_core fast | 670 passed，2 skipped，26 deselected，37.84 秒 |
| nir_core slow | 26 passed，1 skipped，671 deselected，201.81 秒 |
| 根目录 skills 测试 | 51 passed，0.45 秒 |
| 前端单元，直接 Rstest 沙箱外 | 63 文件，529 passed、0 failed，9.204 秒 |
| 相关 Python Ruff / 新文件 format / git diff --check | 通过 |
| 真实运行配置工具解析 | 八个工具解析成功，0 provider calls |

完整 backend 最终与修改前逐 testcase 比较：87 个失败节点均已存在于基线，新增失败为零。基线另一个失败 `tests.test_mcp_file_migration.TestWorkspaceSnapshots::test_changed_workspace_files_detects_created_and_modified_files` 本次通过；该文件时间测试在前几次复跑中有通过/失败波动，未声称本轮修复了它。剩余失败主要涉及 Windows 权限/符号链接、缺少的可选依赖/工具环境、旧 sandbox/MCP/skills/channel 行为；没有扩大本阶段修复范围。最终 node 差异见 `.tmp/phase4-final-failure-diff.json`，完整失败清单见 `.tmp/phase4-full-final-g.log` 和 XML。仓库回归不是全绿，未把既有失败忽略。未运行完整浏览器端到端或服务部署验收。

Fake 验证使用原 Lead 与真正小型合成回归，不用伪造性能代替模型执行。同步/异步各验证 LLM STOP 与预算 RETAIN，训练→反馈→新决定→保留历史→冻结→唯一 Test→可预测工件/报告→审批分离。Fake 来源明确标为 fake_model。六个新模块加既有上下文回归共新增 79 个用例；覆盖取消、并发、重放、失败关闭、哈希损坏、完成缓存保护、导出恢复、30 种阶段×绕行调用组合等。

### 真实 DeepSeek 验证：未通过完整交付

配置解析和服务请求可用，实际模型类是 `PatchedChatDeepSeek`，model_name=`deepseek-flash`。每次独立线程/私有 `.tmp` Home，60×12 合成数据，seed=4909，原划分 seed=42，3 折 CV；不指定模型家族、预处理顺序或固定轮数。进程内使用原 Lead、配置工具及 NIR/DynamicContext 中间件，未测试 Gateway/浏览器传输。额外 smoke cap 为最多 24 次 provider 请求、3 个实验、18 fits、6 个接受动作；搜索额度 18 calls/800000 准入 tokens/180 秒；终结 6 calls/180 秒，前四次 250000、最后一次改为生产默认 500000 tokens。server 额度在验证进程内收紧，未修改本地运行配置。

公开证据保存于 `docs/chemagent-phase4-real-llm-evidence.json`，包括失败尝试，原始记录继续保留在 `.tmp`，没有后写 STOP 或伪造 Test：

| thread 后缀 | 实际结果 | fits / 完整交付 |
| --- | --- | --- |
| 43a3d00ca3d5 | 启动参数/legacy 歧义，GraphRecursionError；实际数据库无安全会话 | 无实验；首版脚本未完整保存失败调用成本，unknown 不记为 0 |
| 71e4fc9bc236 | 成功封存但 prepare/context 循环，预算关闭 | 0 fits；未 Test |
| ba1c2425b632 | 自主 SG+SNV+PLS 网格真实成功；后续 action/只读参数等错误超过修正限额 | 13 fits、1 成功实验；未 Test |
| 025aacfecd87 | PLS 网格→根据真实反馈 SWITCH_MODEL 为固定 SVR，保持 SG+SNV；剩 1 fit，预算终止，只读最终选择因终结额度不足关闭 | **17 fits、2 成功实验**；没有 RETAIN/冻结/Test/交付 |
| f9f991dd63c5 | 初始提案修正后真实 PLS 固定执行；反馈合同嵌套/缺字段/错字段连续错误，安全关闭 | 4 fits、1 成功实验；未 Test |

两轮真实证据具有 provider_model 收据、不同 decision_id/experiment_id、真实配置、Val/CV/拟合次数及 previous_experiment_id 关联。该成功部分不能证明完整交付。最后一次同时发生 token 搜索终止和提案失败的旧来源标记问题已用两个自动化回归修正；历史真实记录保留原样，不回写成新的“成功”。修正后没有追加真实 provider 运行。

真实完整路径失败是提案合同稳定性/终结资源分配问题，不是 provider 不可用，也不能用 Fake 结果冒充。共五次有界尝试后停止追加 API 成本。本阶段性能提高并非验收条件，也未运行 Wheat/Meat/Corn 大实验或论文对比。

## 8. 剩余问题与验收回答

### 能力状态

- **已实现并自动化验证**：有限大预算、累计/不可重置、LLM 自主正常 STOP、预算 RETAIN、历史成功保留、Cal 冻结、唯一 Test、可信模型/报告/ZIP、预测/审批/注册门禁、相关幂等与失败关闭。
- **真实部分验证**：自然语言到实际安全会话、LLM 初始选择、真实训练、反馈切换模型、累计 fits 和预算关闭、非法方案明确拒绝。
- **尚未真实验证成功**：实际 DeepSeek STOP/RETAIN→freeze→Test→交付全部串联；不能宣布真实完整流程验收通过。
- **部分实现/限制**：合作式训练期限；保守 token admission 而非精确计费；失败的部分参数候选记录；交付成本不含后续叙述；不确定 lease 需人工审计；生产 Gateway/浏览器完整运行未验收。
- **用户约束范围**：程序强制校验已绑定的模型、预处理、资源和验证协议约束；其他需求保留用户原文供 LLM 解释。任意自然语言定量质量门槛尚未完整结构化，不应声称执行器可以强制验证所有语义要求。
- **未实现且不在本轮**：金额硬限、严格抢占第三方 fit、通用自动崩溃恢复、Cal+Val refit、跨任务记忆、论文实验、前端重构/第五阶段。

进入后续阶段前，应先把真实 Lead 提案结构/语义可靠性和有限终结资源中的完整交付作为第四阶段待验收工作：依据现有失败收据改善结构化提交指导或兼容接口，保留所有合法性/来源/预算门禁，再另行授权小预算真实验收。不得为了“成功”增加固定模型/固定 STOP、放宽非法参数、回写原理由或手动补一个假 LLM 决策。既有全仓失败需要独立环境/兼容审查，不混入本轮大修。

### 六项验收问题

| 问题 | 基于实际证据的回答 |
| --- | --- |
| 1. 取消自主模式固定失败次数的正常停止？ | **是，新显式模式中。** 质量、平台期和数值失败仅是证据；无效提案限额是安全关闭。旧业务规则保留。 |
| 2. 更大可配置预算内 LLM 自决继续/停止？ | **实现并由自动化验证。** 11 个真实数值实验超旧 10 次入口；质量好仍继续、平台期后多种动作和提前 STOP 均通过，未宣称真实运行 40 轮。 |
| 3. 硬预算停止训练且保留真实来源？ | **是。** 真实两轮 17 fits 后 budget 关闭；终结不再拟合，后续错误保留预算来源的修正测试通过。 |
| 4. LLM 自选历史成功模型而非自动 best？ | **实现并由 Fake Agent/数值执行验证。** 可保留非最新/非最优实验，有证据/硬约束校验；真实保留尚未成功。 |
| 5. STOP 完整接通 freeze、独立 Test、交付？ | **代码已接通，Fake 完整链路通过。** 实际回归模型、Test 一次、导出、预测和门禁均验证；真实 LLM 路径尚未成功。 |
| 6. 一次真实自然语言任务完成全部可追溯流程？ | **本轮没有证明，尚未通过。** 已证实真实两轮反馈闭环，五次完整交付尝试均未完成 Test/报告；不能根据代码或 Fake 推断成功。 |

本轮到此停止，保留全部修改与证据，等待下一步指令。
