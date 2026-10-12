# ChemAgent 第四阶段补充修复与最终验收报告

日期：2026-10-09。仓库：`D:\ChemAgent\nirs-agent`；分支：`dev/Duan`；HEAD：`ef288ea3ec241507c63164828a012a2fd519a440`。

**结论：真实 DeepSeek 已完成一次自然语言任务 → 初始方案 → 数值执行 → 真实反馈 → LLM RETAIN → Freeze → 唯一 Test → 模型与报告交付。** 本次真实停止来源为 `budget`，不是 LLM 主动 STOP。真实运行有一个成功实验、两个已接受决策；没有第二次训练，不把 RETAIN 宣称为已执行的 TUNE/SWITCH_MODEL。LLM 主动 STOP 及多轮优化动作仍由真实 Lead + Fake LLM 回归覆盖；本轮没有单独证明真实 provider 的主动 STOP 分支。

交付流程验收通过不等于模型质量通过：本次合成数据模型的最终质量门禁未通过，用户审批为 pending、未注册。未扩大训练预算、未强制模型/优化动作/停止时机、未自动进入第五阶段。

## 1. 原真实运行失败原因

检查了原实施报告、`chemagent-phase4-real-llm-evidence.json`、五份 `.tmp/phase4-real-*.json`、对应只读 SQLite 状态及当前 Prompt/Skill/工具/预算代码。没有改写这些历史运行记录。

| 排查项 | 实际证据与源码原因 | 修复与回归 |
| --- | --- | --- |
| 自然语言启动混入 legacy 参数 | `43a3d00ca3d5` 出现 `nir_dataset_binding_invalid`、错误的 prepare 与 legacy planning/catalog 路径。原 workflow 工具公开所有任务/旧预算/数据绑定参数；完整交付模式缺少单独的启动 Schema | Delivery 启动视图只列所需字段，绑定 sealed/LLM/rounds/delivery 标志，排除 legacy budget 和未使用的 lineage 参数；接收端与权限仍作独立检查。Fake intake 与完整自然任务回归通过 |
| prepare/context 循环 | `71e4fc9bc236` 19 次请求、0 fits。先前消息压缩删除 prepare 收据，而 workflow 启动收据仍可见。旧 context 的哈希还包含每次变化的调用数/耗时，重复读取生成新版本 | 保留先前已修复的 prepare 收据/当前状态提示；新增按持久历史、阶段和 runtime/知识版本缓存不可变 context。重复读取不重读 NPZ、不重检索、不生成新 revision；当前阶段工具视图在拿到 context 后提供 submit/history |
| action 与 selected alternative 不一致 | `ba1c2425b632` 反馈提案顶层 action 与选中 alternative.action 不同。原 Alternative.action 默认 EXPLORE，统一 Schema 未要求显式填写 | Provider Schema 要求显式 alternative.action，按 initial/feedback/RETAIN 列允许动作；接收端错误明确指出 `alternatives.<index>.action` 及预期值。不能替 LLM 修改 action |
| 字段缺失、错误嵌套、只读参数 | `f9f991dd63c5` 把 optimization 放在 pipeline 内、fixed_params 放在 optimization 内；`ba1c2425b632` 给 ET 提交 random_state。原工具参数只是 `proposal: str`，真正合同只在 context 内，模型参数为自由字典 | 工具支持 proposal 对象，动态工具 Schema 给出正确嵌套、绑定字段与四 adapter 的合法参数值；random_state/n_jobs 等运行设置不进入可写参数。旧字符串兼容，StrictContract/实际 adapter 校验不变 |
| 连续修正仍失败 | 原 feedback Schema 把 uncertainties/hypotheses/not_selected_reasons/expected_fits 标为有默认值，但 submit 在反馈阶段实际必需；组合错误仅提示 `feedback: require ...`，LLM 随后编造 candidate_rejection_reasons 字段 | Feedback Schema 明确 required/minItems/minProperties。语义错误逐字段返回；error 附当前 context、阶段、允许动作、必填字段、剩余修正次数与嵌套说明。非法计划仍拒绝；修正次数上限仍为 2，不静默补字段/参数 |
| 终结 token 额度不足 | `025aacfecd87` 有两个成功实验、17 fits；终结 context/history 两次请求已扣 195686 准入 tokens，原 250000 额度无法准入下一次。终结 context 仍重复完整训练 Schema，工具库存和旧证据也扩大请求 | 终结视图只有 RETAIN 合同，去掉训练定义，终结流程只暴露相应 process 工具。维持既有 6 次调用/500000 tokens，未无限扩容；按完整转换后的工具 Schema 计费而不是仅 args properties |
| 有成功模型仍无法 RETAIN/Freeze/Finalize | 历史 c/e 的无效提案触发 safety_closed；d 的 reserve 耗尽。另一个问题是 reserve_call 在准入时才关闭搜索，但该次 provider 请求已构造旧 feedback/搜索工具，可能用终结额度提交旧动作 | 搜索准入关闭时先保存 budget 终态并返回“需重建请求”，刷新终结状态、丢弃旧 context/tool pairs，要求先取 final context，再按新的请求计费。RETAIN 仍是 LLM 提交零 fit 决策，程序不选 best。Fake 调用预算耗尽分支完整交付通过 |

旧 c/e 的 safety_closed 是真实失败，不从历史强行恢复或替模型补造 RETAIN。旧 d 仍保留原失败记录。新验收使用新线程、新私有目录。

## 2. 修改的代码与接口

| 文件 | 本轮修改 |
| --- | --- |
| `backend/packages/harness/deerflow/community/nir/decision_schema.py`（新增） | 从既有合同派生阶段 Schema、合法参数视图、提交指导、绑定实际 context 的工具 Schema；没有候选排序或模型选择器 |
| 同目录 `decision_contracts.py` | 增加可选 submission_guide；旧合同兼容 |
| 同目录 `decision_context.py` | 对象/字符串提案兼容；保持原始参数的 provenance 校验；context 缓存；阶段合同；具体反馈/action 错误路径 |
| 同目录 `decision_tools.py` | 对象输入、错误修正合同与剩余次数；保持实际 runtime/来源审计和接收端合法性检查 |
| 同目录 `decision_budget.py` | 搜索到终结的延迟准入；失败 provider 类型/HTTP 状态独立保存；不改变有限资源上限 |
| `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py` | 实际同步/异步 Lead 请求的当前阶段工具视图；完整 Schema 准入估算；预算切换请求刷新与证据压缩；真实失败原因保存 |
| `backend/packages/harness/deerflow/agents/lead_agent/prompt.py`、`skills/custom/nir-coordinator/SKILL.md` | 简短说明对象合同、正确嵌套与 context 复用；未靠追加长 Prompt 替代接口修复 |
| `backend/scripts/verify_nir_delivery_llm.py` | 简化自然任务；保存完整实际 tool_calls；增加工件加载预测、冻结身份、报告/ZIP、缓存重放和 Test 后搜索关闭断言 |
| `backend/tests/test_nir_delivery_contract_reliability.py`（新增） | 17 个 Schema/来源/缓存/修正/预算/完整 Lead 交付用例 |
| `README.md`、根与 backend `AGENTS.md` | 同步接口及开发约定 |

本轮没有重建 Agent、优化器、划分/封存系统或 ProcessStore；沿用原 v5 数据库、accepted decision → safe_search、freeze/finalize、报告/工件接口。实际 `config.yaml` 已具备八个工具，解析检查通过；没有修改密钥或该文件（mtime 仍为 2026-10-09 12:54:08）。没有修改前端或 nir_core 数值执行代码。

## 3. 决策可靠性与控制权

实际调用链：DeerFlow Lead factory/Prompt → NIRWorkflowMiddleware 的阶段视图与调用准入 → 真实 DeepSeek → server model-issued receipt → nir_submit_decision 严格验证及不可变保存 → nir_execute_decision 读取已接受 decision_id → 原封存数值服务 → 真实反馈 → 新 context → LLM 决策。

阶段视图决定允许读取/提交/执行/冻结的**流程步骤**，不决定建模动作：initial 仅 EXPLORE 是首轮定义，模型/预处理/参数仍由 LLM 填写；feedback 同时允许 TUNE、CHANGE_PREPROCESS、SWITCH_MODEL、EXPLORE、STOP；硬预算终态只允许零 fit 的 RETAIN。质量门禁、诊断标签和知识相关性不选模型、不强制正常 STOP。

Schema 中只读证据与可提交字段分开。完整模型参数的允许值来自现有四 adapter，预处理合法性继续由现有注册表与科学约束检查。context_id/hash/revision 在实际 provider 工具 Schema 中绑定，接收端再核对真实持久版本。顶层 action 与所选 alternative 的对应关系、真实 Cartesian 成本及动作边界仍由接收端检查。

对象输入没有经过自动修复：按原对象内容核对 server receipt 的 args_hash，再严格验证、保存。伪造 author、改写模型发出的参数、未提交直接执行、legacy/MCP/file 绕行仍被拒绝。执行后反馈不回写旧 rationale。

Fake 用实际 Lead/工具/合成训练验证了错误嵌套被拒绝后显式修正、LLM STOP、fit 预算 RETAIN、调用预算 RETAIN 三条完整链路；不是只测试 JSON 数据类。

真实验收后还修正了错误回执的剩余修正次数显示：配置允许两次修正，第一次错误后应显示 2，第二次后显示 1，第三次错误关闭后为 0。原有拒绝与预算上限没有放宽。该提示修正先写失败回归，再进行最终全库复验；没有追加真实 provider 调用，真实成功记录保持原样。

## 4. 预算与终结保障

任务启动时冻结搜索和终结两套 allowance，准备前 intake 调用转入同一累计 ledger。正常搜索不消费 finalization allowance。硬预算终止保留 `stop_source=budget`，禁止新 fits；已有成功实验由 LLM 明确 RETAIN，仍需现有 STOP evidence 与实验引用。

终结最短需要四次 provider/tool 回合：final context → RETAIN → freeze → finalize。默认六次允许少量历史读取/修正，但不是无限重试。真实本轮用四次，准入 charge 为 **249390/500000**，还剩两次调用；上下文中不再带训练合同。令牌准入仍为保守 UTF-8 字节估算加有界输出，actual usage 独立计数；没有金额上限或严格 wall-time 中断保证。额度不足仍明确关闭，不能补造选择。

完整转换后的 function schema（包括定义、description）进入估算，修复了只统计 tool.args 时漏算定义的问题。搜索额度在模型调用准入处耗尽时，旧请求不直接使用 final bucket；先重建只读终结请求，再准入。provider 失败保存 exception_type/http_status，独立记录 terminal_failure_source；既有 budget 来源及完成的 Test 缓存不被覆盖。

交付完成后的叙述/审批沿用原 runtime 限制，不计入不可变交付成本快照。本次 ledger 纳入 10 次调用（search 6、finalization 4），之后 3 次实际请求也保存在验收证据中；完整真实请求共 13 次，未把 10 次快照宣称为总开销。

## 5. 自动化测试

新增行为先运行失败测试，再实现修复。第一轮 red 为 8 failed/1 passed；参数只读字段回归也先观察到失败。最终全部新增用例随 NIR 回归通过。

| 检查 | 结果与证据 |
| --- | --- |
| 第一至第四阶段全部 backend NIR（最终） | **735 passed、2 skipped、0 failed**；由同次最终完整 backend XML 汇总 |
| 前一次独立 NIR 回归 | 734 passed、2 skipped、0 failed，73.37 秒；`.tmp/phase4-accept-nir1.log/.xml`；之后新增修正次数显示用例 |
| 本轮完整 backend 修改前 | **6391 passed、88 failed、45 skipped**，437.93 秒；`.tmp/phase4-accept-baseline.log/.xml` |
| 本轮完整 backend 修改后（最终） | **6408 passed、88 failed、45 skipped**，378.63 秒；`.tmp/phase4-accept-final.log/.xml` |
| 逐 testcase 对照 | 88 个失败节点完全相同，**新增失败为零**；NIR 新增 17 个通过用例；`.tmp/phase4-acceptance-failure-diff.json` |
| 相关 Python Ruff、format、git diff --check | 通过 |
| 实际 config 的八工具解析 | 通过，0 provider calls |

全库不是全绿。88 个既有失败主要为原 Windows/权限/符号链接、可选依赖及旧 sandbox/MCP/skills/channel 行为，本轮未修复或隐藏。第一至第四阶段相关回归包含参数精确执行、累计资源、硬约束、封存目标扰动、绕行权限、来源绑定、STOP/RETAIN 冻结、单 Test、损坏/取消/并发/缓存/导出恢复及 legacy 工件兼容。

未改前端/数值内核，因此本轮未重复上一轮的 frontend/nir_core 全量测试；没有浏览器端到端或部署验收。

## 6. 真实 DeepSeek 完整运行

获得本次联网/API 授权后只运行了一次：总请求上限 24、累计拟合上限 18、三个实验上限、三折 Cal CV；search 18 calls/800000 准入 tokens/180 秒，finalization 6 calls/500000 tokens/180 秒，单次输出上限 3600、timeout 45 秒、provider retries=0。

使用当前实际配置的 `deepseek-flash` / `PatchedChatDeepSeek`，不是 Fake；原 Lead factory/Prompt、实际注册的八个 NIR 工具、NIR 与 DynamicContext middleware。为独立验证省略无关 Gateway/UI/标题/记忆服务，使用新的 `.tmp/phase4-accept-real-a-home`，没有从程序提前启动工作流或指定模型。

合成数据：60×12，Cal/Val/Test=36/12/12；随机源 4909，数据划分及 KFold seed=42。CV manifest 为 `cv_77338568696792a153131548a89edab80bdd1bd360d53ce26a0a4d21a06e7303`。

| 实际事实 | 结果 |
| --- | --- |
| thread_id | `phase4-real-6bed97e17d19` |
| session_id | `search_e0594759fa0444419bded890185cc2c4` |
| 自主初始选择 | SG smoothing(window=7, order=2) → SNV → PLS；比较了未执行的 Ridge、SVR 候选 |
| 授权优化 | n_components=[1,2,3,4,5]，五个 Cartesian 候选；其他已声明/默认合法参数保持不变 |
| 实际执行 | 五候选×三折+一次 Cal fit = **16 fits**；只训练 PLS，无隐藏 Ridge/SVR 比较 |
| 实际完整 PLS 参数 | copy=true, max_iter=500, n_components=4, scale=false, tol=1e-6 |
| 初始决策 | `decision_d820d4cdf63795b6d17027e1af7860f9` → `exp_d5f9f381702aeeaec48423b9e94c081e` |
| 反馈后 LLM 决策 | `decision_82f662f291cc3f89680a5745c2024c67`，phase=final_selection/action=RETAIN，明确引用上述成功实验 |
| 停止原因 | 剩余 2 fits，不足一个三折候选所需的 4；程序预算终止，**stop_source=budget**；没有伪造主动 STOP |
| Freeze / Test / Export | 指定实验冻结；holdout.npz 只打开一次且处于 durable testing lease；stage=finalized，完整八文件发布成功 |
| 总 provider 请求/耗时 | **13 次 / 43.297 秒** |
| 完整请求 actual usage | input=226425、output=4464，总计 **230889 tokens**，13 次都有 usage；金额未由 provider 提供 |
| 不可变交付 ledger scope | search 6 calls / 397449 charged；finalization 4 calls / 249390 charged；actual input=141561、output=3502 |
| 审批/注册/质量 | user_approved=false、registered=false、quality_passed=false；已交付，等待用户审批 |

两个已接受决策的 origin 均为真实 `lead_agent_model_call`、verification_kind=provider_model，关联同一个真实 run_id、message_id、tool_call_id、响应哈希。实际执行配置与 accepted_spec 相同；Retain 实验与 Freeze 工件一致，原始选择理由未修改。

本次没有非法模型决策需要修正；接口修正能力由 Fake 的真实工具循环证明。另有两次真实 workflow 调用被正确拒绝：准备前尝试 legacy 流程动作、交付后尝试未审批 complete。LLM 最终返回报告而没有越权审批/注册。这些失败 tool_calls 原样保留，不能把该次运行描述为每个调用都成功。

| 指标（真实数值工具） | 范围 | R² | RMSE |
| --- | --- | --- | --- |
| Cal | calibration，n=36 | 0.504403 | 1.202522 |
| Cal CV（OOF） | calibration_cv，n=36 | 0.303920 | 1.425142 |
| Val | validation，n=12 | 0.542181 | 1.437366 |
| Test | final_test，n=12 | 0.437618 | 1.524348 |

不把模型质量较差视为继续看 Test 改模型的理由。该结果只验证自主流程、执行一致性、封存和交付；不是论文方法效果证据。

## 7. STOP/RETAIN → Freeze → Test → Report 验收

**真实预算 RETAIN 完整链路成功；真实主动 STOP 本轮未单独运行，自动化真实 Lead + Fake 的 STOP 完整链路通过。** 没有通过固定规则指定下一模型或自动 best selection 取得验收。

已验证：冻结实验就是 LLM 指定的 retained_experiment_id；没有 Cal+Val 重新训练；最终 holdout 只读一次；重复 finalize 返回缓存，不再次预测 Test；Test 后搜索 context 被拒绝。原工件与导出工件分别加载，对三条新输入预测一致。报告包含全部已接受 decision_id、原始理由/候选、实际配置/参数、执行成本与分范围指标；ZIP 存在，manifest 校验通过。

交付位置：

- [真实运行模型报告](D:/ChemAgent/nirs-agent/.tmp/phase4-accept-real-a-home/threads/phase4-real-6bed97e17d19/user-data/outputs/search_e0594759fa0444419bded890185cc2c4/delivery_96599133527945cba25ee9894e6f0ef4/report.md)
- 同目录 `report.html`、`model.joblib`、`model.joblib.manifest.json`、`metrics.json`、`decisions.json`、`delivery.zip`、`publication.json`。
- [验收证据](D:/ChemAgent/nirs-agent/docs/chemagent-phase4-final-acceptance-evidence.json)：保存原真实调用记录/提案/反馈/最终结果、来源文件 SHA-256、预算 usage 以及全量回归失败对照。原证据仍为 `.tmp/phase4-real-6bed97e17d19.json`；日志为 `.tmp/phase4-accept-real-a.log`，私有持久库在该新线程的 `users/default/threads/.../nir-process/index.sqlite3`。

历史模型、报告、五份原失败证据和原有未提交修改保留；所有新工件落在新私有线程及唯一发布目录。没有 reset/checkout/提交/推送。

## 8. 限制与后续建议

1. 一次有界真实成功证明链路可运行，不代表所有 provider 响应/数据/随机条件下均稳定。真实主动 STOP、真实第二次训练后的 STOP 本轮未验证；本次反馈后的新决策为 RETAIN。无需为本轮验收硬编码更多轮次。
2. LLM 仍可能发出当前 Schema 未列出的工具名，权限层必须继续作为独立防线；本次两次 workflow 拒绝说明该防线仍必要。模型理由的科学正确性不能仅凭 evidence ID 自动证明，报告将其与实际数值证据分开保留。
3. 未选候选是比较提案，不是已执行实验；只有被选择的授权计划进入数值拟合。不能把未执行 Ridge/SVR 写成比较实验结果。
4. context 中预算为创建时的不可变快照；实际累计资源在服务端和 model admission 校验，缓存不刷新额度。终结资源仍有限；大历史/慢服务/异常响应可能安全失败。未知 lease 仍要求人工处理，不自动重发 Test。
5. actual usage 与保守准入 tokens 不同，墙钟合作式检查、无准确金额硬限制。交付后常规叙述/审批开销不在交付成本快照中；本次完整 13 次开销另行保留。
6. 用户硬约束仍为现有结构化模型/管线/预算/协议约束；任意自然语言质量要求尚未全部形式化。模型质量失败和真实用户审批/注册不能由交付完成自动替代。
7. 全库 88 个既有失败、完整服务部署/浏览器验收和大数据表现不属于本次已通过范围。没有新模型、前端重构、大规模真实数据或论文对比实验。

本轮停在第四阶段补充验收；等待用户下一步指令。
