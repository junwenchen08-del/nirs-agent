# ChemAgent 统一自主建模入口与确定性自动交付实施报告

日期：2026-10-10。项目：`D:\ChemAgent\nirs-agent`。

本报告记录第六阶段补充改造。**代码改造、本地回归、Fake LLM 网页交付及一次真实 DeepSeek / Corn 完整验收已完成。真实运行执行 3 个实验、153 fits，LLM 主动 STOP，程序自动 Freeze→唯一 Test→Report/ZIP；20 次真实响应，770282 tokens。** 下文分别标明 Fake 合成数据和真实 Corn 证据，不能混用。

## 1. 实施前检查与保护

检查了实际源码、根目录及 backend/frontend 的 AGENTS.md、决策设计/路线图和第一至第六阶段报告。基于现有 DeerFlow Lead、Gateway、ProcessStore、SearchSessionStore、安全搜索和交付服务扩展，没有建立第二套 Agent、分区管理或实验管理系统。

当前分支为 `dev/Duan`，HEAD 为 `ef288ea3ec241507c63164828a012a2fd519a440`。此前阶段有大量未提交文件。本轮未 checkout/reset/提交/推送。修改前快照 `.tmp/unified-before.json` 记录了 1957 个 Git 已缓存或未忽略的文件；阶段 1–6 文档、真实证据及配置示例共 20 个受保护文件哈希不变。原实验目录未用于本轮测试；测试、浏览器 fixture 和准备中的真实验收分别使用新的隔离 home。

运行配置密钥和用户设置没有被覆盖。本地浏览器检查使用 `.tmp` 中的独立配置，provider 执行被 OfflineOnly 中间件拒绝。随后另获一次成本授权，启用单次隔离真实验收；验收结束后关闭付费入口，页面服务重新以禁止 provider 调用的方式运行，便于只读查看与下载。

源码确认了第六阶段真实 Corn 失败的关键事实：160 fits 任务实际仅 6 fits，累计 Token 准入阻断后续训练；真实 RETAIN/冻结完成，但最终 Test/交付未完成。旧报告中的 18 次响应、494319 tokens 是上一阶段真实运行的证据，**不是本轮运行结果**。旧失败记录、冻结工件和固定数据分区保留。

## 2. 统一自然语言入口

默认网页聊天删除了建模模式组件及 InputBox 的相关 props、渲染和预算表单；没有用 CSS 隐藏。提交时 `modelingChatContext` 过滤旧浏览器设置中的 NIR 控制字段，不修改原设置。普通聊天的模型偏好、输入、上传、取消、图表、过程查看和下载保留。历史预算适配函数作为内部兼容函数保留，普通用户没有配置入口。

实际调用关系为：

```text
默认聊天 / 上传
  → Gateway 生成可信 pending intake 标记（不接受客户端 workflow）
  → 实际 Lead 根据请求、上传和目标判断是否需要执行回归
  → Lead 调用 start_nir_workflow（记录真实 model-issued receipt）
  → web_entry 校验调用来源、owner/thread 和上传路径
  → workflow 绑定 llm_autonomous + sealed_holdout + rounds + delivery
  → 持久化服务器 binding 和统一 policy
  → prepare / context / submit / execute 原有工具闭环
```

后台绑定不依赖用户可编辑的模式字段或仅靠 Prompt 的声明。新 intake 的 provider START Schema 不要求模型填写内部协议、策略和预算开关。真实工具调用来源是任务识别与启动证据；程序没有按 PLS、SNV 等关键词启动训练。

Fake Lead 下，“什么是 SNV”“PLS 与 SVR 区别”和概念解释均保持问答，没有创建 workflow/session 数据库。目标或数据不足时的澄清职责仍在 Lead。真实 Corn 自然语言请求成功进入自主工作流；真实 provider 对各种含糊请求/普通问题的分类稳定性没有穷举。为兼容普通业务，未识别成建模任务前保留通用工具，明确建模/旧 fit 工具已经受 intake 限制；一旦绑定封存工作流，原 legacy、MCP fit、原始文件等绕行阻断继续生效。

`before_agent` 从可信 binding 恢复 workflow；缺失 checkpoint 或伪造较高 revision 的 legacy 状态不能切换已绑定任务。旧会话没有被静默升级政策。已有 workflow 的会话收到新的 pending 建模启动时返回新建会话提示，避免把旧任务改写成新策略。自定义 Agent 入口不被统一默认聊天逻辑改写。

## 3. Token、成本与资源策略

新增统一 policy 为 schema v2：

```json
{
  "profile": "unified",
  "max_llm_tokens": null,
  "final_llm_tokens": null,
  "token_enforcement": "statistics_only",
  "automatic_delivery": true
}
```

`None/null` 表示不启用累计 Token 准入。统一 profile 拒绝重新提交数值 Token 总额；不是用大整数模拟无限。搜索和终结判断统一使用可选 cap 检查，因此累计输入/输出和保守准入代理不触发统一任务的 budget STOP，也不阻断合法终结或自动交付。历史 schema v1 development/autonomous 等数值政策保留原语义。

模型调用数、时间、实验数、fits、高层决策数、无效提案修正次数、重复工具检测、取消和并发保护继续存在。默认统一政策从服务器 autonomous 能力继承并受服务器 ceilings 限制，连续无效结构修正默认最多 5 次，接受合法决策后该计数重置；累计调用/时间等预算不重置。实际政策持久化后不会在恢复时重新获得预算。时间检查仍是原有合作式检查点，**不是可以强杀任意数值库调用的 OS 计时器**。

单次输出、请求超时、provider 上下文窗口和限流能力未移除。阶段 Schema、Skill 投影、有界历史和 DecisionContext 保留。实际 pending START 尝试与已绑定统一建模绕过通用额外摘要 provider；普通概念问答仍使用原有摘要机制。

成本观察复用两种记录：

- NIR ledger：准入估算与实际 input/output、calls、seconds、unknown usage、失败原因分别记录；首个实际 START receipt 只计入一次，避免 prepare 双计数。
- RunJournal：按可信 run_id 获取全调用 snapshot，包含 Lead/辅助来源及 provider/model/request 身份、实际 usage、状态和耗时。只投影允许字段，不导出日志文本、密钥或隐藏思维过程。没有可信 journal 时明确 unavailable，不伪造全调用统计。

报告区分 ledger 估算与 provider 实际 usage；RunJournal 全调用总数不能与 Lead ledger 再相加。自动交付前采集全调用 snapshot，自动交付后不要求真实模型撰写总结。Fake 测试每次故意返回 400 万 input tokens 是检验累计限制关闭的测试数据，不是真实计费或真实 context 大小。

## 4. LLM 控制权与确定性交付

模型家族、预处理及顺序、固定参数/授权网格、下一动作、正常 STOP 和 retained_experiment_id 仍由 Lead 决定。原合法性、用户硬约束、来源审计、执行配置一致性和预算校验保留。数值工具不扩大搜索空间，不额外比较未选择模型，不按固定阈值或失败次数替 LLM 决定动作。

终结链为：

```text
LLM submit STOP / budget RETAIN
  → 原合同与来源校验
  → accepted immutable decision + automatic_delivery queued 先事务持久化
  → complete_terminal_delivery 校验同会话、指定成功实验及终止来源
  → 原 SearchSessionStore.freeze（冻结已有 Cal 模型）
  → 原 finalize（先持久化唯一 Test lease，再访问 Test）
  → 原 decision_delivery + publish_final_delivery
  → Markdown / HTML / joblib / manifest / ZIP
  → ProcessStore / binding 更新，网页按持久事实显示
  → grounded deterministic chat completion（不调用 provider）
```

终结 executor 只接受已持久化的合法 STOP/RETAIN，不接受临时模型/参数。它不从 Val 排名选模型、不重新搜索、不重新拟合 Cal+Val、不修改原决策理由。Fake 测试故意保留此前较差但成功的实验，验证 frozen_experiment_id 等于 LLM 指定 ID，能与 best Val ID 不同。

`stop_source=llm` 与 `stop_source=budget` 保留区别。预算 RETAIN 不增加 fits，也不被报告成主动 STOP。没有合法保留选择时不会自动补造选择。

### 4.1 持久化、并发与恢复

交付使用现有 store 的事务和原 freeze/finalize/export 服务。新增编排 lease 与 queued/freezing/frozen/final_test_pending/final_test_completed/publishing/delivered 事件；失败、取消和 audit_required 另行记录。

- 并发/未知 lease 拒绝执行；只有 Gateway 已确认的 orphan 且 run_id 匹配时才允许恢复该 executor lease。
- 浏览器断开使用现有 RunManager continue 行为；显式取消仍保留真实来源。
- 恢复复用私有 workflow snapshot 和服务器 binding，不从浏览器构造终结身份；私有 snapshot 不进 LLM 上下文/公开审计投影。
- 已完成交付 replay 校验同一模型及已发布工件，返回可信缓存，不追加事件、不重新 Test、不重置审批。
- Test 状态不确定、final_test_failed 或发布中断无法确认时失败关闭；不重新申请第二次 Test。
- Test 已完成但报告生成明确失败时，仅允许重试报告/发布，不重开 Test。
- 已完成交付文件损坏/无法解析时标记 audit_required，不补做模型或 Test。
- 取消在冻结后的检查点发生时留下 Test 封存状态；并不抹除已完成的冻结事实。

本地 fault injection 覆盖了上述路径。真实进程突然终止、磁盘/网络故障组合及跨部署环境恢复未穷举，不能把单元注入说成所有故障均已真实演练。

### 4.2 科学评价与审批

没有改动安全数值核心的模型含义和 Cal/Val/Test 分区。沿用 Cal fold 内拟合预处理、Cal CV 与 Val 搜索、冻结后单次 Test。Test 指标不进入新的搜索 context。已完成 Test 后不会重开搜索。Cal 光谱缓存及只读路由继续复用，第六阶段绘图不重新训练也不读取 Test。

报告由原模板和不可变决策/执行/反馈/最终评价记录生成，不额外调用 LLM，不把未执行候选记成模型实验。Cal、Cal CV OOF、trial mean-fold RMSECV、Val、Test 分别标明。新增成本 snapshot 不改变模型质量结论。

自动交付不是自动批准或注册。网页显示 delivered、pending approval、quality 与 registered 各自事实。原审批与质量失败拒绝注册规则继续执行；没有设置自动 approve/register。

## 5. 新增与修改文件

修改原文件 28 个（与本轮初始快照比较，不将之前阶段未提交代码冒充本轮新增），完整列表及保护哈希见 [local-validation.json](./chemagent-unified-evidence/local-validation.json)。

| 范围 | 文件 | 本轮内容 |
| --- | --- | --- |
| 统一入口 | `backend/.../community/nir/web_entry.py`, `workflow.py`, `decision_origin.py`, `dataset_session.py` | 可信 pending/START、后台绑定、调用来源、intake ledger 衔接 |
| 预算/成本 | `decision_budget.py`, `decision_provider.py`, `decision_delivery.py` | 可选 Token cap、v1 兼容、精确失败原因、RunJournal 成本 snapshot、报告投影 |
| 终结/过程 | `decision_context.py`, `decision_tools.py`, `search_tools.py`, `autonomous_view.py` | accepted terminal 先保存、执行触发、共享发布 adapter、只读自动交付事实 |
| 新增编排 | `backend/packages/harness/deerflow/community/nir/automatic_delivery.py` | 复用冻结/唯一 Test/发布，leases、事件、幂等、失败关闭、恢复 |
| 实际 Agent | `backend/.../agents/middlewares/nir_workflow_middleware.py`, `summarization_middleware.py`, `agents/lead_agent/prompt.py` | intake/阶段 Schema、权限、checkpoint binding 恢复、确定性终态回复、摘要旁路 |
| Gateway | `backend/app/gateway/services.py`, `backend/app/gateway/deps.py` | disconnect continue、既有 orphan 恢复衔接 |
| Skill | `skills/custom/nir-coordinator/SKILL.md` | 统一新入口和已接受终结后的程序交付说明，保留 legacy 条件 |
| 前端 | `frontend/src/components/workspace/input-box.tsx`, `src/app/workspace/chats/[thread_id]/page.tsx`, `src/core/nir/autonomous.ts`, `src/components/workspace/nir/autonomous-process.tsx` | 删除 NIR 控件/提交开关，展示实际交付/成本状态 |
| 删除 | `frontend/src/components/workspace/nir/autonomous-mode.tsx` | 删除已不需要的模式/预算 UI 实现 |
| 新增测试 | `backend/tests/test_nir_unified_policy.py`, `test_nir_unified_agent.py`, `test_nir_automatic_delivery.py`, `frontend/tests/unit/core/nir/unified-entry.test.ts` | 新策略、自然语言 Fake Lead 全链路、终结故障、安全和入口回归 |
| 更新测试 | `backend/tests/test_nir_autonomous_initial_agent.py`, `frontend/tests/unit/core/nir/autonomous.test.ts` | 新 hook 的测试图节点预算、实际交付 UI 断言 |
| 文档 | 根 `README.md`, `AGENTS.md`, `backend/AGENTS.md`, `frontend/AGENTS.md`, 本报告及证据目录 | 当前调用关系、兼容与验证状态 |

上述 `backend/...` 指 `backend/packages/harness/deerflow/`，并非另建同名目录。`.tmp` 的离线浏览器/验收 launcher 和 private YAML 仅是验收支架，不是生产 Agent 或用户配置。

## 6. 自动化与网页验证

所有测试使用隔离 DEER_FLOW_HOME / 唯一 basetemp。没有付费 provider 执行。日志保留了修复过程中的失败，不用最后一次成功删除早先证据。

| 范围 | 结果 | 证据 |
| --- | --- | --- |
| 本轮修改前全后端 | 6521 passed / 85 failed / 45 skipped | `.tmp/unified-baseline2.xml`, `.log` |
| 修改后全后端 | 6548 passed / 86 failed / 45 skipped | `.tmp/unified-full-after1.xml`, `.log` |
| 后端 NIR 全部用例 | **873 passed / 2 skipped** | `.tmp/unified-nir2.xml`, `.log` 中去掉额外 MCP recheck 用例 |
| NIR 全部 + 新增的时序失败复核 | **874 passed / 2 skipped** | 同上，96.10 秒 |
| 独立数值内核全部（包含 slow） | **696 passed / 2 skipped**，214.81 秒 | `.tmp/unified-core.xml`, `.log`，真实验收后隔离执行，不读取 Corn Test/不调用 provider |
| 最后交付/Agent/政策/legacy 专项 | **78 passed** | `.tmp/unified-final-target.xml`, `.log` |
| 最后取消/完整性/编排专项 | **12 passed** | `.tmp/unified-cancel.xml`, `.log` |
| 修改前前端全部单元 | 578 passed，66 文件 | `.tmp/unified-front-baseline.log` |
| 修改后前端全部单元 | **581 passed，67 文件** | `.tmp/unified-front-final2.log` |
| 最后前端导入顺序/清理复核 | **39 passed，2 文件** | `.tmp/unified-front-last-import.log` |
| Python scoped Ruff | All checks passed | `.tmp/unified-ruff-final.log`；最后新增取消测试另行 scoped check 通过 |
| 前端修改文件 Prettier | 通过 | scoped check，6 个文件 |
| Git diff whitespace | 通过 | `git diff --check` |
| TypeScript 全检查 | **未通过**，依赖声明/导出等错误 | `.tmp/unified-front-tsc.log`，包括 Playwright、rstest、React/Radix 等；没有本轮修改前 tsc 快照，不能据此宣称都是既有失败 |
| ESLint scoped | **环境阻断：缺少 unrs-resolver**，exit 1 | `.tmp/unified-front-eslint-final.log`；修正两处导入顺序和未使用变量后只余 4 处 resolver 错误，不能作为 lint 通过证据 |
| 真实 DeepSeek / Corn 新流程 | **完整交付成功**：3 实验/153 fits/20 响应/真实 STOP/单次 Test | [real-corn-evidence.json](./chemagent-unified-evidence/real-corn-evidence.json)，另见第 7 节 |

**全量后端没有全部通过。** baseline 的 85 个失败在 after 中全部保留，涉及既有 Windows POSIX 权限/符号链接、shell/部署及其他公共模块。完整 node IDs、before/after 差集均保存于 local-validation.json。额外一次失败是未修改的 `TestWorkspaceSnapshots::test_changed_workspace_files_detects_created_and_modified_files` 时间戳检测；独立复核以及最终 NIR 合并复核均通过，未修改它来制造绿色结果。最后两项安全测试在全 NIR sweep 后新增，随后由 78/12 项专项覆盖，不声称已包含于较早 sweep。

NIR 的两项符号链接用例在当前 Windows 账户跳过，不计为通过。末次同步 pytest 结束有 pyreadline WinError 析构警告，但 12 项测试和退出码均成功；该警告未当作验证失败消除。

### 6.1 Fake Lead 与数值执行证据

真实 `create_lead_agent` 工厂/中间件/工具链注入 Fake LLM，执行合成数据的两轮 Ridge，共 8 fits；同一工具脚本验证同步/异步 STOP 和预算 RETAIN。模型选择没有来自程序默认 PLS。STOP 指定第一轮 alpha=2，第二轮 alpha=0.2 的 Val 更好；冻结仍是第一轮。执行配置、决策不可变性、反馈关联、context version、来源、资源、只读 GET 和 Test 唯一性均有断言。

额外断言包括无效 START 修正的初次调用计数、恢复缺失/伪造 checkpoint、用户约束、旧 explicit delivery、合法参数/有界网格、辅助调用 snapshot 及既有阶段行为。不同动作的科学合同和数值能力由 NIR 全回归覆盖；本轮不新增模型或动作。

### 6.2 实际网页检查（Fake provider + 合成数据）

隔离 Gateway / Next 开发页面完成以下实际操作：默认输入页面检查、持久化合成数据 fixture 的过程页面查看、ZIP 点击下载、页面刷新与数据库哈希核对。真实 provider 全程禁用。

| 项目 | 实际事实 |
| --- | --- |
| thread | `unified-llm-False` |
| session | `search_6d59dfe6c549491f880612c518455a80` |
| 数据 | 合成 Cal=36 / Val=12 / Test=12 |
| 实验 / fits / 模型响应 | 2 / 8 / 10（Fake） |
| 停止 | STOP，`stop_source=llm`（Fake 来源，不是真实 provider STOP） |
| 保留与冻结 | 同一 `exp_55ffa7a4330e9dad7153f9e4233ece80` |
| 唯一 Test / 发布 | finalized / delivered |
| 审批 / 注册 | pending approval / false |
| 下载 | 实际 ZIP 文件 SHA256 与已发布工件相同 |
| 刷新 | 仍为 2 实验 / 8 fits，持久数据库 SHA256 不变 |

ZIP 包含 report.md、report.html、model.joblib、model manifest、metrics/decisions JSON、delivery_manifest 和 README。报告与模型身份一致，测试验证 joblib 可加载并预测。网页交付并不依赖额外模型解释消息。

保留模型的合成数据页面数值（四舍五入）为 Cal RMSE=0.1856/R²=0.9972，Cal CV OOF RMSE=0.5325/R²=0.9768，Val RMSE=0.0975/R²=0.9971，Test RMSE=0.1694/R²=0.9948/RPD=14.4491。第二轮 Val RMSE=0.0104。**这些只是 Fake 合成数据验证，不是 Corn 模型精度或方法优越性证据。**

浏览器截图：

- [默认聊天入口](./chemagent-unified-evidence/chat-entry.png)
- [两轮决策时间线](./chemagent-unified-evidence/fake-timeline.png)
- [自动交付与审批事实](./chemagent-unified-evidence/fake-delivery.png)
- [刷新后的过程页面](./chemagent-unified-evidence/fake-after-refresh.png)

## 7. 一次已授权真实 Corn 完整验收

本地测试完成后，用户直接回复 **“授权这一次有界真实验收”**。这是本轮新授权，没有延用原第六阶段已经消耗的授权。只运行了一个新的隔离会话，没有失败后追加第二次 provider 尝试。

保持已接受的 Corn/m5 Oil 输入 SHA256 `0850054f2b3c278da0e0f18fe4d90c789e3116d3cb44258c76db2ebe2b572fbd`，80 样本、700 波长（1100–2498 nm），原 Cal/Val/Test=48/16/16，Cal 5 折 KFold/shuffle=True/random_state=42。源未注明 Oil 目标单位，历史同批 Corn 光谱已有使用；本结果是**公开历史数据上的功能验证，不是全新独立确认或方法优越性证明**，没有重新划分 Test。

实际入口是网页上传同一 NPZ，加自然语言要求预测 Oil、自主选择方案与停止、保持既有分区/协议。用户消息没有内部策略开关，没有指定模型、预处理、切换顺序、轮数或强制 STOP。使用实际 Gateway/DeerFlow Lead、当前 `deepseek-flash` 配置的 `PatchedChatDeepSeek`（`chat-deepseek`）与 `https://api.deepseek.com/chat/completions`、真实 NIR 工具和数值训练。四个接受决策的来源均为 `lead_agent_model_call / verification_kind=provider_model`，关联真实 run/message/tool_call 身份，不是 author 字段自述。

| 身份/范围 | 实际值 |
| --- | --- |
| thread | `6ee9fa3a-45fb-4cb5-ad67-ebd5e0c52be3` |
| run | `a24f38c7-39a2-4a2b-b227-d938de54ae50`，status=success |
| session | `search_60222bac96f3469aa6030afc1fb9af3b` |
| 保留/冻结实验 | `exp_31cfe4f52b6810aa76cb966b1fc0964b`（第二轮） |
| 终止决策 | `decision_82b27f44f48e804e27e852ff9c718d90`，STOP / source=llm |
| 实验 / fits / 接受高层动作 | 3 / 153 / 4（含 STOP） |
| provider 实际 usage | input=728024 / output=42258 / total=770282，20 次响应 |
| 辅助调用 | middleware=0 / subagent=0，未启用 title/suggestions/memory/summary provider |
| RunManager 时间 | 14:44:20.354068–14:47:44.037681 UTC，约 203.68 秒 |
| 数值执行总时长 | 3 轮记录合计约 1.44 秒；provider/框架耗时另计 |
| 原始冻结 pipeline SHA256 | `d63ca67b78cca9979928c5c665fef43cbb25e82a0ff39ade8a0254353b631823` |
| 发布包装工件 SHA256 | `1fe8063000a71ebd670783bf2cc311453eead07db67f0232a34e5cf6167b370d` |
| CV manifest | `cv_f8d479ce3cdfb72c8c58f3408a8a07503dde244c240842d80a50b3c6793f1c19` |
| 最终状态 | finalized / automatic_delivery=delivered / approval pending / registered=false |

授权保护为 8 个实验、160 fits、24 个决策；搜索 1800 秒/最多 36 次 Lead 调用，终结最多 6 次/300 秒；全 provider 最多 42 请求、总验收 2400 秒、单次输出 4096 tokens、请求超时 60 秒、自动重试 0。实际已绑定政策为 unified v2、两项 Token cap 均 null。首请求前记录一次性开始标记；没有按累计 Token 判断。实际结果在全部边界以内，终结预算模型调用为 0。

### 7.1 实际动作、反馈与正常停止

| 轮次 | LLM 动作 | 实际管线/授权区域 | 选定潜变量 | fits | Cal CV OOF RMSE | mean-fold RMSECV | Val RMSE / R² |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | EXPLORE | SNV→derivative1，window=11/order=2/index spacing；PLS n_components=1–10 | 9 | 51 | 0.0789257071 | 0.0765905912 | 0.0693926030 / 0.866068676 |
| 2 | CHANGE_PREPROCESS | SNV→derivative2，同 window/order/spacing；PLS n_components=1–15 | 15 | 76 | 0.0552377757 | 0.0529523175 | 0.0565348547 / 0.911102742 |
| 3 | TUNE | 预处理/家族保持第二轮，PLS n_components=16–20 | 20 | 26 | 0.0551731135 | 0.0528588056 | 0.0571063184 / 0.909296482 |
| 4 | STOP | 明确保留第二轮，无新模型执行 | — | 0 | — | — | — |

每轮采用 scale=False、max_iter=500、tol=1e-6；实际返回完整参数另含 copy=True。导数实际参数 delta=1.0。实际预处理顺序与接受合同相同，未执行的 Raw/Ridge/SVR/MSC 等仅为候选比较，没有隐式训练。

第二轮根据第一轮 baseline/noise 诊断和 Val 反馈提出强化基线/曲率处理的假设，实际 Val RMSE 改善 0.0128577483，达到其 ≥0.003 的结构化目标；反馈 ID `feedback_10254e111a6f3473b8ed1361510d453f`。第三轮根据第二轮 15 个潜变量位于授权区域边界扩展至 16–20，实际 Val RMSE 反而增加 0.0005714637，未达到 ≥0.002 的改善目标；反馈 ID `feedback_344c617d5561f5f185bf6b9b7a3d7aae`。没有把失败假设改写成提升，也没有修改原理由。

第二轮质量门禁通过后 LLM 仍选择第三轮 TUNE，证明本次没有按质量阈值自动 STOP。最后 LLM 比较 STOP、MSC 单点和 SVR 单点，依据第三轮 Val 略差、CV 接近平坦、第二轮更简洁及剩余资源选择保留第二轮。**STOP 时仍有 7 fits；其未选候选的固定执行需 6 fits，搜索服务没有因硬预算禁止训练，因此本次是主动 STOP，不是预算 RETAIN。** 同时，这个 STOP 受到剩余预算的实际影响，不能外推为在更宽松预算下也必然停止。

### 7.2 自动终结与独立最终评价

持久事件顺序 queued→freezing→frozen→final_test_pending→final_test_completed→publishing→delivered，均发生在合法 STOP 保存后。Agent 没有调用 nir_freeze_search/nir_finalize_search；没有终结/交付后的 provider 响应。冻结第二轮已有 Cal 模型，无新 fits、无 Cal+Val 重训。最终 Test lease 在读取前持久化，最终评价一次，后续网页只读缓存，没有再打开搜索。

最终保留模型指标：

| 范围 | n | RMSE | R² | RPD |
| --- | ---: | ---: | ---: | ---: |
| Cal | 48 | 0.0027506444 | 0.999763279 | 65.682987532 |
| Cal CV OOF | 48 | 0.0552377757 | 0.904535685 | 3.270778691 |
| Val | 16 | 0.0565348547 | 0.911102742 | 3.463938549 |
| 最终 Test | 16 | 0.0479011281 | 0.899725237 | 3.261508687 |

上述 Cal CV OOF 与所选 trial mean-fold RMSECV=0.0529523175 不同，均保留原范围。最终质量检查 good/passed=True（现有门禁 R²≥0.8、RPD≥3），系统流程成功与门禁达标两项分别成立；未知目标单位、仅 16 个 Val/Test 样本、Cal 与 CV 误差差距很大，仍限制科学/应用解释。不能以该一次结果证明模型优于其他方法。

网页验证真实 Cal 原始/处理曲线（12/48 样本、512/700 原有有序波长，明确 nm）、三轮 Val 趋势、最终 Test 缓存图、实际参数和交付。真实 ZIP 点击下载，文件 SHA256 与发布文件相同；刷新后过程数据库 SHA256 不变，仍为 3 实验/153 fits/单次 Test。

模型是既有 NIR 包装工件，包含 estimator 和实际拟合的预处理状态。加载后使用 **io_tools 的原 preprocessing/wavelength prediction adapters** 对 3 行 Cal 预测，结果 `[3.2653076193, 3.4050547564, 3.6760239184]` 与原冻结 pipeline 完全相同。审计脚本初次直接调用包装内的裸 PLS estimator 是错误检查方式，随后纠正为原预测适配器，记录于证据的 `local_audit_correction`；未改动模型、数据或 Test。这也解释了包装工件与冻结 pipeline 的序列化哈希不同，不是替换模型。

最终模型/报告/原决策证据副本及 hash：

- [模型](./chemagent-unified-evidence/real-corn/model.joblib)、[Markdown 报告](./chemagent-unified-evidence/real-corn/report.md)、[HTML 报告](./chemagent-unified-evidence/real-corn/report.html)、[ZIP](./chemagent-unified-evidence/real-corn/delivery.zip)
- [完整结构化验收证据](./chemagent-unified-evidence/real-corn-evidence.json)（来源、合同、实际执行、Feedback、CV 折、usage、hash、lease 事件与刷新检查）
- [网页最终交付](./chemagent-unified-evidence/real-corn-delivery.png)、[真实光谱/趋势/Test 缓存图](./chemagent-unified-evidence/real-corn-charts.png)、[刷新页面](./chemagent-unified-evidence/real-corn-after-refresh.png)

### 7.3 真实成本、错误与证据限制

本次 20 响应/770282 实际 tokens 没有因累计 Token 终止。保守准入累计代理 2921020 与计费 usage 不能混淆。RunJournal 全调用 usage 与 Lead ledger 完全对账，辅助为 0，自动交付为 0 provider 调用。分阶段工具数从 intake 的 48 收敛至末轮的 2；末次 provider 请求 method knowledge 约 5310 UTF-8 bytes、history 约 21975 bytes，仍需继续关注系统/历史体积。bytes 是记录到的序列化规模，不能称为 tokenizer counts。

**不宣称真实总成本降低。** 上阶段是一次训练的失败流程，本次是三次训练的成功流程，任务/动作不同，没有同条件对照；770282 也高于原 494319。已验证消除 Token 准入阻断、关闭辅助调用和取消交付模型调用；输入总量与错误修正成本仍然较高，未得到可靠货币费用。

真实提案有 6 次拒绝（按持久 reflection 记录）：首轮缺候选 optimization/comparison、selected ID、理由/预期；第二轮缺公共字段以及缺未选候选理由；TUNE 缺 observation evidence/公共字段以及 expected_revision；STOP 一次 JSON EOF（响应字符串截断）。返回具体路径并允许 LLM 修正，均没有执行非法提案或擅自替换默认模型。连续拒绝最多 2 次，合法接受后 repair counter 重置，最后 counter=1 不代表累计只拒绝一次。

还有一次 premature final response 被现有 workflow guard 重新唤起，直到合法 STOP；这是要求完成合法生命周期的反馈，不是程序替选 STOP。初始 MCP 工具 discovery 有 connection closed 警告，所需本地 NIR 工具和数值流程仍正常运行；不将该警告抹去或宣称所有外部 MCP 服务可用。

科学理由存在局限，须保留原记录而不事后修饰：第二轮称首轮最佳位于边界，但真实最佳为 9/1–10，该描述不严格；“powdered corn”没有本文件来源支撑；未训练 SVR 不能据此证明 48 样本不适合 SVR。后续部分文字泛称 Cal-CV 时使用的是 mean-fold 值，不是 OOF。系统记录/报告明确两者，但不自动校正文案为已证事实。当前证据支持任务内动作与反馈关联及严格执行，不支持每个化学计量学论断都已经实验验证。

## 8. 状态、限制与待办

| 状态 | 内容 |
| --- | --- |
| 已实现且本地通过 | 无 NIR 模式/预算 UI、可信新入口、v2 可选 Token cap、真实来源/usage 记录、STOP/RETAIN 自动终结、幂等与失败关闭、只读网页恢复和下载、legacy/审批约束 |
| 已通过实际浏览器检查 | 默认入口无控件；Fake 两轮 STOP 后冻结/单次 Test/报告/ZIP；网页刷新不重训，数据库不变 |
| 已有旧阶段真实证据 | 原 Corn 初始选模、反馈提案、预算 RETAIN 和冻结；原阶段没有最终交付 |
| 本轮已真实验证 | 统一网页自然语言启动、EXPLORE→CHANGE_PREPROCESS→TUNE→STOP、保留第二轮、自动冻结/唯一 Test/Report/ZIP、下载与刷新、provider 实际 usage 对账 |
| 本轮尚未真实验证 | 统一策略预算 RETAIN 的真实交付、更多模型/预处理路径、不同语义/取消/真实崩溃组合；同条件计费/错误率改善 |
| 环境/公共残余 | 全量后端 85 个 baseline 失败；TypeScript 检查未通过；Windows 符号链接跳过；跨真实重启/故障组合覆盖有限 |
| 本轮未开发 | 新模型/Agent、跨任务历史复用、论文对比/消融、前端重构、自动审批注册 |

本轮真实授权已经使用，不自动追加。后续真实验证仍须逐次授权，不能补造选择、重复看 Test 调试或将功能验证改写成方法优越性证明。至此停止本次改造，不自动开展其他研究阶段。

## 9. 十四项验收回答

1. **用户是否不再需要选模式？** 是；控件/组件删除，默认页面实际浏览器确认。
2. **上传+自然语言能否启动完整自主流程？** 是，本轮真实 DeepSeek/Corn 网页已经完成自然语言→三轮训练→主动 STOP→自动交付，没有模式/预算开关。
3. **非建模聊天是否保留？** 三类概念问答 Fake 测试不创建任务，现有普通聊天回归保留；真实 provider 各类普通问题/含糊请求未穷举。
4. **累计 Token 硬预算是否取消？** 新统一 v2 是 null/statistics_only，超大 Fake usage 不阻断；历史 v1 政策保留。
5. **资源允许时能否继续优化？** 本轮实际 770282 tokens/三轮训练未因累计 tokens 停止，calls/fits/时间保护继续有效；不能推断无限运行。
6. **模型/下一动作/正常 STOP 是否仍归 LLM？** 是，本轮四个真实接受决策及实际执行/反馈关联可核验；程序只校验和执行，未训练未选择候选。
7. **STOP 后是否无需 LLM 调交付工具？** 是，真实 STOP 后自动 Freeze→唯一 Test→Report/ZIP，Agent 未调用 freeze/finalize，终结/解释 provider 调用=0。
8. **预算 RETAIN 能否自动交付？** 本地完整链路通过，stop_source 仍 budget。真实统一 RETAIN 交付尚未验证。
9. **Cal/Val/Test 隔离和唯一 Test 是否有效？** 本地 NIR、spy、重放/并发/未知状态失败关闭通过，未更改科学协议；公开 Corn 历史暴露限制另行披露。
10. **报告是否确定性依据真实记录？** 是，真实 Corn 最终报告/审计记录与实际参数、三轮指标、保留身份和费用对账；生成没有模型调用，原理由包括其局限原样保存。
11. **网页能否自动展示下载？** 是，真实最终页面、ZIP 下载 SHA、Cal 对照/Val 趋势和刷新无重训已验证。
12. **审批、质量、legacy 是否保留？** NIR/legacy 专项通过，实际 Fake 页面 pending approval、registered=false；自动交付没有审批权。
13. **真实 Corn 完整验收是否通过？** 本次公开历史数据的功能验收通过，且现有质量门禁通过；不是正式独立确认或比较实验。
14. **尚未覆盖真实主动 STOP/多轮情形？** 本次已经观察到真实三轮优化、第四动作主动 STOP 和完整交付；未覆盖其他模型家族动作/更宽松资源 STOP、真实统一 budget RETAIN、各种故障组合，不外推为全面验证。
