# ChemAgent 第三阶段实施报告

日期：2026-10-09。工作目录：`D:\ChemAgent\nirs-agent`；分支仍为 `dev/Duan`。

**已实现 LLM 根据上一轮真实建模反馈，自主选择下一步优化动作，并由原有数值工具执行。** 新功能通过 `autonomous_rounds=true` 显式启用，继续要求 `decision_strategy="llm_autonomous"` 与 `search_protocol="sealed_holdout"`。支持单目标回归的 PLS、Ridge、SVR、Extra Trees。关闭多轮开关时，第二阶段单次初始执行和 legacy 模式保持兼容。

Fake LLM 覆盖全部动作及同步／异步 Lead Agent 调用链；一次获得联网授权的真实 DeepSeek 验证完成了初始选择、反馈决策和第二次执行。真实第二轮没有改善 Val，系统如实记录这一结果。STOP 只保留 LLM 指定的成功实验并进入待最终确认状态，没有实施第四阶段的最终冻结／Test 生命周期。

## 1. 开发前检查与实际范围

核对了第二阶段报告、三份设计文档，以及 `decision_contracts.py`、`decision_context.py`、`decision_origin.py`、`decision_tools.py`、`dataset_session.py`、`search_tools.py`、`safe_search.py`、`workflow.py`、工作流中间件和 `process.py`。

代码与第二阶段报告一致：已有实际模型调用凭据、Cal-only 上下文、不可变初始决策、四种模型的固定／网格 adapter、先持久化再执行的 lease、Cal/Val 搜索和 Test 封存。当时只允许一个初始决策，完成后停在 `initial_result`；没有多轮动作或 STOP。本地运行配置已经在第二阶段合并了安全搜索和三个决策工具，并非仍停留在示例配置。

起始 HEAD 为 `ef288ea3ec241507c63164828a012a2fd519a440`。第一、二阶段代码和报告原本未提交，本阶段在这些修改上继续开发；未 reset、checkout、提交、推送或覆盖已有实验。相对 HEAD 的整个 diff 包含前两阶段，不能当作本阶段新增范围。

实施按可测试的小任务推进：先新增失败测试，扩展合同／持久化／动作校验，再连接真实工具循环，随后验证权限、预算、Test 不变性和真实服务。没有重建 Agent、数据划分、训练内核或实验存储。

## 2. 新增与修改文件

| 文件 | 第三阶段实际改动 |
|---|---|
| `backend/packages/harness/deerflow/community/nir/decision_contracts.py` | feedback 阶段与五种动作；候选动作、当前实验引用、保留实验、问题／不确定性、未选理由、假设、拟合估计、数值改善目标、重复理由 |
| 同目录 `decision_context.py` | 在原 InitialDecisionService 上扩展多轮上下文、分页历史、动作语义、重复区域、STOP、预算终止、执行后反馈和 CV 配置；保留类名兼容旧调用 |
| 同目录 `decision_origin.py` | 除提交／执行凭据外，复用现有凭据表记录每次实际模型响应的 token 和耗时，区分未知 usage |
| 同目录 `decision_tools.py` | 原三个工具进入多轮循环；新增 `nir_decision_history`；执行仅接收 decision_id；保留首轮与最新执行状态；旧请求重放不替换当前待执行决策 |
| 同目录 `process.py` | v4→v5 最小迁移：移除同一 session 只能有一条决策的 UNIQUE 限制，逐字节保留原 payload／execution，沿用原表和私有索引 |
| 同目录 `dataset_session.py` | 显式多轮策略绑定、累计高层动作上限、停止／预算终态仍标记 Test sealed |
| 同目录 `workflow.py`、`search_tools.py` | 传递 `autonomous_rounds`、`max_decisions`、`max_fits`；实验上限继续复用 max_attempts，准备后不能重新获得预算 |
| 同目录 `response_grounding.py` | 多轮未完成时要求下一次 context／submit／execute；STOP／硬预算终态允许返回结果 |
| `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py` | 新阶段与历史工具权限；持续禁止绕行；捕获模型成本；只在模型输入副本删除旧的完整 governed 调用对 |
| `backend/packages/harness/deerflow/agents/lead_agent/prompt.py`、`agents/thread_state.py` | 多轮实际 Agent 指令、阶段和状态字段；原 checkpoint 消息保留 |
| `skills/custom/nir-coordinator/SKILL.md` | 多轮优先指令、动作比较和停止边界；保留旧任务规则，但不能用于自主模式 |
| `config.example.yaml`、本地忽略文件 `config.yaml` | 仅增加 nir_decision_history 条目；结构化比较确认所有旧条目、密钥和其他设置不变 |
| `backend/scripts/verify_nir_initial_llm.py` | 原验证脚本增加 --multi-round；实际生产 Lead factory／模型工厂／注册工具；两实验、14 fits、请求前限制最多九次模型调用 |
| `backend/tests/test_nir_multiround_decisions.py`、`test_nir_multiround_agent.py`、`test_nir_multiround_context_cost.py`、`test_nir_multiround_legacy_replay.py` | 新增 59 项参数化后测试，覆盖下文要求和旧格式请求重放 |
| `backend/tests/test_nir_process.py` | 更新现有迁移版本断言为 v5 |
| `README.md`、根与 backend `AGENTS.md` | 同步功能、兼容开关、存储版本及开发边界 |
| 本报告、`docs/chemagent-phase3-real-llm-evidence.json` | 新报告及脱敏真实证据；不回写前两阶段报告或证据 |

`safe_search.py` 与四模型参数 adapter 原样复用。没有新增第二套模型搜索、数据分区或实验管理平台，也没有改前端。

## 3. 真实多轮调用链

```text
Lead Agent / 实际配置的模型
  → nir_prepare_search（只绑定一次，Test 私有封存）
  → nir_decision_context
  → LLM 比较候选并选择
  → 模型中间件保存准确的 tool_call_id / 参数哈希 / 响应来源
  → nir_submit_decision：校验 + 保存不可变 DecisionRecord
  → nir_execute_decision(decision_id)
  → SearchSessionStore.search → 原 safe_search / 四模型 adapter
  → 分离保存 Experiment 和 Feedback
  → decision_review → 新 DecisionContext → 再次调用同一 LLM
  → 新动作，或 LLM STOP，或硬预算终止
```

执行接口不接收临时模型／预处理／参数覆盖。内核仍要求已接受决策处于 executing lease，并逐项比较完整配置。直接 `nir_search_model`、legacy 训练、MCP fit、原始文件、脚本、子任务和 workflow 重启均不可绕行；该限制在 STOP 和预算终态仍有效。关闭多轮开关则沿用 `initial_result` 单次终态。

中间件捕获真实 Agent 模型响应，而非相信 `author="llm"`。Fake provider 明确标记 fake_model；真实凭据含模型类／名称、消息、run/call ID、准确参数哈希、响应哈希与 token usage。实际模型选择由 LLM 提交；工具没有模型评分公式、PLS 首轮默认、诊断标签映射或固定阈值切换规则。

## 4. 动作合同与控制边界

| 动作 | 校验与实际控制权 |
|---|---|
| 初始 EXPLORE | LLM 自选 Raw 或有序预处理、模型和固定／网格策略；只执行所选候选 |
| TUNE | 与当前实验保持同一模型家族和规范化预处理；LLM 指定新的合法参数或离散区域 |
| CHANGE_PREPROCESS | 保持模型家族，预处理必须真实改变，并通过原注册表／provider 检查 |
| SWITCH_MODEL | 家族必须改变，规范化预处理保持一致；具体换哪个家族由 LLM 决定 |
| 后续 EXPLORE | 模型或预处理方向必须改变，声明与已尝试方案的区别及探索理由 |
| STOP | 必须显式引用本会话成功实验，所选候选 pipeline／optimization 为 null，expected_fits=0；保存 stopped 记录，等待最终确认 |

反馈 proposal 必须匹配当前 context_id／revision／hash，引用最新 base_experiment_id，并提交观察、证据、不确定性、候选比较、每个未选候选的理由、假设、预期结果和准确拟合估计。可声明 `expected_improvement={metric: val_rmse 或 val_r2, comparison: previous 或 retained, minimum_change: ...}`。每次选择保留实验是 LLM 显式决定；Val 最优实验只作为信息，不会被程序自动保留。

重复检查比较规范化预处理和完整授权候选集合，忽略候选顺序／预算等表面差别；fixed 与单点 grid，以及 alpha 等参数的整数／小数写法不能绕过。保留 Extra Trees max_features 的整数／比例语义差别。历史签名从原始已接受方案重新比较，不回写旧 payload。重复网格候选直接拒绝；确需重复验证时必须明确 replication_reason，并继续计费预算。非法参数、过期上下文、错误动作语义和不足预算返回明确错误，要求 LLM 修正，不自动替换方案。

用户结构化 model_constraint／pipeline_constraint 跨轮持续校验，Raw [] 也能是硬约束。先前 LLM 自选模型没有被提升为用户硬约束，执行失败后可以由 LLM 改选。任意自然语言条件仍需 LLM 正确理解并映射到现有结构化约束／预算；本阶段没有建立通用自然语言约束解析器。

## 5. 动态上下文与成本控制

上下文来自私有持久化决策和独立执行字段，包含当前完整关键配置／实际参数／Cal CV／Val、最近四条精简结果、Val 最优信息、LLM 已保留实验、成功／失败计数、已尝试管线及授权区域、原理由／预期与执行后支持情况、剩余资源、Cal 诊断和有来源的领域卡片。首轮不使用历史，跨任务历史不参与。

方法知识默认只返回少量真实检索结果；不携带 priority 等选择暗示，保留用途、适用条件、限制、参数指引和来源哈希。`knowledge_query` 按需补取；`nir_decision_history(offset, limit<=4, experiment_id)` 在同一 owner/thread/session 内分页查详情。压缩没有生成或改写实验事实。

注册表和 adapter 仍提供真实支持能力。后续上下文引用完整冻结 CV manifest 的 ID、哈希、分折数和配置，避免反复发送行索引；原完整折成员一直保存在 session 与实际实验反馈。显式披露 KFold shuffle=true/random_state=42，GroupKFold shuffle=false/random_state=null，不改变旧 manifest 哈希或历史记录。

新 context 出现后，只在发给模型的消息副本删除更早的完整 governed AI/tool 调用对。保留原用户要求、最新成对调用、其他工具及未完成调用；checkpoint／原消息内容／决策原因均不修改。五轮合成测试的默认 JSON 编码上下文小于 45000 字符。这是小规模测试结果，并非普适 token 硬上限。

真实首四次响应累计 input=64590，低于第二阶段四次响应的 111018；本次八次响应总 input=149067/output=5378，比单轮总量更高。不能声称多轮总成本已经降低；长历史、详细卡片和重复修正仍需进一步优化。

## 6. Decision → Experiment → Feedback

执行前 payload 保存 decision/context/session ID、预定 experiment_id、轮次／动作、完整 proposal、被接受的实际规范配置、运行时版本／provider 绑定、CV manifest ID、授权 fit 预算、前一实验与保留实验引用、决策时间和真实调用来源。

执行后另存 status／execution，包含 decision_id、experiment_id、稳定 feedback_id、实际配置和完整 estimator 参数、Cal CV／Val、相对前轮和保留方案的指标变化、实际 fits／耗时、失败类型／原因、预期数值目标是否 achieved／not_achieved／not_evaluable、CV 配置及累计预算。文本预期原文保留；没有数值目标或没有可比较指标时不会推断“改善已实现”。假设的解释仍由后续 LLM 完成。

失败亦持久化，已用 fits 不退还。前置失败可以是零 fits；uncertain executing lease 不自动重试。执行前 rationale 不被反馈覆盖；已有模型与报告不写入同名路径。v4→v5 的字节保留测试同时验证历史决策 payload 和模型工件原样保存。旧 proposal 重放仅在内存补齐可选字段后比较，返回原 payload；旧会话没有新增预算字段也能执行，不回写历史理由。

## 7. 累计预算与参数执行

区分六项：已接受高层动作、搜索候选、模型实际 fits、实验次数、科学执行时间、LLM 请求／tokens／耗时。非法提案消耗实际 LLM 调用成本，但不增加已接受动作或模型 fits。

成本仍为 `候选数 × 冻结 Cal CV 折数 + 一次 Cal refit`。PLS 固定潜变量数、Ridge alpha、SVR C/gamma/epsilon 等和 Extra Trees 已支持参数严格复用第二阶段 adapter；固定模式不隐式搜索。网格不能更改家族、预处理或授权区域。返回完整实际参数和逐候选结果，校验实际配置与不可变 accepted_spec 相同。

max_attempts（最多十个公开 workflow 实验）、max_fits（最多 200）、max_decisions（最多 24，含 STOP）只在会话准备时绑定。新轮次不恢复额度。硬预算不足以再执行一个固定候选、达到实验上限或接受动作上限时单独记录 hard_budget／耗尽原因，不生成虚假的 LLM STOP、不自动选择最优模型。LLM 主动 STOP 单独记录 llm 来源，并保留它明确指定的成功实验。

LLM usage 来自实际响应；未知 usage 单独计数，金额未由 provider 提供。运行 deadline／取消继续使用既有资源检查，不新增第三方 fit 的强制中断框架。

## 8. Test 封存与兼容性

搜索和 context 只读取 `search.npz`；该文件没有 Test 数组，`holdout.npz` 从未进入模型搜索 API。历史只能读取本会话的搜索反馈，不包含最终测试指标或派生统计。STOP 和 budget_exhausted 的 Test 状态仍为 sealed，禁止调用 freeze/finalize 绕开后续确认。

自动化测试在全部上下文／执行轮次监视 np.load，拒绝 holdout；两份相同 Cal/Val 数据分别封存原 Test y 和放大 100 倍的 Test y，后续两轮模型参数选择、Cal/Val 证据及验证结果完全相同。没有修改已经锁定会话的数据绑定。真实运行的四次加载均为 search.npz，Test 未消费。

保留 legacy 和第二阶段默认路径、已有安全模型工件和报告。旧的含 Test 会话内容并未被擦除，应使用新对话；本协议隔离可信数据执行路径，不能消除用户主动提供的既有测试知识。

## 9. 自动化与真实服务验证

| 验证 | 结果与证据 |
|---|---|
| TDD 初始失败 | 新的 8 项核心测试先失败，证明当时没有多轮能力；`.tmp/phase3-red.log` |
| 分任务验证 | 核心合同／执行、同步与异步真实 Lead 工具循环分别测试后再扩展权限／预算／封存 |
| 最终定向回归 | 112 passed；`.tmp/phase3-release-focused.log`，覆盖 59 新测试及已有初始决策／参数／ProcessStore 测试 |
| 修改前完整 backend | 6253 passed、88 failed、45 skipped；`.tmp/phase3-baseline.xml` |
| 第一轮完整回归 | 6311 passed、87 failed、45 skipped；NIR 637 passed、2 skipped、0 failed；无新增失败 |
| 最终完整回归 | 6312 passed、88 failed、45 skipped；NIR 639 passed、2 skipped、0 failed；`.tmp/phase3-release-full.xml`／`.log` |
| 静态检查 | 修改 Python 文件 Ruff check 与 git diff --check 通过 |
| 真实 DeepSeek | passed；见下表和独立 JSON 证据，不与 Fake 结果混算 |

新增测试覆盖全部动作、Raw／四家族首轮兼容、非 PLS 无隐藏 PLS、实际反馈读取、旧理由不改、预期与反馈关联、质量高／低均不代选动作、失败后改选、非法参数修正、过期／并发决策、固定／网格精确执行、重复与等价区域拒绝、累计 fit／实验／动作预算、用户硬约束、36 个阶段×绕行权限组合、Test y 不变性、迁移保留和输入副本压缩。

完整 backend 并非全绿。第一轮有一个旧的 workspace 文件变化检测用例恢复通过，第二次完整运行有两个旧例恢复通过；最终完整运行与基线的 88 个失败 node ID 集合完全相同，新增失败为零。既有问题集中在 Windows shell、chmod／symlink／权限、sandbox／MCP／渠道和旧服务客户端环境，没有为本阶段改造这些无关功能。最终比较记录位于 `.tmp/phase3-failure-comparison-final.json`。

真实证据：[chemagent-phase3-real-llm-evidence.json](chemagent-phase3-real-llm-evidence.json)。

| 项目 | 真实运行事实 |
|---|---|
| 服务／模型 | 本地已配置 deepseek-flash，实际类 PatchedChatDeepSeek，provider_model 来源；获联网运行授权 |
| 数据／服务范围 | 48×12 合成数据；既有 Lead factory／prompt、配置工具、NIR 和 DynamicContext 中间件；进程内运行，省略无关 Gateway/UI 服务 |
| 初始决策 | LLM 自选 SG(window=7, order=2)＋PLS，潜变量网格 [1,2,3]；实际最优 n_components=1，共 10 fits |
| 初始验证 | Val RMSE=1.3491620455，R²=-0.1395387510 |
| 反馈决策 | LLM 根据弱拟合、斜率／偏差及剩余 4 fits，自选 SWITCH_MODEL：保留 SG，改为 Ridge 固定 alpha=1.0；没有代码强制这一步 |
| 合同修正 | 一次所选候选 action 与顶层动作不一致，明确拒绝且未拟合；LLM 自行修正后接受 |
| 第二轮执行 | 4 fits，完整参数／配置与接受记录一致；Val RMSE=1.3753914300，R²=-0.1842775483 |
| 预期检验 | Val RMSE 变化 +0.0262293845；声明的降低误差目标 not_achieved，未回写旧理由 |
| 完整关联 | 两个不同 context／decision／experiment；第二轮 base_experiment_id 和 Feedback.previous_experiment_id 均引用首轮成功实验 |
| 累计成本 | 14 fits，8 次实际模型响应，input/output=149067/5378 tokens；流程 34.40 s；两次科学执行约 0.125/0.094 s |
| 终态 | 两实验／14 fits 的硬预算终止；hard_budget，未自动保留模型，不是 LLM STOP |
| Test | 只打开 search.npz；未冻结／消费 Test，没有真实 Wheat/Meat/Corn 实验 |

真实初始模型使用了三候选、10 fits，次轮固定候选用 4 fits，均在 LLM 提交和服务器执行的累计 14-fit 许可内。验证提示原曾建议每实验两候选／七 fits，但该建议没有作为服务器硬上限；本次不能宣称执行了这个额外限制。复现脚本已改为清楚要求 LLM 在累计 14 fits 内自行分配，不再宣称未实现的单实验限额。模型选择与两轮执行事实来自原始证据，未修改成预期结果。

## 10. 复现方式

此环境复用 `D:\NIR-Agent\deer-flow\backend\.venv\Scripts\python.exe`，使用仓库 `.tmp/phase1-deps` 中已经安装的 Chemotools 0.4.4；未更改外部环境或模型密钥。使用新 DEER_FLOW_HOME／basetemp，避免旧实验互相影响。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:PYTHONPATH='D:\ChemAgent\nirs-agent\.tmp\phase1-deps;D:\ChemAgent\nirs-agent\backend\packages\harness;D:\ChemAgent\nirs-agent\backend;D:\ChemAgent\nirs-agent'
$env:DEER_FLOW_HOME='D:\ChemAgent\nirs-agent\.tmp\new-phase3-validation-home'
$py='D:\NIR-Agent\deer-flow\backend\.venv\Scripts\python.exe'
& $py -m pytest backend/tests/test_nir_multiround_decisions.py backend/tests/test_nir_multiround_agent.py backend/tests/test_nir_multiround_context_cost.py backend/tests/test_nir_multiround_legacy_replay.py backend/tests/test_nir_initial_decisions.py backend/tests/test_nir_autonomous_initial_agent.py backend/tests/test_nir_authorized_parameters.py backend/tests/test_nir_process.py -q -p no:cacheprovider --basetemp=.tmp/new-phase3-focused
& $py -m pytest backend/tests -q -p no:cacheprovider --basetemp=.tmp/new-phase3-full
# 需真实服务、凭据和明确联网授权，会产生 API 费用：
& $py backend/scripts/verify_nir_initial_llm.py --allow-real-provider --multi-round
```

## 11. 已知限制与后续条件

1. STOP 已持久化，停在 awaiting_final_confirmation；尚未接入完整 LLM STOP→确认→冻结→单次 Test→注册和恢复流程。未越过封存去“补齐”结果。
2. 初始模式默认保持单次执行；只有显式 autonomous_rounds=true 才进入多轮。部署需加载更新后的配置／Prompt／Skill；本次真实测试不是现有 Gateway/UI 的部署验收。
3. 多轮动作由 LLM 决定，不保证性能单调改善。真实次轮变差正是应保留的反馈；没有新增阈值停止或固定模型切换规则。
4. 只适配单目标回归和四模型，预处理继续受原注册表、数据轴和 provider 限制；不支持自主波长选择、轴变更／监督型预处理或复杂新优化器。
5. 文本假设的支持解释留给后续 LLM；工具只检验声明的可计算 Val 改善目标。没有用规则假装理解全部自然语言目标。
6. 上下文已明显缩小每次输入，但累计 token 成本仍较高；没有通用硬 token 上限或货币费用估计。高层动作上限不等于模型修正请求上限；本次真实脚本独立强制九次请求，生产通用 LLM 费用硬限制仍是后续工作。
7. 状态工具按 context→submit→execute 的顺序调用；并行状态变更的完整验收和 executing lease 的崩溃自动恢复编排仍未开展。运行时 deadline 仍为合作式检查，不能强制打断第三方库正在进行的 fit。
8. 原完整 backend 环境失败仍存在；报告没有把它们说成全部通过。历史实验字节保留、legacy NIR 与初始决策回归通过。
9. 没有跨任务历史复用、论文对比、大规模真实数据实验、十二模型适配或前端重构。

后续开发需用户下一步明确指令，并继续基于现有决策、会话、预算、封存与 lease 扩展最终确认生命周期。第三阶段交付在这里停止。
