# ChemAgent 第五阶段实施报告

日期：2026-10-10。仓库：`D:\ChemAgent\nirs-agent`；分支 `dev/Duan`；工作基点 `ef288ea3ec241507c63164828a012a2fd519a440`。

**第五阶段网页自主建模主流程已完成，并通过一次真实 DeepSeek 的网页预算 RETAIN → Freeze → Test → Report 验收。** 真实桌面浏览器完成上传、自然语言启动、初始自主选择、数值训练、依据反馈保留、冻结、独立 Test、模型／HTML 报告／ZIP 下载及刷新恢复。验证使用小型合成数据，实际接受 2 个决策、执行 1 个数值实验、拟合 16 次；本轮没有真实主动 STOP，也没有多次数值实验后的真实交付，不能把 RETAIN 当成 STOP 或第二次训练。

前端全量 573 项通过；修复后的 backend NIR 回归 818 项通过、2 项跳过；完整 backend 回归 6483 项通过、88 项失败、45 项跳过，失败节点与修改前基线的 88 项完全一致。全库并非全绿。下文分别记录验证范围及限制。

开始时工作区已有第一至第四阶段大量未提交修改。本轮没有 reset、checkout、覆盖前四阶段报告或历史模型、提交或推送；测试使用新隔离目录，三次真实网页运行分别创建新会话。没有进入第六阶段、扩展模型、开展论文对比实验或跨任务记忆复用。

## 1. 现有前端审查与复用

| 原有能力 | 入口与来源 | 本阶段复用 |
| --- | --- | --- |
| 自然语言聊天及流式运行 | 默认 `/workspace/chats/[thread_id]`、InputBox、thread hooks、原 LangGraph SDK／Gateway | 增加显式自主模式和预算控件，沿用上传、提交、消息及流式更新 |
| 上传及数据绑定 | 原 owner/thread 上传、挂载、数据集资产接口及 workflow 澄清 | 不另建摄取服务；首轮安全绑定阻止样本行预览和旧任务记忆 |
| 工作流状态 | live `thread.values.nir_workflow`、owner-checked thread state fallback | 读取当前真实 search_session_id、策略及审批／注册状态 |
| 建模过程 Sheet 与独立页面 | `nir-workflow-panel.tsx`、`/workspace/nir/process/[thread_id]`、ProcessWorkspace | 按实际策略／session 分流，保留 legacy 组件 |
| 图表与请求管理 | ProcessChartView、`core/nir/process-{api,hooks}.ts` | 共用 SVG、指标交互、鉴权请求、owner/thread cache key 和 ETag |
| 决策及反馈持久化 | ProcessStore v5、SearchSessionStore、initial_decisions／search_experiments | 新增只读投影，执行前 proposal 和执行后 feedback 分开读取 |
| 最终报告与工件 | 第四阶段 delivery adapter、原报告、ZIP、artifact 服务 | 展示原发布工件并通过原下载接口交付 |

原过程接口主要组织 legacy attempt/step/chart，不能直接表达 autonomous 的完整 proposal、接受记录、feedback 和 session 生命周期。本阶段增加最小只读 adapter，没有重建 Agent、训练服务、分区、决策库或实验管理平台。

## 2. 前后端实际修改文件

路径相对仓库根目录。清单仅列本阶段职责；当前 Git diff 同时包含此前未提交工作，不能将其总量全部归入第五阶段。

| 文件 | 实现内容 |
| --- | --- |
| `backend/packages/harness/deerflow/community/nir/web_entry.py`（新增） | 网页结构化模式校验、有限预算解析及内部可信绑定 |
| `backend/app/gateway/services.py` | 创建 run 前校验；URL thread 为准；防伪造内部 context、已绑定策略降级及 checkpoint rewind |
| `backend/packages/harness/deerflow/community/nir/workflow.py` | 消费网页可信绑定；四个安全字段一致；保留旧 Runtime 无 context 的兼容性 |
| `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py` | 首轮即启用工具权限与模型调用准入；持久预算、决策及执行链沿用原服务 |
| 同目录 `dynamic_context_middleware.py`、`memory_middleware.py`、`uploads_middleware.py` | 网页启动即隔离跨任务记忆和光谱样本预览 |
| 同目录 `tool_output_budget_middleware.py` | 七个封存协议工具的 JSON 在工具输出和历史预算边界保持完整，避免 context/schema 被替换为文件预览 |
| `backend/packages/harness/deerflow/community/nir/autonomous_view.py`（新增） | session、决策、拒绝、反馈、生命周期、图表的有界只读投影 |
| `backend/app/gateway/routers/nir_process.py` | 原 owner/thread 权限下的自主只读路由、session 归属、分页及 ETag |
| `backend/packages/harness/deerflow/community/nir/dataset_session.py` | 原唯一 Test 评价租约内保存有限散点／残差缓存 |
| 同目录 `search_tools.py`、`decision_delivery.py` | 图表大数组不进入 LLM 输出；严格摄取错误及已有科学门禁反馈；发布继续使用原服务 |
| 同目录 `decision_tools.py` | 核验真实模型来源后记录拒绝提案；候选、拒绝和已执行实验分开 |
| `nir_core/io/loaders.py` | 修复明确命名目标列＋数值波长表头被当成样本的问题，保留纯数值首行与缺失值检查 |
| `frontend/src/core/nir/autonomous.ts`（新增） | 有界解码、scope／阶段校验、STOP/budget/RETAIN、预算、保留身份及真实参数 trial 投影 |
| `frontend/src/components/workspace/nir/autonomous-mode.tsx`（新增） | 常规／自主模式；开发验证／自主搜索／自定义四项预算 |
| 同目录 `autonomous-process.tsx`（新增） | 概览、时间线、按需详情、参数 trials、Val 趋势、最终 Test 图、冻结／质量／下载 |
| `frontend/src/components/workspace/input-box.tsx`、`frontend/src/app/workspace/chats/[thread_id]/page.tsx` | 默认桌面聊天提交结构化模式；开始后锁定控件 |
| `frontend/src/components/workspace/nir/nir-workflow-panel.tsx`、`process-workspace.tsx` | 原 Sheet／页面复用自主组件；workflow 与 session 身份一致；修复 legacy 子组件清理父级共享缓存的问题 |
| 同目录 `process-chart.tsx` | 原 SVG 图表增加清楚的数据范围标识 |
| `frontend/src/core/nir/process-api.ts`、`process-hooks.ts` | 原权限与缓存机制下的 ETag、按需读取及有界轮询 |
| `frontend/src/core/messages/reasoning-visibility.ts`（新增）、`components/workspace/messages/message-list.tsx` | 自主页面不显示原始冗长 reasoning；保留工具、原决策理由、使用量和存储记录 |

新增后端测试：`test_nir_web_entry.py`、`test_nir_web_entry_agent.py`、`test_nir_web_checkpoint_guard.py`、`test_nir_autonomous_view.py`、`test_nir_autonomous_router.py`、`test_nir_upload_preview_isolation.py`、`test_nir_web_csv_ingestion.py`、`test_nir_web_ingestion_failure.py`、`test_nir_web_protocol_output.py`。扩展 `test_nir_delivery_agent.py` 的 Fake Lead middleware 注入以验证真实输出预算交互。

新增前端测试：`frontend/tests/unit/core/nir/autonomous.test.ts` 和 `frontend/tests/unit/core/messages/reasoning-visibility.test.ts`。文档更新：`README.md`、根／backend／frontend 的 `AGENTS.md`、本报告及 `docs/chemagent-phase5-real-web-evidence.json`。

四种模型、预处理、参数策略、后续动作及 STOP/RETAIN 的控制权仍来自已有真实 LLM 决策服务。本阶段没有增加隐藏 baseline、固定阈值选模或程序代替 LLM 选择保留实验。

## 3. 网页自主建模流程及真实调用链

1. 在默认桌面新聊天通过原上传入口上传数据，输入目标、单位、样本方向等自然语言要求。
2. 明确选择“自主建模”，配置开发验证、自主搜索或合法自定义预算：实验数、fit 数、高层动作数、搜索秒数。
3. 原 submit 发送 `context.nir_autonomous`；无需把内部协议标志写入自然语言 Prompt。
4. Gateway 在创建 run 前验证策略，绑定 `sealed_holdout`、`llm_autonomous`、`autonomous_rounds=true`、`autonomous_delivery=true`。普通模式不被悄悄改为自主模式；不能由客户端伪造 workflow 或恢复 checkpoint 来重置预算。
5. 实际 DeerFlow Lead 通过原 workflow 与摄取／澄清流程确认任务，prepare 建立原 Cal/Val/Test 分区；Cal-only DecisionContext 提供诊断、知识、能力和剩余预算。
6. LLM 提交合法结构化方案，保存不可变来源与 proposal；执行工具仅接受 decision_id，数值内核严格在授权预处理、模型和参数区域内训练。
7. 后续上下文读取真实 Experiment/Feedback；LLM 继续优化、主动 STOP，或在硬预算结束后使用终结 reserve 提交 RETAIN。程序不替代这一选择。
8. 原 freeze 绑定 LLM 指定成功实验的 Cal 工件；原 finalize 获得唯一 Test 租约后评价并发布模型、报告及 ZIP。搜索不能在最终 Test 后继续。
9. 用户查看原过程 Sheet／独立过程页面并下载工件。导出、质量达标、用户批准和注册为四种不同事实，不自动批准注册。

当前运行配置的模型／工具沿用真实服务；验证仅在 `.tmp/phase5-web-config.yaml` 和隔离 `.tmp/phase5-web-home` 使用有限配置及合成数据。没有覆盖用户现有密钥或原历史状态。真实接口为当前 DeepSeek，模型名 `deepseek-flash`，网页执行档为 Flash。

## 4. 决策过程可视化

任务概览展示 session、当前 stage、实验／fit／接受动作累计、停止原因、冻结和发布状态。准备前、数据失败、提案非法、拟合失败、预算终止、Test 失败及发布失败具有不同提示。进度来自事实，未添加虚构百分比。

时间线每页 10 条接受记录，显示实际 round/action、选择管线、原 rationale、expected_outcome，以及单独保存的执行状态。候选未执行、接受待执行、成功、失败、拒绝、STOP 和 RETAIN 不混同。拒绝记录单独分页，并关联已核验的模型调用；不产生虚构指标。

展开一轮按需读取观察、不确定性、知识证据、候选比较、未选理由、预期、完整 accepted_spec，以及实际参数、Cal CV/Val、CV 配置、比较、预期效果反馈、拟合数、耗时和 trial。没有从指标变化补写原先理由，也没有把内部思维链当成正式决策证据。

Val 趋势只含同 session 成功且 scope 合法的实验。最低 Val RMSE 的“当前最佳”与 LLM 明确保留的实验分别标记；最佳标记只是信息，不改变冻结选择。真实网页已检查指标选择、保留标记和参数 trial 展开。

最终区域显示冻结引用、实际参数、Cal/Cal CV/Val/Test、质量门禁、审批／注册事实及已发布下载。独立页使用交付快照，并结合 owner-checked 当前 workflow 与 session 身份；聊天 Sheet 使用实际当前 workflow。真实低质量模型没有被显示为已批准注册。

## 5. 数据接口与来源映射

路由前缀：`GET /api/threads/{thread_id}/nir-process/autonomous`。全部复用原 thread read 权限及 owner 检查；SQLite 使用 `mode=ro`，不迁移、训练、freeze、finalize 或读取 holdout。无索引的旧会话返回空列表，不为展示创建数据库。

| 路由后缀 | 用途 |
| --- | --- |
| 空后缀 | session 列表，每页 20，offset 分页 |
| `/{session_id}` | 摘要、预算、趋势、停止／冻结／交付；持久字节 revision 与 ETag/304 |
| `/{session_id}/decisions` | 默认 10、最多 20 条，含 total／next_offset |
| `/{session_id}/decisions/{decision_id}` | 同 session 的原 proposal、accepted_spec、执行 feedback／trials |
| `/{session_id}/rejections` | 真实来源的拒绝提案，10 条分页 |
| `/{session_id}/charts/final_test` | finalized 后的合法 Test 点缓存；封存时拒绝，旧缺失记录明确不可用 |

| 页面字段 | 可信来源 | 边界 |
| --- | --- | --- |
| task/session/stage/Test 状态 | search_sessions.payload／task_snapshot | 封存、租约、完成分开 |
| round/action/status、decision→experiment | initial_decisions.payload 及 status/execution 列 | 同 session 精确身份关联 |
| 观察、候选、理由、预期、知识证据 | 不可变 DecisionRecord.proposal | 限长按需读取；缺失不推断 |
| 实际配置、参数、CV、Val、trials、比较、成本 | execution/feedback | 成功指标必须有正确 scope；候选无伪造结果 |
| fits、实验、动作、时间、LLM ledger | session 累计和 delivery_policy | 接受动作≠provider 调用；准入估计≠实际 token 或金额 |
| 停止及 retained experiment | session 与 STOP/RETAIN 决策 | budget 与主动 STOP 分开 |
| freeze/最终参数 | session.freeze | 原 LLM 保留的 Cal 工件，终结不新增 fit |
| Test 指标／图 | finalized session.final | 原唯一租约缓存；GET 不重读 Test 或模型 |
| model/report/ZIP/evidence | 成功 export 工件清单 | 允许 outputs 下的发布文件，禁止私有 session 路径 |
| approval/registration | 相符 workflow 与交付 lifecycle | 发布完成不代表审批／注册 |

前后端 adapter 均做字段、深度、集合、有限数值、scope 和下载白名单校验；不暴露私有 Test 路径、光谱矩阵、分区行号、密钥、完整系统 Prompt 或历史 raw reasoning。ProcessStore 仍为 v5，没有新实验数据库或应用 ORM 迁移。

## 6. 图表、报告与实际下载

| 能力 | 实现及验证 |
| --- | --- |
| Val 性能 | 原 SVG 展示实际 round 对 Val RMSE/R²，成功实验和保留标记；真实网页已切换并查看 |
| 参数优化 | 按需展示真实 Cal CV trial 表与曲线；真实 3 个潜变量候选已显示。mean-fold RMSECV 与 OOF Cal CV RMSE 分开，不假称相同 |
| 最终 Test | 原 finalize 内缓存 prediction/residual 点，最多 2000、确定性采样并标明总体与显示数量；真实 12 点散点／残差图已检查 |
| 光谱图 | legacy 图保留；当前自主 Cal 原始／预处理曲线未持久化，显示“未记录”，不读取文件或 Test 补画 |
| 工件 | model.joblib、report.md、report.html、delivery.zip、metrics.json、decisions.json；沿用原 artifact 接口 |

真实网页已下载模型、HTML 报告和 ZIP。下载文件与发布文件 SHA-256 完全相同；ZIP CRC 合法，包含模型、manifest、报告和完整决策／指标证据。原 HTML artifact 下载安全策略保留，没有为了预览改为执行主动网页内容。

已先验证本地原生模型完整性 manifest，再核对浏览器下载副本字节一致并加载；在 3 行新生成合成输入上得到有限预测，与原发布模型一致。验证禁止额外 fit，未读取 Test 文件。报告包含两条接受 decision_id、Cal CV/Val/Test 范围、预算 RETAIN、参数及生命周期。工件质量较差、未审批、未注册，仍可合法导出研究证据。

## 7. 安全、兼容、恢复及真实失败修复

保留前四阶段的真实来源审计、参数／用户约束、累计预算、Cal/Val 搜索与最终 Test 封存。网页首轮就禁止 legacy 训练、MCP fit、原始文件读取、脚本及跨任务记忆绕行。非法参数仍要求 LLM 修正，不替换默认方案。

三次真实运行均逐次获得人类联网／成本授权；失败记录与会话保留，未追加第四次运行。

| 实际问题 | 源码原因及修复 | 证据 |
| --- | --- | --- |
| 第一次 CSV 摄取失败，无 prepare/fit/Test | `protein,900,…,1600` 的混合表头被当成样本；loader 对明确命名目标＋数值波长进行表头识别，保持有限值科学检查，摄取错误准确终止 | 新 CSV 边界回归及 46 项 CSV／安全入口测试 |
| 第二次 context 循环及提案失败，无 fit/Test | 通用输出中间件把 >12k 的完整 DecisionContext 外置／截断，丢失版本及当前 schema；read_file 被安全协议正确拒绝 | 七个协议工具在两处预算边界保持原 JSON；累计 NIR 模型准入仍收费；8 个先失败后通过的专项测试，122 项协议／交付回归 |
| 独立过程页退回 legacy 视图 | legacy 子组件卸载时删除 owner 的共享 nir-process queries，清除了父组件的自主 session 查询 | 只清理子组件 nir-process-legacy 查询，保留父级清理职责；真实浏览器刷新恢复已通过 |

修复没有开放原始文件权限、降低科学门禁或代替 LLM 选模。旧 Runtime 不含 context 时的兼容性已修复并回归。最终图表不进入后续决策上下文；历史缺失图不补算。

浏览器刷新前后，整个 session JSON、决策／执行 JSON、final JSON 哈希一致；fits=16、实验=1、接受决策=2、Lead 回执=14、run 数=1、发布 generation 均不变。真实工具调用跨 163 个持久 root checkpoints 去重：freeze 1 次、finalize 1 次。真实网页未装文件 IO spy；物理 Test 读取一次由 Fake 集成 spy 和原租约测试验证，不把快照中的推导字段当成独立 IO 计数。

hooks 使用 owner/thread/resource key、AbortSignal、ETag、隐藏标签页暂停及有界轮询／重试；详情、图表按需加载，20+ 记录分页已自动化测试。实际浏览器没有另做 20+ 轮性能实验。

## 8. 自动化、浏览器与真实 LLM 验证

### 8.1 自动化回归

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 修改前完整 backend | 6408 passed、88 failed、45 skipped，383.56 s | `.tmp/phase5-baseline.log`／`.xml` |
| 修改后完整 backend | 6483 passed、88 failed、45 skipped，396.53 s；同 88 失败节点，无新增失败 | `.tmp/phase5-final-backend-b.log`／`.xml`、`.tmp/phase5-baseline-comparison.json` |
| 修复后的全部 backend NIR | 818 passed、2 skipped，79.19 s | `.tmp/phase5-nir-final.log`／`.xml`，含前四阶段及本阶段路径 |
| 输出预算／Schema／网页 Fake 交付 | 122 passed，11.90 s | `.tmp/phase5-protocol-green-b.log`／`.xml` |
| CSV loader／安全入口 | 46 passed | `.tmp/phase5-csv-core.log`／`.xml` |
| 修改前 frontend | 529 passed，63 files | `.tmp/phase5-frontend-baseline.log` |
| 最终 frontend 全量 | 573 passed、0 failed，65 files，6.040 s | `.tmp/phase5-frontend-acceptance.log`，退出码 0；含新增 44 项 |
| TypeScript | `tsc --noEmit --pretty false` 退出码 0 | `.tmp/phase5-typecheck-acceptance.log` |
| Ruff／Prettier／Git 空白检查 | 本阶段受检代码通过 | scoped Ruff check/format、前端 Prettier check、`git diff --check` |
| ESLint | **未完成**：本地依赖缓存缺少 unrs-resolver@1.11.1 的文件 | 未用禁用规则或空检查伪造通过，未网络安装依赖 |

完整 backend 结果在协议输出修复前收口；其后对实际变动运行了完整 NIR 和相关通用中间件的 122 项专项复验，未再以无新关注点为由重复整个 backend。88 项基线失败没有在本任务中逐项治理，不能宣称全库验收全绿。nir_core 本阶段摄取变动由 CSV 回归覆盖，没有把 backend NIR 数量称为整个 nir_core 全库结果。

后端测试覆盖可信模式、预算绑定、checkpoint 防降级、权限／跨用户／session、拒绝记录、分页、ETag、指标 scope、GET 不加载 Test/模型、未批准不注册及 legacy 兼容。Fake 使用小型合成数据与实际 Lead 工具循环，加入生产 ToolOutputBudgetMiddleware，分别走初始选择→TUNE→STOP 和预算 RETAIN→freeze→Test→发布；fit 累计及物理 Test 单次读取／缓存重读有 spy 验证。Fake 不能证明真实 provider 已运行，真实证据单列如下。

### 8.2 三次有界真实网页运行

统一合成 CSV：60 样本、8 波长列、单目标 protein；自然语言明确目标及方向，但没有指定模型家族、预处理、优化动作或正常停止轮次。用户预算 3 实验／18 fits／6 接受动作／240 s；服务器搜索准入 30 次／120 万估计 token，终结预留 6 次／50 万。没有发送真实业务样本。

| 次数 | 结果 | 实际 Lead token（input＋output） |
| --- | --- | --- |
| 第一次，thread `24ca3eb2-1674-4f59-b395-2358e6a2c02b` | 表头摄取失败；0 实验／0 fit／0 Test；不算建模成功 | 242610＋16663＝259273，8 响应 |
| 第二次，thread `cbaf4f19-75eb-47ad-986c-62b11d35eb18` | 原子 context 被通用中间件截断；LLM token 准入耗尽；无成功实验可 RETAIN；0 fit／0 Test | 295825＋3113＝298938，11 响应 |
| 第三次，thread `7d1cd330-1dd8-49db-af29-2574dac46787` | **完整网页预算 RETAIN 交付成功**；1 实验／16 fits／2 接受决策／1 合法 Test | 485843＋6466＝492309，14 响应 |

三次 Lead 实际总计 1,050,520 tokens。标题及建议生成另行调用，不在上述 Lead 统计内；不据此编造 API 金额。第三次持久准入 ledger 搜索 7 次、估计 charged_tokens=855126，终结 4 次、459796，均在各自授权上限。ledger 的实际 usage 合计 342671 tokens；发布后 3 次 Lead 说明共 149638 tokens 使用原普通运行限制，在 Lead 总数中，但不在搜索／终结 ledger 内。网页总使用量包含不同调用，不与该 ledger 混淆。这一成本边界沿用第四阶段，仍是限制。

第三次 session：`search_8ca9788edd5443b1a880ee8b96783e41`；run：`8d03c902-4fc5-430d-a6df-9fc6df835ed7`。

| 决策／执行 | 真实结果 |
| --- | --- |
| 初始决策 | `decision_ffb0876a759f6bf934ecce31bbda47fd`；真实 LLM 自主选择 SG smooth(window=7, order=2)＋mean_center＋PLS |
| 参数授权 | 固定 scale=false、max_iter=500、tol=1e-6；grid n_components=[1,2,3]；max_trials=3、max_fits=16 |
| 数值执行 | `exp_9aabaec250fdcbaed742d319626dd1bd`；真实 5 折 Cal CV×3 候选＋1 Cal 拟合＝16；实际潜变量数 3 |
| 后续决策 | 剩余 2 fits 不足一次合法固定参数执行；stop_source=budget、reason=fits；LLM 读取实际反馈并提交 `decision_eb119d8eb436b8c93572b1c39dfa7c66` RETAIN，明确保留上述实验，新增 fits=0 |
| 冻结／Test／交付 | 原 retained experiment 一致；2026-10-10 02:21:27Z 冻结，02:21:29Z 完成发布；1 次 finalize，12 个 Test 样本 |
| 生命周期 | quality_passed=false、user_approved=false、registered=false；图表、下载和刷新恢复正确 |

| 指标范围 | n | RMSE | R² | RPD |
| --- | --- | --- | --- | --- |
| Cal / calibration | 36 | 1.3008 | 0.6248 | 1.6556 |
| Cal OOF CV / calibration_cv | 36 | 1.5974 | 0.4342 | 1.3483 |
| Val / validation | 12 | 1.5580 | 0.4870 | 1.4582 |
| Test / final_test | 12 | 1.3340 | 0.5840 | 1.6194 |

选择准则使用 Cal mean-fold RMSECV；三个 trial 依次为约 1.8644、1.6412、1.5585，不能将最后的 1.5585 误写为 OOF RMSE 1.5974。质量 food_protein 门槛 R²≥0.85、RPD≥3.5，实际未通过。

浏览器实际检查：上传与自然语言、显式预算、真实决策时间线和详情、参数 trials、Val 指标交互、最终 Test 散点／残差、低质量未审批注册、模型／报告／ZIP 下载、独立页刷新恢复。真实调用没有预设 PLS 或 RETAIN；模型选择与保留由已核验 provider 回执及接受 proposal 证明。

公开有界证据：`docs/chemagent-phase5-real-web-evidence.json`；保留完整私有日志与持久状态于 `.tmp/phase5-gateway-live.log`、`.tmp/phase5-gateway-real-b.log`、`.tmp/phase5-gateway-real-c.log`、`.tmp/phase5-web-home`。下载核验：`.tmp/phase5-download-verification.json`；刷新前后：`.tmp/phase5-browser-{before,after}-refresh.json`。界面截图：`.tmp/phase5-third-{ready,timeline,trials,charts,after-refresh,delivery}.jpg`。原始调用上下文／reasoning／私有分区路径没有复制到公开证据。

## 9. 剩余问题与八项验收回答

| 分类 | 结果与范围 |
| --- | --- |
| 已实现且已验证 | 显式安全网页入口、真实持久决策／反馈、有界只读接口、权限及 scope 自动化、参数 trial 图、Val 趋势、Test 缓存、真实预算 RETAIN 全交付、下载及刷新恢复 |
| 部分覆盖 | STOP、TUNE、CHANGE_PREPROCESS、SWITCH_MODEL、EXPLORE 等已有合同及回归继续有效；网页 Fake STOP 完整链通过。本次真实页面只有初始 EXPLORE 和后续 RETAIN，其他真实动作没有被截图或宣称执行 |
| 尚未验证 | 本阶段真实主动 STOP、多次数值实验后的真实交付；生产鉴权端到端浏览器（真实运行用隔离 localhost owner=default）；20+ 轮浏览器性能、用户后续审批注册、部署级取消／异常恢复的全部分支 |
| 未实现 | 当前自主执行路径 Cal 原始／预处理光谱缓存与对照图；专用参数组合热图。已有缺失图不补造，trial 曲线与表已支持 |
| 基线／环境限制 | 全 backend 88 个既有失败；ESLint 缺 unrs-resolver 文件；当前 Anaconda MCP discovery 缺 chemotools，已记录失败并使用受控本地能力，真实训练使用 pinned runtime |
| 成本／解释限制 | token 准入是保守估计，时间是合作式限制，尚无金额硬限；交付后普通 Lead 说明不计入模型流程 ledger；自由文本说明可出现计数偏差，正式页面／报告以不可变接受记录和执行证据为准 |

模型质量较差是一次自由决策的真实结果；本阶段证明执行与交付链可用，不证明所有数据均能达到质量门槛或所有真实动作已验证。SDK run success 仅表示图结束：前两次业务失败仍记录为失败。

| 最终验收问题 | 严格回答 |
| --- | --- |
| 一：网页上传＋自然语言启动 LLM 自主建模？ | **是，已由真实网页和当前 DeepSeek 完成。** 限默认桌面聊天、小型单目标回归与现有四种模型能力 |
| 二：实时展示每轮真实决策、方案和反馈？ | **是。** 本次实际初始选择＋反馈后 RETAIN 两轮已展示；复杂多次数值实验展示另有 Fake／接口回归，未冒充本次真实运行 |
| 三：查看选择、切换和停止理由？ | **是，展示实际已保存理由、候选和反馈。** 本次真实验证选择与 RETAIN 理由；未发生真实切换，不补造切换证据 |
| 四：区分主动 STOP、budget 和 RETAIN？ | **是。** 自动化覆盖 STOP/预算 RETAIN，真实页面显示 budget→LLM RETAIN；未称真实主动 STOP 通过 |
| 五：图表均为真实 Cal/Val 或合法 Test？ | **是，当前支持图表如此。** scope 自动化＋真实 trial/Val/Test 图已核对；未记录的光谱曲线明确不可用 |
| 六：网页获得最终模型、报告及交付？ | **是。** 三种文件实际下载并校验；模型加载新输入预测成功，报告引用与冻结一致，未自动注册 |
| 七：刷新恢复历史且不重训？ | **是，真实独立页面 reload 已验证。** session/decision/final 哈希、fit、run、Lead 数不变；20+ 分页另有自动化，无长历史浏览器性能结论 |
| 八：真实浏览器＋真实 LLM 完整端到端？ | **是，预算 RETAIN 分支已通过。** 不包括真实主动 STOP、多个数值实验或生产鉴权全部分支 |

本轮第五阶段收口，保留上述限制和所有失败证据；不自动进入下一阶段，也不再追加真实 provider 调用。
