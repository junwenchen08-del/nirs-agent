# Docker 自主决策异常结束诊断与修复（2026-10-11）

## 真实运行结论

用户截图对应 thread `3033f528-0758-4c88-8f4b-ce55b65dad14`、run
`fc5f2935-816f-4005-85bd-703333e548e5`、session
`search_2f29ea638207411fa15c20e2971f072e`。

这是未完成建模交付的异常终止，不是正常 LLM STOP，也不是预算 RETAIN。
通用 RunJournal 的 `success` 仅指 Agent 回合结束，并不证明业务成功。
只读检查持久化数据库、checkpoint 和 ProcessStore；没有重新调用 provider
或读取 holdout.npz。原运行记录保持不变。

实际有 **3 个成功数值实验、168 次拟合**，不是截图摘要中的 4 次训练。

| 实验 | LLM 动作和实际管线 | Cal CV mean-fold RMSE | Cal CV OOF RMSE | Val RMSE | Val R² | Val RPD | fits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exp_7cbc5671038ac3ca9fd4c9f502d48929 | EXPLORE：SNV → derivative1(window=11,order=2) → PLS，n_components=3 | 0.4617051631357832 | 0.4708028936292041 | 0.42087307694531056 | 0.8441995756438008 | 2.6165560582005307 | 61 |
| exp_e3c59a55f0063472b7cf272a25be0783 | CHANGE_PREPROCESS：MSC → 相同 derivative1 → PLS，n_components=3 | 0.4676683059753577 | 0.4720646019842197 | 0.4424855713231294 | 0.827787538802292 | 2.4887545958206188 | 61 |
| exp_dff77be69e44ed766d140f57e1780533 | EXPLORE：SNV → 相同 derivative1 → SVR，C=10,gamma=scale,epsilon=0.1 | 0.6950680737169905 | 0.7249126130292043 | 0.4470389269071354 | 0.8242250309810755 | 2.463405159890055 | 46 |

训练规模为 Cal=48、Val=16，5 折 KFold（shuffle=true、seed=42，有冻结 manifest）。
Test 仍封存，freeze/final 均为 null。没有合法最终保留选择，没有报告交付。
剩余 fits=832，实验额度剩余 37：本次不是训练预算耗尽。

运行所选 food_moisture 质量标准为 Val R²≥0.9、RPD≥4.0，三轮均未通过。
这不意味着应由程序停止；实际 LLM 已根据反馈调整预处理并探索 SVR。
任务中的分析物/单位由此前真实启动提案给出，不把它当作源文件单位已经核验的证据。

## 真实错误及源码原因

1. 三次 `nir_submit_decision` 响应 `finish_reason=length`，output_tokens=4096。
   部分提案缺失 expected_outcome，或把它放入 alternatives；第三次缺少
   observations evidence_refs、alternatives、selected_alternative_id、rationale 等。
   严格合同正确拒绝这些提案，没有猜测补齐。
2. 此后连续三个响应存在 `invalid_tool_calls`：JSON 语法不合法，未进入工具。
   原续接逻辑仅将它视为“提前返回自然语言”，没有告诉 LLM 解析错误。
   两次隐藏提醒后第三次关闭；截图中的“我会重新提交”并不等于工具已执行。
3. 原 guard 只把 workflow checkpoint 设为 blocked；SearchSession 仍为 searching，
   stop_source=null，持久状态不一致，没有清晰的流程失败原因。
4. `_attempt_evidence_records` 仅用 metrics_path 判断最新证据是否已列入历史。
   封存实验的轻量证据没有公开 metrics_path，所以第三个实验被追加一次，显示成“尝试4”。
5. 通用 legacy 摘要只列质量和 RMSECV，未展示 Val、未说明工具续接失败，
   容易误解成质量门禁停止了自主流程。

持久账本：20 次 Lead 模型响应，实际 input=700804、output=35049、total=735853
tokens；搜索 ledger 同样为 20 次，finalization=0。截图显示值约 692.2k，
本报告采用持久账本完整运行口径，不把 UI 数字与该总量直接相加。
没有声称这些失败调用已被退费或本次修复已证明真实 provider 成功。

## 实际修改

- `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py`：
  对非法工具 JSON 给出 parser line/column 反馈，复用原有两次续接上限；
  失败后持久化关闭 session 与统一工作流绑定。Provider 历史投影移除整段
  非法工具参数，原 checkpoint 保留。不解析修补并执行非法方案。
- `backend/packages/harness/deerflow/community/nir/decision_budget.py`：
  `close_incomplete_workflow` 记录 safety_closed / terminal_failure_reason=
  workflow_continuation_limit；非法 JSON source 为 decision_invalid。
  保留既有 budget 来源和已冻结/最终结果，不选择模型、不使用终结 fits。
- `backend/packages/harness/deerflow/community/nir/decision_origin.py`：
  模型响应账本保留 finish_reason、invalid-call identity/参数 hash/字符数；
  不为非法工具调用签发可执行决策凭证，不保存隐藏推理正文。
- `backend/packages/harness/deerflow/community/nir/decision_schema.py`：
  当前 context submission guide 明确根级 expected_outcome 与简洁候选表达。
  两个合法比较候选即可；未限制其模型、预处理或动作，不提高输出资源上限。
- `backend/packages/harness/deerflow/community/nir/response_grounding.py`：
  按 experiment_id 去重，未完成自主流程单独报告原因和 Cal CV/Val 范围，
  不把质量、流程失败或 Test 成功混为一谈。
- `backend/tests/test_nir_malformed_decision_recovery.py`：9 项新回归；
  README、根与 backend AGENTS 同步。

## 自动化验证

- TDD：修改前新测试 5 failed / 1 passed，复现去重、修正反馈和持久关闭缺陷。
- 定向合同/摘要/恢复：57 passed（初始 6 项新测试版本）。
- 全部 NIR：**913 passed**（904 既有 + 9 新测试）。
- 无开发源码挂载的候选镜像：66 passed，包含新恢复、统一入口、自动交付、MAT、Test 封存。
- 随后将新 Fake 测试强化为实际统一网页入口与自动交付：**9 passed**；
  同步/异步均验证修正后两次训练 → STOP → 自动冻结/Test/报告；
  持续失败只保留一次训练、0 Test 读取，session 与 stored_workflow 均关闭。
- Provider-only 压缩测试：20000 字符非法参数从后续输入删除，原始对象保持完整；
  这是字符级机制验证，不能据此宣称真实 API Token 降低了固定百分比。
- 后端广泛离线基线：6515 passed / 21 failed / 20 skipped；
  修改后：6524 passed / **相同 21 failed** / 20 skipped。失败身份逐项比较一致。
  与上一修复相同，9 个会自动下载 Node/readabilipy 依赖的 HTML 提取文件排除；
  不将这个范围声称为不受环境限制的全量绿灯。既有失败涉及 live client provider、
  XHTML MIME、部署环境文件及 Git workspace fixture。
- 本轮新增真实 provider 调用：**0**。没有利用用户 Test 调试本次修复。

## Docker 与历史保护

仅更新 Gateway，前端、provider 密钥、模型选择规则和预算上限不改。
基于已验证本地镜像叠加 5 个源码文件，不安装新依赖。
原 Dockerfile 以 sha256 image ID 作为 FROM 被当作远程镜像名而失败；
改用已保留本地回滚标签构建成功，失败日志也保留。

候选镜像：`sha256:2659a93430709edb8f87ecbf13fcb8c0849b844e35b27ca6f64e8163d13e01e1`。
备份目录：`D:\ChemAgent\docker-backups\json-fix-20261011-163234`。
已部署并核验：Gateway 运行上述镜像；490 个后端/数值源码文件 SHA256
均与当前工作区一致。切换前确认无活动用户任务；1292 个持久文件共
1144257819 bytes，备份复制与原件逐文件 hash 一致。
切换后历史文件无删除/修改，仅 SQLite shared-memory 文件变化，
config.yaml/.env/extensions_config.json hash 均保持原值。
原运行的 20 次调用、735853 tokens、原状态均保留。
公共首页和原聊天页 HTTP 200；受保护 API 在未登录时返回 401，权限未放宽。
部署实际监听 8001，容器内 `/health` HTTP 200、status=healthy；
核验时最初探测基准 compose 的 8002 端口被拒绝，改用运行中 Cmd 确认的 8001。
Ruff 与 git diff --check 通过。

## 限制

这次修复证明错误能被正确反馈、有限恢复与如实关闭，不能保证真实 DeepSeek
从此不会产生截断或非法 JSON。未追加真实有偿验收，不声称真实全流程通过。
原失败运行未迁移、未重写决策或自动冻结实验，截图历史文本也不会被重写。
本报告中的最好 Val 实验只是客观比较；它没有被程序自动确定为最终保留模型。
