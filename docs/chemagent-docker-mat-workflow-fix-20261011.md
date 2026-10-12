# Docker MAT 建模入口与检查后转建模修复（2026-10-11）

## 结论与验收范围

本轮修复了用户截图对应的三个实际问题，并更新正在运行的 Docker Gateway：统一自主回归拒绝 MAT 上传、已完成检查任务不能转入建模、MAT 检查反转已经正确解析的样本方向。

已通过断网 Docker 中的 Fake LLM + 实际 DeerFlow Lead + 实际数值训练完整流程测试，以及实际上传 Corn MAT 的 Cal/CV/Val 适配检查。**本轮未调用真实 DeepSeek，不将这些结果表述为真实 provider 再验收成功。** 本轮没有新增 API 费用，也没有在真实上传数据上评价最终 Test。

原失败会话现在符合安全转入条件，用户可以刷新网页后在原会话重新提出建模要求。若数据目标或单位仍缺少可信说明，仍需正常澄清，不从文件名推断单位。

## 原运行证据与失败原因

- 会话：`87a2336e-7972-4132-b9f3-7ac7404283b3`。
- 原运行：`b071c646-ebcc-4a11-ace7-3cd62c8e8529`。
- 原运行实际 usage：28 次 Lead 响应、1,297,073 tokens。这是原用户运行的消耗，本轮没有追加。
- Gateway 的 `success` 表示 Agent 本次对话运行结束；该次没有模型实验或搜索会话，不能据此认定建模成功。
- 文件：`corn_moisture.mat`，SHA-256 为 `4adcf7cbeb50d98d2a6a118dead951250e38795ab113451591432f0a429befbc`。实际 X 为 80×700，y 是 80 个样本的单目标向量，wv 是 700 点轴，端点为 1100、2500；本轮没有另行确认目标单位或波轴单位。

| 实际错误 | 源码原因 | 修复 |
| --- | --- | --- |
| `Unified regression requires an owned numeric upload` | `validate_unified_start` 的格式白名单没有 `.mat`；`nir_prepare_search` 也只接受 NPZ/CSV/TXT/TSV | 增加受所有者路径检查保护的 MAT；复用已有 `load_mat`、划分函数和 `SearchSessionStore` |
| 省略 validation_goal 时触发封存协议限制 | 默认网页已绑定 sealed_holdout，但启动入口仍要求显式填写原 legacy 验证参数 | 仅在统一入口省略时明确使用 `internal_holdout`；显式外部验证仍被拒绝 |
| `This thread already owns a workflow…` | 任何已有工作流均被同等锁定，已完成且未建模的 inspection 也不能转入回归 | 只允许完成且没有建模证据的 inspection/knowledge 转入；原记录不可变归档，已有搜索／预算不可重启 |
| 检查报告称 700 样本×80 波长 | MAT schema 已归一化为样本在行，`_summarize_array` 又根据矩阵宽度重新猜方向 | 对已经归一化的 v5/结构体/v7.3 MAT 明确使用 samples_in_rows；保留未解析数组的原启发式 |
| Dataset lineage 参数不完整／不一致 | 原运行的部分提案没有原样复制附件字段 | 保留原有严格校验，补充入口／Skill 的准确操作说明，不自动补造 lineage |

此前更新后的 73 项集中测试没有覆盖 MAT 检查后转建模，因此不能证明该路径可运行。本轮新增了专门的失败复现与完整执行回归。

## 本轮代码修改

| 文件 | 实际修改 |
| --- | --- |
| `backend/packages/harness/deerflow/community/nir/web_entry.py` | MAT 上传白名单；完成且未建模任务的严格转换条件；在原 receipt 表保存不可变前序快照和引用；持久状态检查阻止已有搜索／实验重启 |
| `backend/packages/harness/deerflow/community/nir/workflow.py` | 实际模型提交的统一启动可从完成的非建模任务进入；默认明确内部留出；保留 Lead 调用来源校验 |
| `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py` | 为这种转换提供统一启动工具视图和说明；转换前禁止直接 legacy/MCP 拟合绕行 |
| `backend/packages/harness/deerflow/community/nir/search_tools.py` | 用现有 MAT loader 准备封存分区；歧义目标／矩阵返回可修正的 mapping 错误与字段；显式变量映射；不创建重复搜索服务 |
| `nir_core/io/schema.py` | 增加按已解析样本数核对的兼容目标候选信息，避免将波轴误算为第二个目标 |
| `nir_core/io/sniffers.py` | 保留已解析的 MAT 样本方向，修复检查统计 |
| `nir_core/io/loaders.py` | 显式嵌套 MAT 变量使用现有 flatten_mat_leaves 保留 dotted path，与 inspect 的变量名一致 |
| `skills/custom/nir-coordinator/SKILL.md` | 同步 MAT、可修正映射、检查后继续建模与完整附件 lineage 的实际合同 |
| `backend/tests/test_nir_mat_unified_entry.py` | 27 项新回归，包括同步／异步 Lead、多轮实际拟合、STOP、单次最终 Test、歧义修正与权限检查 |
| `README.md`、根与 backend 的 `AGENTS.md` | 同步入口使用说明和约束 |

未修改四模型的选择、参数优化、正常 STOP 的主体。没有固定切换模型或停止规则。模型训练、封存、冻结、Test lease、报告和审批均复用既有实现。没有 reset、checkout、提交、推送或清空旧会话。

## 数据隔离和旧记录

统一回归仍使用 `llm_autonomous + sealed_holdout`，搜索只获取 Cal/Val。MAT 只在可信准备阶段解析，复用现有一次固定划分，后续工具不能重新读取来源以改变分区。歧义只返回变量名和合法映射字段，不把原始矩阵或目标统计塞进 LLM 上下文。

完成的 inspection/knowledge 必须没有模型工件、模型尝试、执行证据、搜索会话或已绑定的自主策略，才能继续。持久库再次校验已有模型尝试、搜索和决策记录。即使当前 state 看起来是检查任务，也不能隐藏已有搜索以重置预算。

原工作流保存到既有 `model_issued_calls` receipt 表，使用内容哈希标识；新工作流记录 `previous_workflow_id`。原先检查事实不回写修正，历史聊天和 ProcessStore 证据保留。新的准备结果提供正确 Cal 信息。

已读真实会话持久状态：inspection/completed，模型尝试=0，搜索会话=0；满足转换条件。原上传文件哈希保持不变。

## 自动化测试

全部在隔离、断网容器中运行，没有挂载真实 API 密钥。以下集合有重叠，不能将数目相加当作不同测试总数。

| 检查 | 结果 |
| --- | --- |
| 初始错误复现 | 新增的最初 18 项在修复前镜像均失败 |
| MAT／统一入口／封存协议集中回归 | 51 passed；含新增 27 项 |
| 全部 backend NIR 回归 | **904 passed**，8 warnings，187.74 秒 |
| 数值内核全部 IO 回归 | **83 passed**，1 个只读 pytest cache 警告 |
| 已部署镜像直接测试（不绑定本地后端源码） | **99 passed**，32.37 秒；含 MAT 两轮实验、STOP、预算 RETAIN、Test 封存和交付恢复 |
| 未筛选的全 backend 测试 | 前后均尝试；网页解析库 readabilipy 在断网环境反复启动 npm 安装，未完成，保留原始日志并停止这些测试容器 |
| 排除相同 9 个依赖 npm 安装文件后的 backend 对照 | 基线：6488 passed / 21 failed / 20 skipped；修复后：6515 passed / **相同 21 failed** / 20 skipped；失败集合一致，新测试增加 27 项，没有新增失败 |
| 静态检查 | 修改文件的新增代码无新增 Ruff 问题；`loaders.py` 原有泛型装饰器的 UP047 仍存在；`git diff --check` 通过 |

21 项既有／测试环境失败包括 live provider 测试、缺失仓库部署文件的检查、XHTML 工件 MIME 断言与 workspace/gitignore 检查；本轮没有修改这些功能，不能报告全后端全绿。完整失败集合和前后摘要保存于备份目录的 `backend-regression-comparison.json`。

为解除断网 npm 阻塞，仅在第二组对照中明确排除了这些测试文件：browserless_client、crawl4ai_tools、exa_tools、fastcrw_tools、firecrawl_tools、groundroute_tools、infoquest_client、jina_client、readability。未筛选运行的日志仍保留；不将排除后的对照称作未经筛选的全量成功。

同步／异步完整路径均使用实际 Lead 工具循环和 Fake LLM：MAT → 准备 → EXPLORE → 实际训练 → 反馈 → TUNE → 实际训练 → STOP → 冻结指定成功实验 → 一次 Test → 模型／报告交付。每条流程两次实验、8 次拟合；冻结实验与提交的保留引用一致。测试同时检查 archive 不变、直接训练绕行拒绝、旧搜索不可重启、重复准备不读原文件、多个目标不被静默选择。

## 实际 Corn 上传文件的离线适配检查

另用截图的真实 MAT 原文件做一次独立只读挂载的受控适配检查，测试目的为验证执行入口。它使用明确的 Raw/Ridge(alpha=1) 测试配置，**不是 LLM 自主决策结果**，不用于证明方法优越性。

- 复用既有划分：Cal/Val/Test=48/16/16，3 折 Cal CV。
- 实际拟合=4，搜索没有读取 holdout.npz，Test 保持封存，未冻结或评价实际数据 Test。
- Cal CV OOF RMSE=0.08838056955558082；平均折 RMSE（RMSECV）=0.08817453963329791，两者不混写。
- Val RMSE=0.048906216562880606，Val R²=0.9978962483022531。
- 来源目标单位仍未独立确认，因此误差不宣称某个具体单位。
- 原文件保持原哈希；运行只写临时容器，不覆盖用户已有实验。

详细证据为 `.tmp/corn-mat-adapter-evidence.json`，本轮副本也保存于部署备份目录。

## Docker 更新和保留检查

- 项目：`nirs-agent-duan`，网页入口 `http://127.0.0.1:2027`。
- 更新服务：仅 Gateway，前端和 nginx 不需要重新构建。
- 运行镜像：`sha256:34b4311f579778f3ebefdc136d80a4ad93812dc585af9f7143bd604691fc2024`。
- 原镜像回滚标签：`nirs-agent-duan-gateway:rollback-mat-20261011-152534`。
- 备份：`D:\ChemAgent\docker-backups\mat-fix-20261011-152534`。
- 常规 Compose 构建遇到 Docker 镜像代理请求 `docker:cli` 的 502。由于本轮没有依赖变更，改为从当前已验证本地镜像制作只复制 7 个修改源码文件的补丁镜像。构建文件与两次构建日志均保存；没有变更依赖、密钥或服务器额度。
- 运行镜像的 **490 个** backend/nir_core 源码文件与工作区 SHA-256 一致；原失败会话状态用新代码只读核验，满足继续条件。
- 停服务后备份并哈希核对 **1282 个文件、1,085,443,219 字节**。启动后核验没有丢失文件，只有 `data/deerflow.db-shm` 的正常共享内存变化；所有模型、报告、实验索引与上传文件没有覆盖。
- `config.yaml`、`.env`、`extensions_config.json` 哈希不变。
- `/health`、`/openapi.json`、首页、workspace 和原聊天页均 HTTP 200；未认证 `/api/models` 返回 401。HTTP 检查没有创建任务。

可按原 Compose 参数叠加备份中的 `rollback-compose.yaml` 回滚镜像。不要把新增数据自动恢复为旧快照。回滚只是恢复修复前代码，MAT 入口缺陷也会随之恢复。

## 剩余限制

本轮真实 DeepSeek 完整 MAT 建模流程未重跑，需要另行逐次 API 成本授权。已经证明的是修复后的 Docker 执行链、科学封存和真实文件适配；不能保证未来每次 provider 都提交合法方案。

实际原会话的失败、错误维度陈述和高 Token 记录保留。新建模应使用当前准备／DecisionContext 的真实数据证据。已有模型或已绑定搜索的旧会话仍应恢复原任务或开启新会话，不能用此转换重置 Test 或预算。

本轮仅修复 Docker MAT 入口与会话转换，没有推进下一阶段研究或添加新模型。
