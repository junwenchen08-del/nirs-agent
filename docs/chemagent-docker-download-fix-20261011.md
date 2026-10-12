# Docker 自动交付下载展示修复（2026-10-11）

## 结论与已有真实运行

用户截图对应的建模并非没有生成下载包。只读审查确认会话
`53a97e49-3406-4957-8bc9-ef323ec6d8e6`、运行
`30381bfa-8343-41c1-b8ad-63373baacf5e` 已真实完成交付。其安全搜索会话为
`search_02940626d20b4c098d6e84d6bca193de`，四次实验累计 309 次拟合；
LLM 主动 STOP 保留 `exp_c48ed9cfd54d5123caead23e45fa43c5`，自动执行器完成
冻结、最终 Test 与出版。当前待用户审批，未注册。

这是用户已运行任务的持久证据；本次修复没有追加真实 provider 请求，没有重新
训练、重新评价 Test 或修改保留实验。不要把本次只读文件验收写成新的 LLM 实验。

工件目录为：
`/mnt/user-data/outputs/search_02940626d20b4c098d6e84d6bca193de/delivery_42d4ef56f4524d03839457fedd21600e/`。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| delivery.zip | 152960 | f882fcdd8079fae8dc2cfbca6e76e03589b61923a38e96b72bb1217094f43f3e |
| model.joblib | 104126 | 927336fd8133c6d2ce831734ba6d40c7f7c4fe5b32d6e96fc25c92d15bef9750 |
| report.html | 19747 | 797abfbb34be30266b5e6dc27d2685ac60e234913967c7c819803a22df335813 |

ZIP 的 CRC 检查通过，含模型、安全清单、Markdown/HTML 报告、指标、决策证据与
交付说明。原文件保持原字节和原路径。

## 原因

1. `nir_submit_decision` 的 STOP/RETAIN 自动交付和自主 `nir_finalize_search`
   仅返回工件路径，没有通过现有 `artifacts` reducer 添加聊天附件；截图会话该通道为 null。
2. 聊天文件卡片原先主要依赖 LLM 的 `present_files` 调用；自动交付后的响应由
   程序确定性生成，因此没有该模型调用。不能为显示卡片而伪造 `present_files`。
3. 原确定性交付消息确实包含三个 Markdown 链接，但自定义 `<a>` 渲染覆盖了默认
   下划线和链接颜色，视觉上近似普通文本。文件生成、publication hash 和下载路由正常。

## 最小修复

- `backend/packages/harness/deerflow/community/nir/search_tools.py`： `_result`
  增加内部显式 `artifacts` 参数，仅成功出版分支提供可信模型/HTML/ZIP 引用。
- `backend/packages/harness/deerflow/community/nir/decision_tools.py`：自动终结交付
  成功后同时加入附件引用。失败响应不添加下载工件。
- `backend/packages/harness/deerflow/agents/middlewares/nir_workflow_middleware.py`：
  确定性交付消息携带真实 `nir_delivery_session_id`，不制造模型调用或改变决策审计。
- `frontend/src/core/artifacts/delivery.ts`：识别服务器交付消息中的 session；旧消息
  仅由已有 ZIP 链接恢复 session 引用。发布是否成功仍由只读接口证实。
- `frontend/src/components/workspace/messages/nir-delivery-files.tsx`：复用现有
  `useProcessResource`、`parseSummary` 和 `ArtifactFileList`，读取 owner/thread
  范围内 `autonomous/{session}`；仅显示 finalized、export succeeded、准确 session
  的允许工件。失败、不匹配和未发布记录不能靠文本生成下载卡片。
- `frontend/src/components/workspace/messages/message-list-item.tsx`：在确定性交付
  消息后显示原有下载卡片，并恢复普通与虚拟工件 Markdown 链接样式。
- `backend/tests/test_nir_unified_agent.py`：增强同步/异步 STOP、预算 RETAIN
  完整链路断言，验证附件合并保留历史文件、身份关联和单次 Test。
- 前端新增 `nir-chat-delivery.test.ts`（4 项）及 `nir-delivery-files.test.ts`
  （3 项），验证旧会话恢复、真实卡片下载 URL、只读刷新和错误边界。
- 更新 README 和三份 AGENTS 说明。

没有重写存储、Agent、协议或页面架构，没有改审批和注册门禁。刷新只读已发布记录，
不更新历史 checkpoint、不载入数值数据、不拟合模型、不评价 Test。

## 验证

| 检查 | 结果 |
| --- | --- |
| 修改前红测：统一 Lead 交付增强断言 | 4 failed / 4 passed，缺 session metadata |
| 修改前前端投影新增测试 | 4 failed，缺少交付恢复能力 |
| 修复后针对性后端检查 | 34 passed |
| 候选 Gateway 生产镜像（未绑定宿主源码） | 55 passed，含统一入口、自动交付、工件安全及封存 |
| 前端全部单元测试 | 588 passed（69 文件，含新增 7 项） |
| 规范修正后的附件/卡片复测 | 24 passed |
| TypeScript | 通过 |
| Python Ruff / git diff --check | 通过 |
| 前端本次文件 ESLint | 0 errors，原 MessageListItem 的 2 个未改动 Hook dependency warnings |
| 前端离线生产编译 | 通过 |
| 真实已有文件经生产下载路由 | ZIP、模型、HTML 均 200，attachment headers 和字节/hash 一致 |
| 下载权限拒绝 | owner check 拒绝返回 404（隐藏存在性）；未登录返回 401 |
| 浏览器实际点击 | 未验证：Chrome 控制连接两次失败，没有替代提取浏览器凭证 |

真实文件下载检查运行在离线容器的内存测试应用，以测试身份注入权限；复用生产
路由及实际 owner-scoped 路径解析，原数据目录只读挂载。它不是生产浏览器 JWT 登录
验收。检查禁止 `np.load` / `joblib.load`，不会读取 Test 数组或加载模型。
证据见 `.tmp/download-real-bytes-router.json`；截图运行的只读审计在
`.tmp/delivery-download-audit.json`。

广泛后端修改前基线：6511 passed、25 failed、20 skipped；其中 4 个失败是新增
交付断言的预期红测，另外 21 项是既有环境/配置失败。基线沿用上一修复的 helper，
额外排除了已通过的 9 项 malformed-decision 测试；修改后会包含这 9 项。
两侧均明确排除会触发 readabilipy/npm 自动安装的 9 个 HTML 抽取测试文件，
所以不能声称无条件完整套件通过。

首次修改后广泛回归在 94% 时容器退出 137，未产生总结，不计为通过；保留日志
`.tmp/download-full-incomplete.log`，随后在前端编译结束后单独重跑。

独立重跑完成：**6524 passed / 21 failed / 20 skipped，433.38 秒**。
逐条比较失败节点，修复前的 21 项既有失败全部保持一致，没有新增失败；4 项交付
红测恢复通过。通过数额外包括基线 helper 排除的 9 项 malformed-decision 测试。
失败属于 live/config、XHTML MIME、部署 extras、隔离环境缺文件/Git fixture 等；
详见 `.tmp/download-full-bounded.log` 与基线日志。本次不扩大修复这些既有问题的范围。

## Docker 更新与历史保护

更新只使用本机镜像及已有依赖，构建网络关闭；不修改配置密钥，不提交或推送。

已更新运行中的两项服务，Nginx 路由保持现状：

| 服务 | 更新后镜像 |
| --- | --- |
| Gateway | e6b317a45fcb3c2fd39586a931095882368009bcc2e1820da1b69be4b7b3c0d0 |
| Frontend | 0600deb2f9e8af0a1665aa15b5567a2cef7e145c1986f6757b870e3e15dde0ef |

Gateway `/health` 返回 healthy，公开网页 `http://127.0.0.1:2027` 返回 200；
运行表仍为 success=34、error=2，没有活动运行。容器 6 个修改源码文件与本地
SHA-256 一致（`.tmp/download-image-source-verification.json`）。

切换前确认无活动运行，停止两项服务后备份当前数据到
`D:\ChemAgent\docker-backups\download-fix-20261011-173341`。
**1312 个文件**逐项比对原路径/大小/SHA-256，备份一致；旧镜像与回退 compose
均保留。服务启动后的保全审计确认文件没有缺失，原配置哈希未改变，仅允许
SQLite `data/deerflow.db-shm` 的共享内存变化。没有恢复旧快照覆盖当前实验。

重新运行只读交付审计，原 4 次实验、309 次拟合、LLM STOP、保留实验、审批状态
与 6 个已发布工件哈希均未改变（`.tmp/delivery-download-audit-after.json`）。
旧 checkpoint 的附件状态不被回写，刷新后的卡片由原发布记录恢复。

## 限制

- 旧会话附件通道仍是其原始历史值；新前端通过只读发布记录显示卡片，不回写旧记录。
- 旧消息中必须有服务器确定性交付标志和可识别的实际 ZIP 链接；其他历史输出继续
  使用原附件/过程页面展示。
- 本次未修订最终回答的通用质量门禁措辞，也不重新评估模型质量。
- 本次不产生额外 LLM/API 费用，不自动审批或注册模型。
