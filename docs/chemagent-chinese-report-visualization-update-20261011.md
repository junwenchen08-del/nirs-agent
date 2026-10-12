# 中文报告与可视化更新（2026-10-11）

本轮仅改进交付展示与中文提示，不修改 LLM 选模权、训练算法、预算、数据分区、审批或注册规则。未调用新的 DeepSeek 请求，未重新训练或评价 Test。

## 实际修改

- `backend/packages/harness/deerflow/community/nir/_report.py`：中文模型、动作、参数策略、质量等级和资源统计；区分 Cal CV 折外预测 RMSE 与折均值 RMSECV；明确实际 CV 配置、各轮反馈与最终质量门禁的分区。
- `report_labels.py`：中文展示词典，保留参数名和原始机器字段，未知内容不猜测。
- `report_figures.py`：使用 Matplotlib 从已有 Cal 光谱缓存、成功实验 Val 指标及已完成 Test 点对生成四张 PNG。没有数据读取、模型加载、拟合或 Test 评价接口。
- `report_assets/`：随附开源 Noto Sans SC 中文字体及 OFL 许可证，Docker 离线生成图表不依赖宿主字体。
- `decision_delivery.py`：复用 `AutonomousView.spectra/chart` 校验过的持久化证据，将图片接入原报告与 `build_delivery`；图表也进入出版完整性清单。
- `delivery.py`：离线 HTML 标题中文化。原有图片内嵌与显式 ZIP 文件清单继续使用，未放宽路径或文件权限。
- `agents/lead_agent/prompt.py`：简洁科学观察、理由和预期跟随用户语言；结构化字段、证据 ID、算法代码和参数名不变。
- 前端 `core/nir/{autonomous,process-display,presentation}.ts` 与 `components/workspace/nir/{autonomous-process,cal-spectra,process-chart,process-facts}.tsx`：中文管线与指标分区、动作、反馈、参数、图例、下载名称；光谱样本以“样本 1”等显示，原身份保留在提示与交互绑定中。历史英文理由可展开查看，未改写持久记录。

## 四类图表

| 图表 | 数据来源 | 边界 |
|---|---|---|
| 原始／预处理后光谱 | 保留实验实际 Cal 拟合状态生成、哈希校验过的缓存 | 至多 12 个样本和 512 个原始轴点，保留轴顺序，不重新拟合 |
| 验证误差趋势 | 本会话持久化成功实验的 Val RMSE | 不包含未执行候选；LLM 保留标记不等同于程序最低误差选模 |
| 最终预测对照 | 冻结后唯一最终 Test 评价已保存的点对 | 不重读 Test；只在 finalized 状态取得 |
| 最终残差 | 同一已保存点对与实际残差 | 检查残差与预测－实测一致，不用于继续搜索 |

`report.md` 使用相对 PNG 引用，需和图文件一起保存；ZIP 包含全部图片与校验清单。HTML 内嵌图片，可离线查看。缺少可信 Cal 缓存时明确标注缺失，不重新生成实验事实。

## 已完成任务的展示修订

- 源会话：`search_02940626d20b4c098d6e84d6bca193de`。
- 源交付：`delivery_42d4ef56f4524d03839457fedd21600e`，原文件未覆盖。
- 中文图文版：`D:\ChemAgent\reports\report-zh-20261011-103803\report.md`；同目录有 `report.html`、`delivery.zip` 与四张图。
- 只从持久证据读取，生成脚本明确禁止 `numpy.load` 与 `joblib.load`；模型、指标、原始决策文件按字节复制，并验证源交付全部文件 SHA-256 未变。
- 五轮简洁理由提供单独的中文展示译述，来源与原提案校验保存在 `presentation-translations.json`；不回写 DecisionRecord，也不将 LLM 当时的机理假设或性能预期视为实际结论。
- 已发布交付重放继续返回原不可变文件，不自动覆盖历史报告。新交付使用新图文模板；已有任务的中文修订版单独提供。

## 验证

| 检查 | 结果 |
|---|---|
| 新增后端测试红灯验证 | 4 项原始缺陷失败，随后修复 |
| 报告、离线交付、自动终结、光谱缓存（候选生产镜像） | 30 项通过 |
| 新增中文前端测试红灯验证 | 3 项原始缺陷失败，随后修复 |
| 全量前端（Docker 实际依赖） | 70 个文件、591 项通过 |
| TypeScript（含实际 contracts） | 通过 |
| 修改文件 ESLint / Ruff / git diff --check | 通过 |
| 后端修改前离线全量基线 | 6524 通过、21 失败、20 跳过 |
| 后端修改后离线全量 | 6528 通过、同样的 21 失败、20 跳过，无新增失败 |
| 原有模型、HTML、ZIP 下载路由 | 字节与附件头一致；其他所有者 404、未登录 401 |
| 历史数据与配置 | 1312 个文件逐一校验，未发生缺失或内容改变，运行配置未变 |
| PNG 视觉检查 | 中文正常显示，四图均使用真实持久证据，单位缺失没有补填 nm |

离线全量仍排除了 9 个会自动调用 npm 安装的 HTML 抽取测试组：browserless_client、crawl4ai_tools、exa_tools、fastcrw_tools、firecrawl_tools、groundroute_tools、infoquest_client、jina_client、readability。21 项已有失败涉及无真实服务的 live 测试、既有 active-content、部署及文档／Git 快照断言，不可称为全量全绿。日志：`.tmp/chinese-report-full-bounded{,-baseline}.log`。

本机前端全量与类型检查遇到依赖／文件访问问题，改用运行容器的完整依赖验证；没有修改包版本或锁文件来掩盖问题。浏览器证实更新后的公共页面恢复，受保护过程页要求登录；本轮未登录账户或重跑付费任务，因此登录态过程页的视觉验收未完成。前端展示有结构化测试和服务端渲染覆盖。

## Docker 更新

- Gateway：`sha256:e5a8fc068792d58c7374cdfb4b3064f63d8a2083c4465a5a45b8eb361ec18c75`。
- Frontend：`sha256:aef2a8d502066dbc873ac548a99c055ede2ab6a556a8ac08dc2f8eee353a111b`。
- 备份与可回滚旧镜像记录：`D:\ChemAgent\docker-backups\chinese-report-20261011-182846`。
- Gateway 健康检查正常，公共 `127.0.0.1:2027` 页面恢复；代理容器内部端口为 2026。

新中文请求的结构化理由已加语言要求，但本轮未产生新的真实 provider 调用；不能据此宣称所有未来模型响应必定中文。历史英文证据仍保留，可追溯且不被修改。没有提交、推送、重置工作树或进入后续研究阶段。
