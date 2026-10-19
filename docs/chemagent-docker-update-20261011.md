# ChemAgent Docker 更新记录（2026-10-11）

本次已将现有 `nirs-agent-duan` 项目的 Gateway 和前端更新为当前工作区源码。正式入口仍为 <http://127.0.0.1:2027>，Nginx 保留原容器和端口。未 reset、checkout、提交或推送；未新增真实 provider 请求、真实建模任务或最终 Test 评价。

## 更新范围与构建结果

沿用 `docker/docker-compose.yaml`、`docker/.cache/compose-duan.yaml` 和原 `.env`；通过 `up -d --no-build --no-deps --force-recreate gateway frontend` 重建两个服务。容器于北京时间 2026-10-11 13:58 创建。

| 服务 | 原镜像 SHA-256 | 当前镜像 SHA-256 |
| --- | --- | --- |
| Gateway | bf8531f7d6740d1c123d78cd624f383a3933546ca7ed0728947c660afa7e26e7 | 8895cfe995633bcdfb07009e916358707dc5094b56c2049864c754b60a12b907 |
| 前端 | 0e0630a1ee0fa232e666b4e2a175a5d099dc2e2201e1115afe5486760616b444 | 96a3e116da1a54a241595a5725e1beae0154771edc7e252a30c4c2345ff54575 |

Gateway 按现有 Dockerfile 和当前锁文件构建成功。原 npm 镜像源和一次官方源构建均发生 `ECONNRESET`；两次失败日志保留。前端 `package.json`、`pnpm-lock.yaml` 与旧镜像的 SHA-256 完全相同，因此本次采用备份目录内的 `frontend-update.Dockerfile`，保留旧镜像的已安装 Linux 依赖，移除原应用目录及旧编译产物，复制当前源码后执行完整 `pnpm build`。生产编译、TypeScript 检查与静态页面生成均通过，没有跳过类型检查或改变依赖版本。

这是本次网络故障下的构建方式；仓库正式前端 Dockerfile 和运行配置未改动。未来使用正式 Dockerfile重新构建仍需可用的依赖下载网络。新旧数值依赖版本一致：NumPy 2.4.4、SciPy 1.18.0、scikit-learn 1.9.0、Chemotools 0.4.4、PyTorch 2.13.0+cpu、langchain-core 1.4.9。

本次仓库改动仅为 `.dockerignore` 排除 `.tmp/`、README 和 AGENTS.md 补充生产更新说明，以及本记录。所有原有未提交业务改动均保留并纳入镜像；本地验收临时状态未打包。

## 持久数据与配置保护

备份目录：`D:\ChemAgent\docker-backups\20261011-134012`。旧镜像保留标签：

- `nirs-agent-duan-gateway:rollback-20261011-134012`
- `nirs-agent-duan-frontend:rollback-20261011-134012`

在切换前两次只读确认活动 run 数为 0。构建期间旧服务继续运行，随后短暂停止 Gateway，对实际挂载的 `backend/.deer-flow` 备份并逐文件比较路径、大小和 SHA-256；再次确认复制期间源文件没有变化后才切换。

- 备份 1,276 个文件，共 931,196,424 bytes；无遗漏、无校验不匹配。
- 启动后 1,275 个原文件 SHA-256 不变。唯一变化为 `data/deerflow.db-shm`，属于 SQLite 启动产生的共享内存变化；没有历史模型、报告或实验文件被修改或删除。
- 主数据库所有表的记录数及 run 状态计数与备份相同：31 个 run（29 success、2 error）、16 条线程元信息、2 个模型版本、1 个数据集、4,010 条 checkpoint、5,887 条 write。
- `.env`、`config.yaml`、`extensions_config.json`、`frontend/.env` 和两份 Compose 文件的 SHA-256 均不变。
- 继续使用原数据、配置和 Skill 挂载，没有删除数据卷或覆盖用户密钥。

备份包含本地配置，保留在本机目录内，不纳入 Git 或公开报告附件。

## 本次验证

| 验证 | 结果 |
| --- | --- |
| 前端生产 build 与 TypeScript | 通过 |
| 新 Gateway 断网临时容器回归 | 73 passed，33.72 秒 |
| 后端应用/框架源码一致性 | 412 个 Python 文件，无差异 |
| 数值内核源码一致性 | 78 个 Python 文件，无差异 |
| 正式前端源码一致性 | 444 个 src 文件，无差异；已删除的模式控件没有残留 |
| 正式运行配置 | 6 个安全工作流/决策工具均已注册且可导入 |
| 统一策略 | v2，累计 Token cap 为 null，automatic_delivery=true |
| `/health` | 200，healthy |
| `/openapi.json` | 200，包含 7 个自主过程路由 |
| 首页、聊天和过程页面 HTTP | 200 |
| 未登录 `/api/models` 与自主过程查询 | 401，认证边界正常 |
| 历史数据、配置与数据库计数 | 通过上述校验 |
| `git diff --check` | 通过 |

离线测试采用 `--network none`，未挂载正式业务数据或真实密钥；覆盖统一策略、真实 Lead 的 Fake LLM 工具循环、STOP/RETAIN 自动交付、网页入口、Test 封存、Cal 光谱缓存及授权参数执行。它们是自动化/Fake 测试，不是新增 DeepSeek 验收。此前全量回归记录中的遗留失败未在本次更新中修复，也没有以这 73 项测试替代全量通过声明。

首次运行配置核对脚本误用了不带 `_tool` 的函数名和错误的 context 工具名，导致审计断言失败；按真实 ToolConfig 名称及注册函数修正后通过，未修改运行配置或业务代码。启动和构建中的既有 Nextra、第三方弃用及语法警告保留在日志中。

本次没有在正式 Docker 上另行提交付费任务或消费 Test，也没有单独进行认证后的真实交付下载验收；相关能力由离线生命周期测试和此前本地真实验收提供证据。

## 证据与回滚

备份目录保存配置备份、完整数据副本、更新状态、逐文件清单、构建日志、离线测试日志、HTTP 检查及源码/数据/运行策略审计 JSON。当前网页可直接使用；浏览器如仍显示旧内容，可刷新页面。

需要回滚代码时，使用保留的原项目与配置，再叠加备份目录中的 `rollback-compose.yaml`：

```powershell
docker compose --project-name nirs-agent-duan --env-file D:\ChemAgent\nirs-agent\.env -f D:\ChemAgent\nirs-agent\docker\docker-compose.yaml -f D:\ChemAgent\nirs-agent\docker\.cache\compose-duan.yaml -f D:\ChemAgent\docker-backups\20261011-134012\rollback-compose.yaml up -d --no-build --no-deps --force-recreate gateway frontend
```

该命令只回滚应用镜像，继续保留现有数据挂载；不要把更新后产生的新记录自动覆盖为旧备份。完整数据副本供需要时另行受控恢复。本轮更新完成，不追加真实 API 调用。
