# ChemAgent

ChemAgent 是面向近红外光谱分析的智能工作台，连接数据管理、预处理、自主建模、过程诊断、报告与模型管理。

## 分析工作流

1. 上传光谱数据，在数据集工作台核对字段映射、波长轴、样本角色和分析区间。
2. 保存解析版本，将数据集挂载到新会话，说明目标变量和分析约束后发送任务。
3. 在建模过程中查看实际执行的轮次、参数、范围明确的指标、预测与残差诊断。
4. 查看报告和最终保留模型，按人工审批流程保存到模型库。
5. 挂载已保存模型，对兼容的未知样品数据进行预测。

ChemAgent 的登录、初始化、侧栏、设置、通知、关于页面和中英文使用文档使用统一品牌。浏览器页签使用分子轮廓与光谱线图标；分享链接指向当前实例。

产品面向中文用户：首次访问默认中文，登录与初始化提示、首页说明、设置、文档导航和日期均提供中文体验。浏览器的英文语言设置不会覆盖产品默认；在外观设置中主动选择的语言会被保存。

## 本地开发

运行环境：Python 3.12+、Node.js 22+、pnpm 10.26.2+、uv。根据 `config.example.yaml` 准备本地配置，密钥与运行数据只保留本地。

```bash
make install
make dev
```

后端位于 `backend/`，算法包位于 `nir_core/`，前端位于 `frontend/`。模块结构和维护规范见各模块的 `AGENTS.md`。

```bash
cd frontend
pnpm check
pnpm test
pnpm build
```

```bash
cd backend
make test
make lint
```

## 使用说明

启动应用后访问 `/zh/docs` 或 `/en/docs`。文档涵盖数据与评价范围、工作台操作、会话记录，以及从数据到模型的完整流程。更多维护说明见 `docs/`；品牌约定见 `docs/chemagent-branding.md`。

## 部署与数据

Docker 配置位于 `docker/`。更新已有部署时，保留现有配置、数据挂载和 Compose 项目，并从审计后的新源码快照构建镜像。兼容性模块、环境变量和持久化标识沿用现有契约，品牌修改不会迁移历史数据。

实验脚本、原始数据、实验输出、日志和本地运行时副本不属于产品发布内容。仓库根目录 `experiments/` 与仓库外的本地备份保持本地使用。

## 项目

[ChemAgent 源码](https://github.com/junwenchen08-del/nirs-agent) · [问题反馈](https://github.com/junwenchen08-del/nirs-agent/issues)

项目许可与第三方版权声明见 [LICENSE](LICENSE)。
