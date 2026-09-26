# NIR-Agent — 近红外光谱智能体

> **版本**：v2.1.0　|　**基础框架**：DeerFlow 2.x（LangGraph + FastAPI）

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](./backend/pyproject.toml)
[![nir-core](https://img.shields.io/badge/nir--core-0.1.0-blue)](./nir_core/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

NIR-Agent 是基于 [DeerFlow](https://github.com/bytedance/deer-flow) 构建的近红外光谱
专用智能体。用户可以通过自然语言完成数据检查、预处理、定量或定性建模、质量评估、
模型注册和新样本预测。

算法计算由独立的确定性 Python 包 `nir_core` 完成，LLM 负责理解需求、选择工具、组织流程
和解释结果。原始光谱矩阵不会直接进入 LLM 上下文。

> 当前部署定位是可信用户使用的本地或单机 Docker 环境。不要直接把默认的 `2026`
> 端口暴露到公网。

## 核心能力

| 能力 | 说明 |
|---|---|
| 数据接入 | MAT v5/v7.3、CSV、TXT；识别样本方向、波长轴、目标列和嵌套矩阵 |
| 数据检查 | 对不唯一的矩阵、方向或字段映射要求人工确认，不静默猜测 |
| 定量分析 | 单目标、多目标、自动划分或具名 Cal/Tuning/Test 分区建模 |
| 定性分析 | PLS-DA、逻辑回归和校准 SVM，支持字符串类别与分组隔离 |
| 预处理 | 21 种目录化方法，Chemotools 优先、原生补充，支持训练边界拟合和预测严格回放 |
| Chemotools MCP | 将固定版本公共工具集作为 MCP 组件提供给智能体，覆盖目录查询、受控执行、诊断和绘图 |
| 校准迁移 | 对同类仪器配对样本执行 DS/PDS/SST，绑定方向、波长轴、仪器和验证证据 |
| 模型选择 | PLS 基线以及受控的 Ridge、SVR、Extra Trees 等候选比较 |
| 波长选择 | 全光谱基线与 CARS 等方案仅使用校准/调优数据比较 |
| 质量控制 | 三集分离、泄漏检查、质量门禁、审批注册和可追溯模型产物 |
| 预测监控 | 产物哈希校验、适用域漂移、连续告警和防篡改预测审计 |
| 可选知识库 | 对已审核文献进行检索，为失败诊断和重试提供证据 |

## Docker 启动

先安装并启动 Docker Desktop，同时确保终端可以使用 Git 和 `make`。Windows 建议使用
Git Bash。Docker 镜像已经包含项目所需的 Python 3.12 和 CPU 版 PyTorch，用户不需要
准备本机 Python 或 Conda 环境。

### 1. 下载项目

```bash
git clone https://github.com/junwenchen08-del/nirs-agent.git
cd nirs-agent
```

### 2. 创建配置文件

```bash
cp config.example.yaml config.yaml
cp .env.example .env
cp extensions_config.example.json extensions_config.json
cp frontend/.env.example frontend/.env
```

这些本地配置均已被 Git 忽略。重复部署时不要用模板覆盖已经配置好的文件。

### 3. 配置模型

在 `config.yaml` 的 `models:` 中启用一个可用模型，并在 `.env` 中填写该供应商的 API
Key。密钥应通过环境变量引用，不要直接写入配置模板、README 或提交记录。

知识库和外部 MCP 均为可选功能，基础 NIR 分析不要求启用。模板会默认启用项目内置的
Chemotools MCP；它随 Gateway 镜像运行，不需要单独部署服务。

### 4. 启动项目

```bash
make docker-start
```

首次启动需要下载并构建镜像。完成后打开
[http://localhost:2026](http://localhost:2026)，首次使用时按页面提示创建管理员账号。

可用下面的命令确认服务状态：

```bash
docker ps --filter "name=deer-flow"
curl http://localhost:2026/health
```

### 5. 查看日志或停止

```bash
make docker-logs
make docker-stop
```

## 基本使用

在网页中新建对话，通过附件按钮上传自己的 MAT、CSV 或 TXT 光谱文件。建议先检查数据
结构，再开始建模。

```text
先检查我刚上传的近红外文件，只报告样本方向、波长轴、目标变量候选和需要确认的映射，暂时不要建模。
```

确认结构后，可以直接提出分析目标：

```text
分析这批光谱并建立蛋白质含量定量模型，使用独立留出集评估。
```

```text
根据类别标签建立定性鉴别模型，并报告 balanced accuracy、macro-F1、MCC 和混淆矩阵。
```

```text
使用已经批准的模型预测新样本，同时检查适用域漂移。
```

智能体会记录当前任务类型、数据确认、建模尝试和审批状态。模型未经用户明确批准不能进入
正式注册阶段；显式选择的模型失败后，也不会在未获得同意时擅自替换模型家族。

## 工作流程

```text
用户需求
  → 数据结构检查与人工确认
  → 科学有效性和泄漏检查
  → 校准/调优/测试划分
  → 预处理、波长与模型选择
  → 独立测试和质量门禁
  → 不达标时反思并生成受约束的重试计划
  → 用户审批
  → 模型注册、预测和漂移监控
```

模型选择和超参数调整只能使用校准集与调优集；最终测试集不会参与候选生成或选择。
具名外部数据只有在来源和分区语义满足要求时才能报告为外部验证，否则必须表述为独立留出。

## 数据接入

- MAT v5 和 v7.3 会递归检查数值矩阵、波长轴及目标变量候选。
- CSV/TXT 会检测编码、分隔符、小数点格式、数值波长表头和字段角色。
- 样本在行和样本在列的常见光谱表均可识别。
- 多个候选得分相同或字段角色不明确时返回 `needs_user_mapping`。
- 数据加载、训练和预测均受文件大小、矩阵规模、内存和运行时间上限保护。

仓库不附带真实业务数据。用户应上传自己拥有合法使用权的数据。

## 预处理方法

`nir_core` 通过统一流水线接口提供以下 21 种预处理方法。新建流水线默认优先调用固定版本
`chemotools==0.4.4` 中经过数值和边界验证的实现；Chemotools 没有等价能力、参数语义不一致
或需要兼容旧模型时，使用项目原生实现：

| 方法 | 作用 | 常用参数 |
|---|---|---|
| `snv` | 标准正态变量变换 | — |
| `robust_snv` | 基于中位数和 MAD 的稳健 SNV | `consistency` |
| `rnv` | Chemotools 稳健正态变量 | `percentile`、`epsilon` |
| `msc` | 多元散射校正 | —（参考谱由训练流水线拟合） |
| `emsc` | 扩展多元散射校正 | `polynomial_order`、`min_multiplicative` |
| `despike` | 检测并替换孤立尖峰 | `window`、`z_threshold` |
| `sg_smooth` | Savitzky-Golay 平滑 | `window`、`order` |
| `whittaker_smooth` | Whittaker 惩罚平滑 | `lambda_` |
| `median_filter` | 中值滤波 | `window_length`、`mode` |
| `derivative1` | 一阶 SG 导数 | `window`、`order`、`delta`、`spacing_mode` |
| `derivative2` | 二阶 SG 导数 | `window`、`order`、`delta`、`spacing_mode` |
| `norris_derivative1` | Norris-Williams 一阶导数 | `gap`、`segment`、`delta` |
| `norris_derivative2` | Norris-Williams 二阶导数 | `gap`、`segment`、`delta` |
| `airpls` | airPLS 基线校正 | `lambda_`、`max_iters` |
| `arpls` | ArPLS 基线校正 | `lambda_`、`ratio`、`max_iters` |
| `asls` | 非对称最小二乘基线校正 | `lambda_`、`p` |
| `rubberband` | 橡皮筋基线校正 | — |
| `detrend` | 多项式去趋势 | — |
| `mean_center` | 均值中心化 | — |
| `autoscale` | 均值中心化和单位方差缩放 | — |
| `normalize` | 行级 L1、L2 或 max 归一化 | `norm` |

典型顺序是：去尖峰 → 基线校正 → 散射校正 → 平滑/导数 → 缩放。稳健 SNV、EMSC、孤立尖峰
去除和 Norris-Williams 导数属于显式选择能力，不会自动加入默认候选流水线。EMSC 参考谱
只在训练数据上拟合，并随预处理状态保存以供预测回放。

运行时算法目录是方法、参数范围和实现版本的唯一依据。智能体可先查询方法清单，再按需
查询单个方法详情或生成有界候选。自动候选只使用校准分区，始终比较原始光谱基线，最终
测试集不会参与候选生成或选择。

新建流水线对 Chemotools 中已完成数值和边界验证的对应方法优先使用固定版本
`chemotools==0.4.4`；项目原生实现只保留给 Chemotools 没有等价能力的 `robust_snv`、
孤立尖峰 `despike`，以及 `normalize(norm="max")` 等语义不重叠分支。模型工件会保存每一步
的实现来源和版本，旧工件没有来源记录时固定按原生实现回放，不会因升级后端而改变预测。
需要临时回滚有原生等价实现的新训练时，可在 `.env` 设置：

```dotenv
NIR_PREPROCESSING_PROVIDER_POLICY=native
```

Docker 镜像会安装并锁定该依赖，使用者不需要本机 Conda 或 `torch1` 环境。

波长轴对齐由独立的 `nir_align_wavelengths` 完成。它会同时更新光谱矩阵和波长轴，要求
输入轴严格单调，并默认禁止外推。它只做轴归一化，不等于校准迁移。

## Chemotools MCP

项目将固定版本 `chemotools==0.4.4` 的公共工具集封装为线程隔离的 stdio MCP 组件。
`extensions_config.example.json` 已默认启用，复制配置后随 Docker Gateway 一起启动。智能体会先
调用能力目录，再读取所选方法的真实构造参数和允许操作，不依赖模型记忆猜测参数。

MCP 目录覆盖 Chemotools 0.4.4 的全部公开预处理类（30 个），并包含校准适配、数据增强、
特征选择、PLS 回归、异常值诊断、物理强度转换、绘图和 Inspector。对外提供的是稳定的
`list_capabilities`、`describe_capability`、`validate_operation`、`fit_estimator`、
`apply_estimator`、`call_function`、`render_plot` 和 `run_inspector` 等受控接口，而不是允许
任意模块名或 Python 代码执行。

MCP 训练对象保存在当前对话工作区，并在再次加载前校验 SHA-256；NumPy 输入禁止 pickle，
读写路径不能越出线程工作区。完整可调用目录不等于全部进入自动候选：普通生产建模仍由
`nir_train_*`/`nir_analyze` 在训练折内拟合预处理；校准迁移、增强、正交投影、特征选择和
异常值诊断默认只允许显式选择。因此新增 MCP 不会绕过现有三集分离、质量门禁和审批注册。

## 跨仪器校准迁移

DS、PDS 和 SST 通过独立工具提供，不会进入普通预处理候选。使用前必须准备同类仪器对
同一批实体样本的配对光谱；默认依据两份 NPZ 中唯一且集合一致的 `sample_names` 配对，
并明确参考仪器、目标仪器和“目标 → 参考”的迁移方向。不同波长轴只允许在共同覆盖范围
内对齐，禁止外推。NIR 与 Raman 等不同光谱模态不能按此协议互相迁移。

智能体先查询 `nir_list_calibration_transfer_methods`，再使用
`nir_fit_calibration_transfer`、`nir_apply_calibration_transfer` 和
`nir_evaluate_calibration_transfer`。工件记录 Chemotools 版本、方法参数、仪器 ID、轴哈希、
配对数据哈希和验证指标，并像模型工件一样在加载前校验 SHA-256（可选强制 HMAC）。只有
光谱误差改善时标记为 `spectrally_validated`。内部配对留出即使通过参考模型 RMSEP 验证，
也只标记为 `model_validated_internal`；只有另行提供不与训练样本重叠的
`validation_source_path` / `validation_target_path`，并同时绑定可信参考模型且迁移后 RMSEP
改善时，才标记为 `production_validated` / `approved`。

## 建模与质量控制

主要训练路径支持：

- 无官方分区数据的自动校准/调优/留出划分；
- 按批次、年份、产地、仪器等字段进行组间隔离；
- 使用既有 Cal/Tuning/Test 分区的固定验证；
- 单目标、多目标回归以及监督分类；
- 训练边界拟合预处理和波长选择；
- PLS 基线与有条件启用的替代模型比较；
- 对重复光谱、冲突参考值和跨分区泄漏进行阻断。

默认质量门禁按应用领域设置：

| 领域 | 最低 R² | 最低 RPD |
|---|---:|---:|
| 水分 | 0.90 | 4.0 |
| 蛋白质 | 0.85 | 3.5 |
| 制药 | 0.85 | 3.0 |
| 饲料 | 0.75 | 2.5 |
| 土壤 | 0.70 | 2.0 |
| 默认 | 0.80 | 3.0 |

显著 bias 即使在 R² 和 RPD 达标时也会阻止通过。内部独立留出不能描述成外部验证。

## 模型产物与预测安全

- 模型、指标和训练数据通过 SHA-256 清单绑定，注册时再次核验。
- 预测只接受指定输出目录中由 NIR 流程生成且清单验证通过的产物。
- 部署可通过 `NIR_ARTIFACT_SIGNING_KEY` 和 `NIR_REQUIRE_SIGNED_ARTIFACTS=1` 强制签名。
- 预测结果包含训练域适用性检查，并可连续累计漂移状态。
- 预测审计只记录必要摘要，不保存原始光谱和逐样本预测值。
- 审计事件使用哈希链和同步落盘；链损坏或无法持久化时失败关闭。

## 项目结构

```text
nirs-agent/
├── backend/            # FastAPI Gateway、智能体运行时和 NIR 工具桥接
├── frontend/           # Next.js 网页界面
├── nir_core/           # 确定性 NIR 算法包和测试
├── skills/             # 智能体技能
├── docker/             # Docker Compose 和 Nginx 配置
├── config.example.yaml
├── extensions_config.example.json
└── README.md
```

## 测试

运行完整的 `nir_core` 测试：

```bash
cd nir_core
pytest
```

仅运行 Chemotools MCP 契约、执行和安全测试：

```bash
cd nir_core
pytest tests/test_mcp -q
```

运行后端 NIR 回归：

```bash
cd backend
make test-nir
```

CI 会分别执行生产代码 Ruff 检查、快速测试和慢速计算回归。正式测试保留在仓库中，并优先
使用程序生成的合成数据。

## 知识库（可选）

知识库默认不启用，不影响数据检查、预处理、训练和预测。需要启用时，在 `.env` 中设置
`UV_EXTRAS=knowledge`，并按 [.env.example](./.env.example) 配置对应的
`NIR_KNOWLEDGE_*` 参数。

知识文档默认以 `draft` 状态导入，只有补全必要来源信息并审核发布后才会被智能体检索。
更换嵌入模型或向量维度时，应使用新的索引路径和索引版本。

## 公开仓库边界

公开仓库不包含真实 MAT 数据及压缩包、本地参考项目、个人资料、内部修改方案、实施日志、
临时测试输出、上传文件、模型产物、数据库、运行日志和知识库索引。正式源码、配置模板和
自动化测试保留；数据来源、再分发授权和隐私合规由数据提供者负责。

## 致谢

- [DeerFlow](https://github.com/bytedance/deer-flow)：智能体运行时、编排、沙箱和技能系统。
- [LangChain](https://github.com/langchain-ai/langchain) 与
  [LangGraph](https://github.com/langchain-ai/langgraph)：底层智能体框架。
- [Chemotools](https://chemotools.org/)：化学计量学预处理、建模、诊断和适配工具集。
- NIR 算法实现使用或参考了 scikit-learn、SciPy 等开源科学计算项目。

## 许可证

本项目基于 [MIT License](./LICENSE) 开源；DeerFlow 框架同样采用 MIT License。
