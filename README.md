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

单目标回归与 MAT 多子集分析现默认交付**完整图文报告（HTML）和统一下载包（ZIP）**。
报告分别说明校准、调参和最终留出/外部验证指标，包含实际选择依据、诊断图与适用范围。
HTML 可离线打开或通过浏览器打印为 PDF；ZIP 保留模型、安全清单、完整指标、图表与 SHA-256 文件清单，不包含原始上传数据。
桌面网页将已有结果的模型、指标和辅助图片默认折叠。旧文件不会自动重写或删除。
实现范围与验证见 [交付物改进说明](docs/nir-delivery-improvements.md)。

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
| 跨会话复用（可选） | 上传数据集自动入库；将已审批、已注册模型提升到用户模型库并安全挂载到新会话 |
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

`config.example.yaml` 中的 `nir_library.enabled` 默认关闭。可选能力包括受所有权保护的
数据集和模型库：启用后默认自动保存网页成功上传的 CSV/TXT/MAT、维护版本化解析
Profile，并把已通过科学门禁、用户审批和 `nir_register_model` 注册的模型完整提升到长期
目录。新会话挂载时会再次校验模型、metrics、训练数据绑定和可选 HMAC，返回现有
`nir_predict` 可直接使用的路径；原验证范围会原样保留，内部留出验证不会被标成外部验证
或生产批准。`nir_inspect` 不会自动保存数据，模型也不会仅因注册成功而自动长期保存。
数据集保存由上传接口完成，不依赖智能体或建模成功；相同内容复用同一用户库中的记录。
上传后显示入库结果，入库失败会明确提示且保留当前上传。`nir_library.auto_save_uploads=false`
可关闭自动入库；手动保存仍要求明确意图。新会话始终重新审查数据，不继承旧建模结论。
启用后，侧栏会显示“数据集库”和“模型库”；在新对话输入 `/datasets` 可搜索已保存的
数据集并发起挂载。聊天选择器只挂载文件并显示已选数据集；用户输入要求并手动发送后才启动智能体。
输入 `/models` 可搜索已保存的模型并选择精确版本；挂载后上传新数据或通过 `/datasets` 选择数据，手动发送预测要求即可复用该模型。选择模型不会自动发送消息。
用户可查看用量、Profile、使用历史和模型血缘，向新会话挂载资产，
并执行受控归档/删除。物理删除要求先归档、再输入精确资产编号；
仍被模型引用的数据集会拒绝删除。删除长期库版本不会删除已经挂载到旧会话的副本，也不
替代部署者的备份保留策略。普通上传、建模、线程注册和预测不受该开关影响。模板版本为
21，已有 `config.yaml` 可运行 `make config-upgrade` 合并新工具条目，不要直接覆盖本地密钥
配置。

### 3. 配置模型

在 `config.yaml` 的 `models:` 中启用一个可用模型，并在 `.env` 中填写该供应商的 API
Key。密钥应通过环境变量引用，不要直接写入配置模板、README 或提交记录。

知识库和外部 MCP 均为可选功能，基础 NIR 分析不要求启用。模板会默认启用项目内置的
Chemotools MCP；它随 Gateway 镜像运行，不需要单独部署服务。
`nir_*` 工具属于独立的 `nir` 工具组；即使 `sandbox.allow_host_bash: false`，
`nir_workflow`、数据检查和建模工具也应可用。已有配置若将它们归入 `bash`，
请将这些工具的 `group` 改为 `nir`，并在 `tool_groups` 中添加 `nir` 后重启 Gateway。

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

第一次使用建议先阅读[《NIR-Agent 用户使用说明》](./docs/nir-user-guide.md)，其中包含从
Docker 启动、数据检查、建模审批到跨会话预测的完整操作示例和可直接复制的对话模板。

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

当当前对话存在 NIR 工作流时，聊天页右上角会显示 `NIR` 状态入口。桌面端的
**ChemAgent 建模过程解释面板**包含“建模过程”“选择依据”“最终结果”三个页签：
可查看数据审查、历次尝试与指标趋势、反思和重试计划，以及预处理、波长和模型候选的
训练内比较与记录中的选择原因。“选择依据”先显示本轮方案、通俗选择原因和候选误差；
原始候选 ID、原因代码与规则可按需展开查看。候选误差仅用于训练内选方案，最终表现以
“最终结果”为准。最终结果明确区分质量门禁、内部留出或外部验证、审批、注册和长期
Model Library 入库；缺少结构化证据时显示未记录，不推断成功。有会话产物时
可从面板直接打开图表与报告列表。普通聊天不会显示该入口；面板只展示当前线程的有界
结构化工作流证据，不展示隐藏推理、原始光谱、完整证据 JSON 或宿主机文件路径。正在运行
的任务优先使用实时工作流状态；旧会话
若未从实时流恢复该字段，会通过带所有权校验的线程状态接口读取最近检查点。

### 跨会话资产库（可选）

在 `config.yaml` 中把 `nir_library.enabled` 设为 `true` 并重启服务后，侧栏会出现：

- `/workspace/nir/datasets`：保存、重命名、查看解析 Profile 与使用历史、挂载、归档和删除数据集；
- `/workspace/nir/models`：查看模型版本、验证范围、指标和来源，并执行挂载、归档和删除。

页面只调用当前登录用户范围内的 API，不把原始光谱、模型二进制或宿主机路径放进浏览器
缓存。空间卡片中的“库写入准入门槛”只决定是否接受新的长期资产，不是对普通上传、线程
临时副本或沙箱输出的严格总配额。归档只隐藏资产，不释放磁盘；完成精确确认的物理删除才
回收长期库目录。验证范围说明模型如何被评估，并不等于模型已获生产部署批准。

对话中要求保存当前上传时，智能体会在数据检查后、离开数据审计阶段前完成保存。跨会话
预测会先挂载数据集和模型，再在同一个新工作流中重新检查并标准化输入，最后调用预测；不会
继承旧会话的检查结论、漂移连续计数或预测审计。预测成功后，最终总结可以引用本工作流
已经绑定并验证的精确模型路径；其他模型路径仍会被回答门禁拒绝。开启模型库后，经过用户审批并成功注册的模型
会自动保存到模型库，注册返回值明确给出保存结果。`nir_library.auto_save_registered_models=false` 可关闭自动保存；
明确要求不保存时会跳过。关闭自动保存或补保存历史模型时，仍可明确要求“保存模型到模型库供以后使用”。
已完成的审批注册工作流也可补保存，服务端仍会核验原注册版本、模型、指标和训练数据绑定。

Docker 部署时，长期资产位于实际 `DEER_FLOW_HOME` 卷的 `users/<用户>/nir-datasets` 和
`users/<用户>/nir-models`。SQLite 部署需要把解析后的 `database.sqlite_dir` 中数据库与
`users/` 目录作为同一恢复点备份；PostgreSQL 部署需要让数据库快照与资产卷保持同一时间
点。建议先停止或冻结库写入，再备份数据库和资产卷，并定期用一份合成数据及模型做恢复
演练。不要把真实数据卷、数据库或密钥提交到 Git。

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
  → 模型注册并自动保存到跨会话模型库（启用时）
  → 新数据预测、漂移监控和防篡改审计
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

预处理选择采用 MCP-first 协议：智能体先查询 Chemotools MCP 能力目录和候选详情，再用项目
运行时注册表检查方法 ID、参数边界、执行顺序和工件回放兼容性。Gateway 会在
分析计划确认以及 `nir_preprocess` 或建模工具执行前，检查本工作流是否已经尝试过 MCP 目录
查询；目录不可用时会保留失败记录并允许受治理的原生回退。自动候选只使用校准分区，始终比较原始光谱基线，
最终测试集不会参与候选生成或选择。

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

## 建模过程探索

聊天中的 NIR 面板提供“查看完整建模过程”，进入独立页面查看实际训练证据：
数据分布、划分、同一样本预处理前后光谱、波长区间、候选参数、预测与残差。
支持联动缩放、选点、候选分页及尝试切换；候选名称展示实际方法与处理顺序。刷新保留当前步骤。图中说明显示样本数和变量数，
全量指标单独展示。只有数据、划分和评估口径一致的调参分数进入趋势；留出集不参与排名。

新增证据从本版本之后的单目标回归训练开始记录；旧会话继续展示已有摘要，并提示当时未记录。
图表文件损坏时独立降级，不影响已完成的模型结果。权限沿用会话归属检查。
后端 `NIR_PROCESS_ENABLED=false` 可关闭记录和读取；前端构建时
`NEXT_PUBLIC_NIR_PROCESS_ENABLED=false` 可隐藏入口。配置默认开启。
实现、证据来源和验证范围见 [可视化实现说明](docs/nir-process-visualization-implementation.md)。

## Chemotools MCP

项目将固定版本 `chemotools==0.4.4` 的公共工具集封装为线程隔离的 stdio MCP 组件。
`extensions_config.example.json` 已默认启用，复制配置后随 Docker Gateway 一起启动。智能体会先
调用能力目录，再读取所选方法的真实构造参数和允许操作，不依赖模型记忆猜测参数。
目录查询必须先于预处理或建模执行并单独完成，不能与执行工具并行发起。

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

建模过程网页为中文用户显示实际候选方法与参数、交叉验证及调参指标中文名称，
数值保留适当精度。“查看中”和“建模采用”分别标记当前查看与最终选择。
候选内部编号由有序步骤与参数生成，仅用于追溯，不代表排名或质量；原始编号、
全精度数值及未知字段收在可展开的技术记录中。缺少配置的历史候选不会推测方法。

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

## 算法知识引导的首轮预处理

方法知识库内置 94 张已整理卡片：保留全部 21 种已接入预处理和 4 种回归参考，
覆盖 Chemotools 官网导航的 70 个文档条目及当前 0.4.4 MCP 的全部 84 项公开能力。
已有卡片复用对应算法，辅助函数、常量和示例数据单独标明。每张卡包含来源、用途、
适用条件、局限与参数说明；官网参数与当前 MCP 参数分别展示。
数据划分后，使用校准数据诊断检索方法用途、参数约束与来源，优先评估相关候选；
原始光谱保留为同批对照，最终由校准内部交叉验证选择。检索无依据或不可用时记录规则回退。
此方法知识库使用本地结构化全文检索，无需下载嵌入模型；论文语义检索仍是独立的可选功能。

`nir_analyze` 默认启用；CSV 自动划分/固定分区训练默认 `pipeline_steps="auto"`。
NPZ 的 `nir_train_model` 使用原始数据时需显式传入 `pipeline_steps="auto"`；其原有
`None` 语义保留为已预处理输入。显式 JSON 流水线保持指定方案，`[]` 表示使用原始光谱。
网页建模过程和报告记录数据诊断、方法来源及实际比较结果。
在“设置 → 知识库 → 方法知识库”可编辑用途、来源、诊断标签和候选参数组合，
测试检索、保存草稿、启用/停用，以及恢复内置内容。管理员维护共享方法卡；
修改持久保存于应用数据目录，新的建模自动读取，无需启动论文 RAG 或重建镜像。
新增卡片只能关联已经接入的算法，参数仍由运行时注册表校验。
网页可按方法类型、算法分类及名称筛选。MCP 扩展卡片供规划检索和显式调用，其参数
定义只读，执行前由 MCP describe/validate 检查；未暴露的官网函数标为仅文档。
模型卡片用于规划检索，其参考参数编辑不修改现有自动
模型搜索网格。项目自有稳健 SNV 和去尖峰卡片明确区分 SciPy 原语参考与项目实现。
升级时补入新内置卡片与缺失字段，保留已有网页编辑和停用状态。
详见 [方法知识检索实现说明](docs/nir-method-knowledge.md)。

## 论文知识库（可选）

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
