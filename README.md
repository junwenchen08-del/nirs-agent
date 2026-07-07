# NIR-Agent — 近红外光谱专有智能体

> **版本**: v2.0 | **基于**: DeerFlow v2.0 (LangGraph + FastAPI)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](./backend/pyproject.toml)
[![nir-core](https://img.shields.io/badge/nir--core-0.1.0-blue)](./nir_core/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

NIR-Agent 是一个基于 [DeerFlow v2.0](https://github.com/bytedance/deer-flow) 框架构建的**近红外光谱（NIR）专用智能体**，实现了完整的化学计量学分析工作流：数据加载 → 预处理优化 → 建模评估 → 反思闭环 → 报告生成。

---

## 设计哲学

```
┌──────────────────────────────────────────────────────────────┐
│  确定性核心 + LLM编排壳                                         │
│                                                              │
│  算法 → 确定性 Python 函数（零 LLM 参与）                         │
│  LLM → 理解意图、选择策略、解释结果、决定重试                        │
│                                                              │
│  光谱数据矩阵永不进入 LLM 上下文                                  │
│  LLM 只看到摘要、指标、路径、决策日志                                │
└──────────────────────────────────────────────────────────────┘
```

核心原则：
- **确定性兜底**：所有数学运算在 `nir_core` 中实现，LLM 无法跳过或错误执行
- **零框架侵入**：NIR 配置由 `nir_core/config.py` 自管理，不修改 DeerFlow 的 `AppConfig`
- **强制三集分离**：训练集、验证集、测试集在加载后立即分离，验证集绝不参与任何参数选择
- **反思闭环**：当模型质量不达标时自动尝试不同预处理组合，最多 3 次回溯

---

## 总体架构

```
用户输入: "分析这批玉米光谱，建立蛋白质含量模型"
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│              DeerFlow Lead Agent                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  nir-coordinator Skill (编排指令 + 质量门禁)             │  │
│  │  - 步骤指引: 加载→预处理→建模→评估→报告                   │  │
│  │  - 质量门禁: 按领域分级的 R²/RPD 阈值                     │  │
│  │  - 反思闭环: 最多3次回溯, 每次更换预处理组合              │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│         ┌────────────────┼────────────────┐                  │
│         ▼                ▼                 ▼                  │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐             │
│  │ nir_load │   │ nir_preprocess│   │nir_train │             │
│  │ _data    │   │               │   │_model    │             │
│  │ @tool    │   │ @tool         │   │@tool     │             │
│  └────┬─────┘   └──────┬───────┘   └────┬─────┘             │
│       │                │                │                     │
│  ┌────┴────────────────┴────────────────┴──────────────────┐ │
│  │           nir_core (独立确定性Python包)                    │ │
│  │  io/  │  preprocess/  │  model/  │  utils/  │ plotting/  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         知识库 Skills (领域知识注入)                        │ │
│  │  nir-knowledge/                                          │ │
│  │  ├── preprocessing-guide.md                              │ │
│  │  ├── modeling-guide.md                                   │ │
│  │  ├── metrics-interpretation.md                           │ │
│  │  ├── troubleshooting.md                                  │ │
│  │  └── chemometrics-rules.md                               │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
输出: Markdown 分析报告 + 模型文件 + 可视化图表
```

---

## 项目目录结构

```
deer-flow/
├── config.yaml                          # DeerFlow 主配置（NIR 工具 + 子代理注册）
│
├── skills/custom/                       # NIR 自定义技能
│   ├── nir-coordinator/SKILL.md         # 主编排技能（/nir 命令激活）
│   ├── nir-io/SKILL.md                  # 数据加载技能
│   ├── nir-preprocess/SKILL.md          # 预处理技能
│   ├── nir-model/SKILL.md               # 建模技能
│   ├── nir-predict/SKILL.md             # 预测技能
│   ├── nir-gallery/SKILL.md             # 可视化技能
│   └── nir-knowledge/                   # 知识库技能
│       ├── SKILL.md
│       └── docs/
│           ├── preprocessing-guide.md
│           ├── modeling-guide.md
│           ├── metrics-interpretation.md
│           ├── troubleshooting.md
│           └── chemometrics-rules.md
│
├── backend/packages/harness/deerflow/community/nir/   # DeerFlow 工具桥接层
│   ├── __init__.py
│   └── tools.py                         # 8 个 @tool 函数
│
└── nir_core/                            # 独立确定性算法包
    ├── pyproject.toml
    ├── __init__.py
    ├── models.py                         # Pydantic 数据模型
    ├── config.py                         # NIR 配置（领域分级阈值）
    ├── io/                               # 数据I/O
    │   ├── loaders.py                    # load_mat, load_csv, auto_detect_and_load
    │   ├── sniffers.py                   # 格式/结构嗅探
    │   ├── validators.py                 # 数据校验
    │   └── writers.py                    # save_npz, save_csv
    ├── preprocess/                       # 预处理算法
    │   ├── scatter.py                    # SNV, MSC
    │   ├── smoothing.py                  # SG 平滑, 导数
    │   ├── baseline.py                   # airPLS, ASLS, 去趋势
    │   ├── scaling.py                    # 均值中心化, 自缩放, 归一化
    │   └── pipeline.py                   # 预处理流水线编排
    ├── model/                            # 建模算法
    │   ├── pls.py                        # PLS 回归
    │   ├── pcr.py                        # PCR 回归
    │   ├── svr.py                        # SVR + GridSearchCV
    │   ├── ensemble.py                   # 集成预测
    │   ├── evaluation.py                 # 三集分离, 嵌套CV, 指标计算
    │   └── selection.py                  # CARS, SPA 波长选择
    ├── utils/                            # 工具模块
    │   ├── metrics.py                    # RMSE, R², RPD, evaluate_quality
    │   ├── validation.py                 # 异常检测, should_retry
    │   ├── drift.py                      # 马氏距离漂移检测
    │   └── registry.py                   # ModelRegistry 版本管理
    ├── plotting/                         # 可视化
    │   ├── spectra.py                    # 光谱图
    │   ├── model_diag.py                 # 预测vs参考, 残差图
    │   └── gallery.py                    # 对比画廊 HTML
    └── tests/                            # 测试套件
        ├── conftest.py
        ├── generators.py                 # 合成数据生成器
        ├── test_io/
        ├── test_preprocess/
        ├── test_model/
        ├── test_utils/
        ├── test_plotting/
        └── test_integration/
```

---

## 工具清单

通过 `backend/.../community/nir/tools.py` 注册的 8 个 `@tool` 函数：

| 工具名 | 功能 | 调用方式 |
|--------|------|---------|
| `nir_load_data` | 加载光谱文件（.mat/.csv/.txt），标准化为 .npz | `nir_load_data(file_path)` |
| `nir_inspect` | 预览文件结构（不完整加载） | `nir_inspect(file_path)` |
| `nir_preprocess` | 单步预处理（11种方法） | `nir_preprocess(input_path, method, output_path)` |
| `nir_train_model` | 建模（PLS/PCR/SVR），含三集分离+CV+质量门禁 | `nir_train_model(input_path, method, domain)` |
| `nir_predict` | 使用已训练模型预测新样本（可选漂移检测） | `nir_predict(model_path, data_path)` |
| `nir_analyze` | 一键端到端分析（不含反思闭环） | `nir_analyze(data_path, domain)` |
| `nir_reflect` | 确定性反思决策（should_retry + get_next_pipeline） | `nir_reflect(metrics, domain, attempt, history)` |
| `nir_compare` | 多预处理流水线并行对比 | `nir_compare(data_path, pipelines, method)` |

---

## 预处理方法

`nir_core` 提供 11 种确定性预处理方法：

| 方法 | 说明 | 参数 |
|------|------|------|
| `snv` | 标准正态变量变换 | — |
| `msc` | 多元散射校正 | reference (可选) |
| `sg_smooth` | Savitzky-Golay 平滑 | window, order |
| `derivative1` | 一阶 SG 导数 | window, order |
| `derivative2` | 二阶 SG 导数 | window, order |
| `airpls` | 自适应迭代重加权惩罚最小二乘基线校正 | lambda_, max_iters |
| `asls` | 非对称最小二乘基线校正 | lambda_, p |
| `detrend` | 去趋势（多项式拟合） | — |
| `mean_center` | 均值中心化 | — |
| `autoscale` | 自缩放（均值+单位方差） | — |
| `normalize` | 行级 L1/L2/max 归一化 | norm |

推荐预处理顺序：散射校正 → 平滑 → 导数 → 基线校正 → 缩放

---

## 质量门禁

质量门禁阈值按应用领域分级，小样本自动放宽：

| 领域 | R²_min | RPD_min | 说明 |
|------|--------|---------|------|
| food_moisture | 0.90 | 4.0 | 水分（通常容易建模） |
| food_protein | 0.85 | 3.5 | 蛋白质 |
| pharma | 0.85 | 3.0 | 制药 |
| feed | 0.75 | 2.5 | 饲料 |
| soil | 0.70 | 2.0 | 土壤（难度大） |
| default | 0.80 | 3.0 | 默认 |

样本量 < 100 时自动放宽（R² - 0.10, RPD - 0.5）。

---

## 两种执行模式

### 快速模式

直接调用 `nir_analyze` 工具，单次完成全流程。不触发反思闭环。

```
/nir 分析这批光谱数据，建立蛋白质含量模型
```

### 分步模式（含反思闭环）

当需要质量优化时，由 `nir-coordinator` Skill 的决策树驱动反思：

```
尝试1: SNV + SG平滑 + 均值中心化 → R²=0.72
   ↓ 不达标
尝试2: MSC + 1阶导数 + 自缩放 → R²=0.85
   ↓ 达标，通过
```

---

## 快速开始

### 1. 环境准备

NIR-Agent 基于 DeerFlow 运行，需要先完成 DeerFlow 的基础配置：

```bash
# 克隆项目
git clone https://gitee.com/starlightsir/nirs-agent.git
cd nirs-agent

# DeerFlow 配置
make setup    # 交互式配置向导

# 安装 nir_core 算法包
cd nir_core
pip install -e .
cd ..
```

### 2. 启动服务

```bash
# Docker 方式（推荐）
make docker-init
make docker-start

# 本地开发方式
make install
make dev
```

### 3. 使用 NIR-Agent

访问 http://localhost:2026，输入 `/nir` 前缀激活 NIR 分析技能：

```
/nir 加载 /mnt/user-data/uploads/corn.mat 并建立蛋白质含量的 PLS 模型
```

---

## nir_core 独立使用

`nir_core` 可以脱离 DeerFlow 独立使用：

```python
from nir_core.io.loaders import auto_detect_and_load
from nir_core.preprocess.pipeline import PreprocessingPipeline, PreprocessingStep
from nir_core.model.pls import train_pls, predict_pls
from nir_core.model.evaluation import split_dataset, compute_metrics
from nir_core.utils.metrics import evaluate_quality

# 加载光谱数据
data = auto_detect_and_load("corn.mat")

# 三集分离
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(data.X, data.y)

# 预处理
pipeline = PreprocessingPipeline([
    PreprocessingStep(method="snv"),
    PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
    PreprocessingStep(method="mean_center"),
])
X_train_pp = pipeline.apply(X_train)

# 建模
model, n_comp, cv_results = train_pls(X_train_pp, y_train)
y_pred = predict_pls(model, pipeline.apply(X_test))
metrics = compute_metrics(y_test, y_pred)

# 质量评估
quality = evaluate_quality(metrics, domain="food_protein", n_samples=len(y_test))
print(f"质量评级: {quality['grade']}, 通过: {quality['passed']}")
```

---

## 测试

```bash
cd nir_core

# 运行全部单元测试
pytest

# 运行 Phase 1 端到端验证
python scripts/validate_phase1.py

# 指定模块测试
pytest tests/test_preprocess/test_snv.py -v
pytest tests/test_model/test_pls.py -v
```

---

## 知识库

NIR-Agent 内置化学计量学领域知识库，通过 `nir-knowledge` Skill 注入 Agent 上下文：

| 文档 | 内容 |
|------|------|
| `chemometrics-rules.md` | 化学计量学硬规则（三集分离、CV选成分、预处理顺序） |
| `preprocessing-guide.md` | 预处理方法选择决策树 |
| `modeling-guide.md` | PLS/PCR/SVR 选择指南 |
| `metrics-interpretation.md` | 指标解读（RMSEC/RMSECV/RMSEP、RPD分级） |
| `troubleshooting.md` | 常见问题排查（过拟合、漂移、异常样本） |

---

## 致谢

- **[DeerFlow v2.0](https://github.com/bytedance/deer-flow)**：本项目基于 ByteDance 开发的 DeerFlow 超级智能体框架构建，感谢其出色的 Agent 运行时、子代理编排、沙箱执行和技能系统。
- **[LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://github.com/langchain-ai/langgraph)**：DeerFlow 的底层框架。
- NIR 算法实现参考了化学计量学经典文献和开源项目（scikit-learn、scipy 等）。

---

## 许可证

本项目基于 [MIT License](./LICENSE) 开源。DeerFlow 框架本身同样遵循 MIT License。
