---
name: nir-preprocess
description: >-
  Apply NIR preprocessing methods (despiking, SNV/robust SNV, MSC/EMSC,
  SG or Norris-Williams derivatives, baseline correction, scaling) and align
  wavelength axes. Activates via /nir-preprocess.
allowed-tools:
  - read_file
  - write_file
  - ls
  - nir_list_preprocessing_methods
  - nir_describe_preprocessing_method
  - nir_recommend_preprocessing
  - nir_preprocess
  - nir_align_wavelengths
  - chemotools_health
  - chemotools_list_capabilities
  - chemotools_describe_capability
  - chemotools_validate_operation
  - chemotools_fit_estimator
  - chemotools_apply_estimator
  - chemotools_call_function
  - chemotools_render_plot
  - chemotools_run_inspector
  - chemotools_get_artifact_metadata
---

# NIR 预处理技能

## 用途
对 .npz 光谱数据应用单一预处理方法，输出预处理后的 .npz。

## Chemotools MCP

固定版本 `chemotools==0.4.4` 已作为独立 MCP 组件提供。每个新的预处理选择都必须先调用
`chemotools_list_capabilities(category=...)` 查询 MCP 运行时目录，再用
`chemotools_describe_capability(capability_id=...)` 读取运行时构造参数和允许操作；禁止按记忆
猜测 Python 包参数。不得把目录查询与预处理执行放在同一批并行调用中。MCP 目录覆盖
Chemotools 的全部公开预处理类，以及校准适配、数据增强、
特征选择、PLS 回归、异常值诊断、物理转换、绘图和 Inspector。

Chemotools MCP 的“完整可调用”不等于“全部自动候选”。普通生产建模仍使用
`nir_train_*`/`nir_analyze` 的训练边界和 `pipeline_steps`，这样预处理只在训练折拟合。尚未进入
NIR 注册表的 Chemotools 方法只允许显式实验；不得先对完整数据调用
`chemotools_fit_estimator`，再把结果随机划分后宣称无泄漏建模。校准迁移、数据增强、投影和
异常值算法默认均为显式调用。

选择顺序固定为：Chemotools MCP 目录 → 所选能力详情 → 项目注册表兼容性检查 → 受控执行。
Chemotools 有等价且语义兼容的能力时优先使用；只有 MCP 调用失败、上游无等价能力、物理轴
语义不匹配或回放旧工件时才允许使用原生实现，并说明回退原因。正式训练继续通过
`nir_train_*` / `nir_analyze` 在训练分区内拟合，不能为了“走 MCP”而先处理完整数据。

## 可用方法

Chemotools MCP 目录是候选选择的第一来源；项目运行时算法目录负责方法 ID、训练边界和工件
兼容性。完成 MCP 查询后，调用 `nir_list_preprocessing_methods` 确认当前项目可执行的方法；
准备构造带参数的步骤前，再调用
`nir_describe_preprocessing_method(method=...)` 获取合法参数、范围、顺序、自动候选级别和实现版本。禁止根据记忆猜参数。下面的表用于快速阅读，不取代运行时目录。

需要解释“为什么推荐这些预处理”时，可调用
`nir_recommend_preprocessing(input_path=..., budget="standard")`。它只返回有界诊断和候选，不返回原始光谱；生产训练仍会在校准分区内重新生成候选，不能把该探索结果当成验证证据。

| 方法 | 说明 | 关键参数 |
|------|------|---------|
| snv | 标准正态变量变换（逐行标准化） | 无 |
| robust_snv | 稳健 SNV（中位数和 MAD） | consistency=1.4826 |
| rnv | Chemotools 稳健正态变量 | percentile=25, epsilon=1e-10 |
| msc | 多元散射校正 | 无 |
| emsc | 扩展多元散射校正 | polynomial_order=0..3 |
| despike | 去除孤立仪器尖峰 | window=5, z_threshold=6.0 |
| sg_smooth | Savitzky-Golay 平滑 | window=11, order=2 |
| whittaker_smooth | Whittaker 惩罚平滑 | lambda_=1e4 |
| median_filter | 中值滤波（仅显式选择） | window_length=3, mode=nearest |
| derivative1 | 一阶导数（SG） | window=11, order=2 |
| derivative2 | 二阶导数（SG） | window=11, order=2 |
| norris_derivative1 | Norris-Williams 一阶导数 | gap=3, segment=5, delta=1.0 |
| norris_derivative2 | Norris-Williams 二阶导数 | gap=3, segment=5, delta=1.0 |
| airpls | airPLS 基线校正 | lambda_=1e7 |
| arpls | ArPLS 基线校正 | lambda_=1e4, ratio=0.01 |
| asls | 非对称最小二乘基线 | lambda_=1e5, p=0.001 |
| rubberband | 橡皮筋基线校正（仅显式选择） | 无 |
| detrend | 去趋势（二次多项式） | 无 |
| mean_center | 均值中心化 | 无 |
| autoscale | 自缩放（均值+单位方差） | 无 |
| normalize | 归一化 | norm=l2 |

## 调用示例

单方法（向后兼容）：
```
nir_preprocess(
  input_path="/mnt/user-data/workspace/data.npz",
  method="snv",
  output_path="/mnt/user-data/workspace/data_snv.npz"
)
```

★ v3 多步流水线（原子执行，推荐）：
```
nir_preprocess(
  input_path="/mnt/user-data/workspace/data.npz",
  pipeline_steps='["snv","sg_smooth","mean_center"]',
  output_path="/mnt/user-data/workspace/data_preprocessed.npz"
)
```

含参数的方法：
```
nir_preprocess(
  input_path="/mnt/user-data/workspace/data.npz",
  method="sg_smooth",
  output_path="/mnt/user-data/workspace/data_sg.npz",
  window=11,
  order=2
)
```

## 预处理顺序与互斥规则

统一顺序为：去尖峰 → 基线/趋势校正 → 散射校正 → 平滑或导数 → 缩放。

- `snv`、`robust_snv`、`rnv`、`msc`、`emsc` 五选一。
- SG 一/二阶导数和 Norris-Williams 一/二阶导数四选一。
- `mean_center`、`autoscale`、`normalize` 三选一。
- `median_filter`、`rubberband` 等显式方法不会自动进入候选；RNV、ArPLS 和
  Whittaker 只在诊断命中且候选预算允许时比较。

## 多步预处理

使用一次原子 `pipeline_steps` 调用，不要为每一步生成中间文件：
```
nir_preprocess(
  input_path="data.npz",
  pipeline_steps='[{"method":"despike","params":{"window":5}},"emsc",{"method":"norris_derivative1","params":{"gap":3,"segment":5}},"mean_center"]',
  output_path="final.npz"
)
```

生产建模不能先对全量数据执行状态预处理后再划分。MSC、EMSC、AirPLS、ArPLS、
均值中心化和自动缩放必须通过训练工具的 `pipeline_steps` 在训练分区内拟合。独立
`nir_preprocess` 主要用于探索、可视化和导出。

## 波长轴对齐

波长对齐会同时改变 `X` 和 `wv`，必须使用独立工具，不能作为普通 `pipeline_steps` 步骤：

```text
nir_align_wavelengths(
  input_path="/mnt/user-data/uploads/data.npz",
  reference_path="/mnt/user-data/uploads/reference_axis.npz",
  output_path="/mnt/user-data/workspace/data_aligned.npz",
  kind="linear",
  allow_extrapolation=false
)
```

默认禁止外推。输入或参考文件缺少 `wv`、轴不单调、存在重复波长或目标范围超出输入范围时必须停止并报告。

DS、PDS、SST 需要两台仪器测量的配对样本，不属于本技能的普通预处理。遇到跨仪器
模型复用需求时使用独立的 NIR 校准迁移工具，禁止把迁移方法写进 `pipeline_steps`。

## 注意
- window 必须为奇数且 > order
- 导数 deriv 必须 ≤ order
- 不要在此技能中建模，建模用 nir-model
