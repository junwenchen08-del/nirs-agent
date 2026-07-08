---
name: nir-preprocess
description: >-
  Apply NIR preprocessing methods (SNV, MSC, SG smoothing, derivatives, airPLS,
  asLS, detrend, scaling, normalization) to spectral .npz data. Activates via /nir-preprocess.
allowed-tools:
  - read_file
  - write_file
  - ls
  - nir_preprocess
---

# NIR 预处理技能

## 用途
对 .npz 光谱数据应用单一预处理方法，输出预处理后的 .npz。

## 可用方法
| 方法 | 说明 | 关键参数 |
|------|------|---------|
| snv | 标准正态变量变换（逐行标准化） | 无 |
| msc | 多元散射校正 | 无 |
| sg_smooth | Savitzky-Golay 平滑 | window=11, order=2 |
| derivative1 | 一阶导数（SG） | window=11, order=2 |
| derivative2 | 二阶导数（SG） | window=11, order=2 |
| airpls | airPLS 基线校正 | lambda_=1e7 |
| asls | 非对称最小二乘基线 | lambda_=1e5, p=0.001 |
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

## 预处理顺序建议
散射校正 → 平滑 → 导数 → 基线校正 → 缩放。
即：snv/msc → sg_smooth → derivative → airpls → mean_center。

## 多步预处理
需多次调用 nir_preprocess，每次输入上一次的输出：
```
1. nir_preprocess(in=data.npz, method=snv, out=step1.npz)
2. nir_preprocess(in=step1.npz, method=sg_smooth, out=step2.npz)
3. nir_preprocess(in=step2.npz, method=mean_center, out=final.npz)
```

## 注意
- window 必须为奇数且 > order
- 导数 deriv 必须 ≤ order
- 不要在此技能中建模，建模用 nir-model
