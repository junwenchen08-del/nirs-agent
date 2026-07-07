---
name: nir-model
description: >-
  Train and evaluate chemometric calibration models (PLS/PCR/SVR) on preprocessed
  NIR .npz data. Performs three-way split, inner CV component selection, full
  metric evaluation, and quality gating. Activates via /nir-model.
allowed-tools:
  - bash
  - read_file
  - write_file
  - ls
  - nir_train_model
---

# NIR 建模技能

## 用途
对预处理后的 .npz 数据（含 X 和 y）建立回归模型，返回完整评估指标和质量判断。

## 调用示例
```
nir_train_model(
  input_path="/mnt/user-data/workspace/data_preprocessed.npz",
  method="pls",
  domain="food_protein",
  model_output="/mnt/user-data/workspace/model.pkl",
  metrics_output="/mnt/user-data/workspace/metrics.json",
  test_ratio=0.20,
  val_ratio=0.10,
  cv_folds=10
)
```

## 方法选择
| 方法 | 适用场景 |
|------|---------|
| pls | 首选。处理共线性，线性关系 |
| pcr | 关注主成分得分时 |
| svr | 非线性关系 |

## 领域选择（影响质量门禁阈值）
| domain | R²_min | RPD_min |
|--------|--------|---------|
| food_moisture | 0.90 | 4.0 |
| food_protein | 0.85 | 3.5 |
| pharma | 0.85 | 3.0 |
| feed | 0.75 | 2.5 |
| soil | 0.70 | 2.0 |
| default | 0.80 | 3.0 |

样本量 < 100 时自动放宽。

## 返回字段
JSON 含：method, n_components, R2_val, RPD, RMSEC, RMSECV, RMSEP, grade, passed, action, thresholds_used, model_path, metrics_path。

## 质量判断
- `passed=true`：质量达标，模型可用
- `action=retry_preprocessing`：不达标，建议换预处理（由 coordinator 反思闭环驱动）
- `action=investigate_data`：质量差，检查数据

## 注意
- 本工具不执行反思闭环，只返回单次结果
- 三集分离自动完成，无数据泄露
- 模型序列化为 .pkl，指标存 .json
- 不要在此技能中做预处理，输入应为已预处理的 .npz
