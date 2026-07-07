---
name: nir-predict
description: >-
  Predict reference values for new NIR spectra using a trained model. Supports
  optional drift detection. Activates via /nir-predict.
allowed-tools:
  - bash
  - read_file
  - write_file
  - ls
  - nir_predict
---

# NIR 预测技能

## 用途
用已训练的模型（.pkl）对新光谱（.npz）预测参考值。

## 调用示例
```
nir_predict(
  model_path="/mnt/user-data/workspace/model.pkl",
  data_path="/mnt/user-data/workspace/new_spectra.npz",
  output_path="/mnt/user-data/workspace/predictions.csv",
  detect_drift=true
)
```

## 参数
| 参数 | 说明 |
|------|------|
| model_path | nir_train_model 生成的 .pkl 模型 |
| data_path | 预处理后的新数据 .npz（须与训练时预处理一致） |
| output_path | 可选，预测结果 CSV |
| detect_drift | 是否检测漂移（马氏距离） |

## 返回
JSON 含：n_samples, prediction_mean/min/max, drift（可选）。

## 注意
- 新数据必须经过与训练相同的预处理
- drift 检测当前用输入数据自身作参考（无训练矩阵时）；真实漂移评估需训练集
- 预测结果保存为 CSV（sample_index, prediction）
