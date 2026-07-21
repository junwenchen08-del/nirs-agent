---
name: nir-predict
description: >-
  Predict reference values for new NIR spectra using a trained model. Supports
  optional drift detection. Activates via /nir-predict.
allowed-tools:
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
  model_path="/mnt/user-data/outputs/model.pkl",
  data_path="/mnt/user-data/workspace/new_spectra.npz",
  output_path="/mnt/user-data/workspace/predictions.csv",
  detect_drift=true
)
```

## 参数
| 参数 | 说明 |
|------|------|
| model_path | NIR 建模工具在 `/mnt/user-data/outputs` 生成且带完整性清单的 .pkl 模型 |
| data_path | 预处理后的新数据 .npz（须与训练时预处理一致） |
| output_path | 可选，预测结果 CSV |
| detect_drift | 是否用训练域 PCA T²/Q 参考检测适用域漂移 |

## 返回
JSON 含：n_samples, prediction_mean/min/max, drift（可选）。

## 注意
- 新数据必须经过与训练相同的预处理
- 仅加载 NIR 工具生成并通过 SHA-256（生产环境可选 HMAC）校验的模型；拒绝 uploads/workspace 中的 pickle
- 新模型保存训练模型空间的 PCA T²/Q 参考；旧模型没有参考时返回 `drift.available=false`，不会伪造自参照漂移
- NPZ 不允许 pickle-backed object arrays；旧文件需重新导出为数值/Unicode 数组
- 预测结果保存为 CSV（sample_index, prediction）
