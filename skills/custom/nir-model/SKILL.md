---
name: nir-model
description: >-
  Train and evaluate calibration models (PLS/PCR/SVR/RF/GBM/Ridge/Lasso/KNN/MLP/CNN)
  on preprocessed NIR .npz data. Performs three-way split, inner CV hyper-parameter
  selection, full metric evaluation, and quality gating. Activates via /nir-model.
allowed-tools:
  - read_file
  - write_file
  - ls
  - nir_train_model
---

# NIR 建模技能

## 用途
对 .npz 数据（含 X 和 y）建立回归模型，返回完整评估指标和质量判断。
★ v3: 支持 pipeline_steps 防泄露预处理模式和 cv_strategy 自适应CV。

## 两种模式

### 模式 A：防泄露模式（★ v3 推荐）
传入 pipeline_steps，工具内部先划分再预处理，避免数据泄露：
```
nir_train_model(
  input_path="/mnt/user-data/workspace/data.npz",
  method="pls",
  pipeline_steps='["snv","sg_smooth","mean_center"]',
  domain="food_protein",
  model_output="/mnt/user-data/workspace/model.pkl",
  metrics_output="/mnt/user-data/workspace/metrics.json"
)
```

### 模式 B：已预处理模式（向后兼容）
输入为已预处理的 .npz：
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

### 波长选择
`nir_train_model` 支持在防泄漏预处理之后、最终建模之前执行训练集内波长选择：
```
nir_train_model(
  input_path="/mnt/user-data/workspace/data.npz",
  method="pls",
  pipeline_steps='["snv","sg_smooth"]',
  wavelength_selection="cars",
  wavelength_selection_params='{"n_mc_samples":50,"n_folds":5}',
  model_output="/mnt/user-data/workspace/model.pkl",
  metrics_output="/mnt/user-data/workspace/metrics.json"
)
```

| wavelength_selection | 参数 | 说明 |
|----------------------|------|------|
| none | `{}` | 默认，全波长建模 |
| cars | `n_mc_samples`, `n_folds`, `random_state` | CARS 竞争性自适应重加权采样 |
| spa | `n_min`, `n_max` | SPA 连续投影算法，适合少量代表性波长 |
| manual | `indices` 或波长 `ranges` | 使用已知波段，如 `{"ranges":[[900,1200],[1450,1650]]}` |

选择器只在训练集上拟合，同一组 `selected_indices` 用于验证集、测试集和后续预测。
返回结果和 metrics 中包含 `wavelength_selection`、`n_wavelengths_original`、
`n_wavelengths_model`。启用预处理或选择时，保存的 `.pkl` v2 artifact 会携带
已拟合预处理流水线和波长选择元数据；`nir_predict` 对原始光谱按“预处理后选择”
的顺序自动复用，已预处理输入可设置 `input_preprocessed=true`。

## 方法选择
| 方法 | 类型 | 适用场景 |
|------|------|---------|
| pls | 经典 | 首选。处理共线性，线性关系。输出VIP和回归系数 |
| pcr | 经典 | 关注主成分得分时 |
| svr | 经典 | 非线性关系 |
| rf | 机器学习 | 随机森林，鲁棒性强，适合非线性、高维数据 |
| et | 机器学习 | 极端随机树，方差更低，适合噪声较大的数据 |
| gbm | 机器学习 | 梯度提升，精度高，适合竞赛级别建模 |
| ridge | 机器学习 | L2正则化线性回归，适合共线性强但变量都重要的场景 |
| lasso | 机器学习 | L1正则化线性回归，自带变量选择，适合高维稀疏数据 |
| elasticnet | 机器学习 | L1+L2混合正则化，兼顾变量选择和稳定性 |
| knn | 机器学习 | K近邻，简单直观，适合局部相似性强的数据 |
| mlp | 深度学习 | 多层感知机，适合中等复杂度非线性映射 |
| cnn | 深度学习 | 一维卷积神经网络，自动提取局部光谱特征，需安装PyTorch |

## CV 策略（★ v3）
| cv_strategy | 行为 |
|-------------|------|
| auto（默认） | n_samples<20→LOOCV, <50→5折, else cv_folds |
| loocv | 强制留一法 |
| fixed | 使用 cv_folds 参数 |

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
JSON 含：method, n_components, preprocessing, R2_val, RPD, RMSEC, RMSECV, RMSEP,
vip_summary（PLS）, coef_summary（PLS）, cv_strategy, grade, passed, action,
thresholds_used, model_path, metrics_path。

## 质量判断
- `passed=true`：质量达标，模型可用
- `action=retry_preprocessing`：不达标，建议换预处理（由 coordinator 反思闭环驱动）
- `action=investigate_data`：质量差，检查数据

## 注意
- 本工具不执行反思闭环，只返回单次结果
- 三集分离自动完成，无数据泄露
- ★ v3: 使用 pipeline_steps 时预处理参数仅在训练集拟合，防泄露
- 模型序列化为 .pkl，指标存 .json
- PLS 模型额外输出 VIP 分数和回归系数用于可解释性分析
