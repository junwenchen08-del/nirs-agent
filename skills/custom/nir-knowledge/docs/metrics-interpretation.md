# 指标解读指南

## RMSE 系列

| 指标 | 含义 | 理想 |
|------|------|------|
| RMSEC | 训练集RMSE | 小，但远小于RMSEP警示过拟合 |
| RMSECV | 交叉验证RMSE | 反映泛化，选成分数依据 |
| RMSEP | 测试集RMSE | 最关键，最终评估 |

### 关系解读
- RMSEC < RMSEP < RMSECV：正常（CV更严）
- RMSEC << RMSECV：过拟合，减成分数
- RMSEP >> RMSECV：验证集不具代表性 或 过拟合验证
- RMSEC ≈ RMSEP ≈ RMSECV：良好泛化

## R²（决定系数）
- >0.90：优秀
- 0.80-0.90：良好
- 0.60-0.80：一般
- <0.60：差

## RPD（Ratio of Performance to Deviation）
RPD = std(y) / RMSEP

| RPD | 等级 |
|-----|------|
| >3.0 | 优秀，可定量 |
| 2.5-3.0 | 良好 |
| 2.0-2.5 | 可粗略筛查 |
| 1.5-2.0 | 仅可区分高低 |
| <1.5 | 不可用 |

## bias（系统偏差）
- bias = mean(y_pred - y_true)
- 应接近 0
- |bias| > 0.1×range(y)：系统误差，需偏差校正

## slope（回归斜率）
- 预测值对参考值的回归斜率
- 应接近 1.0
- slope < 1：预测范围被压缩（回归到均值）
- slope > 1：预测范围被放大

## 残差分析
- 残差应随机分布，无趋势
- 残差 vs 预测值有趋势：异方差或非线性
- 残差直方图应近似正态

## 领域阈值（nir_core 配置）
| 领域 | R²_min | RPD_min |
|------|--------|---------|
| food_moisture | 0.90 | 4.0 |
| food_protein | 0.85 | 3.5 |
| pharma | 0.85 | 3.0 |
| feed | 0.75 | 2.5 |
| soil | 0.70 | 2.0 |
| default | 0.80 | 3.0 |

样本量<100 自动放宽（R²-0.10, RPD-0.5）。
