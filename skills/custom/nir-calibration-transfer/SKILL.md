---
name: nir-calibration-transfer
description: >-
  Transfer a reference NIR calibration model between compatible instruments
  using paired spectra, DS, PDS, or SST. Activates for calibration transfer,
  instrument standardization, model transfer, or cross-instrument deployment.
allowed-tools:
  - nir_list_calibration_transfer_methods
  - nir_fit_calibration_transfer
  - nir_apply_calibration_transfer
  - nir_evaluate_calibration_transfer
---

# NIR 校准迁移

## 边界

校准迁移不是普通预处理。只有同类光谱仪器对同一批实体样本的成对测量，才可学习
“目标仪器 → 参考仪器”映射。不得把 NIR 与 Raman 等不同模态互相迁移，也不得把
`ds`、`pds`、`sst` 放入 `pipeline_steps`。

## 固定步骤

1. 调用 `nir_list_calibration_transfer_methods` 获取 DS/PDS/SST 的实际参数。
2. 确认参考和目标 NPZ 都有 `X`、严格单调的 `wv`、唯一的 `sample_names`。
3. 明确 `source_instrument_id` 是参考模型所属仪器，`target_instrument_id` 是待部署仪器。
4. 默认使用 `pairing_mode="sample_names"`。仅在用户明确确认逐行一一对应时才允许
   `pairing_mode="row_order"`。
5. 用 `nir_fit_calibration_transfer` 在配对训练子集拟合；没有独立验证文件时只执行内部
   配对留出，波长轴不同时允许在覆盖范围内线性对齐，始终禁止外推。
6. 只有光谱误差改善时返回 `spectrally_validated`。这不等于生产批准。
7. 生产部署必须另行传入不与训练样本重叠的 `validation_source_path`、
   `validation_target_path`，以及可信 `reference_model_path`、`reference_model_id`；验证源
   NPZ 必须含参考 y。内部留出通过只能得到 `model_validated_internal`，不能宣称生产批准。
   只有独立验证上迁移后的 RMSEP 改善并返回 `production_validated` / `approved`，才可作为
   生产迁移工件。
8. 用 `nir_apply_calibration_transfer` 处理目标仪器新光谱；仪器 ID 和目标轴必须与工件一致。
9. 独立配对批次使用 `nir_evaluate_calibration_transfer` 复核 RMSE、bias、slope 和 R²。

## 方法选择

- DS：第一基线，参数最少，先试。
- PDS：局部波段差异明显时使用；`window_length` 和 `n_components` 只能基于训练/调优证据选择。
- SST：系统性低维空间差异较复杂时使用；`n_components` 不能查看最终评估集选择。

所有迁移工件只能从 `/mnt/user-data/outputs` 加载，并在反序列化前校验 SHA-256；生产环境
要求签名时同样执行 HMAC 校验。
