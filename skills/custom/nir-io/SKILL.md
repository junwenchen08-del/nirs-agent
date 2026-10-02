---
name: nir-io
description: >-
  Load and inspect NIR spectral data from .mat / .csv / .txt files. Auto-detects
  format and structure, standardizes to .npz. Activates via /nir-io.
allowed-tools:
  - read_file
  - write_file
  - ls
  - nir_load_data
  - nir_inspect
  - nir_dataset_list
  - nir_dataset_get
  - nir_dataset_history
  - nir_dataset_save
  - nir_dataset_attach
---

# NIR 数据加载技能

## 用途
当用户上传 .mat/.csv/.txt 光谱文件，需要加载、检查或标准化时使用。

## 工作流

### Step 0：需要跨会话复用时

- 用户提到“之前保存的数据”但没有当前线程文件时，先用 `nir_dataset_list`，同名或多个候选时让用户按 ID 消歧；再用 `nir_dataset_get` 查看 Profile 摘要。
- 选定后调用 `nir_dataset_attach`。它只把经过哈希校验的副本放回当前线程 uploads；必须对返回的 `virtual_path` 重新执行 `nir_inspect`，不得继承旧会话的检查或建模结论。
- `profile_id` 只有 confirmed 状态才能随挂载复用；不确定时不传 Profile，按原始文件重新检查。
- 只有用户最新消息明确要求长期/跨会话保存时才调用 `nir_dataset_save`。普通上传和 `nir_inspect` 不自动入库。
- 已明确要求保存时，顺序必须是：启动工作流 → `nir_inspect` → `nir_dataset_save` → `record_audit`。保存必须在仍处于 `data_audit` 阶段时完成，不得拖到计划、执行或审查阶段。

### Step 1: 预览文件结构
```
nir_inspect(file_path="/mnt/user-data/uploads/data.mat")
```
返回 JSON：格式、形状、样本数、波长数、值范围、是否有 NaN。不完整加载，快速预览。

### Step 2: 加载并标准化
```
nir_load_data(
  file_path="/mnt/user-data/uploads/data.mat",
  output_path="/mnt/user-data/workspace/data.npz"
)
```
自动检测格式（mat v5/v7/v7.3、csv/txt）、排列方向（行=样本 or 列=样本），标准化为 .npz（含 X/y/wv）。

## 输出 .npz 结构
| 键 | 形状 | 说明 |
|----|------|------|
| X | (n_samples, n_wavelengths) | 光谱矩阵 |
| y | (n_samples,) | 参考值（若无则为空） |
| wv | (n_wavelengths,) | 波长（若无则为空） |

## 注意
- 不要读取 Python 脚本，只调用工具
- .npz 是 nir-preprocess 和 nir-model 的标准输入
- 大文件（>1GB）先用 nir_inspect 预览再加载
