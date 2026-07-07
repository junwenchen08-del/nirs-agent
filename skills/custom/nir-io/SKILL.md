---
name: nir-io
description: >-
  Load and inspect NIR spectral data from .mat / .csv / .txt files. Auto-detects
  format and structure, standardizes to .npz. Activates via /nir-io.
allowed-tools:
  - bash
  - read_file
  - write_file
  - ls
  - nir_load_data
  - nir_inspect
---

# NIR 数据加载技能

## 用途
当用户上传 .mat/.csv/.txt 光谱文件，需要加载、检查或标准化时使用。

## 工作流

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
