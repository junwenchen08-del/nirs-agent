---
name: nir-gallery
description: >-
  Generate visualization plots for NIR analysis results (spectra, predicted vs
  reference, residuals, drift heatmap, comparison gallery). Activates via /nir-gallery.
allowed-tools:
  - bash
  - read_file
  - write_file
  - ls
---

# NIR 可视化技能

## 用途
为 NIR 分析结果生成可视化图表。

## 可视化类型

### 1. 光谱图
用 bash 调用 nir_core 的绘图函数：
```bash
python -c "
import matplotlib; matplotlib.use('Agg')
from nir_core.io.loaders import auto_detect_and_load
from nir_core.plotting.spectra import plot_raw_spectra
import base64
data = auto_detect_and_load('/mnt/user-data/workspace/data.npz')
png = plot_raw_spectra(data)
open('/mnt/user-data/workspace/spectra.png','wb').write(base64.b64decode(png))
"
```

### 2. 预测vs参考图
```bash
python -c "
import matplotlib; matplotlib.use('Agg')
from nir_core.plotting.model_diag import plot_predicted_vs_reference
import numpy as np, json, base64
m = json.load(open('/mnt/user-data/workspace/metrics.json'))
# 需 y_ref/y_pred，可从 metrics 或重算
"
```

### 3. 多模型对比画廊
```bash
python -c "
from nir_core.plotting.gallery import generate_comparison_gallery
from nir_core.models import ModelResult
results = [ModelResult(method='pls', n_components=3, metrics={'RMSEP':0.1,'R2':0.95,'RPD':5.0})]
html = generate_comparison_gallery(results)
open('/mnt/user-data/workspace/gallery.html','w').write(html)
"
```

## 注意
- matplotlib 须用 Agg backend（无显示环境）
- 图表返回 base64 PNG 或 HTML，保存到 /mnt/user-data/workspace/
- nir_analyze 已自动生成 predicted_vs_reference.png，可直接引用
