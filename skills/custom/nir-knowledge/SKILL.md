---
name: nir-knowledge
description: >-
  NIR spectroscopy domain knowledge base. Not triggered by a slash command;
  loaded as reference by nir-coordinator. Contains chemometrics rules,
  preprocessing/modeling guides, metrics interpretation, and troubleshooting.
allowed-tools:
  - read_file
  - ls
---

# NIR 知识库

本技能是知识库入口，不通过 / 命令激活。nir-coordinator 在分析前会读取本目录下的文档。

## 文档清单
- `docs/chemometrics-rules.md` — 化学计量学硬规则（必读）
- `docs/preprocessing-guide.md` — 预处理方法选择指南
- `docs/modeling-guide.md` — 建模方法选择指南
- `docs/metrics-interpretation.md` — 指标解读指南
- `docs/troubleshooting.md` — 常见问题排查

## 使用方式
路径前缀：`/mnt/skills/custom/nir-knowledge/docs/`
```
read_file(file_path="/mnt/skills/custom/nir-knowledge/docs/chemometrics-rules.md")
```
