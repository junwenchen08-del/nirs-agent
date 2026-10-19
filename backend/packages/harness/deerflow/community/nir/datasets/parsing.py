"""Bounded, deterministic choices for visual library parsing; no guessed groups."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from nir_core.io.schema import flatten_mat_leaves, infer_csv_mapping, infer_mat_mapping, parse_number, profile_csv

MAX_FIELDS = 256
MAX_ELEMENTS = 5_000_000


def _id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def _mat_leaves(path):
    from .preview import MAX_FILE_BYTES, _check_mat_size

    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("文件超过 64 MB，暂不支持可视化解析。")
    if _check_mat_size(path):
        import h5py

        def read(node):
            return {name: read(node[name]) for name in node} if isinstance(node, h5py.Group) else np.asarray(node[...])

        with h5py.File(path, "r") as root:
            raw = {name: read(root[name]) for name in root if name != "#refs#"}
    else:
        from scipy.io import loadmat

        raw = {key: value for key, value in loadmat(path, squeeze_me=True, struct_as_record=False).items() if not key.startswith("__")}
    leaves = flatten_mat_leaves(raw)
    if any(np.iscomplexobj(value) for value in leaves.values()):
        raise ValueError("复数数据暂不支持可视化解析，请检查原文件。")
    if len(leaves) > MAX_FIELDS or any(len(name) > 512 for name in leaves) or sum(value.size for value in leaves.values()) > MAX_ELEMENTS:
        raise ValueError("文件包含过多字段或超过 500 万个数据值，请拆分后再解析。")
    return raw, leaves


def _near(field, parent):
    prefix = field.rpartition(".")[0]
    return prefix == parent or not prefix or parent.startswith(prefix + ".")


def _named(field, prefixes):
    return field.rsplit(".", 1)[-1].lower().startswith(prefixes)


def _mat_catalog(path):
    raw, leaves = _mat_leaves(path)
    matrices = {name: value for name, value in leaves.items() if value.ndim == 2 and min(value.shape) >= 2}
    groups = {}
    for name, value in matrices.items():
        if _named(name, ("x", "spectra", "absorbance")):
            groups.setdefault(name.rpartition(".")[0], []).append((name, value))
    layouts = []

    def add(names, transpose=False):
        arrays = [leaves[name].T if transpose else leaves[name] for name in names]
        samples, features = arrays[0].shape[0], sum(value.shape[1] for value in arrays)
        if any(value.shape[0] != samples for value in arrays) or samples > 5000 or features > 20000:
            return
        parent = names[0].rpartition(".")[0]
        targets = []
        for field, value in leaves.items():
            if field in names or not _near(field, parent) or _named(field, ("wv", "wn", "wave", "lambda", "axis")):
                continue
            if value.ndim == 1 and len(value) == samples:
                columns = [None]
            elif value.ndim == 2 and value.shape[0] == samples and value.shape[1] <= 16:
                columns = list(range(value.shape[1]))
            else:
                continue
            for column in columns:
                spec = {"field": field, "column": column}
                targets.append({"id": _id(spec), "label": field + (f" · 指标 {column + 1}" if column is not None else ""), "_spec": spec})
        axes, chosen = [], []
        for name, array in zip(names, arrays, strict=True):
            suffix = re.search(r"\d+$", name)
            options = [(field, value) for field, value in leaves.items() if value.ndim == 1 and len(value) == array.shape[1] and _near(field, parent) and np.isfinite(value).all()]
            preferred = [(field, value) for field, value in options if _named(field, ("wv", "wn", "wave", "lambda", "axis"))]
            if suffix:
                matched = [(field, value) for field, value in preferred if field.endswith(suffix[0])]
                preferred = matched or preferred
            if len(names) == 1:
                axes = [{"id": _id([field]), "label": field, "_fields": [field]} for field, _ in options]
            if len(preferred) == 1:
                chosen.append(preferred[0][0])
        if len(names) > 1 and len(chosen) == len(names):
            axes = [{"id": _id(chosen), "label": " + ".join(chosen), "_fields": chosen}]
        preferred_targets = [item for item in targets if _named(item["_spec"]["field"], ("y", "target", "ref", "label", "content"))]
        default_target = preferred_targets[0]["id"] if len(preferred_targets) == 1 else None
        layout_id = _id({"x": names, "transpose": transpose})
        layouts.append(
            {
                "id": layout_id,
                "group": parent or "文件内的数据",
                "label": "全部光谱（" + " + ".join(name.rsplit(".", 1)[-1] for name in names) + "）" if len(names) > 1 else names[0].rsplit(".", 1)[-1] + ("（样本在列）" if transpose else ""),
                "sample_count": samples,
                "feature_count": features,
                "targets": targets,
                "axes": axes,
                "default_target": default_target,
                "default_axis": _id(chosen) if len(chosen) == len(names) else None,
                "_x": names,
                "_transpose": transpose,
            }
        )

    combined = set()
    for _group, entries in groups.items():
        if len(entries) > 1 and len({value.shape[0] for _, value in entries}) == 1:
            names = sorted(name for name, _ in entries)
            add(names)
            combined.update(names)
    inferred = infer_mat_mapping(raw)
    for name in matrices:
        transpose = bool(inferred.get("transpose")) if inferred.get("x_variable") == name else False
        add([name], transpose)
        if name not in combined and matrices[name].shape[0] != matrices[name].shape[1]:
            add([name], not transpose)
    if len(layouts) > 128:
        raise ValueError("可选择的光谱矩阵过多，请先拆分文件。")
    suggested = None
    parents = {item["group"] for item in layouts}
    if len(parents) == 1:
        matching = next((item for item in layouts if len(item["_x"]) > 1), None)
        if matching is None and inferred.get("status") == "auto":
            matching = next((item for item in layouts if item["_x"] == [inferred["x_variable"]] and item["_transpose"] == inferred["transpose"]), None)
        if matching is None:
            evidenced = [item for item in layouts if item["default_target"] is not None and item["default_axis"] is not None]
            if len(evidenced) == 1:
                matching = evidenced[0]
        if matching:
            suggested = {"layout_id": matching["id"], "target_id": matching["default_target"], "axis_id": matching["default_axis"]}
    return layouts, leaves, suggested


def _public_layout(item):
    return {key: ([{k: v for k, v in option.items() if not k.startswith("_")} for option in value] if key in {"targets", "axes"} else value) for key, value in item.items() if not key.startswith("_")}


def apply_header(table, has_header=None):
    if has_header is not None and has_header != table.has_header:
        rows = table.rows if has_header else [table.headers, *table.rows]
        headers = rows[0] if has_header else [f"第 {i + 1} 列" for i in range(table.n_columns)]
        rows = rows[1:] if has_header else rows
        fractions = [sum(parse_number(row[i], str(table.dialect["decimal"])) is not None for row in rows) / max(1, len(rows)) for i in range(table.n_columns)]
        table = replace(table, headers=headers, rows=rows, has_header=has_header, total_rows=table.total_rows + (-1 if has_header else 1), mapping=infer_csv_mapping(headers, rows, fractions, str(table.dialect["decimal"]), has_header))
    return table


def _table(path, has_header=None):
    table = profile_csv(str(path), max_rows=202)
    if table.total_rows > 5001 or table.total_rows * table.n_columns > MAX_ELEMENTS:
        raise ValueError("当前解析支持最多 5000 个样本和 500 万个数据值。")
    if has_header is None and not table.has_header and table.rows:
        from .workspace import LABEL_NAMES, USE_NAMES

        if any(cell.lower().strip() in LABEL_NAMES | USE_NAMES | {"样本编号", "样本名称", "含量", "浓度", "水分", "蛋白质", "脂肪"} for cell in table.rows[0]):
            has_header = True
    return apply_header(table, has_header)


def describe_file(path: Path):
    if path.suffix.lower() == ".mat":
        layouts, _leaves, suggested = _mat_catalog(path)
        return {"format": "mat", "layouts": [_public_layout(item) for item in layouts], "columns": [], "suggested": suggested}
    from .preview import MAX_FILE_BYTES

    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("文件超过 64 MB，暂不支持可视化解析。")
    table = _table(path)
    roles = table.mapping
    spectral = roles.get("spectral_columns") or []
    from .workspace import LABEL_NAMES

    named_targets = [i for i, name in enumerate(table.headers) if name.lower().strip() in LABEL_NAMES | {"含量", "浓度", "水分", "蛋白质", "脂肪", "糖分", "淀粉", "密度", "酸值"}]
    named_samples = [i for i, name in enumerate(table.headers) if name.strip() in {"样本编号", "样本名称", "样品编号", "样品名称"}]
    roles = {**roles, "target_columns": list(dict.fromkeys([*roles.get("target_columns", []), *named_targets])), "sample_id_columns": roles.get("sample_id_columns") or named_samples}
    return {
        "format": "table",
        "layouts": [],
        "columns": [{"index": i, "name": name[:120] or f"第 {i + 1} 列"} for i, name in enumerate(table.headers)],
        "suggested": {
            "spectral_start": min(spectral) if spectral else None,
            "spectral_end": max(spectral) if spectral else None,
            "target_columns": roles.get("target_columns", []),
            "sample_column": (roles.get("sample_id_columns") or [None])[0],
            "transpose": roles.get("transpose", False),
            "coordinate_column": roles.get("wavelength_column"),
            "has_header": table.has_header,
        },
    }


def resolve_selection(path: Path, selection: dict, task_type: str):
    if task_type not in {"calibration", "classification", "inspection"}:
        raise ValueError("请选择有效的分析用途。")
    if path.suffix.lower() == ".mat":
        layouts, _leaves, _suggested = _mat_catalog(path)
        layout = next((item for item in layouts if item["id"] == selection.get("layout_id")), None)
        if layout is None:
            raise ValueError("请选择文件中的数据组和光谱范围。")
        target = next((item["_spec"] for item in layout["targets"] if item["id"] == selection.get("target_id")), None)
        axis = next((item["_fields"] for item in layout["axes"] if item["id"] == selection.get("axis_id")), None)
        if selection.get("target_id") is not None and target is None or selection.get("axis_id") is not None and axis is None:
            raise ValueError("所选目标值或光谱坐标与此数据组不匹配。")
        if task_type != "inspection" and target is None:
            raise ValueError("请为定量分析或分类识别选择目标值。")
        return {"library_parse": {"format": "mat", "x_fields": layout["_x"], "transpose": layout["_transpose"], "target": target, "axis_fields": axis, "selection": selection}}
    table = _table(path, selection.get("has_header"))
    start, end = selection.get("spectral_start"), selection.get("spectral_end")
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < table.n_columns:
        raise ValueError("请选择有效的光谱起始列和结束列。")
    targets = selection.get("target_columns", [])
    sample = selection.get("sample_column")
    coordinate = selection.get("coordinate_column")
    transpose = selection.get("transpose", False)
    if type(transpose) is not bool or not isinstance(targets, list) or len(targets) > 16 or len(set(targets)) != len(targets):
        raise ValueError("目标列或行列方向无效。")
    ids = [*targets, *([sample] if sample is not None else []), *([coordinate] if transpose and coordinate is not None else [])]
    if any(type(i) is not int or not 0 <= i < table.n_columns for i in ids) or any(start <= i <= end for i in ids) or sample in targets:
        raise ValueError("光谱列、目标列和样本名称列不能重复，请调整选择。")
    if transpose and (coordinate is None or targets) or not transpose and task_type != "inspection" and not targets:
        raise ValueError("请确认光谱坐标列或目标列；样本在列时仅支持查看光谱。")
    spectral = list(range(start, end + 1))
    axis = [parse_number(table.headers[i]) for i in spectral] if table.has_header else []
    mapping = {
        "status": "confirmed_mapping",
        "has_header": table.has_header,
        "spectral_columns": spectral,
        "target_columns": targets,
        "sample_id_columns": [sample] if sample is not None else [],
        "transpose": transpose,
        "library_parse": {"format": "table", "selection": selection},
    }
    if transpose:
        mapping["wavelength_column"] = coordinate
    elif axis and all(value is not None for value in axis):
        mapping["wavelengths"] = axis
    return mapping


def mapped_mat_data(path: Path, settings: dict):
    _raw, leaves = _mat_leaves(path)
    names = settings["x_fields"]
    X = np.concatenate([np.asarray(leaves[name], dtype=float).T if settings.get("transpose") else np.asarray(leaves[name], dtype=float) for name in names], axis=1)
    spec = settings.get("target")
    y = None
    if spec:
        values = leaves[spec["field"]]
        y = np.asarray(values if spec["column"] is None else values[:, spec["column"]], dtype=float).reshape(-1)
        if len(y) != len(X):
            raise ValueError("目标值数量与样本不匹配。")
    fields = settings.get("axis_fields")
    axis = np.concatenate([np.asarray(leaves[name], dtype=float).reshape(-1) for name in fields]) if fields else None
    if axis is not None and (len(axis) != X.shape[1] or not np.isfinite(axis).all()):
        raise ValueError("光谱坐标与变量数量不匹配。")
    return SimpleNamespace(X=X, y=y, wv=axis, y_names=[spec["field"].rsplit(".", 1)[-1]] if spec else None, sample_names=[])
