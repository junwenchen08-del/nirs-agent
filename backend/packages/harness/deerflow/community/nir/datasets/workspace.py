"""Owner UI workbook data and deterministic exports; never modifies source values."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from nir_core.io.loaders import load_mat
from nir_core.io.schema import infer_csv_mapping, parse_number, profile_csv

from .preview import MAX_FILE_BYTES, _check_mat_size

MAX_ROWS = 5000
MAX_ELEMENTS = 5_000_000
MAX_FEATURES = 20_000
MAX_CURVES = 24
MAX_POINTS = 512
USAGES = ("unassigned", "train", "validation", "test", "excluded")
EXPORT_HEADERS = ("_nir_workspace_sample", "_nir_workspace_usage")
LABEL_NAMES = {"class", "label", "category", "species", "cultivar", "variety", "品种", "类别", "分类", "等级"}
USE_NAMES = {"set", "split", "partition", "usage", "用途", "样本用途", "数据集划分"}
USE_CODES = {"train": 1, "training": 1, "cal": 1, "calibration": 1, "训练集": 1, "校正": 1, "校正集": 1, "val": 2, "validation": 2, "tuning": 2, "验证": 2, "验证集": 2, "test": 3, "测试": 3, "测试集": 3, "excluded": 4, "不参与": 4}


class WorkspaceError(ValueError):
    """A safe user-facing parsing or selection error."""


@dataclass
class Workbook:
    X: np.ndarray
    axis: np.ndarray | None
    names: list[str]
    fields: list[str]
    values: list[list[str | float | None]]
    usages: list[int]
    mapping: dict
    groups: list[str] | None = None

    @property
    def coordinates(self):
        return self.axis if self.axis is not None else np.arange(1, self.X.shape[1] + 1, dtype=float)


def _cell(value):
    if isinstance(value, (float, int, np.number)):
        return float(value) if np.isfinite(value) else None
    return str(value)


def read_workbook(path: Path, profile: dict | None = None) -> Workbook:
    """Materialize bounded rows with exact original sample/variable indices."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise WorkspaceError("文件超过 64 MB，暂不支持表格编辑。可以继续挂载原文件分析。")
    mapping = dict((profile or {}).get("mapping") or {})
    settings = mapping.pop("workspace", None)
    groups = None
    if path.suffix.lower() == ".mat":
        parsed = mapping.get("library_parse", {})
        hdf5 = _check_mat_size(path)
        options = {key: mapping[key] for key in ("x_var", "y_var", "wv_var", "subset", "transpose") if key in mapping}
        if parsed.get("format") == "mat":
            from .parsing import mapped_mat_data

            data = mapped_mat_data(path, parsed)
        elif hdf5:
            from nir_core.io.loaders import _load_mat_v73

            data = _load_mat_v73(str(path), options.get("x_var"), options.get("y_var"), options.get("wv_var"), options.get("subset"), options.get("transpose", False))
        else:
            data = load_mat(str(path), **options)
        X, axis = np.asarray(data.X, dtype=float), data.wv
        names = list(data.sample_names or [])
        targets = np.asarray(data.y) if data.y is not None else np.empty((X.shape[0], 0))
        if targets.ndim == 1:
            targets = targets[:, None]
        fields = [data.y_names[i] if data.y_names and i < len(data.y_names) else f"目标 {i + 1}" for i in range(targets.shape[1])]
        if len(fields) > 16:
            raise WorkspaceError("目标字段超过 16 个，暂不支持工作台编辑，请继续使用原文件分析。")
        values = [[_cell(v) for v in row] for row in targets[:, :16]]
        fields = fields[:16]
        usages = [0] * X.shape[0]
    else:
        # Profile a prefix first; reject oversized allocations before full parsing.
        table = profile_csv(str(path), max_rows=202)
        if table.total_rows > MAX_ROWS + 1 or table.total_rows * table.n_columns > MAX_ELEMENTS:
            raise WorkspaceError("当前表格支持最多 5000 个样本和 500 万个数据值。请拆分较大的文件后再编辑。")
        table = profile_csv(str(path), max_rows=None)
        from .parsing import apply_header

        table = apply_header(table, mapping.get("has_header"))
        if "has_header" not in mapping and not table.has_header and table.rows and any(cell.lower().strip() in LABEL_NAMES | USE_NAMES for cell in table.rows[0]):
            headers, rows = table.rows[0], table.rows[1:]
            fractions = [sum(parse_number(row[i], str(table.dialect["decimal"])) is not None for row in rows) / max(1, len(rows)) for i in range(table.n_columns)]
            table = replace(table, headers=headers, rows=rows, has_header=True, total_rows=len(rows), mapping=infer_csv_mapping(headers, rows, fractions, str(table.dialect["decimal"]), True))
        roles = {**table.mapping, **mapping}
        if roles.get("status") == "needs_user_mapping" and not mapping:
            raise WorkspaceError("请先在数据概览的解析设置中确认光谱列和目标列，再打开编辑工作台。")
        spectral = list(roles.get("spectral_columns") or [])
        targets = list(roles.get("target_columns") or [])
        labels = [i for i, name in enumerate(table.headers) if name.lower().strip() in LABEL_NAMES]
        use_cols = [i for i, name in enumerate(table.headers) if name.lower().strip() in USE_NAMES]
        group_cols = [i for i, name in enumerate(table.headers) if name.lower().strip() in {"group", "sample_group", "batch", "batch_id", "组号", "批次", "重复组"}]
        if len(group_cols) > 1 or len(set(targets + labels)) > 16:
            raise WorkspaceError("存在多个分组字段或过多目标列，暂不支持工作台编辑，请继续使用原文件分析。")
        targets = targets if mapping.get("library_parse") else list(dict.fromkeys([*targets, *labels]))
        spectral = [i for i in spectral if i not in targets and i not in use_cols and i not in group_cols]
        ids = roles.get("sample_id_columns") or []
        if not spectral or any(type(i) is not int or not 0 <= i < table.n_columns for i in [*spectral, *targets, *ids]):
            raise WorkspaceError("无法确认光谱列，请检查解析设置。")
        decimal = str(table.dialect["decimal"])
        X = np.asarray([[parse_number(row[i], decimal) if parse_number(row[i], decimal) is not None else np.nan for i in spectral] for row in table.rows], dtype=float)
        axis = np.asarray(roles["wavelengths"], dtype=float) if roles.get("wavelengths") else None
        if roles.get("transpose"):
            X = X.T
            coordinate = roles.get("wavelength_column")
            if type(coordinate) is not int or not 0 <= coordinate < table.n_columns:
                raise WorkspaceError("请确认光谱坐标列。")
            axis = np.asarray([parse_number(row[coordinate], decimal) for row in table.rows], dtype=float)
            names = [table.headers[i] for i in spectral]
            fields, values, usages = [], [[] for _ in names], [0] * len(names)
        else:
            names = [row[ids[0]] for row in table.rows] if ids else []
            fields = [table.headers[i][:120] for i in targets[:16]]
            categorical = (profile or {}).get("task_type") == "classification"
            values = [[_cell(row[i]) if categorical or i in labels else parse_number(row[i], decimal) for i in targets[:16]] for row in table.rows]
            usages = [USE_CODES.get(row[use_cols[0]].lower().strip(), 0) for row in table.rows] if use_cols else [0] * len(table.rows)
            if group_cols:
                groups = [str(parse_number(row[group_cols[0]], decimal)) if parse_number(row[group_cols[0]], decimal) is not None else row[group_cols[0]] for row in table.rows]
        mapping = {**roles, "spectral_columns": spectral, "target_columns": targets}
    if X.ndim != 2 or not min(X.shape) or X.shape[0] > MAX_ROWS or X.shape[1] > MAX_FEATURES or X.size > MAX_ELEMENTS:
        raise WorkspaceError("数据为空或超出工作台支持的规模，请检查文件。")
    if axis is not None:
        axis = np.asarray(axis, dtype=float).reshape(-1)
        if len(axis) != X.shape[1] or not np.all(np.isfinite(axis)):
            raise WorkspaceError("光谱坐标与数据不匹配，请确认解析设置。")
    if len(names) != len(X):
        names = [f"样本 {i + 1}" for i in range(len(X))]
    workbook = Workbook(X, axis, names, fields, values, usages, mapping, groups)
    if settings:
        checked = validate_settings(workbook, settings["usages"], settings["range"])
        workbook.usages = checked["usages"]
    return workbook


def validate_settings(book: Workbook, usages, band_range) -> dict:
    if not isinstance(usages, list) or len(usages) != len(book.X) or any(type(code) is not int or code not in range(5) for code in usages):
        raise WorkspaceError("样本用途数量或标签无效，请刷新后重试。")
    if book.groups:
        _validate_group_roles(book.groups, usages)
    if not isinstance(band_range, list) or len(band_range) != 2 or any(type(v) not in (int, float) or not np.isfinite(v) for v in band_range):
        raise WorkspaceError("请输入有效的波段起止值。")
    low, high = map(float, band_range)
    coordinates = book.coordinates
    if low > high or low < float(coordinates.min()) or high > float(coordinates.max()) or not np.any((coordinates >= low) & (coordinates <= high)):
        raise WorkspaceError("所选范围必须在文件坐标内，并至少包含一个光谱变量。")
    return {"usages": usages.copy(), "range": [low, high]}


def workbook_summary(book: Workbook, profile: dict | None) -> dict:
    settings = (profile or {}).get("mapping", {}).get("workspace") or {}
    band = settings.get("range") or [float(book.coordinates.min()), float(book.coordinates.max())]
    return {
        "profile_id": (profile or {}).get("id"),
        "version": (profile or {}).get("profile_version", 0),
        "saved": bool(settings),
        "sample_count": len(book.X),
        "feature_count": book.X.shape[1],
        "axis_kind": "axis" if book.axis is not None else "index",
        "axis": book.coordinates.tolist(),
        "fields": book.fields + (["批次或重复组"] if book.groups else []),
        "usages": book.usages,
        "range": band,
        "selected_feature_count": int(np.count_nonzero((book.coordinates >= band[0]) & (book.coordinates <= band[1]))),
    }


def workbook_rows(book: Workbook, offset: int, limit: int, query: str = "", usage: int | None = None) -> dict:
    indices = [i for i, name in enumerate(book.names) if (not query or query.casefold() in name.casefold()) and (usage is None or book.usages[i] == usage)]
    return {
        "total": len(indices),
        "offset": offset,
        "rows": [
            {"index": i, "name": book.names[i][:200], "values": [value[:200] if isinstance(value, str) else value for value in book.values[i]] + ([book.groups[i][:200]] if book.groups else [])} for i in indices[offset : offset + limit]
        ],
    }


def workbook_spectra(book: Workbook, indices: list[int] | None = None) -> dict:
    chosen = np.linspace(0, len(book.X) - 1, min(MAX_CURVES, len(book.X)), dtype=int).tolist() if indices is None else indices
    if len(chosen) > MAX_CURVES or any(type(i) is not int or not 0 <= i < len(book.X) for i in chosen) or len(set(chosen)) != len(chosen):
        raise WorkspaceError("一次最多查看 24 条不同样本的光谱。")
    points = np.linspace(0, book.X.shape[1] - 1, min(MAX_POINTS, book.X.shape[1]), dtype=int)
    return {"axis": book.coordinates[points].tolist(), "total": len(book.X), "point_count": book.X.shape[1], "curves": [{"index": i, "name": book.names[i][:200], "values": [_cell(v) for v in book.X[i, points]]} for i in chosen]}


def write_workspace_csv(book: Workbook, settings: dict, destination: Path, *, parsing_only=False) -> None:
    """Deterministic UTF-8 export with full selected values and fixed row roles."""
    selected = np.flatnonzero((book.coordinates >= settings["range"][0]) & (book.coordinates <= settings["range"][1]))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                *EXPORT_HEADERS,
                *(["_nir_workspace_parsing_only"] if parsing_only else []),
                *(["_nir_workspace_group"] if book.groups else []),
                *(f"target_{i}:{name}" for i, name in enumerate(book.fields)),
                *((str(float(book.coordinates[i])) if book.axis is not None else f"index:{i + 1}") for i in selected),
            ]
        )
        for row, code in enumerate(settings["usages"]):
            writer.writerow([book.names[row], USAGES[code], *(["1"] if parsing_only else []), *([book.groups[row]] if book.groups else []), *book.values[row], *(float(book.X[row, i]) for i in selected)])
    if destination.stat().st_size > MAX_FILE_BYTES:
        raise WorkspaceError("所选数据导出后超过 64 MB，请缩小光谱范围后再挂载。")


def load_workspace_export(path: str) -> dict | None:
    """Recognize UI exports and preserve any explicitly assigned membership."""
    if Path(path).suffix.lower() != ".csv":
        return None
    with open(path, "rb") as handle:
        prefix = handle.read(128).removeprefix(b"\xef\xbb\xbf")
    if not prefix.startswith((EXPORT_HEADERS[0] + "," + EXPORT_HEADERS[1] + ",").encode()):
        return None
    if Path(path).stat().st_size > MAX_FILE_BYTES:
        raise WorkspaceError("保存版本超过支持的文件大小。")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader, [])
        if tuple(headers[:2]) != EXPORT_HEADERS:
            return None
        rows = list(reader)
    if not rows or len(rows) > MAX_ROWS or len(rows) * len(headers) > MAX_ELEMENTS:
        raise WorkspaceError("保存版本为空或超过支持的规模。")
    targets = [i for i, name in enumerate(headers) if name.startswith("target_")]
    group_column = headers.index("_nir_workspace_group") if "_nir_workspace_group" in headers else None
    parsing_column = headers.index("_nir_workspace_parsing_only") if "_nir_workspace_parsing_only" in headers else None
    if parsing_column is not None and any(row[parsing_column] != "1" or row[1] != "unassigned" for row in rows):
        raise WorkspaceError("仅解析版本不能覆盖已指定的样本用途。")
    features = [i for i in range(2, len(headers)) if i not in targets and i not in {group_column, parsing_column}]
    if not features or any(len(row) != len(headers) for row in rows):
        raise WorkspaceError("保存版本的数据列不完整。")
    usage = np.asarray([row[1] for row in rows])
    if any(value not in USAGES for value in usage):
        raise WorkspaceError("保存版本包含无效的样本用途。")
    groups = [row[group_column] for row in rows] if group_column is not None else None
    if groups:
        _validate_group_roles(groups, [USAGES.index(value) for value in usage])
    index_axis = [headers[i].startswith("index:") for i in features]
    if any(index_axis) and not all(index_axis):
        raise WorkspaceError("保存版本的光谱坐标类型不一致。")
    return {
        "X": np.asarray([[float(row[i]) for i in features] for row in rows]),
        "targets": np.asarray([[row[i] for i in targets] for row in rows]),
        "target_names": [headers[i] for i in targets],
        "wv": None if all(index_axis) else np.asarray([float(headers[i]) for i in features]),
        "usage": usage,
        "groups": np.asarray(groups, dtype=str) if groups else None,
        "names": [row[0] for row in rows],
        "parsing_only": parsing_column is not None,
    }


def has_fixed_partitions(data: dict) -> bool:
    """Any explicit role, including exclusion, prevents automatic repartitioning."""
    return bool(np.any(data["usage"] != "unassigned"))


def fixed_regression_partitions(data: dict) -> dict:
    if data["targets"].ndim != 2 or data["targets"].shape[1] != 1:
        raise WorkspaceError("定量分析需要选择一个目标值列。")
    if not has_fixed_partitions(data):
        result = {"X": data["X"], "y": np.asarray(data["targets"][:, 0], dtype=float)}
        if data["wv"] is not None:
            result["wv"] = data["wv"]
        if data.get("groups") is not None:
            result["groups"] = data["groups"]
        return result
    if np.any(data["usage"] == "unassigned"):
        raise WorkspaceError("还有待分配的样本，请先分配用途或标为不参与。")
    result = {"wv": data["wv"]} if data["wv"] is not None else {}
    for part, role in (("cal", "train"), ("val", "validation"), ("test", "test")):
        mask = data["usage"] == role
        if not mask.any():
            raise WorkspaceError("请为训练集、验证集和测试集分别分配样本。")
        result["X_" + part], result["y_" + part] = data["X"][mask], np.asarray(data["targets"][mask, 0], dtype=float)
        if data.get("groups") is not None:
            result["groups_" + part] = data["groups"][mask]
    return result


def _validate_group_roles(groups: list[str], usages: list[int]) -> None:
    seen = {}
    for group, code in zip(groups, usages, strict=True):
        if code not in {1, 2, 3}:
            continue
        if not group.strip():
            raise WorkspaceError("分组信息缺失，请先补齐原文件中的批次或重复组。")
        if seen.setdefault(group, code) != code:
            raise WorkspaceError("同一批次或重复组的样本不能同时用于训练、验证或测试，请一起调整这些样本的用途。")
