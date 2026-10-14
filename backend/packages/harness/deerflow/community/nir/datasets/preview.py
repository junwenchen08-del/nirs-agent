"""Bounded dataset summaries for the owner's library UI, never model evidence."""

from __future__ import annotations

import struct
import warnings
import zlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from nir_core.io.loaders import load_mat
from nir_core.io.schema import infer_csv_mapping, parse_number, profile_csv

MAX_FILE_BYTES = 64 * 1024**2
MAX_SAMPLES = 512
MAX_POINTS = 256
_LABEL_NAMES = {"class", "classes", "label", "labels", "category", "species", "cultivar", "variety", "品种", "类别", "分类", "等级"}


def _empty(message: str, task: str = "unknown") -> dict:
    return {
        "status": "needs_mapping",
        "message": message,
        "analysis_type": task,
        "sample_count": None,
        "feature_count": None,
        "preview_sample_count": 0,
        "missing_ratio": None,
        "axis_kind": "index",
        "spectrum": [],
        "targets": [],
        "fields": [],
    }


def _number(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def _target(values: np.ndarray, name: str, categorical: bool) -> dict:
    if categorical:
        counts = Counter(str(value) for value in values if str(value).strip() and str(value).lower() not in {"nan", "none", "na"})
        top = counts.most_common(12)
        if len(counts) > 12:
            top.append(("其他类别", sum(counts.values()) - sum(count for _, count in top)))
        return {"name": name[:100], "kind": "classification", "bars": [{"label": label[:80], "count": count} for label, count in top]}
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not numeric.size:
        return {"name": name[:100], "kind": "regression", "bars": []}
    counts, edges = np.histogram(numeric, bins=min(10, max(1, int(np.sqrt(numeric.size)))))
    return {"name": name[:100], "kind": "regression", "bars": [{"label": f"{edges[i]:.4g}–{edges[i + 1]:.4g}", "count": int(count)} for i, count in enumerate(counts)]}


def build_preview(path: Path, profile: dict | None = None) -> dict:
    """Read a verified immutable source; return aggregates, not individual spectra."""
    profile = profile or {}
    task = {"calibration": "regression", "classification": "classification"}.get(profile.get("task_type"), "unknown")
    mapping = profile.get("mapping", {})
    if path.stat().st_size > MAX_FILE_BYTES:
        return _empty("文件较大，暂不生成图表。仍可将文件挂载到对话中分析。", task)
    fields: list[str] = []
    targets: list[tuple[str, np.ndarray, bool]] = []
    try:
        if path.suffix.lower() == ".mat":
            hdf5 = _check_mat_size(path)
            allowed = {key: mapping[key] for key in ("x_var", "y_var", "wv_var", "subset", "transpose") if key in mapping}
            if mapping.get("library_parse", {}).get("format") == "mat":
                from .parsing import mapped_mat_data

                data = mapped_mat_data(path, mapping["library_parse"])
            elif hdf5:
                # Real MATLAB 7.3 files put their HDF5 signature after a
                # userblock. Reuse the core reader after bounded preflight.
                from nir_core.io.loaders import _load_mat_v73

                data = _load_mat_v73(str(path), allowed.get("x_var"), allowed.get("y_var"), allowed.get("wv_var"), allowed.get("subset"), allowed.get("transpose", False))
            else:
                data = load_mat(str(path), **allowed)
            X, axis = data.X, data.wv
            sample_count, feature_count = X.shape
            rows = np.linspace(0, sample_count - 1, min(sample_count, MAX_SAMPLES), dtype=int)
            X = X[rows]
            if data.y is not None:
                y = np.asarray(data.y)
                if y.ndim == 1:
                    y = y[:, None]
                for i in range(min(y.shape[1], 6)):
                    name = data.y_names[i] if data.y_names and i < len(data.y_names) else f"目标 {i + 1}"
                    categorical = task == "classification" or name.lower() in _LABEL_NAMES
                    targets.append((name, y[rows, i], categorical))
        else:
            table = profile_csv(str(path), max_rows=MAX_SAMPLES + 1)
            from .parsing import apply_header

            table = apply_header(table, mapping.get("has_header"))
            # Text classification headers are outside the numeric training
            # inference contract. Interpret recognized labels only for display.
            if "has_header" not in mapping and not table.has_header and table.rows and any(cell.lower().strip() in _LABEL_NAMES for cell in table.rows[0]):
                headers, rows = table.rows[0], table.rows[1:]
                fractions = [sum(parse_number(row[i], str(table.dialect["decimal"])) is not None for row in rows) / max(1, len(rows)) for i in range(table.n_columns)]
                table = replace(table, headers=headers, rows=rows, has_header=True, total_rows=table.total_rows - 1, mapping=infer_csv_mapping(headers, rows, fractions, str(table.dialect["decimal"]), True))
            roles = {**table.mapping, **mapping}
            columns = roles.get("spectral_columns", [])
            target_columns = roles.get("target_columns", [])
            label_columns = [i for i, name in enumerate(table.headers) if name.lower().strip() in _LABEL_NAMES]
            target_columns = target_columns if mapping.get("library_parse") else list(dict.fromkeys([*target_columns, *label_columns]))
            if roles.get("status") == "needs_user_mapping" and not mapping:
                result = _empty("暂时无法确定哪些列是光谱和目标值，请先确认解析设置。", task)
                result["fields"] = [name[:120] for name in table.headers[:30]]
                return result
            if not columns or any(type(i) is not int or i < 0 or i >= table.n_columns for i in [*columns, *target_columns]):
                return _empty("暂时无法确定光谱列，请先确认解析设置。", task)
            columns = [i for i in columns if i not in target_columns]
            if len(table.rows) * len(columns) > 5_000_000:
                return _empty("数据结构较大，暂不生成图表。仍可挂载到对话分析。", task)
            decimal = str(table.dialect["decimal"])
            X = np.asarray([[parse_number(row[i], decimal) if parse_number(row[i], decimal) is not None else np.nan for i in columns] for row in table.rows], dtype=float)
            axis = np.asarray(roles["wavelengths"], dtype=float) if roles.get("wavelengths") else None
            if roles.get("transpose"):
                X = X.T
                axis_column = roles.get("wavelength_column")
                axis = np.asarray([parse_number(row[axis_column], decimal) for row in table.rows], dtype=float) if type(axis_column) is int else None
                sample_count, feature_count = len(columns), table.total_rows
            else:
                sample_count, feature_count = table.total_rows, len(columns)
                for i in target_columns[:6]:
                    categorical = task == "classification" or i in label_columns
                    values = np.asarray([row[i] if categorical else (parse_number(row[i], decimal) if parse_number(row[i], decimal) is not None else np.nan) for row in table.rows])
                    targets.append((table.headers[i], values, categorical))
            fields = [name[:120] for name in table.headers[:30]]
        if X.ndim != 2 or min(X.shape) == 0 or X.size > 5_000_000:
            return _empty("数据结构较大或不完整，暂不生成预览。", task)
        rows = np.linspace(0, X.shape[0] - 1, min(X.shape[0], MAX_SAMPLES), dtype=int)
        indices = np.linspace(0, X.shape[1] - 1, min(X.shape[1], MAX_POINTS), dtype=int)
        summary = X[np.ix_(rows, indices)]
        summary = np.where(np.isfinite(summary), summary, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, low, high = np.nanmean(summary, axis=0), np.nanmin(summary, axis=0), np.nanmax(summary, axis=0)
        axis_ok = axis is not None and len(axis) == X.shape[1] and np.all(np.isfinite(axis))
        spectrum = [{"x": float(axis[i]) if axis_ok else int(i + 1), "mean": _number(mean[j]), "low": _number(low[j]), "high": _number(high[j])} for j, i in enumerate(indices)]
        if task == "unknown" and targets:
            kinds = {categorical for _, _, categorical in targets}
            if len(kinds) == 1:
                task = "classification" if True in kinds else "regression"
        return {
            "status": "ready",
            "message": "图表为数据概览，基于预览样本的汇总统计。",
            "analysis_type": task,
            "sample_count": int(sample_count),
            "feature_count": int(feature_count),
            "preview_sample_count": len(rows),
            "missing_ratio": float(np.mean(~np.isfinite(summary))),
            "axis_kind": "axis" if axis_ok else "index",
            "spectrum": spectrum,
            "targets": [_target(values, name, categorical) for name, values, categorical in targets],
            "fields": fields,
        }
    except (ValueError, TypeError, IndexError, KeyError, OSError, RuntimeError):
        return _empty("当前文件的结构无法直接预览，请在解析设置中确认光谱矩阵和目标列。", task)


def _check_mat_size(path: Path) -> bool:
    """Reject oversized numeric MAT allocations and HDF5 external references."""
    try:
        import h5py

        hdf5 = h5py.is_hdf5(path)
    except ImportError:
        hdf5 = False
    if not hdf5:
        _check_mat_expansion(path)
        from scipy.io import whosmat

        if sum(int(np.prod(shape)) for _, shape, kind in whosmat(path) if kind not in {"struct", "cell", "char"}) > 5_000_000:
            raise ValueError("Preview allocation limit exceeded")
        return False
    with h5py.File(path, "r") as root:
        visited = set()
        elements = 0

        def inspect(group):
            nonlocal elements
            if group.id in visited:
                return
            visited.add(group.id)
            for name in group:
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise ValueError("External or symbolic HDF5 references are not previewed")
                node = group[name]
                if isinstance(node, h5py.Group):
                    inspect(node)
                else:
                    elements += int(np.prod(node.shape))
                    if elements > 5_000_000:
                        raise ValueError("Preview allocation limit exceeded")

        inspect(root)
    return True


def _check_mat_expansion(path: Path) -> None:
    """Bound MAT v5 decompression before allocating nested struct arrays."""
    with path.open("rb") as handle:
        header = handle.read(128)
        if len(header) != 128 or header[-2:] not in {b"IM", b"MI"}:
            return  # MAT v4 is preflighted by whosmat.
        endian = "<" if header[-2:] == b"IM" else ">"
        total = 0
        while tag := handle.read(8):
            if len(tag) != 8:
                raise ValueError("Invalid MAT element")
            kind, size = struct.unpack(endian + "II", tag)
            if kind != 15:  # miCOMPRESSED
                total += size
                handle.seek(size + (-size % 8), 1)
            else:
                decoder = zlib.decompressobj()
                remaining = size
                while remaining:
                    chunk = handle.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("Truncated MAT element")
                    remaining -= len(chunk)
                    while chunk:
                        total += len(decoder.decompress(chunk, 1024 * 1024))
                        if total > MAX_FILE_BYTES:
                            raise ValueError("MAT expanded size limit exceeded")
                        chunk = decoder.unconsumed_tail
            if total > MAX_FILE_BYTES:
                raise ValueError("MAT expanded size limit exceeded")
