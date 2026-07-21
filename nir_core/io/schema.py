"""Deterministic schema inference for heterogeneous CSV and MATLAB files.

The helpers in this module never execute user code.  They expose the evidence
used to map a source file onto the canonical ``X`` / ``y`` / ``wv`` contract so
callers can auto-load high-confidence layouts and request clarification for
ambiguous ones.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TARGET_TOKENS = (
    "active",
    "api",
    "assay",
    "content",
    "concentration",
    "target",
    "reference",
    "moisture",
    "protein",
    "fat",
    "sugar",
    "brix",
    "starch",
    "nitrogen",
    "dry matter",
    "dry_matter",
    "dm",
    "ssc",
    "soluble solids",
    "ph",
)
_SAMPLE_ID_TOKENS = (
    "sample",
    "sample_id",
    "sampleid",
    "specimen",
    "object",
    "objlabel",
    "id",
    "name",
)
_AXIS_TOKENS = (
    "wavelength",
    "wavenumber",
    "wave",
    "lambda",
    "axis",
    "wn",
    "wv",
    "nm",
    "cm-1",
    "cm^-1",
)
_X_TOKENS = (
    "x",
    "spectra",
    "spectrum",
    "signal",
    "absorbance",
    "reflectance",
    "nir",
    "data",
)
_Y_TOKENS = (
    "y",
    "ref",
    "reference",
    "target",
    "label",
    "chemistry",
    "assay",
    "content",
)
_MISSING = {"", "na", "n/a", "nan", "null", "none", "-"}


def detect_text_encoding(filepath: str) -> str:
    """Return a practical encoding for a delimited text file."""
    raw = Path(filepath).read_bytes()[:131072]
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for encoding in ("utf-8", "gb18030"):
        try:
            raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return encoding
    return "latin-1"


def _delimiter_score(lines: list[str], delimiter: str) -> tuple[float, int]:
    counts: list[int] = []
    for line in lines:
        if delimiter == "whitespace":
            counts.append(len(line.split()))
        else:
            counts.append(len(next(csv.reader([line], delimiter=delimiter))))
    if not counts:
        return 0.0, 0
    values, frequencies = np.unique(counts, return_counts=True)
    best_index = int(np.argmax(frequencies))
    width = int(values[best_index])
    consistency = float(frequencies[best_index]) / len(counts)
    return (width * consistency if width > 1 else 0.0), width


def sniff_csv_dialect(filepath: str) -> dict[str, str | None]:
    """Infer encoding, delimiter, and decimal mark from several rows."""
    encoding = detect_text_encoding(filepath)
    with open(filepath, encoding=encoding, errors="replace") as handle:
        lines = [line.rstrip("\r\n") for line in handle if line.strip()][:30]
    candidates = [",", ";", "\t", "|", "whitespace"]
    scored = [
        (candidate, *_delimiter_score(lines, candidate)) for candidate in candidates
    ]
    delimiter, score, _width = max(scored, key=lambda item: item[1])
    if score <= 0:
        delimiter = "whitespace"

    comma_decimals = 0
    dot_decimals = 0
    for line in lines[1:]:
        cells = (
            line.split()
            if delimiter == "whitespace"
            else next(csv.reader([line], delimiter=delimiter))
        )
        comma_decimals += sum(
            bool(re.fullmatch(r"[+-]?\d+,\d+(?:[eE][+-]?\d+)?", cell.strip()))
            for cell in cells
        )
        dot_decimals += sum(
            bool(re.fullmatch(r"[+-]?\d+\.\d+(?:[eE][+-]?\d+)?", cell.strip()))
            for cell in cells
        )
    decimal = "," if delimiter != "," and comma_decimals > dot_decimals else "."
    return {
        "encoding": encoding,
        "delimiter": None if delimiter == "whitespace" else delimiter,
        "decimal": decimal,
    }


def parse_number(value: str, decimal: str = ".") -> float | None:
    """Parse one cell using the inferred decimal mark."""
    text = value.strip().replace("\u2212", "-")
    if text.lower() in _MISSING:
        return None
    if decimal == ",":
        text = text.replace(".", "").replace(",", ".")
    try:
        result = float(text)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def _axis_header_value(value: str) -> float | None:
    text = value.strip().lower()
    direct = parse_number(text)
    if direct is not None:
        return direct
    match = re.search(
        r"(?<![a-z])([0-9]{3,5}(?:\.[0-9]+)?)(?:\s*(?:nm|cm-?1|cm\^-1))?", text
    )
    return float(match.group(1)) if match else None


def _token_match(name: str, tokens: tuple[str, ...]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return any(
        normalized == token
        or normalized.startswith(f"{token}_")
        or f"_{token}_" in f"_{normalized}_"
        for token in tokens
    )


@dataclass
class CSVProfile:
    dialect: dict[str, str | None]
    headers: list[str]
    rows: list[list[str]]
    has_header: bool
    total_rows: int
    n_columns: int
    mapping: dict[str, Any]


def _read_delimited_rows(
    filepath: str, dialect: dict[str, str | None], max_rows: int | None
) -> list[list[str]]:
    delimiter = dialect["delimiter"]
    encoding = str(dialect["encoding"])
    rows: list[list[str]] = []
    with open(filepath, encoding=encoding, errors="replace", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = (
                line.split()
                if delimiter is None
                else next(csv.reader([line], delimiter=str(delimiter)))
            )
            rows.append([cell.strip() for cell in row])
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _infer_has_header(rows: list[list[str]], decimal: str) -> bool:
    if len(rows) < 2:
        return False
    first, rest = rows[0], rows[1 : min(len(rows), 12)]
    if not first:
        return False
    for index, cell in enumerate(first):
        if not cell:
            continue
        if _token_match(cell, _TARGET_TOKENS + _SAMPLE_ID_TOKENS + _AXIS_TOKENS):
            return True
        if parse_number(cell, decimal) is None:
            below = [row[index] for row in rest if index < len(row)]
            if (
                below
                and sum(parse_number(value, decimal) is not None for value in below)
                / len(below)
                >= 0.6
            ):
                return True
    return False


def profile_csv(filepath: str, *, max_rows: int | None = 201) -> CSVProfile:
    """Profile a delimited file and infer column roles with confidence."""
    dialect = sniff_csv_dialect(filepath)
    rows = _read_delimited_rows(filepath, dialect, max_rows)
    if not rows:
        raise ValueError(f"File {filepath!r} is empty.")
    n_columns = max(len(row) for row in rows)
    decimal = str(dialect["decimal"])
    has_header = _infer_has_header(rows, decimal)
    headers = (
        rows[0] + [""] * (n_columns - len(rows[0]))
        if has_header
        else [f"column_{i}" for i in range(n_columns)]
    )
    data_rows = rows[1:] if has_header else rows
    padded = [row + [""] * (n_columns - len(row)) for row in data_rows]
    numeric_fractions: list[float] = []
    for index in range(n_columns):
        values = [row[index] for row in padded]
        present = [value for value in values if value.strip().lower() not in _MISSING]
        numeric = sum(parse_number(value, decimal) is not None for value in present)
        numeric_fractions.append(numeric / len(present) if present else 0.0)

    mapping = infer_csv_mapping(headers, padded, numeric_fractions, decimal, has_header)
    total_rows = len(data_rows)
    if max_rows is not None and len(rows) >= max_rows:
        with open(
            filepath, encoding=str(dialect["encoding"]), errors="replace"
        ) as handle:
            total_rows = sum(bool(line.strip()) for line in handle) - int(has_header)
    return CSVProfile(
        dialect, headers, padded, has_header, total_rows, n_columns, mapping
    )


def infer_csv_mapping(
    headers: list[str],
    rows: list[list[str]],
    numeric_fractions: list[float],
    decimal: str,
    has_header: bool,
) -> dict[str, Any]:
    """Infer spectra, target, sample-id, and orientation from a CSV preview."""
    numeric_columns = [
        index for index, fraction in enumerate(numeric_fractions) if fraction >= 0.9
    ]
    sample_ids = [
        index
        for index, name in enumerate(headers)
        if _token_match(name, _SAMPLE_ID_TOKENS) and index not in numeric_columns
    ]
    targets = [
        index
        for index, name in enumerate(headers)
        if _token_match(name, _TARGET_TOKENS) and index in numeric_columns
    ]
    axis_values = [_axis_header_value(name) for name in headers]
    spectral = [
        index
        for index, value in enumerate(axis_values)
        if value is not None and index in numeric_columns
    ]

    first_name = headers[0] if headers else ""
    first_values = [parse_number(row[0], decimal) for row in rows if row]
    first_numeric = bool(first_values) and all(
        value is not None for value in first_values
    )
    first_axis = _token_match(first_name, _AXIS_TOKENS)
    remaining_numeric = len(numeric_columns) >= max(2, len(headers) - 1) and all(
        index in numeric_columns for index in range(1, len(headers))
    )
    if first_axis and first_numeric and remaining_numeric:
        axis = np.asarray(first_values, dtype=float)
        monotonic = axis.size > 1 and (
            np.all(np.diff(axis) >= 0) or np.all(np.diff(axis) <= 0)
        )
        if monotonic:
            return {
                "status": "auto",
                "confidence": 0.96,
                "clarification_required": False,
                "transpose": True,
                "wavelength_column": 0,
                "spectral_columns": list(range(1, len(headers))),
                "target_columns": [],
                "sample_id_columns": [],
                "candidate_numeric_columns": numeric_columns,
                "evidence": [
                    "axis_named_first_column",
                    "monotonic_axis",
                    "numeric_sample_columns",
                ],
            }

    if has_header and len(spectral) >= 3:
        unassigned_numeric = [
            index
            for index in numeric_columns
            if index not in spectral and index not in targets
        ]
        if unassigned_numeric and not targets:
            return {
                "status": "needs_user_mapping",
                "confidence": 0.65,
                "clarification_required": True,
                "transpose": False,
                "spectral_columns": spectral,
                "target_columns": [],
                "sample_id_columns": sample_ids[:1],
                "candidate_numeric_columns": numeric_columns,
                "unassigned_numeric_columns": unassigned_numeric,
                "evidence": ["numeric_spectral_headers", "unassigned_numeric_columns"],
            }
        confidence = 0.96 if targets else 0.88
        return {
            "status": "auto",
            "confidence": confidence,
            "clarification_required": False,
            "transpose": False,
            "spectral_columns": spectral,
            "target_columns": targets,
            "sample_id_columns": sample_ids[:1],
            "wavelengths": [float(axis_values[index]) for index in spectral],
            "candidate_numeric_columns": numeric_columns,
            "evidence": [
                "numeric_spectral_headers",
                *(["named_target"] if targets else []),
            ],
        }

    # Preserve the established empty-corner layout detector in loaders.py.
    if headers and headers[0] == "" and len(spectral) >= 3:
        return {
            "status": "legacy_auto_layout",
            "confidence": 0.9,
            "clarification_required": False,
            "transpose": False,
            "spectral_columns": spectral,
            "target_columns": [0],
            "sample_id_columns": [],
            "candidate_numeric_columns": numeric_columns,
            "evidence": ["empty_corner", "numeric_spectral_headers"],
        }

    if not has_header and len(numeric_columns) == len(headers):
        return {
            "status": "spectra_only",
            "confidence": 0.8,
            "clarification_required": False,
            "transpose": False,
            "spectral_columns": numeric_columns,
            "target_columns": [],
            "sample_id_columns": [],
            "candidate_numeric_columns": numeric_columns,
            "evidence": ["all_numeric_matrix"],
        }

    return {
        "status": "needs_user_mapping",
        "confidence": 0.5,
        "clarification_required": True,
        "transpose": False,
        "spectral_columns": spectral,
        "target_columns": targets,
        "sample_id_columns": sample_ids[:1],
        "candidate_numeric_columns": numeric_columns,
        "evidence": ["column_roles_ambiguous"],
    }


def csv_profile_numeric_matrix(profile: CSVProfile) -> np.ndarray:
    """Convert all profiled data cells to floats, using NaN for text/missing."""
    decimal = str(profile.dialect["decimal"])
    return np.asarray(
        [
            [
                np.nan if (value := parse_number(cell, decimal)) is None else value
                for cell in row
            ]
            for row in profile.rows
        ],
        dtype=float,
    )


def _unwrap_mat_value(value: object) -> object:
    current = value
    while (
        isinstance(current, np.ndarray)
        and current.dtype == object
        and current.size == 1
    ):
        current = current.reshape(-1)[0]
    return current


def flatten_mat_leaves(root: dict[str, object]) -> dict[str, np.ndarray]:
    """Recursively expose numeric MATLAB leaves under dotted paths."""
    leaves: dict[str, np.ndarray] = {}

    def visit(value: object, path: str) -> None:
        value = _unwrap_mat_value(value)
        if hasattr(value, "_fieldnames"):
            for name in value._fieldnames:
                visit(getattr(value, name), f"{path}.{name}" if path else name)
            return
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, f"{path}.{name}" if path else str(name))
            return
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and array.size:
            leaves[path] = np.squeeze(array)

    for name, value in root.items():
        visit(value, str(name))
    return leaves


def _name_score(path: str, tokens: tuple[str, ...]) -> float:
    leaf = path.rsplit(".", 1)[-1]
    return 1.0 if _token_match(leaf, tokens) else 0.0


def _monotonic_score(values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size < 3 or not np.isfinite(vector).all():
        return 0.0
    diffs = np.diff(vector)
    return 1.0 if np.all(diffs >= 0) or np.all(diffs <= 0) else 0.0


def infer_mat_mapping(root: dict[str, object]) -> dict[str, Any]:
    """Infer a unique MAT X/y/wv mapping using names and shape compatibility."""
    leaves = flatten_mat_leaves(root)
    matrices = {
        path: value
        for path, value in leaves.items()
        if value.ndim == 2 and min(value.shape) >= 2
    }
    vectors = {
        path: value.reshape(-1) for path, value in leaves.items() if value.ndim == 1
    }
    if not matrices:
        return {
            "status": "needs_user_mapping",
            "confidence": 0.0,
            "clarification_required": True,
            "x_candidates": [],
            "y_candidates": list(vectors),
            "wv_candidates": list(vectors),
            "evidence": ["no_numeric_2d_matrix"],
        }

    hypotheses: list[dict[str, Any]] = []
    for x_path, matrix in matrices.items():
        for transpose in (False, True):
            n_samples, n_features = (
                (matrix.shape[1], matrix.shape[0]) if transpose else matrix.shape
            )
            y_candidates = [
                path for path, value in vectors.items() if value.size == n_samples
            ]
            wv_candidates = [
                path for path, value in vectors.items() if value.size == n_features
            ]
            y_path = max(
                y_candidates,
                key=lambda path: (
                    _name_score(path, _Y_TOKENS),
                    -list(vectors).index(path),
                ),
                default=None,
            )
            wv_path = max(
                wv_candidates,
                key=lambda path: (
                    _name_score(path, _AXIS_TOKENS),
                    _monotonic_score(vectors[path]),
                    -list(vectors).index(path),
                ),
                default=None,
            )
            score = 0.45 + 0.12 * _name_score(x_path, _X_TOKENS)
            evidence = ["numeric_2d_matrix"]
            if y_path is not None:
                score += 0.13 + 0.05 * _name_score(y_path, _Y_TOKENS)
                evidence.append("sample_length_vector")
            if wv_path is not None:
                score += 0.13 + 0.07 * max(
                    _name_score(wv_path, _AXIS_TOKENS),
                    _monotonic_score(vectors[wv_path]),
                )
                evidence.append("feature_length_axis")
            if n_features >= n_samples and (y_path is not None or wv_path is not None):
                score += 0.05
                evidence.append("nir_wide_matrix")
            hypotheses.append(
                {
                    "score": min(score, 0.99),
                    "x_variable": x_path,
                    "y_variable": y_path,
                    "wv_variable": wv_path,
                    "transpose": transpose,
                    "shape": [int(n_samples), int(n_features)],
                    "evidence": evidence,
                }
            )

    hypotheses.sort(key=lambda item: item["score"], reverse=True)
    best = hypotheses[0]
    tied_x = sorted(
        {
            item["x_variable"]
            for item in hypotheses
            if abs(item["score"] - best["score"]) < 1e-9
        }
    )
    orientation_tie = any(
        item["x_variable"] == best["x_variable"]
        and item["transpose"] != best["transpose"]
        and abs(item["score"] - best["score"]) < 1e-9
        for item in hypotheses
    )
    ambiguous = len(tied_x) > 1
    spectra_only_orientation = orientation_tie and not ambiguous
    status = (
        "needs_user_mapping"
        if ambiguous
        else ("spectra_only" if spectra_only_orientation else "auto")
    )
    return {
        "status": status,
        "confidence": round(
            0.5
            if ambiguous
            else (0.65 if spectra_only_orientation else float(best["score"])),
            3,
        ),
        "clarification_required": ambiguous,
        "x_variable": None if ambiguous else best["x_variable"],
        "y_variable": None if ambiguous else best["y_variable"],
        "wv_variable": None if ambiguous else best["wv_variable"],
        "transpose": False if ambiguous else best["transpose"],
        "shape": best["shape"],
        "x_candidates": tied_x if ambiguous else list(matrices),
        "y_candidates": [
            path
            for path in vectors
            if vectors[path].size
            in {matrix.shape[0] for matrix in matrices.values()}
            | {matrix.shape[1] for matrix in matrices.values()}
        ],
        "wv_candidates": [
            path for path in vectors if _monotonic_score(vectors[path]) > 0
        ],
        "evidence": (
            ["equal_matrix_candidates"]
            if ambiguous
            else (
                ["single_matrix", "samples_in_rows_convention"]
                if spectra_only_orientation
                else best["evidence"]
            )
        ),
    }


def load_inferred_mat_arrays(
    root: dict[str, object], mapping: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Materialize arrays for a high-confidence mapping returned above."""
    if mapping.get("status") != "auto" or not mapping.get("x_variable"):
        raise ValueError(
            "MAT field mapping is ambiguous; inspect the file and provide x_var/y_var/wv_var explicitly."
        )
    leaves = flatten_mat_leaves(root)
    X = np.asarray(leaves[str(mapping["x_variable"])], dtype=float)
    if mapping.get("transpose"):
        X = X.T
    y_path = mapping.get("y_variable")
    wv_path = mapping.get("wv_variable")
    y = np.asarray(leaves[str(y_path)], dtype=float).reshape(-1) if y_path else None
    wv = np.asarray(leaves[str(wv_path)], dtype=float).reshape(-1) if wv_path else None
    return X, y, wv
