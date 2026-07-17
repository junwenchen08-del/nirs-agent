"""Format and structure sniffers for spectral files.

These helpers inspect files without fully loading them, enabling the agent
to preview data shape/quality before committing to a load strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MATLAB v5/v7 files start with a 124-byte text header followed by a version
# word. The first ~116 bytes are human-readable text. HDF5 (v7.3) files use
# the 8-byte magic number below.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_MAT_V5_MAGIC_OFFSET = 124  # version word offset for v5 .mat


def _read_magic(filepath: str, n_bytes: int = 128) -> bytes:
    """Read the first ``n_bytes`` of a file (best effort)."""
    with open(filepath, "rb") as fh:
        return fh.read(n_bytes)


def detect_format(filepath: str) -> str:
    """Detect the on-disk format of a spectral file.

    Detection priority:
      1. HDF5 magic bytes -> ``"mat"`` (MATLAB v7.3).
      2. MATLAB v5/v7 text header pattern -> ``"mat"``.
      3. File extension ``.mat`` -> ``"mat"``.
      4. File extension ``.csv`` -> ``"csv"``.
      5. File extension ``.txt`` -> ``"txt"``.
      6. Otherwise fall back to extension (lowercased, no dot) or ``"txt"``.

    Args:
        filepath: Path to the file to inspect.

    Returns:
        One of ``"mat"``, ``"csv"``, ``"txt"``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = Path(filepath).suffix.lower()
    # Quick path for common extensions before touching the file.
    if suffix in (".csv", ".txt"):
        return suffix.lstrip(".")
    if suffix == ".mat":
        # Distinguish v7.3 (HDF5) from v5/v7 via magic bytes.
        try:
            magic = _read_magic(filepath, n_bytes=8)
        except OSError:
            return "mat"
        if magic == _HDF5_MAGIC:
            return "mat"
        return "mat"

    # Unknown extension: sniff by magic bytes.
    try:
        magic = _read_magic(filepath, n_bytes=8)
    except OSError:
        magic = b""
    if magic == _HDF5_MAGIC:
        return "mat"
    return "txt"


def detect_structure(X: np.ndarray) -> str:
    """Infer whether samples are stored in rows or columns.

    Heuristic:
      - Let ``ratio = max(n_rows, n_cols) / min(n_rows, n_cols)``.
      - If ``n_rows > n_cols`` and ``ratio > 1.5`` -> ``"samples_in_rows"``.
      - If ``n_cols > n_rows`` and ``ratio > 1.5`` -> ``"samples_in_columns"``.
      - Otherwise default to ``"samples_in_rows"`` (NIR convention).

    Args:
        X: A 2D array.

    Returns:
        ``"samples_in_rows"`` or ``"samples_in_columns"``.

    Raises:
        ValueError: If ``X`` is not 2D.
    """
    if X.ndim != 2:
        raise ValueError(
            f"detect_structure expects a 2D array, got shape {X.shape!r}"
        )
    n_rows, n_cols = X.shape
    if n_rows == 0 or n_cols == 0:
        return "samples_in_rows"
    lo, hi = (n_rows, n_cols) if n_rows <= n_cols else (n_cols, n_rows)
    ratio = hi / lo
    if n_rows > n_cols and ratio > 1.5:
        return "samples_in_rows"
    if n_cols > n_rows and ratio > 1.5:
        return "samples_in_columns"
    return "samples_in_rows"


def inspect_file(filepath: str) -> str:
    """Produce a JSON preview of a spectral file without fully loading it.

    Reads only what is necessary to populate the preview fields:
      - ``format``: detected format ("mat" | "csv" | "txt").
      - ``shape``: ``[n_rows, n_cols]`` of the data block.
      - ``estimated_samples``: best-guess sample count.
      - ``estimated_wavelengths``: best-guess wavelength count.
      - ``structure``: ``"samples_in_rows"`` | ``"samples_in_columns"``.
      - ``value_range``: ``[min, max]`` of the data (or ``null`` if unknown).
      - ``has_nan``: whether NaNs were detected.

    For ``.mat`` files, only the first candidate 2D variable is inspected
    (the heuristic used by :func:`nir_core.io.loaders.load_mat`). For CSV/TXT,
    the file is parsed with numpy but only shape/statistics are reported.

    Args:
        filepath: Path to the file.

    Returns:
        A JSON string (``json.dumps(..., ensure_ascii=False)``).

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file cannot be parsed at all.
    """
    fmt = detect_format(filepath)

    if fmt == "mat":
        info = _inspect_mat(filepath)
    else:
        info = _inspect_csv_like(filepath)

    info["format"] = fmt
    return json.dumps(info, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal inspectors
# ---------------------------------------------------------------------------

def _inspect_csv_like(filepath: str) -> dict:
    """Inspect a CSV/TXT file by parsing it with numpy (header-sniffed).

    Only the first 200 data rows are loaded for statistics; total row count
    is obtained by line-counting to avoid loading very large files entirely.
    """
    delimiter = _sniff_delimiter(filepath)
    has_header = _sniff_header(filepath, delimiter)

    # Count total data rows without parsing floats (cheap line scan).
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        total_rows = sum(1 for line in fh if line.strip())
    if has_header:
        total_rows -= 1
    if total_rows < 0:
        total_rows = 0

    # Load at most 200 rows for shape/statistics preview.
    try:
        preview = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=1 if has_header else 0,
            dtype=float,
            max_rows=200,
        )
    except TypeError:
        # Older numpy without max_rows — fall back to full load.
        preview = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=1 if has_header else 0,
            dtype=float,
        )
    if preview.ndim == 1:
        preview = preview.reshape(1, -1)
    elif preview.ndim == 0:
        raise ValueError(f"Could not parse a 2D data block from {filepath!r}")

    n_cols = preview.shape[1]
    n_data_rows = total_rows if total_rows > 0 else preview.shape[0]

    structure = detect_structure(preview)
    if structure == "samples_in_rows":
        n_samples, n_wavelengths = n_data_rows, n_cols
    else:
        n_wavelengths, n_samples = n_cols, n_data_rows

    finite = preview[np.isfinite(preview)]
    if finite.size == 0:
        value_range: list[float] | None = None
    else:
        value_range = [float(finite.min()), float(finite.max())]

    # ★ v3.8: Detect non-numeric metadata columns. When a column is almost
    # entirely NaN in the preview (which is how genfromtxt signals non-numeric
    # strings like "Set"/"Season"/"Cultivar"), it's a metadata column that
    # would leak into X as NaN. Surface these so the agent can pass x_cols to
    # nir_load_data to skip them.
    non_numeric_cols: list[int] = []
    if preview.shape[0] > 1 and preview.shape[1] > 1:
        n_rows = preview.shape[0]
        for col_idx in range(preview.shape[1]):
            col = preview[:, col_idx]
            nan_count = int(np.sum(~np.isfinite(col)))
            # A column is "non-numeric" if >90% of its cells are NaN (not
            # just missing values — true string columns are ~100% NaN).
            # Use a high threshold so sparse missing-value columns aren't
            # flagged as metadata.
            if nan_count / n_rows > 0.9:
                non_numeric_cols.append(int(col_idx))

    return {
        "shape": [int(n_data_rows), int(n_cols)],
        "estimated_samples": int(n_samples),
        "estimated_wavelengths": int(n_wavelengths),
        "structure": structure,
        "value_range": value_range,
        "has_nan": bool(np.isnan(preview).any()),
        # The presence of a NaN at the (0, 0) corner is the strongest single
        # signal that the file uses the row-label + column-label layout
        # (empty corner where row label meets column label). The agent uses
        # this to decide whether to expect y / wv auto-separation.
        "corner_is_nan": bool(
            preview.shape[0] > 0 and preview.shape[1] > 0 and not np.isfinite(preview[0, 0])
        ),
        # ★ v3.8: Columns that are almost entirely NaN (= non-numeric string
        # columns like Set/Season/Region/Cultivar). The agent should pass
        # x_cols to nir_load_data to skip them — otherwise they leak into X
        # as NaN and break PLS/SVR. Empty list means no metadata columns
        # detected (or n_rows <= 1, in which case detection is unreliable).
        "non_numeric_columns": non_numeric_cols,
    }


def _inspect_mat(filepath: str) -> dict:
    """Inspect a .mat file by locating its first 2D numeric variable.

    ★ v3.7: When the file contains a MATLAB struct, the inspector returns
    struct metadata (available subsets, fields) instead of crashing on
    ``np.isfinite`` for object arrays. This lets ``nir_inspect`` guide the
    agent to call ``nir_load_data(subset=...)`` correctly.
    """
    # Try HDF5 first; on failure fall back to scipy.io.loadmat (v5/v7).
    if _is_hdf5(filepath):
        return _inspect_mat73(filepath)
    return _inspect_mat_v5(filepath)


def _is_hdf5(filepath: str) -> bool:
    """Return True if the file has the HDF5 magic bytes."""
    try:
        return _read_magic(filepath, n_bytes=8) == _HDF5_MAGIC
    except OSError:
        return False


def _inspect_mat_v5(filepath: str) -> dict:
    """Inspect a v5/v7 .mat file via scipy.io.loadmat.

    Detects MATLAB structs (mat_struct / object arrays) and delegates to
    :func:`_summarize_struct` when one is found. Falls back to the legacy
    "pick first 2D array" path for flat files.
    """
    from scipy.io import loadmat

    raw = loadmat(filepath, squeeze_me=False, struct_as_record=False)
    candidates: dict[str, object] = {
        k: v for k, v in raw.items() if not k.startswith("__")
    }

    # Detect a top-level struct wrapper. scipy returns mat_struct objects
    # inside a (1,1) object array when squeeze_me=False; with squeeze_me=True
    # they come back as bare mat_struct. We use duck-typing (_fieldnames) to
    # avoid importing mat_struct.
    struct_obj = _find_top_struct(candidates)
    if struct_obj is not None:
        return _summarize_struct(struct_obj)

    # Flat file: pick the largest 2D numeric array (legacy behaviour).
    flat: dict[str, np.ndarray] = {}
    for k, v in candidates.items():
        arr = np.asarray(v)
        if arr.ndim >= 1:
            flat[k] = arr
    arr = _pick_first_2d(flat)
    if arr is None:
        raise ValueError(
            f"No suitable 2D numeric variable found in {filepath!r}"
        )
    return _summarize_array(arr)


def _inspect_mat73(filepath: str) -> dict:
    """Inspect a v7.3 (HDF5) .mat file via h5py (lazy import)."""
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            "Reading MATLAB v7.3 (HDF5) files requires the 'h5py' package, "
            f"but it could not be imported: {exc!s}. Install it via "
            "`pip install h5py`."
        ) from exc

    with h5py.File(filepath, "r") as fh:
        # Detect HDF5 groups (MATLAB structs in v7.3 files).
        groups = {k: fh[k] for k in fh.keys() if isinstance(fh[k], h5py.Group)}
        if groups:
            # Pick the top-level group with the most leaf datasets.
            top_name = max(groups, key=lambda k: len(groups[k].keys()))
            struct_dict = {
                sub: np.asarray(groups[top_name][sub][...])
                for sub in groups[top_name].keys()
            }
            return _summarize_struct(struct_dict)

        # Flat file: pick the largest 2D dataset.
        flat: dict[str, np.ndarray] = {}
        for key in fh.keys():
            try:
                flat[key] = np.asarray(fh[key][...])
            except Exception:
                continue
        arr = _pick_first_2d(flat)
        if arr is None:
            raise ValueError(
                f"No suitable 2D numeric variable found in {filepath!r}"
            )
        return _summarize_array(arr)


def _find_top_struct(candidates: dict[str, object]) -> object | None:
    """Find a top-level MATLAB struct among loadmat candidates.

    scipy.io.loadmat with ``struct_as_record=False`` returns either a bare
    ``mat_struct`` instance (squeeze_me=True) or a ``(1,1) object`` array
    wrapping one (squeeze_me=False). We detect both via the ``_fieldnames``
    attribute without importing mat_struct (keeps the dependency lazy).
    """
    for value in candidates.values():
        # Unwrap (1,1) object arrays.
        if isinstance(value, np.ndarray) and value.dtype == object:
            if value.size == 1:
                value = value.flat[0]
        if hasattr(value, "_fieldnames") and isinstance(getattr(value, "_fieldnames"), list):
            return value
    return None


def _summarize_struct(struct: object) -> dict:
    """Build an inspect_file info dict for a MATLAB struct.

    Returns a dict with the struct's fields categorised as ``subsets``
    (nested structs, e.g. R562/R568), ``x_fields`` (2D arrays matching X*
    pattern), ``y_fields`` (1D arrays matching Y* pattern), and
    ``wv_fields`` (1D arrays matching wn*/wv* pattern). The standard
    ``shape`` / ``estimated_samples`` / ``estimated_wavelengths`` fields
    are populated from the first detected X block so the agent gets a
    preview of the data scale.

    When the struct contains nested sub-structs, the summary is built from
    the first sub-struct (so the agent sees the per-subset shape), and the
    full list of available subsets is returned in ``available_subsets``.
    """
    fields: dict[str, object] = dict(_iter_struct_fields(struct))

    # Categorise fields.
    sub_structs: dict[str, object] = {}
    x_fields: list[str] = []
    y_fields: list[str] = []
    wv_fields: list[str] = []
    arrays: dict[str, np.ndarray] = {}

    for name, value in fields.items():
        if hasattr(value, "_fieldnames") or isinstance(value, dict):
            sub_structs[name] = value
            continue
        arr = np.asarray(value)
        if arr.ndim < 1:
            continue
        arrays[name] = arr
        lname = name.lower()
        if arr.ndim == 2 and _matches_any(lname, ("x", "spectra", "absorbance", "data")):
            x_fields.append(name)
        elif arr.ndim == 1:
            if _matches_any(lname, ("y", "ref", "reference", "target", "label")):
                y_fields.append(name)
            elif _matches_any(lname, ("wn", "wv", "wavelength", "lambda", "wave")):
                wv_fields.append(name)

    info: dict = {
        "is_struct": True,
        "struct_fields": sorted(fields.keys()),
        "x_fields": sorted(x_fields),
        "y_fields": sorted(y_fields),
        "wv_fields": sorted(wv_fields),
    }

    if sub_structs:
        info["available_subsets"] = sorted(sub_structs.keys())
        # Inspect the first sub-struct to report its shape.
        first_name = sorted(sub_structs.keys())[0]
        first = sub_structs[first_name]
        first_fields = dict(_iter_struct_fields(first))
        first_arrays = {
            k: np.asarray(v) for k, v in first_fields.items()
            if np.asarray(v).ndim >= 1
        }
        # Look for X1/X2/... in the sub-struct; concatenate column counts.
        sub_x_names = sorted(
            n for n, v in first_arrays.items()
            if v.ndim == 2 and _matches_any(n.lower(), ("x", "spectra", "absorbance", "data"))
        )
        if sub_x_names:
            total_cols = sum(first_arrays[n].shape[1] for n in sub_x_names)
            total_rows = first_arrays[sub_x_names[0]].shape[0]
            info["shape"] = [int(total_rows), int(total_cols)]
            info["estimated_samples"] = int(total_rows)
            info["estimated_wavelengths"] = int(total_cols)
            info["structure"] = "samples_in_rows"
            info["x_fields"] = sub_x_names
            info["subset_x_fields"] = {first_name: sub_x_names}
            # Collect y / wv from the sub-struct too. Ravel 2D (1, n) arrays
            # to 1D so we catch Y stored as a row vector (scipy with
            # squeeze_me=False keeps the extra dimension).
            sub_y = []
            for n, v in first_arrays.items():
                v1 = np.asarray(v).ravel()
                if v1.ndim == 1 and v1.shape[0] == total_rows and _matches_any(n.lower(), ("y", "ref", "reference", "target", "label")):
                    sub_y.append(n)
            sub_wv = [
                n for n, v in arrays.items()
                if np.asarray(v).ravel().ndim == 1 and _matches_any(n.lower(), ("wn", "wv", "wavelength", "lambda", "wave"))
            ]
            info["y_fields"] = sub_y
            info["wv_fields"] = sub_wv
            # value_range from the first X block (sample to keep preview cheap).
            first_x = first_arrays[sub_x_names[0]]
            finite = first_x[np.isfinite(first_x)]
            info["value_range"] = (
                [float(finite.min()), float(finite.max())]
                if finite.size > 0 else None
            )
            info["has_nan"] = bool(np.isnan(first_x).any())
        else:
            info["shape"] = None
            info["estimated_samples"] = None
            info["estimated_wavelengths"] = None
            info["structure"] = "struct"
            info["value_range"] = None
            info["has_nan"] = False
        return info

    # Flat struct with X/Y/wn siblings (no nested sub-structs).
    if x_fields:
        # Use the concatenated column count when multiple X blocks exist.
        total_cols = sum(arrays[n].shape[1] for n in x_fields)
        total_rows = arrays[x_fields[0]].shape[0]
        info["shape"] = [int(total_rows), int(total_cols)]
        info["estimated_samples"] = int(total_rows)
        info["estimated_wavelengths"] = int(total_cols)
        info["structure"] = "samples_in_rows"
        first_x = arrays[x_fields[0]]
        finite = first_x[np.isfinite(first_x)]
        info["value_range"] = (
            [float(finite.min()), float(finite.max())]
            if finite.size > 0 else None
        )
        info["has_nan"] = bool(np.isnan(first_x).any())
    else:
        # No X-named field; fall back to the largest 2D array if any.
        two_d = {k: v for k, v in arrays.items() if v.ndim == 2}
        if two_d:
            arr = max(two_d.values(), key=lambda a: a.size)
            summary = _summarize_array(arr)
            info.update(summary)
        else:
            info["shape"] = None
            info["estimated_samples"] = None
            info["estimated_wavelengths"] = None
            info["structure"] = "struct"
            info["value_range"] = None
            info["has_nan"] = False
    return info


def _iter_struct_fields(struct: object):
    """Yield (field_name, value) pairs from a MATLAB struct (v5 or dict).

    ★ v3.7: When scipy.io.loadmat is called with ``squeeze_me=False``, nested
    structs are returned as ``(1,1) object`` arrays wrapping a ``mat_struct``.
    We transparently unwrap such wrappers so downstream code sees the actual
    struct / array.
    """
    if hasattr(struct, "_fieldnames"):
        for name in struct._fieldnames:
            value = getattr(struct, name)
            # Unwrap (1,1) object arrays containing a mat_struct (nested struct
            # returned by scipy with squeeze_me=False).
            if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
                inner = value.flat[0]
                if hasattr(inner, "_fieldnames"):
                    value = inner
            yield name, value
        return
    if isinstance(struct, dict):
        yield from struct.items()
        return


def _matches_any(name_lower: str, prefixes: tuple[str, ...]) -> bool:
    """Return True if *name_lower* starts with any of *prefixes*."""
    return any(name_lower.startswith(p) for p in prefixes)


def _pick_first_2d(candidates: dict[str, np.ndarray]) -> np.ndarray | None:
    """Pick the first variable that is 2D (prefer larger arrays)."""
    two_d = {k: v for k, v in candidates.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    if not two_d:
        return None
    # Prefer the array with the most elements (heuristic for the spectra block).
    best_key = max(two_d, key=lambda k: two_d[k].size)
    return two_d[best_key]


def _summarize_array(arr: np.ndarray) -> dict:
    """Build the inspect_file info dict for a 2D array."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    structure = detect_structure(arr)
    if structure == "samples_in_rows":
        n_samples, n_wavelengths = arr.shape
    else:
        n_wavelengths, n_samples = arr.shape
    finite = arr[np.isfinite(arr)]
    value_range = (
        [float(finite.min()), float(finite.max())]
        if finite.size > 0
        else None
    )
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "estimated_samples": int(n_samples),
        "estimated_wavelengths": int(n_wavelengths),
        "structure": structure,
        "value_range": value_range,
        "has_nan": bool(np.isnan(arr).any()),
    }


# ---------------------------------------------------------------------------
# CSV/TXT sniffing helpers (also used by loaders.py)
# ---------------------------------------------------------------------------

def _sniff_delimiter(filepath: str) -> str:
    """Sniff the most likely delimiter for a CSV/TXT file.

    Tries ``,`` ``\\t`` ``;`` and whitespace; picks the one yielding the
    most columns on the first non-empty line.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip() == "":
                continue
            counts = {
                d: (line.count(d) if d != "whitespace" else len(line.split()))
                for d in [",", "\t", ";", "whitespace"]
            }
            best = max(counts, key=counts.get)
            if best == "whitespace":
                return None  # numpy treats None as whitespace
            return best
    return ","


def _sniff_header(filepath: str, delimiter: str | None) -> bool:
    """Return True if the first non-empty row is mostly non-numeric.

    A single leading label cell (e.g. an empty corner cell or a 'y' tag in a
    wavelength row) does NOT count as a header: such rows are commonly used
    to carry wavelength metadata alongside numeric data. Only when more than
    half of the cells fail to parse as floats do we treat the row as a true
    text header.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip() == "":
                continue
            cells = (
                line.split() if delimiter is None else line.split(delimiter)
            )
            non_empty = [c.strip() for c in cells if c.strip() != ""]
            if not non_empty:
                continue
            non_numeric = 0
            for cell in non_empty:
                try:
                    float(cell)
                except ValueError:
                    non_numeric += 1
            return non_numeric * 2 > len(non_empty)
    return False
