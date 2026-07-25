"""Loaders for spectral files (.mat, .csv, .txt) into :class:`SpectralData`.

All loaders are pure functions: they read from disk and return a new
:class:`~nir_core.models.SpectralData` instance; no global state is mutated.
"""

from __future__ import annotations

import os
import re
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar, cast

import numpy as np

from nir_core.io.resources import ResourceBudget
from nir_core.io.schema import (
    csv_profile_numeric_matrix,
    infer_mat_mapping,
    load_inferred_mat_arrays,
    profile_csv,
)
from nir_core.io.sniffers import (
    _detect_labeled_matrix_layout,
    _sniff_delimiter,
    _sniff_header,
    detect_format,
)
from nir_core.models import SpectralData

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_Loader = TypeVar("_Loader", bound=Callable[..., SpectralData])


def _bounded_loaded_data(loader: _Loader) -> _Loader:
    """Apply post-parse matrix and memory limits to every public loader."""

    @wraps(loader)
    def wrapped(*args, **kwargs) -> SpectralData:
        data = loader(*args, **kwargs)
        y = np.asarray(data.y) if data.y is not None else None
        target_count = 0 if y is None else (1 if y.ndim <= 1 else int(y.shape[1]))
        ResourceBudget().check_array(
            np.asarray(data.X),
            target_count=target_count,
            peak_multiplier=3.0,
            stage="loaded_spectral_matrix",
        )
        return data

    return cast(_Loader, wrapped)


@_bounded_loaded_data
def load_mat(
    filepath: str,
    x_var: str | None = None,
    y_var: str | None = None,
    wv_var: str | None = None,
    subset: str | None = None,
    transpose: bool = False,
) -> SpectralData:
    """Load a MATLAB ``.mat`` file into :class:`SpectralData`.

    Supports both v5/v7 (via :func:`scipy.io.loadmat`) and v7.3 / HDF5 (via
    ``h5py``, imported lazily). When variable names are not supplied, the
    loader applies a shape-based heuristic to identify ``X`` (2D matrix),
    ``y`` (1D, length == number of samples) and ``wv`` (1D, length == number
    of wavelengths).

    ★ v3.7: MATLAB struct (``object`` dtype) variables are now supported.
    Many public NIR datasets (e.g. Open-Nirs-Datasets) store data as nested
    structs:

    .. code-block:: text

        Melamine_Dataset (struct)
        ├── wn1: (225,)              ← wavelength vector for X1
        ├── wn2: (121,)              ← wavelength vector for X2
        ├── R562 (struct)            ← subset "R562"
        │   ├── X1: (3032, 225)
        │   ├── X2: (3032, 121)
        │   └── Y:  (3032,)
        ├── R568 (struct)            ← subset "R568"
        └── ...

    When the file contains a struct, the loader:

    1. Flattens the struct recursively, collecting 2D arrays (``X*``),
       1D arrays whose length matches the sample count (``Y`` / ``y*``), and
       1D arrays whose length matches the wavelength count (``wn*`` / ``wv*``).
    2. If multiple nested sub-structs exist (like R562 / R568 / R861 / R862),
       picks one via the ``subset`` parameter. When ``subset`` is None and
       multiple candidates exist, raises ``ValueError`` listing the available
       subsets so the agent can re-call with ``subset=...``.
    3. Concatenates multiple ``X*`` blocks column-wise when their sample
       counts match (e.g. X1 from 600-1100nm + X2 from 1100-2500nm), and
       concatenates the corresponding ``wn*`` vectors.

    Args:
        filepath: Path to the ``.mat`` file.
        x_var: Optional explicit variable name for the spectra matrix.
        y_var: Optional explicit variable name for reference values.
        wv_var: Optional explicit variable name for wavelengths.
        subset: Optional name of a nested sub-struct to load (e.g.
            ``"R562"``). Only used when the top-level variable is a struct
            with multiple sub-structs. When ``None`` and the struct has
            exactly one sub-struct, that one is used automatically.

    Returns:
        A :class:`SpectralData` with ``original_format="mat"``.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If the file cannot be parsed, required variables are
            missing / malformed, or ``subset`` is required but not given.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    ResourceBudget().check_file(filepath, stage="mat_file_preflight")

    # Detect HDF5 magic to decide between loadmat and h5py.
    with open(filepath, "rb") as fh:
        magic = fh.read(8)
    if magic == b"\x89HDF\r\n\x1a\n":
        return _load_mat_v73(filepath, x_var, y_var, wv_var, subset, transpose)
    return _load_mat_v5(filepath, x_var, y_var, wv_var, subset, transpose)


@_bounded_loaded_data
def load_csv(
    filepath: str,
    delimiter: str | None = None,
    x_cols: str | None = None,
    y_col: int | None = None,
    wv_row: int | None = None,
    auto_layout: bool = True,
    y_cols: list[int] | None = None,
) -> SpectralData:
    """Load a CSV/TXT spectral file into :class:`SpectralData`.

    The delimiter is auto-detected (``\\t`` ``,`` ``;`` or whitespace) when
    not supplied. A header row is detected when the first non-empty row
    contains non-numeric cells.

    Layout assumptions (when explicit selectors are not given):
      - The matrix body is numeric.
      - Wavelengths may live in the first row (``wv_row=0``).
      - Reference values may live in the first column (``y_col=0``).
      - Otherwise the whole numeric block is treated as ``X``.

    When ``auto_layout=True`` (the default) and the caller did NOT pass
    explicit ``y_col`` / ``wv_row``, the loader inspects the top-left corner
    and the first row / first column to detect a common NIR-style layout:

        ,<wv_1>,<wv_2>,...,<wv_n>
        <y_1>,<a_1_1>,<a_1_2>,...,<a_1_n>
        <y_2>,<a_2_1>,<a_2_2>,...,<a_2_n>
        ...

    Detection signals (all must hold):
      - Cell (0, 0) is empty / NaN (the "corner" where row label meets
        column label).
      - The first row (excluding the corner) is fully numeric with values in
        the typical NIR wavelength range ``[600, 3000] nm`` (or any strictly
        increasing positive sequence).
      - The first column (excluding the corner) is fully numeric and its
        value range differs significantly from the inner block (typical
        reference values such as moisture 0-100% vs. absorbance 0-3).

    If any check fails, the loader falls back to treating the whole block as
    ``X`` with no ``y`` and no ``wv`` (preserves prior behavior). Pass
    ``auto_layout=False`` to disable detection entirely.

    Args:
        filepath: Path to the CSV/TXT file.
        delimiter: Optional explicit delimiter (``","``, ``"\\t"`` ...).
            ``None`` triggers auto-detection.
        x_cols: ★ v3.8 Column selector for the spectra block ``X``, used to
            skip metadata columns. Supports slice-like syntax:
            ``"8:"`` (col 8 to end), ``"8:314"`` (half-open), ``":8"``,
            ``"8,9,10"`` (explicit list), or ``":"`` (all). When ``y_col``
            is inside the selected range it is automatically excluded from
            ``X``. When omitted, all columns except ``y_col`` are kept
            (preserves prior behavior — which is problematic for files with
            leading metadata columns like the Anderson 2020 mango dataset
            where cols 0-7 are Set/Season/Region/...).
        y_col: Index of one reference-value column. ``None`` disables y.
        y_cols: Explicit reference-value columns for multi-component data.
            Mutually exclusive with ``y_col``. Multi-component loading is
            intentionally explicit and is not auto-detected.
        wv_row: Index of the wavelength row. ``None`` disables wv.
        auto_layout: When True and ``y_col``/``wv_row`` are not given, try
            to auto-detect a row-label + column-label layout from the
            top-left corner.

    Returns:
        A :class:`SpectralData` with ``original_format="csv"``.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If the file cannot be parsed as a numeric table.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    ResourceBudget().check_file(filepath, stage="csv_file_preflight")
    if y_col is not None and y_cols is not None:
        raise ValueError("y_col and y_cols are mutually exclusive")

    if (
        auto_layout
        and y_col is None
        and y_cols is None
        and wv_row is None
        and x_cols is None
    ):
        profile = profile_csv(filepath, max_rows=201)
        if profile.mapping["status"] == "auto":
            return _load_inferred_csv_profile(profile, filepath)

    if delimiter is None:
        delimiter = _sniff_delimiter(filepath)
    has_header = _sniff_header(filepath, delimiter)
    header_candidate = (
        _read_header_cells(filepath, delimiter) if y_cols is not None else []
    )
    if not has_header and y_cols is not None:
        selected_headers = [
            header_candidate[index]
            for index in y_cols
            if 0 <= index < len(header_candidate)
        ]
        if len(selected_headers) == len(y_cols) and all(
            value.strip() and not _is_float_text(value) for value in selected_headers
        ):
            has_header = True
    header_cells = (
        header_candidate
        if has_header and header_candidate
        else (_read_header_cells(filepath, delimiter) if has_header else [])
    )

    arr = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=1 if has_header else 0,
        dtype=float,
    )
    if arr.ndim == 0:
        raise ValueError(f"File {filepath!r} does not contain a data table.")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Auto-detect row-label + column-label layout when caller didn't pin
    # y_col / wv_row explicitly. See docstring for the heuristic.
    if auto_layout and y_col is None and y_cols is None and wv_row is None:
        detected_wv_row, detected_y_col = _detect_labeled_layout(arr)
        wv_row = detected_wv_row
        y_col = detected_y_col

    header_wv: np.ndarray | None = None
    if has_header and header_cells:
        feature_indices = _resolve_csv_feature_indices(
            len(header_cells),
            y_col=y_col,
            y_cols=y_cols,
            x_cols=x_cols,
        )
        try:
            candidate = np.asarray(
                [float(header_cells[index]) for index in feature_indices],
                dtype=float,
            )
        except (ValueError, IndexError):
            candidate = np.empty(0, dtype=float)
        if candidate.size == len(feature_indices) and np.isfinite(candidate).all():
            header_wv = candidate

    # A mixed text/numeric first row (target names followed by wavelengths)
    # is consumed as a header above. Treat an explicit wv_row=0 as referring
    # to that raw first row rather than deleting the first sample after it.
    split_wv_row = None if header_wv is not None and wv_row == 0 else wv_row
    X, y, wv = _split_csv_block(
        arr,
        y_col=y_col,
        y_cols=y_cols,
        wv_row=split_wv_row,
        x_cols=x_cols,
    )
    if header_wv is not None:
        wv = header_wv

    y_names = None
    if y_cols is not None:
        y_names = [
            header_cells[index].strip()
            if index < len(header_cells) and header_cells[index].strip()
            else f"y{i}"
            for i, index in enumerate(y_cols)
        ]

    return SpectralData(
        X=X,
        y=y,
        y_names=y_names,
        wv=wv,
        sample_names=[],
        source_file=str(filepath),
        original_format="csv",
    )


def _load_inferred_csv_profile(profile, filepath: str) -> SpectralData:
    """Materialize a high-confidence CSV schema profile."""
    mapping = profile.mapping
    sample_id_columns = [int(index) for index in mapping["sample_id_columns"]]
    if (
        profile.total_rows > len(profile.rows)
        and profile.dialect["decimal"] == "."
        and not mapping.get("transpose")
        and not sample_id_columns
    ):
        feature_columns = [int(index) for index in mapping["spectral_columns"]]
        target_columns = [int(index) for index in mapping["target_columns"]]
        selected_columns = sorted(set(feature_columns + target_columns))
        selected = np.genfromtxt(
            filepath,
            delimiter=profile.dialect["delimiter"],
            skip_header=1 if profile.has_header else 0,
            usecols=selected_columns,
            dtype=float,
            encoding=profile.dialect["encoding"],
        )
        if selected.ndim == 1:
            selected = selected.reshape(1, -1)
        local_index = {
            original: index for index, original in enumerate(selected_columns)
        }
        X = selected[:, [local_index[index] for index in feature_columns]]
        y: np.ndarray | None = None
        y_names: list[str] | None = None
        if target_columns:
            y_matrix = selected[:, [local_index[index] for index in target_columns]]
            y = y_matrix[:, 0] if len(target_columns) == 1 else y_matrix
            y_names = [profile.headers[index] for index in target_columns]
        return SpectralData(
            X=X,
            y=y,
            y_names=y_names,
            wv=np.asarray(mapping.get("wavelengths"), dtype=float),
            sample_names=[],
            source_file=str(filepath),
            original_format="csv",
        )

    if profile.total_rows > len(profile.rows):
        profile = profile_csv(filepath, max_rows=None)
        mapping = profile.mapping
        sample_id_columns = [int(index) for index in mapping["sample_id_columns"]]
    matrix = csv_profile_numeric_matrix(profile)
    if mapping.get("transpose"):
        wavelength_column = int(mapping["wavelength_column"])
        feature_columns = [int(index) for index in mapping["spectral_columns"]]
        X = matrix[:, feature_columns].T
        wv = matrix[:, wavelength_column]
        return SpectralData(
            X=X,
            y=None,
            wv=wv,
            sample_names=[profile.headers[index] for index in feature_columns],
            source_file=str(filepath),
            original_format="csv",
        )

    feature_columns = [int(index) for index in mapping["spectral_columns"]]
    target_columns = [int(index) for index in mapping["target_columns"]]
    X = matrix[:, feature_columns]
    y: np.ndarray | None = None
    y_names: list[str] | None = None
    if target_columns:
        y_matrix = matrix[:, target_columns]
        y = y_matrix[:, 0] if len(target_columns) == 1 else y_matrix
        y_names = [profile.headers[index] for index in target_columns]
    sample_names = (
        [row[sample_id_columns[0]] for row in profile.rows] if sample_id_columns else []
    )
    return SpectralData(
        X=X,
        y=y,
        y_names=y_names,
        wv=np.asarray(mapping.get("wavelengths"), dtype=float),
        sample_names=sample_names,
        source_file=str(filepath),
        original_format="csv",
    )


def _read_header_cells(filepath: str, delimiter: str | None) -> list[str]:
    """Read the first non-empty header row without changing numeric parsing."""
    with open(filepath, encoding="utf-8-sig") as file:
        line = next(
            (candidate.strip("\r\n") for candidate in file if candidate.strip()), ""
        )
    if not line:
        return []
    if delimiter is None or delimiter.isspace():
        return re.split(r"\s+", line.strip())
    import csv

    return next(csv.reader([line], delimiter=delimiter))


def _is_float_text(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# NIR wavelength range covers visible-NIR (~400 nm) to mid-IR (~25000 nm),
# but practical instruments used for chemometrics (NIR / FT-NIR) almost
# always report wavelengths in roughly 600-3000 nm. We use this as a soft
# signal for "this row looks like wavelengths, not spectra".
_NIR_WAVELENGTH_MIN_NM = _env_float("NIR_WAVELENGTH_MIN_NM", 600.0)
_NIR_WAVELENGTH_MAX_NM = _env_float("NIR_WAVELENGTH_MAX_NM", 3000.0)


def _detect_labeled_layout(
    arr: np.ndarray,
) -> tuple[int | None, int | None]:
    """Detect a labelled NIR-style layout from the top-left corner of ``arr``.

    Returns ``(wv_row, y_col)`` to feed into :func:`_split_csv_block`. Returns
    ``(None, None)`` when the signals are inconclusive — callers should then
    treat the whole block as ``X``.
    """
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return None, None

    # Signal 1: the corner cell (0, 0) must be NaN. CSV with a leading empty
    # cell in the first row is the canonical "label x wavelength" layout.
    if np.isfinite(arr[0, 0]):
        return None, None

    # Signal 2: the wavelength row (rest of row 0) is all finite,
    # monotonically non-decreasing (within floating-point tolerance),
    # and in the NIR range.
    wv_row_values = arr[0, 1:]
    if not np.isfinite(wv_row_values).all():
        return None, None
    wv_min, wv_max = float(wv_row_values.min()), float(wv_row_values.max())
    if not (
        wv_min >= _NIR_WAVELENGTH_MIN_NM * 0.5  # half-baked safety margin
        and wv_max <= _NIR_WAVELENGTH_MAX_NM * 1.5
    ):
        # Not in NIR wavelength territory. Could be wavenumbers, frequency,
        # or just a numeric header. Refuse to guess.
        return None, None
    # Monotonic non-decreasing check (small tolerance for FP noise).
    diffs = np.diff(wv_row_values.astype(float))
    if not (diffs >= -1e-6).all():
        return None, None

    # Signal 3: the y column (rest of column 0) is all finite. The inner
    # block must also have at least one finite value so the spectra block
    # is non-degenerate.
    y_col_values = arr[1:, 0]
    if not np.isfinite(y_col_values).all():
        return None, None
    inner_block = arr[1:, 1:]
    inner_finite = inner_block[np.isfinite(inner_block)]
    if inner_finite.size == 0:
        return None, None

    # When signals 1 (corner NaN) and 2 (NIR wavelength row) are already
    # confident, the first column is almost certainly the reference-value
    # column: typical NIR reference values (moisture %, pH, protein, oil
    # content, …) never fall in the NIR wavelength band themselves. Only
    # reject when the first column's values look like NIR wavelengths,
    # which would indicate a samples-in-columns layout where the first
    # column is wavelengths rather than reference values.
    #
    # The previous range-ratio heuristic (rejecting when y and spectra had
    # similar dynamic ranges, ratio in [0.3, 3.0]) silently misclassified
    # legitimate layouts where reference values and absorbances share a
    # similar numeric scale — e.g. pH 6–8 (range 2.0) vs absorbance 0.2–1.5
    # (range 1.3), ratio ≈ 1.54, was wrongly rejected and y got merged
    # into X with data.y = None.
    y_min, y_max = float(y_col_values.min()), float(y_col_values.max())
    if y_min >= _NIR_WAVELENGTH_MIN_NM * 0.5 and y_max <= _NIR_WAVELENGTH_MAX_NM * 1.5:
        return None, None

    return 0, 0


def auto_detect_and_load(filepath: str, *, x_cols: str | None = None) -> SpectralData:
    """Detect the file format by extension/magic and dispatch to a loader.

    ``.mat`` -> :func:`load_mat`; ``.csv`` and ``.txt`` (and anything else)
    -> :func:`load_csv`.

    Args:
        filepath: Path to the file.
        x_cols: ★ v3.8 Optional column selector forwarded to ``load_csv``
            when the file is CSV/TXT. Ignored for .mat. See
            :func:`load_csv` for syntax.

    Returns:
        A :class:`SpectralData` with ``original_format`` set to ``"mat"``
        or ``"csv"``/``"txt"`` depending on the detected format.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If parsing fails.
    """
    fmt = detect_format(filepath)
    if fmt == "mat":
        return load_mat(filepath)
    data = load_csv(filepath, x_cols=x_cols)
    if fmt == "txt":
        # load_csv tags everything as "csv"; correct the format tag for
        # .txt files so downstream metadata is accurate.
        data = data.model_copy(update={"original_format": "txt"})
    return data


# ---------------------------------------------------------------------------
# MAT loading internals
# ---------------------------------------------------------------------------


def _load_mat_v5(
    filepath: str,
    x_var: str | None,
    y_var: str | None,
    wv_var: str | None,
    subset: str | None,
    transpose: bool,
) -> SpectralData:
    """Load a v5/v7 .mat via scipy.io.loadmat."""
    from scipy.io import loadmat

    raw = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    # Drop MATLAB metadata keys. Keep everything else as-is (including
    # mat_struct objects — they are unwrapped later by _flatten_mat_struct).
    variables = {k: v for k, v in raw.items() if not k.startswith("__")}
    if (
        x_var is None
        and y_var is None
        and wv_var is None
        and subset is None
        and not transpose
    ):
        labeled = _load_labeled_matrix_bundle(variables, filepath)
        if labeled is not None:
            return labeled
        mapping = infer_mat_mapping(variables)
        if mapping["status"] == "auto":
            X, y, wv = load_inferred_mat_arrays(variables, mapping)
            y_path = mapping.get("y_variable")
            return SpectralData(
                X=X,
                y=y,
                y_names=[str(y_path).rsplit(".", 1)[-1]] if y_path else None,
                wv=wv,
                sample_names=[],
                source_file=str(filepath),
                original_format="mat",
            )
        x_candidates = [str(name) for name in mapping.get("x_candidates", [])]
        if x_candidates and not all(
            _name_matches(name.rsplit(".", 1)[-1].lower(), _X_NAME_RE)
            for name in x_candidates
        ):
            raise ValueError(
                "MAT spectra mapping is ambiguous. Candidate matrices: "
                f"{x_candidates}. Inspect the file and provide x_var/y_var/wv_var."
            )
    X, y, wv = _resolve_mat_variables(
        variables, x_var, y_var, wv_var, subset, transpose
    )
    return SpectralData(
        X=X,
        y=y,
        wv=wv,
        sample_names=[],
        source_file=str(filepath),
        original_format="mat",
    )


def _load_labeled_matrix_bundle(
    variables: dict[str, object],
    filepath: str,
) -> SpectralData | None:
    """Load a detected PLS-Toolbox labeled matrix without metadata leakage."""
    layout = _detect_labeled_matrix_layout(variables)
    if layout is None:
        return None

    matrix = np.asarray(variables[layout["matrix_variable"]], dtype=float)
    X = matrix[:, layout["spectral_indices"]]
    target_indices = layout["target_indices"]
    y: np.ndarray | None = None
    y_names: list[str] | None = None
    if target_indices:
        y_matrix = matrix[:, target_indices]
        y = y_matrix[:, 0] if len(target_indices) == 1 else y_matrix
        y_names = [layout["labels"][index] for index in target_indices]

    return SpectralData(
        X=X,
        y=y,
        y_names=y_names,
        wv=np.asarray(layout["wavelengths"], dtype=float),
        sample_names=layout["sample_names"],
        source_file=str(filepath),
        original_format="mat",
    )


def _load_mat_v73(
    filepath: str,
    x_var: str | None,
    y_var: str | None,
    wv_var: str | None,
    subset: str | None,
    transpose: bool,
) -> SpectralData:
    """Load a v7.3 (HDF5) .mat via h5py (lazy import)."""
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            f"Reading MATLAB v7.3 (HDF5) files requires the 'h5py' package, but it could not be imported: {exc!s}. Install it via `pip install h5py` (optional dependency of nir-core)."
        ) from exc

    def read_node(node):
        if isinstance(node, h5py.Group):
            return {name: read_node(node[name]) for name in node.keys()}
        return np.asarray(node[...])

    with h5py.File(filepath, "r") as fh:
        root: dict[str, object] = {
            name: read_node(fh[name]) for name in fh.keys() if name != "#refs#"
        }

    if (
        x_var is None
        and y_var is None
        and wv_var is None
        and subset is None
        and not transpose
    ):
        mapping = infer_mat_mapping(root)
        if mapping["status"] == "auto":
            X, y, wv = load_inferred_mat_arrays(root, mapping)
            y_path = mapping.get("y_variable")
            return SpectralData(
                X=X,
                y=y,
                y_names=[str(y_path).rsplit(".", 1)[-1]] if y_path else None,
                wv=wv,
                sample_names=[],
                source_file=str(filepath),
                original_format="mat",
            )

    X, y, wv = _resolve_mat_variables(root, x_var, y_var, wv_var, subset, transpose)
    return SpectralData(
        X=X,
        y=y,
        wv=wv,
        sample_names=[],
        source_file=str(filepath),
        original_format="mat",
    )


# Recognised attribute / key name patterns for struct fields.
_X_NAME_RE = ("x", "spectra", "absorbance", "data")  # prefix match, case-insensitive
_Y_NAME_RE = ("y", "ref", "reference", "target", "label")  # prefix match
_WV_NAME_RE = ("wn", "wv", "wavelength", "lambda", "wave")  # prefix match


def _is_struct_like(value: object) -> bool:
    """Return True if *value* looks like a MATLAB struct (v5 or v7.3)."""
    # scipy.io.loadmat with struct_as_record=False produces mat_struct
    # instances (or object ndarrays containing them). We check by attribute
    # name rather than importing mat_struct to keep the dependency lazy.
    if hasattr(value, "_fieldnames") and isinstance(
        getattr(value, "_fieldnames"), list
    ):
        return True
    # h5py groups are already converted to dicts by _load_mat_v73; a dict
    # of mixed dict/ndarray values is also treated as a struct.
    if isinstance(value, dict):
        return any(isinstance(v, dict) or np.ndarray is type(v) for v in value.values())
    return False


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
            if (
                isinstance(value, np.ndarray)
                and value.dtype == object
                and value.size == 1
            ):
                inner = value.flat[0]
                if hasattr(inner, "_fieldnames"):
                    value = inner
            yield name, value
        return
    if isinstance(struct, dict):
        yield from struct.items()
        return


def _flatten_mat_struct(
    root: dict[str, object],
    subset: str | None,
) -> tuple[dict[str, np.ndarray], str | None]:
    """Flatten a MATLAB struct into a flat ``{name: ndarray}`` dict.

    Handles two patterns:

    1. **Flat struct with X / Y / wn siblings** — e.g. ``{X: (n,p), Y: (n,), wn: (p,)}``.
       Returns the dict as-is after unwrapping the outer struct.
    2. **Nested struct with multiple sub-structs** — e.g.
       ``{wn1, wn2, R562: {X1, X2, Y}, R568: {...}, ...}``.
       Picks the requested ``subset`` (or the only sub-struct when ``subset``
       is None) and merges its fields with the parent's wavelength vectors.

    Returns:
        (flat_variables, chosen_subset_name). The flat dict contains only
        ``np.ndarray`` values, with the chosen subset's fields lifted to the
        top level alongside any sibling wavelength arrays from the parent.

    Raises:
        ValueError: When multiple sub-structs exist and ``subset`` is None,
            or when ``subset`` is given but not found.
    """
    # Locate the outer struct (if any). A struct is a single top-level
    # entry that is either a mat_struct (has _fieldnames) or a dict
    # containing nested dicts/ndarrays.
    struct_keys = [k for k, v in root.items() if _is_struct_like(v)]
    non_struct_keys = [k for k, v in root.items() if k not in struct_keys]

    if not struct_keys:
        # No struct — everything is already flat ndarrays. Return as-is.
        flat: dict[str, np.ndarray] = {}
        for k, v in root.items():
            arr = np.asarray(v)
            if arr.ndim >= 1:
                flat[k] = arr
        return flat, None

    # If there's exactly one struct and no sibling numeric arrays, descend
    # into it and re-flatten. (This is the common "single top-level wrapper"
    # case, e.g. Melamine_Dataset.mat has one struct named Melamine_Dataset.)
    if len(struct_keys) == 1 and not non_struct_keys:
        outer_name = struct_keys[0]
        outer = root[outer_name]
        # Recurse: treat the struct's fields as a new root.
        inner_root = dict(_iter_struct_fields(outer))
        return _flatten_mat_struct(inner_root, subset)

    # We're now at a level where we have either:
    # (a) one struct + sibling numeric arrays (e.g. {wn1, wn2, R562(struct)})
    # (b) multiple sibling structs (e.g. {R562, R568, R861, R862})
    # (c) a mix of both.
    #
    # Collect sub-structs (candidate subsets) and sibling arrays (wavelength
    # vectors that should be merged with the chosen subset).
    sub_structs: dict[str, object] = {}
    sibling_arrays: dict[str, np.ndarray] = {}
    for k, v in root.items():
        if _is_struct_like(v):
            sub_structs[k] = v
        else:
            arr = np.asarray(v)
            if arr.ndim >= 1:
                sibling_arrays[k] = arr

    if not sub_structs:
        # All siblings are arrays → flat struct, return as-is.
        return sibling_arrays, None

    # Pick the subset.
    if len(sub_structs) == 1 and subset is None:
        chosen_name = next(iter(sub_structs))
    elif subset is not None:
        if subset not in sub_structs:
            raise ValueError(
                f"subset {subset!r} not found in MAT struct. Available subsets: {sorted(sub_structs)}"
            )
        chosen_name = subset
    else:
        # Multiple sub-structs and no subset → ambiguous, ask the caller.
        raise ValueError(
            f"MAT struct contains {len(sub_structs)} sub-structs ({sorted(sub_structs)}). Specify one via the `subset` parameter."
        )

    chosen = sub_structs[chosen_name]
    flat = dict(sibling_arrays)  # start with sibling wavelength vectors
    for name, value in _iter_struct_fields(chosen):
        arr = np.asarray(value)
        if arr.ndim >= 1:
            flat[name] = arr
    return flat, chosen_name


def _concat_multi_x(
    flat: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Concatenate multiple X* blocks column-wise when sample counts match.

    Datasets like Open-Nirs-Datasets store spectra in two wavelength ranges
    (e.g. X1: 600-1100nm, X2: 1100-2500nm). This helper:

    1. Identifies all 2D arrays whose names match the ``X*`` pattern.
    2. Verifies they share the same row count (sample count).
    3. Concatenates them column-wise in name order (X1, X2, X3, ...).
    4. Concatenates the matching ``wn*`` / ``wv*`` vectors in the same order.

    Returns:
        (X, wv, x_names_joined) where ``wv`` is None when no matching
        wavelength vector is found, and ``x_names_joined`` is a string like
        ``"X1+X2"`` describing the concatenation (for downstream reporting).

    When only one X* block exists, returns it unchanged with its matching wv.
    """
    x_names = sorted(
        n
        for n, v in flat.items()
        if isinstance(v, np.ndarray)
        and v.ndim == 2
        and _name_matches(n.lower(), _X_NAME_RE)
    )
    if not x_names:
        raise ValueError(
            f"No 2D X* array found in MAT struct. Available fields: {sorted(flat)}"
        )

    # Single X block — straightforward.
    if len(x_names) == 1:
        X = np.asarray(flat[x_names[0]], dtype=float)
        return X, None, x_names[0]

    # Multiple X blocks — verify sample counts match, then concat.
    sample_counts = {flat[n].shape[0] for n in x_names}
    if len(sample_counts) != 1:
        raise ValueError(
            f"X* blocks {x_names} have mismatched sample counts {sample_counts}; cannot concatenate."
        )
    X = np.concatenate([flat[n] for n in x_names], axis=1).astype(float)

    # Try to find matching wavelength vectors (wn1↔X1, wn2↔X2, ...).
    wv_pieces: list[np.ndarray] = []
    wv_complete = True
    for xname in x_names:
        # Extract trailing digits (X1 → "1", X2 → "2").
        suffix = "".join(c for c in xname if c.isdigit())
        wv_candidates: list[np.ndarray] = []
        for wv_pattern in _WV_NAME_RE:
            # wn1, wv1, wavelength1 ...
            key = f"{wv_pattern}{suffix}" if suffix else wv_pattern
            if key in flat:
                wv_candidates.append(np.asarray(flat[key]).ravel())
        if not wv_candidates:
            wv_complete = False
            break
        wv_pieces.append(wv_candidates[0])

    wv: np.ndarray | None = None
    if wv_complete and wv_pieces:
        wv = np.concatenate(wv_pieces).astype(float)
        # Trim or warn if length mismatch (defensive — should not happen
        # for well-formed datasets, but keeps the loader robust).
        if wv.shape[0] > X.shape[1]:
            wv = wv[: X.shape[1]]
        elif wv.shape[0] < X.shape[1]:
            import warnings

            warnings.warn(
                f"Concatenated wv length ({wv.shape[0]}) < X columns ({X.shape[1]}); wavelength labels may be incomplete.",
                UserWarning,
                stacklevel=2,
            )

    return X, wv, "+".join(x_names)


def _name_matches(name_lower: str, prefixes: tuple[str, ...]) -> bool:
    """Return True if *name_lower* starts with any of *prefixes*."""
    return any(name_lower.startswith(p) for p in prefixes)


def _resolve_mat_variables(
    variables: dict[str, object],
    x_var: str | None,
    y_var: str | None,
    wv_var: str | None,
    subset: str | None,
    transpose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Pick X/y/wv from a dict of MATLAB variables (now with struct support).

    ★ v3.7: When the dict contains struct-like values, they are flattened
    via :func:`_flatten_mat_struct` before the standard heuristic kicks in.
    """
    if not variables:
        raise ValueError("MAT file contains no variables.")

    # Check if any value is struct-like; if so, flatten first.
    has_struct = any(_is_struct_like(v) for v in variables.values())
    if has_struct:
        flat, chosen_subset = _flatten_mat_struct(variables, subset)
        # Replace variables with the flattened dict for the rest of the
        # resolution. (chosen_subset is reported via the SpectralData
        # source_file, not returned here — callers can inspect it.)
        variables = flat  # type: ignore[assignment]

    # If explicit x_var/y_var/wv_var are given, use the legacy path.
    if x_var is not None or y_var is not None or wv_var is not None or transpose:
        return _resolve_explicit(variables, x_var, y_var, wv_var, transpose)

    # Heuristic path. When multiple X* blocks exist (after struct flattening),
    # concatenate them; otherwise pick the largest 2D array.
    x_candidates = {
        k: v
        for k, v in variables.items()
        if isinstance(v, np.ndarray)
        and v.ndim == 2
        and _name_matches(k.lower(), _X_NAME_RE)
    }
    if len(x_candidates) >= 2:
        X, wv_from_x, _xname = _concat_multi_x(variables)  # type: ignore[arg-type]
        n_samples = X.shape[0]
        n_wavelengths = X.shape[1]
        y = _heuristic_y(variables, n_samples, exclude_name=None)
        # If _concat_multi_x already built a wv, prefer it; otherwise heuristic.
        if wv_from_x is not None:
            wv = wv_from_x
        else:
            wv = _heuristic_wv(variables, n_wavelengths)
        return X, y, wv

    # Single-X (or no X-named) path: fall back to the original heuristic.
    variables_nd = {k: np.asarray(v) for k, v in variables.items()}
    X = _heuristic_x(variables_nd)
    n_samples, n_wavelengths = X.shape
    y = _heuristic_y(variables_nd, n_samples, exclude_name=None)
    wv = _heuristic_wv(variables_nd, n_wavelengths)
    return X, y, wv


def _resolve_explicit(
    variables: dict[str, np.ndarray],
    x_var: str | None,
    y_var: str | None,
    wv_var: str | None,
    transpose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Pick X/y/wv using explicit variable names (legacy behaviour)."""

    def resolve_name(name: str | None) -> str | None:
        if name is None or name in variables:
            return name
        leaf_name = name.rsplit(".", 1)[-1]
        return leaf_name if leaf_name in variables else name

    x_var = resolve_name(x_var)
    y_var = resolve_name(y_var)
    wv_var = resolve_name(wv_var)
    if x_var is not None:
        if x_var not in variables:
            raise ValueError(
                f"Variable {x_var!r} (x_var) not found in MAT file. Available: {sorted(variables)}"
            )
        X = np.asarray(variables[x_var])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"Variable {x_var!r} must be 2D, got shape {X.shape!r}.")
    else:
        X = _heuristic_x(variables)

    if transpose:
        X = X.T

    n_samples, n_wavelengths = X.shape

    y: np.ndarray | None = None
    if y_var is not None:
        if y_var not in variables:
            raise ValueError(
                f"Variable {y_var!r} (y_var) not found in MAT file. Available: {sorted(variables)}"
            )
        y = np.asarray(variables[y_var]).ravel()
        if y.shape[0] != n_samples:
            raise ValueError(
                f"y_var {y_var!r} length {y.shape[0]} != n_samples {n_samples}."
            )
    else:
        y = _heuristic_y(variables, n_samples, exclude_name=None)

    wv: np.ndarray | None = None
    if wv_var is not None:
        if wv_var not in variables:
            raise ValueError(
                f"Variable {wv_var!r} (wv_var) not found in MAT file. Available: {sorted(variables)}"
            )
        wv = np.asarray(variables[wv_var]).ravel()
        if wv.shape[0] != n_wavelengths:
            raise ValueError(
                f"wv_var {wv_var!r} length {wv.shape[0]} != n_wavelengths {n_wavelengths}."
            )
    else:
        wv = _heuristic_wv(variables, n_wavelengths)

    return X, y, wv


def _heuristic_x(variables: dict[str, np.ndarray]) -> np.ndarray:
    """Pick the spectra matrix: the 2D variable with the most elements."""
    candidates = {
        k: v for k, v in variables.items() if isinstance(v, np.ndarray) and v.ndim == 2
    }
    if not candidates:
        raise ValueError(
            f"No 2D variable found in MAT file; cannot identify spectra matrix. Available: {sorted(variables)}"
        )
    best = max(candidates, key=lambda k: candidates[k].size)
    return np.asarray(candidates[best], dtype=float)


def _heuristic_y(
    variables: dict[str, np.ndarray],
    n_samples: int,
    exclude_name: str | None,
) -> np.ndarray | None:
    """Pick reference values: 1D variable whose length matches n_samples.

    Prefers names matching the Y* pattern (case-insensitive) when multiple
    candidates exist, so that ``Y`` is chosen over an unrelated 1D array
    of the same length (e.g. an unrelated per-sample weight).
    """
    # First pass: collect all 1D arrays matching n_samples.
    matches: list[tuple[str, np.ndarray]] = []
    for name, v in variables.items():
        if name == exclude_name:
            continue
        v1 = np.asarray(v).ravel()
        if v1.ndim == 1 and v1.shape[0] == n_samples and v1.size > 0:
            matches.append((name, v1))
    if not matches:
        return None
    # Prefer names that look like "Y" / "y" / "ref" / "target".
    y_named = [m for m in matches if _name_matches(m[0].lower(), _Y_NAME_RE)]
    if y_named:
        return y_named[0][1].astype(float)
    return matches[0][1].astype(float)


def _heuristic_wv(
    variables: dict[str, np.ndarray], n_wavelengths: int
) -> np.ndarray | None:
    """Pick wavelengths: 1D variable whose length matches n_wavelengths.

    Prefers names matching the wn/wv/wavelength pattern when multiple
    candidates exist.
    """
    matches: list[tuple[str, np.ndarray]] = []
    for name, v in variables.items():
        v1 = np.asarray(v).ravel()
        if v1.ndim == 1 and v1.shape[0] == n_wavelengths and v1.size > 0:
            matches.append((name, v1))
    if not matches:
        return None
    wv_named = [m for m in matches if _name_matches(m[0].lower(), _WV_NAME_RE)]
    if wv_named:
        return wv_named[0][1].astype(float)
    return matches[0][1].astype(float)


# ---------------------------------------------------------------------------
# CSV block splitting
# ---------------------------------------------------------------------------


def _parse_x_cols(
    spec: str,
    n_cols: int,
    *,
    drop_col: int | None = None,
) -> list[int]:
    """Parse an ``x_cols`` selector spec into a list of column indices.

    Supported syntax (Python-slice-like, but returns an explicit list):
    - ``"8:"``        → columns 8 to end
    - ``"8:314"``     → columns 8 to 313 (half-open, Python-style)
    - ``":8"``        → columns 0 to 7
    - ``"8,9,10"``    → explicit list
    - ``"-8:"``       → last 8 columns onward (i.e. n_cols-8 to end)
    - ``":"``         → all columns (equivalent to omitting x_cols)

    When ``drop_col`` is given, that index is removed from the result (used
    when ``y_col`` is inside the ``x_cols`` range — the y column should not
    appear in ``X``).

    Args:
        spec: The selector string.
        n_cols: Total number of columns in the array.
        drop_col: Optional column index to exclude from the result.

    Returns:
        Sorted list of column indices.

    Raises:
        ValueError: If the spec is malformed or indices are out of range.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("x_cols spec is empty")

    indices: list[int]

    if "," in spec:
        # Explicit list: "8,9,10"
        try:
            indices = [int(p.strip()) for p in spec.split(",") if p.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid x_cols list {spec!r}: {exc}") from exc
    elif ":" in spec:
        # Slice-like: "start:stop" (step not supported — rare need)
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid x_cols slice {spec!r}; use 'start:stop'")
        start_s, stop_s = parts[0].strip(), parts[1].strip()

        # Handle negative start (e.g. "-8:" → last 8 columns)
        start = int(start_s) if start_s else 0
        if start < 0:
            start = max(0, n_cols + start)

        # Handle stop
        if stop_s:
            stop = int(stop_s)
            if stop < 0:
                stop = max(0, n_cols + stop)
        else:
            stop = n_cols

        indices = list(range(start, stop))
    else:
        # Single column index
        try:
            idx = int(spec)
        except ValueError as exc:
            raise ValueError(f"Invalid x_cols spec {spec!r}: {exc}") from exc
        if idx < 0:
            idx = n_cols + idx
        indices = [idx]

    # Validate range
    for i in indices:
        if not (0 <= i < n_cols):
            raise ValueError(f"x_cols index {i} out of range for {n_cols} columns")

    # Empty selection is almost certainly a user error (e.g. x_cols="10:"
    # on a 3-column file). Raise rather than silently producing an empty X.
    if not indices:
        raise ValueError(
            f"x_cols spec {spec!r} selected 0 columns from {n_cols} available; check the indices/range."
        )

    # Drop y_col if requested
    if drop_col is not None:
        indices = [i for i in indices if i != drop_col]

    return sorted(indices)


def _resolve_csv_feature_indices(
    n_cols: int,
    *,
    y_col: int | None,
    y_cols: list[int] | None,
    x_cols: str | None,
) -> list[int]:
    """Resolve spectral columns against the original CSV column positions."""
    target_cols = set(y_cols or ([] if y_col is None else [y_col]))
    if x_cols is None:
        indices = [index for index in range(n_cols) if index not in target_cols]
    else:
        indices = [
            index for index in _parse_x_cols(x_cols, n_cols) if index not in target_cols
        ]
    if not indices:
        raise ValueError("x_cols selected only reference columns; no spectra remain")
    return indices


def _split_csv_block(
    arr: np.ndarray,
    y_col: int | None,
    wv_row: int | None,
    x_cols: str | None = None,
    y_cols: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Carve (X, y, wv) out of a numeric 2D block using selectors.

    Order of operations: the wavelength row is extracted first (this also
    removes the empty corner cell that sits at the intersection of the
    wavelength row and the y column), then the y column is pulled from the
    remaining rows. The wavelength vector is then stripped of any leading
    non-finite placeholder (the empty corner cell).

    ``x_cols`` (★ v3.8) allows selecting a subset of columns as the spectra
    block ``X``, skipping metadata columns. When ``x_cols`` is provided it
    takes precedence over the "keep all non-y columns" default. Syntax:
    - ``"8:"``        → columns 8 to end
    - ``"8:314"``     → columns 8 to 313 (Python half-open)
    - ``":8"``        → columns 0 to 7
    - ``"8,9,10"``    → explicit list
    - ``"-8:"``       → last 8 columns onward (rare; mainly for symmetry)
    """
    X_full = arr.astype(float)
    y: np.ndarray | None = None
    wv: np.ndarray | None = None

    if wv_row is not None and X_full.shape[0] > 0:
        if not (0 <= wv_row < X_full.shape[0]):
            raise ValueError(
                f"wv_row={wv_row} out of range for {X_full.shape[0]} rows."
            )

    if y_cols is not None and X_full.shape[1] > 0:
        target_cols = [int(index) for index in y_cols]
        if not target_cols:
            raise ValueError("y_cols must select at least one column")
        if len(set(target_cols)) != len(target_cols):
            raise ValueError("y_cols must not contain duplicate columns")
        bad = [index for index in target_cols if not 0 <= index < X_full.shape[1]]
        if bad:
            raise ValueError(
                f"y_cols {bad} out of range for {X_full.shape[1]} columns."
            )
    elif y_col is not None and X_full.shape[1] > 0:
        if not (0 <= y_col < X_full.shape[1]):
            raise ValueError(
                f"y_col={y_col} out of range for {X_full.shape[1]} columns."
            )

    feature_indices = _resolve_csv_feature_indices(
        X_full.shape[1],
        y_col=y_col,
        y_cols=y_cols,
        x_cols=x_cols,
    )
    data_rows = np.delete(X_full, wv_row, axis=0) if wv_row is not None else X_full
    if y_cols is not None:
        y = data_rows[:, [int(index) for index in y_cols]].astype(float)
    elif y_col is not None:
        y = data_rows[:, y_col].astype(float)
    X = data_rows[:, feature_indices]

    if wv_row is not None:
        candidate_wv = X_full[wv_row, feature_indices].astype(float)
        if np.isfinite(candidate_wv).all():
            wv = candidate_wv
        else:
            import warnings

            warnings.warn(
                "The selected wavelength row contains non-numeric values in spectral columns; wavelength labels were omitted to avoid misalignment with X.",
                UserWarning,
                stacklevel=2,
            )
            wv = None

    return X, y, wv
