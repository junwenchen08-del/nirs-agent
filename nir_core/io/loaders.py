"""Loaders for spectral files (.mat, .csv, .txt) into :class:`SpectralData`.

All loaders are pure functions: they read from disk and return a new
:class:`~nir_core.models.SpectralData` instance; no global state is mutated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nir_core.models import SpectralData
from nir_core.io.sniffers import _sniff_delimiter, _sniff_header, detect_format


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mat(
    filepath: str,
    x_var: str | None = None,
    y_var: str | None = None,
    wv_var: str | None = None,
) -> SpectralData:
    """Load a MATLAB ``.mat`` file into :class:`SpectralData`.

    Supports both v5/v7 (via :func:`scipy.io.loadmat`) and v7.3 / HDF5 (via
    ``h5py``, imported lazily). When variable names are not supplied, the
    loader applies a shape-based heuristic to identify ``X`` (2D matrix),
    ``y`` (1D, length == number of samples) and ``wv`` (1D, length == number
    of wavelengths).

    Args:
        filepath: Path to the ``.mat`` file.
        x_var: Optional explicit variable name for the spectra matrix.
        y_var: Optional explicit variable name for reference values.
        wv_var: Optional explicit variable name for wavelengths.

    Returns:
        A :class:`SpectralData` with ``original_format="mat"``.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If the file cannot be parsed or required variables are
            missing / malformed.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect HDF5 magic to decide between loadmat and h5py.
    with open(filepath, "rb") as fh:
        magic = fh.read(8)
    if magic == b"\x89HDF\r\n\x1a\n":
        return _load_mat_v73(filepath, x_var, y_var, wv_var)
    return _load_mat_v5(filepath, x_var, y_var, wv_var)


def load_csv(
    filepath: str,
    delimiter: str | None = None,
    x_cols: str | None = None,
    y_col: int | None = None,
    wv_row: int | None = None,
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

    Args:
        filepath: Path to the CSV/TXT file.
        delimiter: Optional explicit delimiter (``","``, ``"\\t"`` ...).
            ``None`` triggers auto-detection.
        x_cols: Reserved for future column-range selectors; currently unused.
        y_col: Index of the reference-value column. ``None`` disables y.
        wv_row: Index of the wavelength row. ``None`` disables wv.

    Returns:
        A :class:`SpectralData` with ``original_format="csv"``.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
        ValueError: If the file cannot be parsed as a numeric table.
    """
    del x_cols  # reserved; not yet implemented

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if delimiter is None:
        delimiter = _sniff_delimiter(filepath)
    has_header = _sniff_header(filepath, delimiter)

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

    X, y, wv = _split_csv_block(arr, y_col=y_col, wv_row=wv_row)

    return SpectralData(
        X=X,
        y=y,
        wv=wv,
        sample_names=[],
        source_file=str(filepath),
        original_format="csv",
    )


def auto_detect_and_load(filepath: str) -> SpectralData:
    """Detect the file format by extension/magic and dispatch to a loader.

    ``.mat`` -> :func:`load_mat`; ``.csv`` and ``.txt`` (and anything else)
    -> :func:`load_csv`.

    Args:
        filepath: Path to the file.

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
    data = load_csv(filepath)
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
) -> SpectralData:
    """Load a v5/v7 .mat via scipy.io.loadmat."""
    from scipy.io import loadmat

    raw = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    # Drop MATLAB metadata keys.
    variables = {
        k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")
    }
    X, y, wv = _resolve_mat_variables(variables, x_var, y_var, wv_var)
    return SpectralData(
        X=X,
        y=y,
        wv=wv,
        sample_names=[],
        source_file=str(filepath),
        original_format="mat",
    )


def _load_mat_v73(
    filepath: str,
    x_var: str | None,
    y_var: str | None,
    wv_var: str | None,
) -> SpectralData:
    """Load a v7.3 (HDF5) .mat via h5py (lazy import)."""
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            "Reading MATLAB v7.3 (HDF5) files requires the 'h5py' package, "
            f"but it could not be imported: {exc!s}. Install it via "
            "`pip install h5py` (optional dependency of nir-core)."
        ) from exc

    variables: dict[str, np.ndarray] = {}
    with h5py.File(filepath, "r") as fh:
        for key in fh.keys():
            try:
                variables[key] = np.asarray(fh[key][...])
            except Exception:
                continue

    X, y, wv = _resolve_mat_variables(variables, x_var, y_var, wv_var)
    return SpectralData(
        X=X,
        y=y,
        wv=wv,
        sample_names=[],
        source_file=str(filepath),
        original_format="mat",
    )


def _resolve_mat_variables(
    variables: dict[str, np.ndarray],
    x_var: str | None,
    y_var: str | None,
    wv_var: str | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Pick X/y/wv from a dict of MATLAB variables using hints + heuristics."""
    if not variables:
        raise ValueError("MAT file contains no variables.")

    # X: explicit or heuristic.
    if x_var is not None:
        if x_var not in variables:
            raise ValueError(
                f"Variable {x_var!r} (x_var) not found in MAT file. "
                f"Available: {sorted(variables)}"
            )
        X = variables[x_var]
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(
                f"Variable {x_var!r} must be 2D, got shape {X.shape!r}."
            )
    else:
        X = _heuristic_x(variables)

    n_samples, n_wavelengths = X.shape

    # y: explicit or heuristic (1D, length == n_samples).
    if y_var is not None:
        if y_var not in variables:
            raise ValueError(
                f"Variable {y_var!r} (y_var) not found in MAT file. "
                f"Available: {sorted(variables)}"
            )
        y = np.asarray(variables[y_var]).ravel()
        if y.shape[0] != n_samples:
            raise ValueError(
                f"y_var {y_var!r} length {y.shape[0]} != n_samples "
                f"{n_samples}."
            )
    else:
        y = _heuristic_y(variables, n_samples, exclude_name=None)

    # wv: explicit or heuristic (1D, length == n_wavelengths).
    if wv_var is not None:
        if wv_var not in variables:
            raise ValueError(
                f"Variable {wv_var!r} (wv_var) not found in MAT file. "
                f"Available: {sorted(variables)}"
            )
        wv = np.asarray(variables[wv_var]).ravel()
        if wv.shape[0] != n_wavelengths:
            raise ValueError(
                f"wv_var {wv_var!r} length {wv.shape[0]} != n_wavelengths "
                f"{n_wavelengths}."
            )
    else:
        wv = _heuristic_wv(variables, n_wavelengths)

    return X, y, wv


def _heuristic_x(variables: dict[str, np.ndarray]) -> np.ndarray:
    """Pick the spectra matrix: the 2D variable with the most elements."""
    candidates = {k: v for k, v in variables.items() if v.ndim == 2}
    if not candidates:
        raise ValueError(
            "No 2D variable found in MAT file; cannot identify spectra "
            f"matrix. Available: {sorted(variables)}"
        )
    best = max(candidates, key=lambda k: candidates[k].size)
    return np.asarray(candidates[best], dtype=float)


def _heuristic_y(
    variables: dict[str, np.ndarray],
    n_samples: int,
    exclude_name: str | None,
) -> np.ndarray | None:
    """Pick reference values: 1D variable whose length matches n_samples."""
    for name, v in variables.items():
        if name == exclude_name:
            continue
        v1 = np.asarray(v).ravel()
        if v1.ndim == 1 and v1.shape[0] == n_samples and v1.size > 0:
            return v1.astype(float)
    return None


def _heuristic_wv(
    variables: dict[str, np.ndarray], n_wavelengths: int
) -> np.ndarray | None:
    """Pick wavelengths: 1D variable whose length matches n_wavelengths."""
    for name, v in variables.items():
        v1 = np.asarray(v).ravel()
        if v1.ndim == 1 and v1.shape[0] == n_wavelengths and v1.size > 0:
            return v1.astype(float)
    return None


# ---------------------------------------------------------------------------
# CSV block splitting
# ---------------------------------------------------------------------------

def _split_csv_block(
    arr: np.ndarray,
    y_col: int | None,
    wv_row: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Carve (X, y, wv) out of a numeric 2D block using selectors.

    Order of operations: the wavelength row is extracted first (this also
    removes the empty corner cell that sits at the intersection of the
    wavelength row and the y column), then the y column is pulled from the
    remaining rows. The wavelength vector is then stripped of any leading
    non-finite placeholder (the empty corner cell).
    """
    X = arr.astype(float)
    y: np.ndarray | None = None
    wv: np.ndarray | None = None

    if wv_row is not None and X.shape[0] > 0:
        if not (0 <= wv_row < X.shape[0]):
            raise ValueError(
                f"wv_row={wv_row} out of range for {X.shape[0]} rows."
            )
        wv = X[wv_row, :].astype(float)
        X = np.delete(X, wv_row, axis=0)

    if y_col is not None and X.shape[1] > 0:
        if not (0 <= y_col < X.shape[1]):
            raise ValueError(
                f"y_col={y_col} out of range for {X.shape[1]} columns."
            )
        y = X[:, y_col].astype(float)
        X = np.delete(X, y_col, axis=1)

    # Clean the wavelength vector: drop the empty corner cell (NaN/Inf) that
    # sits at the (wv_row, y_col) intersection, and trim to the column count.
    if wv is not None:
        finite_mask = np.isfinite(wv)
        if not finite_mask.all():
            wv = wv[finite_mask]
        if wv.shape[0] > X.shape[1]:
            wv = wv[: X.shape[1]]

    return X, y, wv
