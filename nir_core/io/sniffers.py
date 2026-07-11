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
    }


def _inspect_mat(filepath: str) -> dict:
    """Inspect a .mat file by locating its first 2D numeric variable."""
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
    """Inspect a v5/v7 .mat file via scipy.io.loadmat."""
    from scipy.io import loadmat

    raw = loadmat(filepath, squeeze_me=False, struct_as_record=False)
    candidates = {
        k: np.asarray(v)
        for k, v in raw.items()
        if not k.startswith("__") and np.asarray(v).ndim >= 1
    }
    arr = _pick_first_2d(candidates)
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
        candidates: dict[str, np.ndarray] = {}
        for key in fh.keys():
            try:
                arr = np.asarray(fh[key][...])
            except Exception:
                continue
            candidates[key] = arr
        arr = _pick_first_2d(candidates)
        if arr is None:
            raise ValueError(
                f"No suitable 2D numeric variable found in {filepath!r}"
            )
        return _summarize_array(arr)


def _pick_first_2d(candidates: dict[str, np.ndarray]) -> np.ndarray | None:
    """Pick the first variable that is 2D (prefer larger arrays)."""
    two_d = {k: v for k, v in candidates.items() if v.ndim == 2}
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
