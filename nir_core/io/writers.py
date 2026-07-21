"""Writers for :class:`~nir_core.models.SpectralData`.

These helpers persist spectral data to disk. Only file-writing side effects
are permitted; no in-memory state is mutated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nir_core.models import SpectralData


def save_npz(data: SpectralData, filepath: str) -> None:
    """Save a :class:`SpectralData` to a NumPy ``.npz`` archive.

    The archive always contains the keys ``X``, ``y``, ``wv``,
    ``y_names``, ``sample_names``, ``source_file`` and ``original_format``. When ``y`` or
    ``wv`` are ``None`` on the input, an empty float array of shape ``(0,)``
    is stored as a placeholder so round-trips remain key-stable.

    Args:
        data: The :class:`SpectralData` to persist.
        filepath: Destination ``.npz`` path. Parent directories are created.

    Raises:
        ValueError: If ``data.X`` is not a 2D numpy array.
    """
    if not isinstance(data.X, np.ndarray) or data.X.ndim != 2:
        raise ValueError("SpectralData.X must be a 2D numpy array to save.")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(data.X, dtype=float)
    y = (
        np.asarray(data.y, dtype=float)
        if data.y is not None
        else np.empty(0, dtype=float)
    )
    wv = (
        np.asarray(data.wv, dtype=float)
        if data.wv is not None
        else np.empty(0, dtype=float)
    )
    # Unicode arrays are deliberately used instead of object arrays so readers
    # can keep ``allow_pickle=False`` at the user-data trust boundary.
    sample_names = np.asarray(data.sample_names, dtype=str)
    y_names = np.asarray(data.y_names or [], dtype=str)

    np.savez_compressed(
        filepath,
        X=X,
        y=y,
        y_names=y_names,
        wv=wv,
        sample_names=sample_names,
        source_file=np.asarray(data.source_file, dtype=str),
        original_format=np.asarray(data.original_format, dtype=str),
    )


def save_csv(data: SpectralData, filepath: str) -> None:
    """Save a :class:`SpectralData` to a CSV file.

    Layout:
      - First row: wavelength header (if ``data.wv`` is present), else a
        blank cell followed by column indices.
      - First column: reference values (if ``data.y`` is present), else a
        running sample index.
      - Remaining cells: the spectral matrix ``X``.

    Args:
        data: The :class:`SpectralData` to persist.
        filepath: Destination ``.csv`` path. Parent directories are created.

    Raises:
        ValueError: If ``data.X`` is not a 2D numpy array or if ``wv``/``y``
            shapes are inconsistent with ``X``.
    """
    if not isinstance(data.X, np.ndarray) or data.X.ndim != 2:
        raise ValueError("SpectralData.X must be a 2D numpy array to save.")
    n_samples, n_wavelengths = data.X.shape

    if data.wv is not None and np.asarray(data.wv).shape[0] != n_wavelengths:
        raise ValueError(
            "wv length must match the number of columns of X "
            f"(got {np.asarray(data.wv).shape[0]} vs {n_wavelengths})."
        )
    if data.y is not None and np.asarray(data.y).shape[0] != n_samples:
        raise ValueError(
            "y length must match the number of rows of X "
            f"(got {np.asarray(data.y).shape[0]} vs {n_samples})."
        )

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(data.X, dtype=float)
    has_y = data.y is not None
    has_wv = data.wv is not None
    y = np.asarray(data.y, dtype=float) if has_y else None
    y_2d = y.reshape(-1, 1) if y is not None and y.ndim == 1 else y
    wv = np.asarray(data.wv, dtype=float) if has_wv else None

    rows: list[list[str]] = []

    # Optional wavelength row (placed first, before any data row).
    if has_wv or (has_y and y_2d is not None and y_2d.shape[1] > 1):
        n_targets = y_2d.shape[1] if y_2d is not None else 1
        is_multi_target = n_targets > 1
        names = data.y_names or [f"y{i}" for i in range(n_targets)]
        wv_row: list[str] = names if has_y and is_multi_target else [""]
        if not has_wv:
            wv_row.extend(str(i) for i in range(n_wavelengths))
        else:
            wv_row.extend(f"{v:.10g}" for v in wv)
        rows.append(wv_row)

    # Data rows. A leading y column is added only when y is present, so a
    # spectra-only file round-trips as a plain numeric matrix.
    for i in range(n_samples):
        row: list[str] = []
        if y_2d is not None:
            row.extend(f"{value:.10g}" for value in y_2d[i])
        row.extend(f"{v:.10g}" for v in X[i])
        rows.append(row)

    with open(filepath, "w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write(",".join(row))
            fh.write("\n")
