"""Small, JSON-safe helpers for describing a spectral axis."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_spectral_axis(values: Any) -> dict[str, float | str | None]:
    """Return the stored endpoint order and monotonic direction of an axis.

    A min/max range cannot distinguish ascending from descending storage.  The
    first and last values are therefore reported separately and direction is
    derived from every adjacent difference.
    """

    if values is None:
        return {
            "axis_first": None,
            "axis_last": None,
            "axis_direction": "missing",
        }

    axis = np.asarray(values, dtype=float).ravel()
    if axis.size == 0:
        return {
            "axis_first": None,
            "axis_last": None,
            "axis_direction": "missing",
        }

    first = float(axis[0]) if np.isfinite(axis[0]) else None
    last = float(axis[-1]) if np.isfinite(axis[-1]) else None
    if not np.isfinite(axis).all():
        direction = "invalid"
    elif axis.size < 2:
        direction = "single_point"
    else:
        differences = np.diff(axis)
        if np.all(differences > 0):
            direction = "ascending"
        elif np.all(differences < 0):
            direction = "descending"
        elif np.any(differences == 0):
            direction = "duplicate_values"
        else:
            direction = "non_monotonic"

    return {
        "axis_first": first,
        "axis_last": last,
        "axis_direction": direction,
    }
