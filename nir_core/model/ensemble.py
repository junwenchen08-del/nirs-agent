"""Ensemble aggregation of model predictions.

Provides simple but robust strategies for combining the outputs of
multiple base learners:

- ``"mean"``: arithmetic mean (equal weighting).
- ``"weighted"``: weighted average, weights normalised to sum to 1.
- ``"median"``: per-sample median (robust to outliers).
"""

from __future__ import annotations

import numpy as np


def ensemble_predict(
    predictions: list[np.ndarray],
    weights: list[float] | None = None,
    method: str = "mean",
) -> np.ndarray:
    """Aggregate a list of model predictions into a single array.

    Args:
        predictions: List of 1-D arrays, each of shape (n_samples,). All
            arrays must share the same length. The list may contain any
            number of models >= 1.
        weights: Optional list of per-model weights. Required when
            ``method == "weighted"``; ignored otherwise. Weights need not
            be normalised -- they are rescaled to sum to 1 internally.
        method: Aggregation strategy, one of:

            - ``"mean"``: ``mean(predictions, axis=0)``.
            - ``"weighted"``: ``sum(w_i * pred_i)`` with normalised weights.
            - ``"median"``: ``median(predictions, axis=0)`` (per-sample).

    Returns:
        1-D array of shape (n_samples,) containing the aggregated
        predictions.

    Raises:
        ValueError: If ``predictions`` is empty, lengths mismatch, an
            unknown method is requested, or ``weights`` is missing / has
            the wrong length / sums to a non-positive value for the
            ``"weighted"`` method.
    """
    if not predictions:
        raise ValueError("predictions must contain at least one array")

    # Validate equal lengths and coerce to 2-D float array (n_models, n_samples).
    arrays = [np.asarray(p, dtype=float).ravel() for p in predictions]
    n = arrays[0].size
    for i, a in enumerate(arrays):
        if a.size != n:
            raise ValueError(f"prediction {i} has length {a.size}, expected {n}")
    stacked = np.vstack(arrays)  # shape (n_models, n_samples)

    if method == "mean":
        return stacked.mean(axis=0)

    if method == "median":
        return np.median(stacked, axis=0)

    if method == "weighted":
        if weights is None:
            raise ValueError('weights must be provided when method="weighted"')
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != stacked.shape[0]:
            raise ValueError(
                f"weights length ({w.size}) != number of predictions ({stacked.shape[0]})"
            )
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                f"weights must be finite and have a positive sum; got sum={total}"
            )
        w_norm = w / total
        # Weighted sum over the model axis.
        return (stacked * w_norm[:, None]).sum(axis=0)

    raise ValueError(
        f"Unknown method {method!r}; expected one of ['mean', 'weighted', 'median']"
    )


__all__ = ["ensemble_predict"]
