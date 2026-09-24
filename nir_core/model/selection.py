"""Wavelength (variable) selection algorithms.

Implements two complementary, widely-used NIR variable-selection methods:

- **CARS** (Competitive Adaptive Reweighted Sampling, Li et al. 2009):
  combines Monte Carlo sampling, an exponentially decreasing function
  (EDF) for coarse variable pruning, and adaptive reweighted sampling
  (ARS) for fine selection, with cross-validation picking the optimal
  iteration.
- **SPA** (Successive Projections Algorithm): greedily selects variables
  whose projections onto the already-selected subspace are minimally
  collinear, evaluating candidate starting wavelengths by RMSE.

Both functions return ``(X_selected, selected_indices)`` where
``selected_indices`` is a sorted ``list[int]`` and ``X_selected`` has shape
``(n_samples, n_selected)``.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nir_core.utils.metrics import rmse


def _safe_n_components(X: np.ndarray, y: np.ndarray, max_components: int) -> int:
    """Pick a safe PLS component count for the given data shape."""
    n_samples, n_wavelengths = X.shape
    if n_wavelengths < 1:
        return 1
    return max(1, min(int(max_components), n_samples - 1, n_wavelengths))


def _pls_cv_rmse(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    n_folds: int,
    random_state: int,
) -> float:
    """K-fold CV RMSE for a PLS model on ``X``/``y``.

    Returns ``+inf`` if the model cannot be fit (e.g. too few samples).
    """
    n_samples = X.shape[0]
    if n_samples < 3 or X.shape[1] < 1:
        return float("inf")
    nc = max(1, min(int(n_components), n_samples - 1, X.shape[1]))
    eff_folds = max(2, min(int(n_folds), n_samples - 1))
    kf = KFold(n_splits=eff_folds, shuffle=True, random_state=random_state)
    rmses: list[float] = []
    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if X_tr.shape[0] <= nc:
            continue
        try:
            m = PLSRegression(n_components=nc, scale=False)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_val).ravel()
            rmses.append(rmse(y_val, pred))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"CARS CV fold skipped (n_components={nc}): {exc}",
                stacklevel=2,
            )
            continue
    if not rmses:
        return float("inf")
    return float(np.mean(rmses))


def cars_wavelength_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_mc_samples: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """CARS wavelength selection (Li et al. 2009).

    Algorithm outline:

    1. Fit an initial PLS on the full data to obtain regression
       coefficients; their absolute values rank wavelength importance.
    2. Run ``n_mc_samples`` Monte Carlo iterations. In each iteration a
       random 80% subsample is drawn, a PLS is fit, and the per-wavelength
       coefficient magnitudes are accumulated.
    3. An **exponentially decreasing function** (EDF) determines the
       retention ratio ``r_i`` at iteration ``i``:
       ``r_i = a * exp(-k * i) + b`` with ``a,b,k`` chosen so that
       ``r_0 = 1`` and ``r_{N-1} = 2 / n_wavelengths``.
    4. At each iteration EDF sets the target subset size
       ``ceil(r_i * n_wavelengths)``. An **adaptive reweighted sampling**
       (ARS) step samples that number of variables from the full wavelength
       pool with probability proportional to weight^2, so weaker variables
       can be eliminated competitively.
    5. Each candidate subset is evaluated by ``n_folds``-fold CV RMSE.
    6. The subset with the smallest CV RMSE is returned.

    Args:
        X: Spectra, shape (n_samples, n_wavelengths).
        y: Reference values, shape (n_samples,).
        n_mc_samples: Number of Monte Carlo / EDF iterations. Must be >= 2.
        n_folds: CV folds used to evaluate each candidate subset.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(X_selected, selected_indices)``:

        - ``X_selected``: array of shape ``(n_samples, n_selected)``.
        - ``selected_indices``: sorted list of 0-based column indices.

    Raises:
        ValueError: If inputs are degenerate or ``n_mc_samples`` < 2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n_samples, n_wavelengths = X.shape
    if n_samples != y.shape[0]:
        raise ValueError(f"X rows ({n_samples}) != y length ({y.shape[0]})")
    if n_mc_samples < 2:
        raise ValueError(f"n_mc_samples must be >= 2, got {n_mc_samples}")

    rng = np.random.default_rng(random_state)

    # Step 1: initial PLS on the full data to get coefficient magnitudes.
    init_nc = _safe_n_components(X, y, max_components=min(10, n_wavelengths))
    try:
        init_pls = PLSRegression(n_components=init_nc, scale=False)
        init_pls.fit(X, y)
        init_coef = np.abs(np.asarray(init_pls.coef_).ravel())
    except Exception:  # noqa: BLE001
        # Fallback: use correlation with y as importance.
        init_coef = np.abs(
            np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(n_wavelengths)])
        )
    init_coef = np.nan_to_num(init_coef, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 2: Monte Carlo sampling to accumulate weights.
    weights = np.zeros(n_wavelengths, dtype=float)
    n_sub = min(n_samples, max(2, round(0.8 * n_samples)))
    mc_nc = _safe_n_components(
        X[:n_sub], y[:n_sub], max_components=min(10, n_wavelengths)
    )
    for _ in range(n_mc_samples):
        idx = rng.choice(n_samples, size=n_sub, replace=False)
        Xs, ys = X[idx], y[idx]
        try:
            m = PLSRegression(n_components=mc_nc, scale=False)
            m.fit(Xs, ys)
            coef = np.abs(np.asarray(m.coef_).ravel())
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"CARS MC sample skipped (n_components={mc_nc}): {exc}",
                stacklevel=2,
            )
            continue
        weights += coef

    # Combine initial + MC weights; guard against all-zero.
    total_weights = init_coef + weights
    if not np.any(total_weights > 0):
        # Degenerate: keep all wavelengths.
        selected = list(range(n_wavelengths))
        return X[:, selected].copy(), selected

    # Step 3: EDF retention ratios.
    N = int(n_mc_samples)
    i_arr = np.arange(N, dtype=float)
    # r_0 = 1, r_{N-1} = 2/n_wavelengths.
    r_end = max(2.0 / n_wavelengths, 1.0 / n_wavelengths)
    # Solve r_i = a * exp(-k * i) + b with r_0 = 1, r_{N-1} = r_end,
    # and choose k so the decay is smooth (k = -ln(r_end) / (N-1), b=0,
    # a=1). If N == 1 we already returned earlier; guard anyway.
    if N > 1:
        k_edf = -np.log(max(r_end, 1e-6)) / (N - 1)
    else:
        k_edf = 0.0
    a_edf = 1.0
    b_edf = 0.0
    r_arr = a_edf * np.exp(-k_edf * i_arr) + b_edf
    # Clamp retention ratio to [2/n_wavelengths, 1].
    r_arr = np.clip(r_arr, 2.0 / n_wavelengths, 1.0)

    # Step 4 + 5: iterate, retain, ARS, evaluate.
    best_rmse = float("inf")
    best_indices: list[int] = list(range(n_wavelengths))
    seen_subsets: set = set()

    for it in range(N):
        r_i = float(r_arr[it])
        n_keep = max(1, int(np.ceil(r_i * n_wavelengths)))
        n_keep = min(n_keep, n_wavelengths)

        # ARS: sample n_keep competitors from the full variable pool with
        # probability proportional to weight^2. Sampling from a pool larger
        # than the requested subset is essential; drawing all retained
        # variables would make the probability weights a no-op.
        w2 = total_weights**2
        s = float(np.sum(w2))
        if s <= 0:
            probs = np.ones(n_wavelengths) / n_wavelengths
        else:
            # Keep a tiny non-zero floor so weighted sampling without
            # replacement remains feasible when most coefficients are zero.
            floor = np.finfo(float).eps * max(float(np.max(w2)), 1.0)
            probs = (w2 + floor) / float(np.sum(w2 + floor))
        chosen = np.sort(
            rng.choice(
                n_wavelengths,
                size=n_keep,
                replace=False,
                p=probs,
            )
        )

        key = tuple(int(c) for c in chosen)
        if key in seen_subsets:
            continue
        seen_subsets.add(key)

        Xs = X[:, chosen]
        nc = _safe_n_components(Xs, y, max_components=min(10, len(chosen)))
        # Use identical folds for every candidate so CV scores are directly
        # comparable; changing the split per iteration can select a subset
        # merely because it received an easier validation fold.
        rmse_i = _pls_cv_rmse(Xs, y, nc, n_folds, random_state)
        if rmse_i < best_rmse:
            best_rmse = rmse_i
            best_indices = [int(c) for c in chosen]

    # Always evaluate the full-wavelength baseline as a safety net. Selection
    # must not replace it unless at least one candidate has lower CV RMSE.
    full_nc = _safe_n_components(X, y, max_components=min(10, n_wavelengths))
    full_rmse = _pls_cv_rmse(X, y, full_nc, n_folds, random_state)
    if full_rmse < best_rmse:
        best_indices = list(range(n_wavelengths))

    best_indices = sorted({int(i) for i in best_indices})
    X_selected = X[:, best_indices].copy()
    return X_selected, best_indices


def _spa_forward(X: np.ndarray, start: int, n_select: int) -> list[int]:
    """Run the SPA forward selection starting at column ``start``.

    At each step, pick the unselected column whose projection onto the
    orthogonal complement of the already-selected subspace has the
    largest norm (i.e. maximally orthogonal / minimally collinear).

    Args:
        X: Data matrix (n_samples, n_wavelengths).
        start: Index of the starting wavelength.
        n_select: Total number of wavelengths to select (including start).

    Returns:
        List of selected column indices in selection order.
    """
    n_wavelengths = X.shape[1]
    selected = [int(start)]
    remaining = [j for j in range(n_wavelengths) if j != start]
    # Normalise columns to unit norm so projection comparisons are fair.
    col_norms = np.linalg.norm(X, axis=0)
    col_norms_safe = np.where(col_norms > 0, col_norms, 1.0)

    while len(selected) < n_select and remaining:
        # Orthonormalise the currently selected columns.
        Q = X[:, selected] / col_norms_safe[selected]
        # QR orthonormalisation via modified Gram-Schmidt for stability.
        Q, _ = np.linalg.qr(Q)
        # Project each remaining candidate onto the orthogonal complement.
        # Projection coefficient magnitude tells how much of the candidate
        # is NOT explained by the selected subspace.
        best_j = -1
        best_proj = -1.0
        for j in remaining:
            v = X[:, j] / col_norms_safe[j]
            # Residual = v - Q @ (Q.T @ v)
            proj = Q @ (Q.T @ v)
            resid = v - proj
            nrm = float(np.linalg.norm(resid))
            if nrm > best_proj:
                best_proj = nrm
                best_j = j
        if best_j < 0:
            break
        selected.append(int(best_j))
        remaining.remove(int(best_j))
    return selected


def spa_wavelength_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_min: int = 1,
    n_max: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Successive Projections Algorithm (SPA) wavelength selection.

    For every possible starting wavelength, runs the forward SPA
    selection to build candidate subsets of sizes ``n_min..n_max``, then
    evaluates each candidate subset by the RMSE of a PLS model fit on the
    selected wavelengths (using a quick leave-one-out style or simple
    correlation criterion). The subset with the best evaluation is
    returned.

    Args:
        X: Spectra, shape (n_samples, n_wavelengths).
        y: Reference values, shape (n_samples,).
        n_min: Minimum subset size (>= 1).
        n_max: Maximum subset size. If ``None``, defaults to
            ``min(10, n_wavelengths // 10)`` (clamped to >= n_min and
            <= n_wavelengths).

    Returns:
        Tuple ``(X_selected, selected_indices)`` with ``selected_indices``
        a sorted list of 0-based column indices.

    Raises:
        ValueError: On degenerate input or invalid size bounds.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n_samples, n_wavelengths = X.shape
    if n_samples != y.shape[0]:
        raise ValueError(f"X rows ({n_samples}) != y length ({y.shape[0]})")
    if n_min < 1:
        raise ValueError(f"n_min must be >= 1, got {n_min}")
    if n_wavelengths == 0:
        raise ValueError("X has zero wavelengths")
    if n_min > n_wavelengths:
        raise ValueError(
            f"n_min ({n_min}) cannot exceed the number of wavelengths ({n_wavelengths})"
        )

    if n_max is None:
        n_max = min(10, max(3, n_wavelengths // 3))
    n_max = int(n_max)
    n_min = int(n_min)
    n_max = max(n_max, n_min)
    n_max = min(n_max, n_wavelengths)

    if n_samples < 2:
        raise ValueError("SPA requires at least 2 samples")

    best_rmse = float("inf")
    best_indices: list[int] = [0]

    # Pre-compute column normalisation; guard zero-variance columns.
    col_std = X.std(axis=0)
    valid_cols = np.where(col_std > 0)[0]
    if len(valid_cols) == 0:
        return X.copy(), list(range(n_wavelengths))

    # For efficiency we evaluate each candidate subset using a simple
    # linear-least-squares fit (no PLS) on the selected columns; RMSE of
    # the in-sample fit is a fast proxy. To avoid overfitting bias we use
    # leave-one-out CV when n_samples is small.
    for start in valid_cols:
        for n_sel in range(n_min, n_max + 1):
            indices = _spa_forward(X, int(start), n_sel)
            Xs = X[:, indices]
            if Xs.shape[1] == 0:
                continue
            # Fast evaluation: ordinary least squares with leave-one-out
            # error via the hat matrix diagonal (analytic LOO).
            try:
                # Add a column of ones for the intercept.
                Xa = np.hstack([Xs, np.ones((n_samples, 1))])
                # Use lstsq for the full fit.
                beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
                resid = y - Xa @ beta
                # Hat matrix diagonal via the pseudo-inverse: h = diag(Xa (Xa^T Xa)^-1 Xa^T)
                # Press statistic (predictive residual error sum of squares):
                XtX_inv = np.linalg.pinv(Xa.T @ Xa)
                H_diag = np.einsum("ij,jk,ik->i", Xa, XtX_inv, Xa)
                # Guard against h_ii == 1 (would divide by zero).
                denom = np.where(np.abs(1.0 - H_diag) < 1e-10, 1e-10, 1.0 - H_diag)
                press = np.sum((resid / denom) ** 2)
                rmse_loo = float(np.sqrt(press / n_samples))
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"SPA evaluation skipped (n_sel={n_sel}): {exc}",
                    stacklevel=2,
                )
                continue
            if rmse_loo < best_rmse:
                best_rmse = rmse_loo
                best_indices = [int(i) for i in indices]

    best_indices = sorted({int(i) for i in best_indices})
    X_selected = X[:, best_indices].copy()
    return X_selected, best_indices


__all__ = ["cars_wavelength_selection", "spa_wavelength_selection"]
