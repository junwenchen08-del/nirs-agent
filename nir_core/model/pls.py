"""Partial Least Squares (PLS) regression trainer and predictor.

``train_pls`` performs automatic selection of the optimal number of
latent components via K-fold cross-validation when ``n_components`` is
left unspecified. The final model is retrained on the full training
set using the best component count.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, LeaveOneOut

from nir_core.utils.metrics import rmse


def _resolve_cv_splitter(
    cv_strategy: str,
    cv_folds: int,
    n_samples: int,
    random_state: int = 42,
):
    """Return a CV splitter based on ``cv_strategy``.

    - ``"auto"``: n_samples < 20 → LOOCV; n_samples < 50 → 5-fold; else cv_folds.
    - ``"loocv"``: always Leave-One-Out.
    - ``"fixed"`` (or any other): use ``cv_folds`` K-fold.
    """
    if cv_strategy == "loocv":
        return LeaveOneOut(), "loocv"
    if cv_strategy == "auto":
        if n_samples < 20:
            return LeaveOneOut(), "loocv"
        eff_folds = 5 if n_samples < 50 else max(2, min(int(cv_folds), n_samples - 1))
        return KFold(n_splits=eff_folds, shuffle=True, random_state=random_state), f"{eff_folds}-fold"
    # fixed / default
    eff_folds = max(2, min(int(cv_folds), n_samples - 1))
    return KFold(n_splits=eff_folds, shuffle=True, random_state=random_state), f"{eff_folds}-fold"


def _safe_max_components(n_samples: int, n_wavelengths: int, max_components: int) -> int:
    """Clamp ``max_components`` to ``min(n_samples-1, n_wavelengths)``.

    PLS requires at least one sample more than components and cannot use
    more components than wavelengths.
    """
    return max(1, min(int(max_components), int(n_samples) - 1, int(n_wavelengths)))


def train_pls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int | None = None,
    max_components: int = 20,
    cv_folds: int = 10,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, int, dict]:
    """Train a PLS regression model, optionally auto-selecting components.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        n_components: If given, train directly with this many latent
            components (no CV search is performed). If ``None``, search
            over ``1..max_components`` using K-fold cross-validation and
            pick the count that minimises mean RMSECV.
        max_components: Upper bound for the component search. Clamped to
            ``min(n_samples-1, n_wavelengths)`` for numerical stability.
        cv_folds: Number of K-fold splits used during component search.
            Ignored when ``n_components`` is given.
        cv_strategy: CV strategy: ``"auto"`` (default) adapts to sample
            size (<20 → LOOCV, <50 → 5-fold, else cv_folds), ``"loocv"``
            forces Leave-One-Out, ``"fixed"`` uses ``cv_folds``.
        random_state: Seed for KFold shuffling.

    Returns:
        Tuple ``(model, best_n_components, cv_results)`` where:

        - ``model``: a fitted :class:`sklearn.cross_decomposition.PLSRegression`.
        - ``best_n_components``: the number of latent variables used.
        - ``cv_results``: dict with keys:
          - ``"n_components"``: list of component counts tested.
          - ``"mean_rmse_cv"``: list of mean RMSECV for each count.
          - ``"best_n_components"``: selected count.
          - ``"std_rmse_cv"``: list of std RMSECV for each count.
          - ``"cv_strategy"``: the effective strategy used.
          When ``n_components`` is given explicitly, the lists contain a
          single entry and ``best_n_components`` equals the input.

    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train rows ({X_train.shape[0]}) != y_train length ({y_train.shape[0]})")
    n_samples, n_wavelengths = X_train.shape

    # ---- Fixed component count: train directly. -----------------------
    if n_components is not None:
        nc = int(n_components)
        if nc < 1:
            raise ValueError(f"n_components must be >= 1, got {nc}")
        nc = min(nc, _safe_max_components(n_samples, n_wavelengths, nc))
        model = PLSRegression(n_components=nc, scale=False)
        model.fit(X_train, y_train)
        cv_results = {
            "n_components": [nc],
            "mean_rmse_cv": [float("nan")],
            "std_rmse_cv": [float("nan")],
            "best_n_components": nc,
            "cv_strategy": "fixed",
        }
        return model, nc, cv_results

    # ---- Auto-select via K-fold CV. -----------------------------------
    upper = _safe_max_components(n_samples, n_wavelengths, max_components)
    n_comp_list = list(range(1, upper + 1))

    splitter, strategy_label = _resolve_cv_splitter(cv_strategy, cv_folds, n_samples, random_state)

    mean_rmse: list[float] = []
    std_rmse: list[float] = []

    for nc in n_comp_list:
        fold_rmse: list[float] = []
        for train_idx, val_idx in splitter.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            # Guard: each fold must have enough samples for nc components.
            if X_tr.shape[0] <= nc:
                continue
            try:
                m = PLSRegression(n_components=nc, scale=False)
                m.fit(X_tr, y_tr)
                pred = m.predict(X_val).ravel()
                fold_rmse.append(rmse(y_val, pred))
            except Exception as exc:
                # Numerical failure on this fold/component combination.
                warnings.warn(
                    f"PLS CV fold skipped (n_components={nc}): {exc}",
                    stacklevel=2,
                )
                continue
        if not fold_rmse:
            # No fold succeeded -> treat as worst.
            mean_rmse.append(float("inf"))
            std_rmse.append(0.0)
        else:
            mean_rmse.append(float(np.mean(fold_rmse)))
            std_rmse.append(float(np.std(fold_rmse, ddof=1)) if len(fold_rmse) > 1 else 0.0)

    # Best = minimum mean RMSECV. Ties broken by smallest component count
    # (more parsimonious model).
    mean_arr = np.asarray(mean_rmse)
    finite_mask = np.isfinite(mean_arr)
    if not finite_mask.any():
        # Fallback: use 1 component.
        best_idx = 0
    else:
        # Among finite values pick the min; ties -> smallest index.
        finite_vals = np.where(finite_mask, mean_arr, np.inf)
        best_idx = int(np.argmin(finite_vals))

    best_n = int(n_comp_list[best_idx])

    # Retrain on the full training set.
    model = PLSRegression(n_components=best_n, scale=False)
    model.fit(X_train, y_train)

    cv_results = {
        "n_components": n_comp_list,
        "mean_rmse_cv": mean_rmse,
        "std_rmse_cv": std_rmse,
        "best_n_components": best_n,
        "cv_strategy": strategy_label,
    }
    return model, best_n, cv_results


def predict_pls(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained PLS model.

    Args:
        model: A fitted :class:`PLSRegression` (or any object exposing
            ``.predict``).
        X: Spectra of shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predicted values, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


def compute_vip(model: PLSRegression, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Variable Importance in Projection (VIP) scores.

    VIP measures the contribution of each wavelength variable to the PLS
    model. A VIP > 1.0 indicates an above-average contribution.

    Reference: Wold, S., Sjostrom, M., Eriksson, L. (2001).
    *PLS-regression: a basic tool of chemometrics.*

    Args:
        model: A fitted :class:`PLSRegression`.
        X: Training spectra (n_samples, n_wavelengths).
        y: Training references (n_samples,).

    Returns:
        VIP scores, shape (n_wavelengths,). All values >= 0.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_comp = model.n_components

    # Extract PLS weights (W) and scores (T).
    # sklearn PLSRegression stores weights as model.x_weights_ (n_features, n_comp).
    W = model.x_weights_  # (n_features, n_comp)
    # x_scores_ = T (n_samples, n_comp)
    T = model.x_scores_
    # y_loadings_ gives the y-variance explained per component.
    q = model.y_loadings_.ravel()  # (n_comp,)

    # SSY per component: sum of squares of y explained by each component.
    ssy = np.zeros(n_comp)
    for a in range(n_comp):
        ssy[a] = float(np.sum((T[:, a].ravel() * q[a]) ** 2))

    total_ssy = float(np.sum(ssy))
    if total_ssy <= 0:
        return np.ones(X.shape[1])

    # W^2 normalized per component (column-wise).
    w2 = W**2
    w2_sum = w2.sum(axis=0)  # (n_comp,)
    w2_sum_safe = np.where(w2_sum > 0, w2_sum, 1.0)
    w2_norm = w2 / w2_sum_safe  # (n_features, n_comp)

    # VIP = sqrt(p * sum_a(ssy_a * w2_norm[:,a]) / total_ssy)
    p = float(X.shape[1])
    vip = np.sqrt(p * np.sum(ssy[None, :] * w2_norm, axis=1) / total_ssy)
    return vip


def get_regression_coefficients(model: PLSRegression) -> np.ndarray:
    """Return the regression coefficients (B) from a fitted PLS model.

    The coefficients map spectral variables to the predicted reference
    value: ``y_pred = X @ B``. They are useful for interpreting which
    wavelengths contribute positively or negatively to the prediction.

    Args:
        model: A fitted :class:`PLSRegression`.

    Returns:
        1-D array of regression coefficients, shape (n_wavelengths,).
    """
    coef = np.asarray(model.coef_, dtype=float)
    return coef.ravel()


__all__ = ["train_pls", "predict_pls", "compute_vip", "get_regression_coefficients"]
