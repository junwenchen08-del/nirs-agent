"""Principal Component Regression (PCR).

PCR = PCA (dimensionality reduction) followed by Ordinary Least Squares
(:class:`sklearn.linear_model.LinearRegression`) on the scores. Component
selection mirrors :mod:`nir_core.model.pls`: when ``n_components`` is
``None`` the optimal count is chosen by K-fold cross-validation that
minimises mean RMSECV.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from nir_core.model.pls import _resolve_cv_splitter
from nir_core.utils.metrics import rmse


@dataclass
class PCRModel:
    """Container holding the fitted PCA + LinearRegression pair.

    Attributes:
        pca: Fitted :class:`sklearn.decomposition.PCA`.
        lr: Fitted :class:`sklearn.linear_model.LinearRegression`.
        n_components: Number of principal components used.
    """

    pca: PCA
    lr: LinearRegression
    n_components: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` through PCA and predict with the linear model.

        Args:
            X: Spectra, shape (n_samples, n_wavelengths).

        Returns:
            Predictions, shape (n_samples, 1) to match sklearn convention.
        """
        X = np.asarray(X, dtype=float)
        scores = self.pca.transform(X)
        return self.lr.predict(scores)


def _safe_max_components(n_samples: int, n_wavelengths: int, max_components: int) -> int:
    """Clamp ``max_components`` for PCA stability."""
    return max(1, min(int(max_components), int(n_samples) - 1, int(n_wavelengths)))


def train_pcr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int | None = None,
    max_components: int = 20,
    cv_folds: int = 10,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, int, dict]:
    """Train a PCR model, optionally auto-selecting components.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        n_components: If given, train directly with this many PCs. If
            ``None``, search ``1..max_components`` and pick the count that
            minimises mean RMSECV.
        max_components: Upper bound for the search, clamped for stability.
        cv_folds: K-fold splits for the CV search.
        cv_strategy: CV strategy: ``"auto"`` adapts to sample size,
            ``"loocv"`` forces Leave-One-Out, ``"fixed"`` uses cv_folds.
        random_state: Seed for KFold shuffling and PCA ``svd_solver``.

    Returns:
        Tuple ``(model, best_n_components, cv_results)``:

        - ``model``: a :class:`PCRModel` instance fitted on the full
          training set with the selected number of components.
        - ``best_n_components``: number of PCs used.
        - ``cv_results``: dict with ``"n_components"``, ``"mean_rmse_cv"``,
          ``"std_rmse_cv"``, ``"best_n_components"``.

    Raises:
        ValueError: On shape mismatch.
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
        pca = PCA(n_components=nc, random_state=random_state)
        scores = pca.fit_transform(X_train)
        lr = LinearRegression()
        lr.fit(scores, y_train)
        model = PCRModel(pca=pca, lr=lr, n_components=nc)
        cv_results = {
            "n_components": [nc],
            "mean_rmse_cv": [float("nan")],
            "std_rmse_cv": [float("nan")],
            "best_n_components": nc,
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
            if X_tr.shape[0] <= nc:
                continue
            try:
                pca = PCA(n_components=nc, random_state=random_state)
                sc_tr = pca.fit_transform(X_tr)
                lr = LinearRegression()
                lr.fit(sc_tr, y_tr)
                sc_val = pca.transform(X_val)
                pred = lr.predict(sc_val).ravel()
                fold_rmse.append(rmse(y_val, pred))
            except Exception as exc:
                warnings.warn(
                    f"PCR CV fold skipped (n_components={nc}): {exc}",
                    stacklevel=2,
                )
                continue
        if not fold_rmse:
            mean_rmse.append(float("inf"))
            std_rmse.append(0.0)
        else:
            mean_rmse.append(float(np.mean(fold_rmse)))
            std_rmse.append(float(np.std(fold_rmse, ddof=1)) if len(fold_rmse) > 1 else 0.0)

    mean_arr = np.asarray(mean_rmse)
    finite_mask = np.isfinite(mean_arr)
    if not finite_mask.any():
        best_idx = 0
    else:
        finite_vals = np.where(finite_mask, mean_arr, np.inf)
        best_idx = int(np.argmin(finite_vals))
    best_n = int(n_comp_list[best_idx])

    # Refit on the full training set.
    pca = PCA(n_components=best_n, random_state=random_state)
    scores = pca.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(scores, y_train)
    model = PCRModel(pca=pca, lr=lr, n_components=best_n)

    cv_results = {
        "n_components": n_comp_list,
        "mean_rmse_cv": mean_rmse,
        "std_rmse_cv": std_rmse,
        "best_n_components": best_n,
        "cv_strategy": strategy_label,
    }
    return model, best_n, cv_results


def predict_pcr(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained PCR model.

    Args:
        model: A fitted :class:`PCRModel` (or any object with ``.predict``).
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


__all__ = ["PCRModel", "train_pcr", "predict_pcr"]
