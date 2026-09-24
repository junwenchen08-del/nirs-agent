"""Support Vector Regression (SVR) with RBF kernel.

Hyper-parameters ``C`` and ``gamma`` are tuned via
:class:`sklearn.model_selection.GridSearchCV`. Feature standardisation is
performed internally via a :class:`sklearn.pipeline.Pipeline` so the
returned model is fully self-contained (caller passes raw spectra).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from nir_core.model.pls import _resolve_cv_splitter

# Default search grids (kept as module constants so tests can inspect them).
DEFAULT_C_GRID: list[float] = [0.1, 1.0, 10.0, 100.0]
DEFAULT_GAMMA_GRID: list = ["scale", 0.01, 0.1, 1.0]


def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train an SVR (RBF) model with grid-searched hyper-parameters.

    The grid covers ``C in [0.1, 1, 10, 100]`` and
    ``gamma in ['scale', 0.01, 0.1, 1]``. Features are standardised
    (zero mean, unit variance) inside a sklearn :class:`Pipeline`, so the
    returned estimator accepts raw spectra directly.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` adapts to sample size,
            ``"loocv"`` forces Leave-One-Out, ``"fixed"`` uses cv_folds.
        random_state: Seed for the KFold splitter used by GridSearchCV.

    Returns:
        Tuple ``(best_model, grid_results)``:

        - ``best_model``: a fitted :class:`sklearn.pipeline.Pipeline`
          (``StandardScaler`` -> ``SVR``) using the best hyper-parameters,
          refit on the full ``X_train``.
        - ``grid_results``: dict with keys:
          - ``"best_params"``: the chosen ``{C, gamma}`` dict.
          - ``"best_score"``: negative-MSE score from sklearn (higher better).
          - ``"best_rmse"``: RMSE corresponding to ``best_score``.
          - ``"cv_folds"``: number of folds used.
          - ``"param_grid"``: the grid that was searched.

    Raises:
        ValueError: On shape mismatch.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train rows ({X_train.shape[0]}) != y_train length ({y_train.shape[0]})"
        )
    n_samples = X_train.shape[0]

    # Build the pipeline so the scaler is fitted only on training data
    # inside each CV fold (sklearn handles this correctly).
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])

    param_grid = {
        "svr__C": DEFAULT_C_GRID,
        "svr__gamma": DEFAULT_GAMMA_GRID,
    }

    eff_folds = max(2, min(int(cv_folds), n_samples - 1))
    splitter, strategy_label = _resolve_cv_splitter(
        cv_strategy, cv_folds, n_samples, random_state
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=splitter,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_score = float(grid.best_score_)  # negative MSE
    best_rmse = float(np.sqrt(-best_score)) if np.isfinite(best_score) else float("inf")

    grid_results = {
        "best_params": dict(grid.best_params_),
        "best_score": best_score,
        "best_rmse": best_rmse,
        "cv_folds": eff_folds,
        "cv_strategy": strategy_label,
        "param_grid": {
            "C": list(DEFAULT_C_GRID),
            "gamma": list(DEFAULT_GAMMA_GRID),
        },
    }
    return grid.best_estimator_, grid_results


def predict_svr(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained SVR pipeline.

    Args:
        model: A fitted object exposing ``.predict`` (typically the
            :class:`Pipeline` returned by :func:`train_svr`).
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


__all__ = ["DEFAULT_C_GRID", "DEFAULT_GAMMA_GRID", "predict_svr", "train_svr"]
