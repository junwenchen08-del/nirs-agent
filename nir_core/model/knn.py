"""K-Nearest Neighbors (KNN) regression.

Uses :class:`sklearn.neighbors.KNeighborsRegressor` wrapped in a
:class:`sklearn.pipeline.Pipeline` with :class:`StandardScaler` (KNN is
distance-based, so feature scaling is essential). Hyper-parameters
``n_neighbors`` and ``weights`` are tuned via grid search.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nir_core.model.pls import _resolve_cv_splitter

# Default search grids.
DEFAULT_N_NEIGHBORS: list[int] = [3, 5, 7, 10, 15]
DEFAULT_WEIGHTS: list[str] = ["uniform", "distance"]


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train a KNN model with grid-searched hyper-parameters.

    The grid covers ``n_neighbors in [3, 5, 7, 10, 15]`` and
    ``weights in ['uniform', 'distance']``. Features are standardised
    inside a sklearn :class:`Pipeline` (KNN is distance-based).

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` adapts to sample size,
            ``"loocv"`` forces Leave-One-Out, ``"fixed"`` uses cv_folds.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(best_model, grid_results)``.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train rows ({X_train.shape[0]}) != y_train length ({y_train.shape[0]})"
        )
    n_samples = X_train.shape[0]

    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())])

    param_grid = {
        "knn__n_neighbors": DEFAULT_N_NEIGHBORS,
        "knn__weights": DEFAULT_WEIGHTS,
    }

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

    best_score = float(grid.best_score_)
    best_rmse = float(np.sqrt(-best_score)) if np.isfinite(best_score) else float("inf")

    grid_results = {
        "best_params": dict(grid.best_params_),
        "best_score": best_score,
        "best_rmse": best_rmse,
        "cv_strategy": strategy_label,
        "param_grid": {
            "n_neighbors": list(DEFAULT_N_NEIGHBORS),
            "weights": list(DEFAULT_WEIGHTS),
        },
    }
    return grid.best_estimator_, grid_results


def predict_knn(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained KNN pipeline.

    Args:
        model: A fitted :class:`Pipeline` (StandardScaler -> KNN).
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


__all__ = ["DEFAULT_N_NEIGHBORS", "DEFAULT_WEIGHTS", "predict_knn", "train_knn"]
