"""Multi-Layer Perceptron (MLP) regression.

Uses :class:`sklearn.neural_network.MLPRegressor` wrapped in a
:class:`sklearn.pipeline.Pipeline` with :class:`StandardScaler`. This is a
basic feed-forward neural network suitable for tabular/spectral data.

Hyper-parameters ``hidden_layer_sizes`` and ``alpha`` (L2 regularization)
are tuned via grid search.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nir_core.model.pls import _resolve_cv_splitter

# Default search grids.
DEFAULT_HIDDEN_LAYERS: list[tuple] = [
    (64,),
    (128, 64),
    (256, 128, 64),
]
DEFAULT_ALPHA_GRID: list[float] = [0.0001, 0.001, 0.01]


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train an MLP (neural network) model with grid-searched hyper-parameters.

    The grid covers ``hidden_layer_sizes in [(64,), (128,64), (256,128,64)]``
    and ``alpha in [0.0001, 0.001, 0.01]``. Features are standardised
    inside a sklearn :class:`Pipeline` (neural networks need scaled input).

    Uses the Adam optimiser with early stopping (``max_iter=500``,
    ``early_stopping=True``).

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

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = {
        "mlp__hidden_layer_sizes": DEFAULT_HIDDEN_LAYERS,
        "mlp__alpha": DEFAULT_ALPHA_GRID,
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
        "best_params": {
            "hidden_layer_sizes": str(grid.best_params_["mlp__hidden_layer_sizes"]),
            "alpha": grid.best_params_["mlp__alpha"],
        },
        "best_score": best_score,
        "best_rmse": best_rmse,
        "cv_strategy": strategy_label,
        "param_grid": {
            "hidden_layer_sizes": [str(h) for h in DEFAULT_HIDDEN_LAYERS],
            "alpha": list(DEFAULT_ALPHA_GRID),
        },
    }
    return grid.best_estimator_, grid_results


def predict_mlp(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained MLP pipeline.

    Args:
        model: A fitted :class:`Pipeline` (StandardScaler -> MLPRegressor).
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


__all__ = ["DEFAULT_ALPHA_GRID", "DEFAULT_HIDDEN_LAYERS", "predict_mlp", "train_mlp"]
