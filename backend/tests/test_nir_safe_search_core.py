"""Real Cal/Val model execution, with fold-local preprocessing."""

import numpy as np
import pytest
from nir_core.model.safe_search import fit_search_model
from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import PreprocessingPipeline


def arrays():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(48, 8))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.05, size=48)
    return X[:32], y[:32], X[32:], y[32:]


@pytest.mark.parametrize("family", ["pls", "ridge", "svr", "et"])
def test_four_families_train_without_a_holdout_argument(family):
    result = fit_search_model(*arrays(), method=family, steps=[], cv_folds=3, max_components=3)
    assert result.metrics["val"]["scope"] == "validation"
    assert result.metrics["train_cv"]["scope"] == "calibration_cv"
    assert "test" not in result.metrics and "RMSEP" not in result.metrics
    assert np.isfinite(result.metrics["val"]["RMSE"])
    assert result.model.predict(arrays()[2]).shape == (16,)


def test_preprocessing_is_fitted_inside_each_cv_fold(monkeypatch):
    seen = []
    original = PreprocessingPipeline.fit

    def spy(self, X, wv=None):
        seen.append(X.copy())
        return original(self, X, wv)

    monkeypatch.setattr(PreprocessingPipeline, "fit", spy)
    X, y, X_val, y_val = arrays()
    fit_search_model(X, y, X_val, y_val, method="pls", steps=[PreprocessingStep(method="mean_center")], cv_folds=3, max_components=2)
    assert len(seen) == 7  # 2 candidates * 3 folds + final Cal fit
    assert all(part.shape[0] < len(X) for part in seen[:-1])
    assert np.array_equal(seen[-1], X)
    assert all(not np.isin(part[:, 0], X_val[:, 0]).any() for part in seen)


def test_groups_do_not_cross_internal_cv_folds():
    X, y, X_val, y_val = arrays()
    groups = np.repeat(np.arange(8), 4)
    result = fit_search_model(X, y, X_val, y_val, method="pls", steps=[], cv_folds=4, max_components=2, groups=groups)
    for train, valid in result.folds:
        assert set(groups[train]).isdisjoint(set(groups[valid]))


def test_unsupported_family_fails_explicitly():
    with pytest.raises(ValueError, match="supported"):
        fit_search_model(*arrays(), method="cnn", steps=[])


def test_direct_execution_rejects_cal_val_overlap():
    X, y, _, _ = arrays()
    with pytest.raises(ValueError, match="overlap"):
        fit_search_model(X, y, X[:4], y[:4], method="pls", steps=[])


def test_cv_rejects_duplicate_spectra_without_isolated_groups():
    X, y, X_val, y_val = arrays()
    X[1], y[1] = X[0], y[0]
    with pytest.raises(ValueError, match="CV folds"):
        fit_search_model(X, y, X_val, y_val, method="pls", steps=[], cv_folds=3)
