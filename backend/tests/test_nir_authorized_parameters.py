"""Authorized parameter plans execute exactly one declared family and space."""

import numpy as np
import pytest
from nir_core.model.safe_search import fit_search_model
from nir_core.models import PreprocessingStep


def data():
    rng = np.random.default_rng(907)
    X = rng.normal(size=(48, 16))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.03, size=48)
    return X[:32], y[:32], X[32:], y[32:]


def plan(params, *, space=None):
    return {"strategy": "grid" if space else "fixed", "fixed_params": params, "search_space": space or {}, "max_trials": 4, "max_fits": 20}


@pytest.mark.parametrize("family,params", [("pls", {"n_components": 2, "scale": True}), ("ridge", {"alpha": 3.7}), ("svr", {"C": 2.4, "gamma": 0.03, "epsilon": 0.07}), ("et", {"n_estimators": 11, "max_depth": 3, "min_samples_leaf": 2})])
def test_fixed_parameters_are_not_researched_or_ignored(family, params):
    fitted = fit_search_model(*data(), method=family, steps=[], cv_folds=3, parameter_plan=plan(params))
    assert len(fitted.trials) == 1 and fitted.fits == 4
    for key, value in params.items():
        assert fitted.parameters[key] == value
    assert fitted.metrics["test_status"] == "sealed"
    assert len(fitted.cv_manifest["folds"]) == 3


def test_grid_stays_inside_authorized_svr_space_without_hidden_pls(monkeypatch):
    import nir_core.model.safe_search as core

    monkeypatch.setattr(core, "PLSRegression", lambda *a, **k: pytest.fail("Hidden PLS fit"))
    fitted = fit_search_model(*data(), method="svr", steps=[PreprocessingStep(method="snv"), PreprocessingStep(method="mean_center")], cv_folds=3, parameter_plan=plan({"epsilon": 0.05}, space={"C": [1.0, 3.0], "gamma": ["scale", 0.1]}))
    assert len(fitted.trials) == 4 and fitted.fits == 13
    assert all(t["parameters"]["C"] in [1.0, 3.0] and t["parameters"]["gamma"] in ["scale", 0.1] and t["parameters"]["epsilon"] == 0.05 for t in fitted.trials)


@pytest.mark.parametrize("family,params", [("pls", {"n_components": 99}), ("ridge", {"alpha": -1}), ("svr", {"epsilon": -1}), ("svr", {"ignored": 1}), ("et", {"n_jobs": 8})])
def test_illegal_parameters_fail_before_any_fit(family, params):
    with pytest.raises(ValueError, match="parameter|n_components"):
        fit_search_model(*data(), method=family, steps=[], cv_folds=3, parameter_plan=plan(params), checkpoint=lambda _: pytest.fail("Fit before validation"))


def test_budget_is_checked_before_grid_execution():
    with pytest.raises(ValueError, match=r"2 candidates \* 3 Cal CV folds \+ 1 Cal refit requires 7 fits"):
        fit_search_model(*data(), method="ridge", steps=[], cv_folds=3, parameter_plan={**plan({}, space={"alpha": [1.0, 2.0]}), "max_fits": 3}, checkpoint=lambda _: pytest.fail("Budget exceeded"))
