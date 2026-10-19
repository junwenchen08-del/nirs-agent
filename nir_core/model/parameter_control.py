"""Execution schemas for four adapters, independent of knowledge-card ranking."""

from __future__ import annotations

import math
from itertools import product


def _number(default, low, high, *, integer=False, nullable=False):
    return {
        "default": default,
        "minimum": low,
        "maximum": high,
        "type": "integer" if integer else "number",
        "nullable": nullable,
    }


def _choice(default, values):
    return {"default": default, "type": "enum", "values": values}


SCHEMAS = {
    "pls": {
        "n_components": _number(2, 1, 20, integer=True),
        "scale": _choice(False, [False, True]),
        "max_iter": _number(500, 50, 5000, integer=True),
        "tol": _number(1e-6, 1e-12, 0.01),
    },
    "ridge": {
        "alpha": _number(1.0, 0.0, 1e6),
        "fit_intercept": _choice(True, [False, True]),
        "tol": _number(1e-4, 1e-12, 0.01),
        "max_iter": _number(None, 50, 100000, integer=True, nullable=True),
    },
    "svr": {
        "C": _number(1.0, 1e-8, 1e6),
        "gamma": {**_number("scale", 1e-12, 1e4), "strings": ["scale", "auto"]},
        "epsilon": _number(0.1, 0.0, 1e4),
        "kernel": _choice("rbf", ["rbf", "linear", "poly", "sigmoid"]),
        "degree": _number(3, 1, 6, integer=True),
        "coef0": _number(0.0, -100.0, 100.0),
        "tol": _number(0.001, 1e-12, 0.1),
        "shrinking": _choice(True, [False, True]),
        "max_iter": _number(10000, 50, 100000, integer=True),
    },
    "et": {
        "n_estimators": _number(100, 1, 500, integer=True),
        "max_depth": _number(None, 1, 64, integer=True, nullable=True),
        "min_samples_split": _number(2, 2, 40, integer=True),
        "min_samples_leaf": _number(1, 1, 20, integer=True),
        "max_features": {**_number(1.0, 1e-8, 1.0), "strings": ["sqrt", "log2"]},
        "bootstrap": _choice(False, [False, True]),
        "criterion": _choice("squared_error", ["squared_error", "friedman_mse"]),
    },
}


def model_capabilities():
    import sklearn
    from .safe_search import _estimator

    result = {}
    for name, schema in SCHEMAS.items():
        model = _estimator(
            name, {key: value["default"] for key, value in schema.items()}
        )
        estimator = model.steps[-1][1] if name in {"ridge", "svr"} else model
        operational = {
            key: value
            for key, value in estimator.get_params(deep=False).items()
            if key not in schema
        }
        result[name] = {
            "parameters": schema,
            "strategies": ["fixed", "grid"],
            "internal_scaler": name in {"ridge", "svr"},
            "source": "nir_core.model.parameter_control",
            "sklearn_version": sklearn.__version__,
            "operational_parameters": operational,
        }
    return result


def _validate(name, value, spec):
    valid = value is None and spec.get("nullable")
    if spec["type"] == "enum":
        valid = any(
            type(value) is type(option) and value == option for option in spec["values"]
        )
    elif isinstance(value, str):
        valid = value in spec.get("strings", [])
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        valid = (
            math.isfinite(value)
            and spec["minimum"] <= value <= spec["maximum"]
            and (spec["type"] != "integer" or isinstance(value, int))
        )
    if not valid:
        raise ValueError(f"parameter.{name}: invalid value; expected {spec}")


def authorized_candidates(method, plan, *, component_upper, cv_folds):
    if method not in SCHEMAS:
        raise ValueError("parameter.model_method: unsupported family")
    if set(plan) != {
        "strategy",
        "fixed_params",
        "search_space",
        "max_trials",
        "max_fits",
    }:
        raise ValueError("parameter.plan: missing or unknown fields")
    fixed, space = plan["fixed_params"], plan["search_space"]
    if (
        not isinstance(fixed, dict)
        or not isinstance(space, dict)
        or set(fixed) & set(space)
    ):
        raise ValueError("parameter.search_space: must be disjoint from fixed_params")
    if (
        plan["strategy"] not in {"fixed", "grid"}
        or (plan["strategy"] == "fixed" and space)
        or (plan["strategy"] == "grid" and not space)
    ):
        raise ValueError(
            "parameter.strategy: fixed requires empty space; grid requires explicit space"
        )
    schema = SCHEMAS[method]
    for name in fixed.keys() | space.keys():
        if name not in schema:
            raise ValueError(f"parameter.{name}: unsupported parameter for {method}")
    if method == "pls" and "n_components" not in fixed and "n_components" not in space:
        raise ValueError(
            "parameter.n_components: explicit PLS count or grid is required"
        )
    for name, value in fixed.items():
        _validate(name, value, schema[name])
    for name, values in space.items():
        if not isinstance(values, list) or not 1 <= len(values) <= 20:
            raise ValueError(f"parameter.{name}: expected 1..20 discrete values")
        for value in values:
            _validate(name, value, schema[name])
    count = math.prod(len(values) for values in space.values())
    for budget in ("max_trials", "max_fits"):
        if type(plan[budget]) is not int or not 1 <= plan[budget] <= (
            40 if budget == "max_trials" else 200
        ):
            raise ValueError(f"parameter.{budget}: invalid budget")
    if count > plan["max_trials"] or count * cv_folds + 1 > plan["max_fits"]:
        raise ValueError(
            f"parameter.budget: {count} candidates * {cv_folds} Cal CV folds + 1 Cal refit requires {count * cv_folds + 1} fits; "
            f"proposal authorizes max_trials={plan['max_trials']} and max_fits={plan['max_fits']}. "
            "Explicitly correct the proposal within the session budget; fixed still requires CV+refit."
        )
    base = {name: spec["default"] for name, spec in schema.items()} | fixed
    names = sorted(space)
    candidates = [
        base | dict(zip(names, values, strict=True))
        for values in product(*(space[name] for name in names))
    ]
    if method == "pls" and any(
        item["n_components"] > component_upper for item in candidates
    ):
        raise ValueError(
            f"parameter.n_components: must be <= {component_upper} for frozen calibration folds/axis"
        )
    return candidates
