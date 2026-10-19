"""Installed method bindings for references; this is not an execution registry."""

from __future__ import annotations

import math
from importlib.metadata import version

MODEL_REFERENCES = {
    "model_pls": (
        "PLS 偏最小二乘回归",
        "sklearn.cross_decomposition.PLSRegression",
        {
            "n_components": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "maximum": 1000,
                "description_zh": "潜变量数；实际上限由训练样本和变量数决定",
            }
        },
    ),
    "model_ridge": (
        "Ridge 岭回归",
        "sklearn.linear_model.Ridge",
        {
            "alpha": {
                "type": "number",
                "default": 1.0,
                "minimum": 1e-12,
                "maximum": 1e6,
                "description_zh": "L2 正则化强度；参考组合的项目边界",
            }
        },
    ),
    "model_svr": (
        "SVR 支持向量回归",
        "sklearn.svm.SVR",
        {
            "C": {
                "type": "number",
                "default": 1.0,
                "minimum": 1e-12,
                "maximum": 1e6,
                "description_zh": "误差惩罚强度",
            },
            "gamma": {
                "type": "string",
                "default": "scale",
                "choices": ["scale", "auto"],
                "description_zh": "核宽度规则；数值 gamma 的现有训练网格另见参数说明",
            },
        },
    ),
    "model_extra_trees": (
        "Extra Trees 极端随机树回归",
        "sklearn.ensemble.ExtraTreesRegressor",
        {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "maximum": 2000,
                "description_zh": "树数量；参考组合的项目边界",
            },
            "max_depth": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description_zh": "最大深度；留空表示官方默认不限深度",
            },
        },
    ),
}


def runtime_reference(method_id: str) -> dict:
    from nir_core.preprocess.registry import METHOD_REGISTRY

    if method_id.startswith("chemotools."):
        from .method_inventory import inventory, mcp_facts

        entry = inventory()["entries"].get(method_id)
        if entry is None:
            raise ValueError("method is not in the reviewed MCP inventory")
        return {
            "method_id": method_id,
            "title": entry["name"],
            "method_kind": "mcp_reference",
            "provider": "chemotools",
            "provider_class": method_id,
            "runtime_provider_version": inventory()["provider_version"],
            "regular_axis_required": False,
            "runtime_parameters": {},
            **mcp_facts(method_id),
        }
    if method_id in MODEL_REFERENCES:
        title, provider_class, schema = MODEL_REFERENCES[method_id]
        return {
            "method_id": method_id,
            "title": title,
            "method_kind": "modeling",
            "provider": "scikit-learn",
            "provider_class": provider_class,
            "runtime_provider_version": version("scikit-learn"),
            "regular_axis_required": False,
            "runtime_parameters": schema,
        }
    spec = METHOD_REGISTRY.get(method_id)
    if spec is None or spec.status != "stable":
        raise ValueError("method is not an installed stable algorithm")
    providers = spec.detailed_dict()["available_providers"]
    provider = next(
        (
            p
            for p in providers
            if p["provider"] == "chemotools" and p["compatibility"] == "verified"
        ),
        None,
    )
    if provider is None:
        provider = next(
            (
                p
                for p in providers
                if p["provider"] == "native" and p["compatibility"] == "native"
            ),
            None,
        )
    if provider is None:
        raise ValueError("method has no verified or native provider")
    return {
        "method_id": method_id,
        "title": spec.display_name_zh,
        "method_kind": "preprocessing",
        "provider": provider["provider"],
        "provider_class": provider["provider_class"],
        "runtime_provider_version": provider["provider_version"]
        or provider["implementation_version"],
        "regular_axis_required": spec.requires_regular_axis,
        "runtime_parameters": spec.detailed_dict()["parameter_schema"],
    }


def validate_reference_params(method_id: str, params: dict) -> tuple[bool, str]:
    if method_id.startswith("chemotools."):
        runtime_reference(method_id)
        return (
            (True, "")
            if not params
            else (
                False,
                "MCP parameters are read-only references; use the MCP describe/validate tools for execution",
            )
        )
    if method_id not in MODEL_REFERENCES:
        from nir_core.preprocess.registry import validate_method_params

        return validate_method_params(method_id, params)
    schema = runtime_reference(method_id)["runtime_parameters"]
    if set(params) - set(schema):
        return False, "unknown model reference parameter"
    for name, value in params.items():
        field = schema[name]
        if field["type"] in {"number", "integer"}:
            if type(value) not in {int, float} or not math.isfinite(value):
                return False, f"{name} must be a finite number"
            if field["type"] == "integer" and type(value) is not int:
                return False, f"{name} must be an integer"
            if not field["minimum"] <= value <= field["maximum"]:
                return False, f"{name} exceeds project reference bounds"
        elif value not in field["choices"]:
            return False, f"{name} has an unsupported value"
    return True, ""
