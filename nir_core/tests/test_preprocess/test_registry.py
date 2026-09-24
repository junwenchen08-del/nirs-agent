"""Contract tests for the preprocessing method registry."""

from __future__ import annotations

import re

EXPECTED_METHODS = {
    "snv",
    "robust_snv",
    "rnv",
    "msc",
    "emsc",
    "despike",
    "sg_smooth",
    "derivative1",
    "derivative2",
    "norris_derivative1",
    "norris_derivative2",
    "airpls",
    "arpls",
    "asls",
    "detrend",
    "mean_center",
    "autoscale",
    "normalize",
    "rubberband",
    "median_filter",
    "whittaker_smooth",
}


def test_registry_contains_every_existing_public_method_once():
    from nir_core.preprocess.registry import METHOD_REGISTRY

    assert set(METHOD_REGISTRY) == EXPECTED_METHODS
    assert len(METHOD_REGISTRY) == len(EXPECTED_METHODS)
    assert all(key == spec.method_id for key, spec in METHOD_REGISTRY.items())


def test_registry_is_the_source_of_pipeline_compatibility_views():
    from nir_core.preprocess.pipeline import PRESTEP_METHODS
    from nir_core.preprocess.registry import METHOD_REGISTRY

    assert set(PRESTEP_METHODS) == set(METHOD_REGISTRY)
    assert all(
        PRESTEP_METHODS[method_id] is spec.implementation
        for method_id, spec in METHOD_REGISTRY.items()
    )


def test_registry_metadata_is_complete_and_defaults_validate():
    from nir_core.preprocess.registry import METHOD_REGISTRY, validate_method_params

    valid_levels = {"default", "conditional", "explicit_only", "disabled"}
    valid_statuses = {"stable", "experimental", "unavailable", "deprecated"}

    for spec in METHOD_REGISTRY.values():
        assert spec.provider in {"native", "chemotools"}
        assert spec.display_name_zh
        assert spec.summary_zh
        assert spec.category
        assert spec.implementation_version
        assert spec.auto_level in valid_levels
        assert spec.status in valid_statuses
        assert callable(spec.implementation)
        defaults = {
            name: parameter.default
            for name, parameter in spec.parameter_schema.items()
            if parameter.has_default
        }
        assert validate_method_params(spec.method_id, defaults) == (True, "")


def test_registry_snapshot_and_hash_are_deterministic_and_serializable():
    import json

    from nir_core.preprocess.registry import catalog_hash, catalog_snapshot

    first = catalog_snapshot()
    second = catalog_snapshot()
    assert first == second
    assert [item["method_id"] for item in first["methods"]] == sorted(EXPECTED_METHODS)
    assert '"implementation":' not in json.dumps(first, ensure_ascii=False)
    assert re.fullmatch(r"[0-9a-f]{64}", catalog_hash())
    assert catalog_hash() == catalog_hash()


def test_registry_query_returns_compact_or_detailed_records():
    from nir_core.preprocess.registry import describe_method, list_methods

    compact = list_methods(auto_levels={"default"})
    assert compact
    assert all(item["auto_level"] == "default" for item in compact)
    assert all("parameter_schema" not in item for item in compact)

    detailed = describe_method("airpls")
    assert detailed["method_id"] == "airpls"
    assert detailed["parameter_schema"]["lambda_"]["default"] == 1e7


def test_verified_chemotools_providers_are_catalog_defaults():
    from nir_core.preprocess.registry import (
        describe_method,
        get_provider_implementation,
    )

    verified = {
        "snv",
        "msc",
        "emsc",
        "sg_smooth",
        "derivative1",
        "derivative2",
        "airpls",
        "arpls",
        "asls",
        "mean_center",
        "autoscale",
        "normalize",
        "norris_derivative1",
        "norris_derivative2",
        "detrend",
        "rnv",
        "rubberband",
        "median_filter",
        "whittaker_smooth",
    }
    for method_id in verified:
        details = describe_method(method_id)
        providers = {item["provider"]: item for item in details["available_providers"]}
        assert details["preferred_provider"] == "chemotools"
        if "native" in providers:
            assert providers["native"]["is_default"] is False
        assert providers["chemotools"]["is_default"] is True
        assert providers["chemotools"]["compatibility"] == "verified"
        assert providers["chemotools"]["provider_version"] == "0.4.4"
        assert callable(
            get_provider_implementation(method_id, "chemotools").implementation
        )

    native_only = describe_method("robust_snv")
    assert native_only["preferred_provider"] == "native"
    assert [item["provider"] for item in native_only["available_providers"]] == [
        "native"
    ]

    assert describe_method("emsc")["parameter_schema"]["method"]["choices"] == [
        "mean",
        "median",
    ]


def test_registry_rejects_unknown_methods_and_invalid_parameter_values():
    import pytest

    from nir_core.preprocess.registry import describe_method, validate_method_params

    with pytest.raises(KeyError, match="Unknown preprocessing method"):
        describe_method("not-a-method")

    valid, reason = validate_method_params("sg_smooth", {"window": 10})
    assert not valid
    assert "奇数" in reason

    valid, reason = validate_method_params("normalize", {"norm": "bad"})
    assert not valid
    assert "可选值" in reason

    valid, reason = validate_method_params("norris_derivative1", {"gap": 4})
    assert not valid
    assert "奇数" in reason

    valid, reason = validate_method_params("median_filter", {"window_length": 4})
    assert not valid
    assert "奇数" in reason
