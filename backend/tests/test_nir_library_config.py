"""Feature-flag and capacity-contract tests for the NIR asset library."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from deerflow.config.app_config import AppConfig
from deerflow.config.nir_library_config import NIRLibraryConfig

_SANDBOX = {"use": "deerflow.sandbox.local:LocalSandboxProvider"}


def test_library_is_disabled_by_default() -> None:
    config = NIRLibraryConfig()

    assert config.enabled is False
    assert config.max_dataset_file_size == 50 * 1024 * 1024
    assert config.max_user_dataset_bytes == 5 * 1024**3
    assert config.max_user_model_bytes == 5 * 1024**3
    assert config.max_total_bytes_for_library_writes == 15 * 1024**3
    assert config.max_model_versions_per_id == 20


@pytest.mark.parametrize("field", ["max_dataset_file_size", "max_user_dataset_bytes", "max_user_model_bytes", "max_total_bytes_for_library_writes", "max_model_versions_per_id", "staging_ttl_hours"])
def test_positive_capacity_fields_reject_zero(field: str) -> None:
    with pytest.raises(ValidationError):
        NIRLibraryConfig.model_validate({field: 0})


def test_enabled_library_rejects_memory_database() -> None:
    with pytest.raises(ValidationError, match="persistent database"):
        AppConfig.model_validate({"sandbox": _SANDBOX, "database": {"backend": "memory"}, "nir_library": {"enabled": True}})


def test_disabled_library_keeps_memory_database_available() -> None:
    config = AppConfig.model_validate({"sandbox": _SANDBOX, "database": {"backend": "memory"}})
    assert config.nir_library.enabled is False


def test_enabled_library_accepts_sqlite_database() -> None:
    config = AppConfig.model_validate({"sandbox": _SANDBOX, "database": {"backend": "sqlite"}, "nir_library": {"enabled": True}})
    assert config.nir_library.enabled is True


def test_example_config_keeps_staged_library_disabled_and_registers_asset_tools() -> None:
    example = Path(__file__).resolve().parents[2] / "config.example.yaml"
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(raw)

    assert raw["config_version"] == 21
    assert config.nir_library.enabled is False
    assert config.nir_library.max_total_bytes_for_library_writes == 15 * 1024**3
    configured_tools = {tool.name: tool.use for tool in config.tools}
    assert {
        name: configured_tools[name]
        for name in (
            "nir_dataset_list",
            "nir_dataset_get",
            "nir_dataset_history",
            "nir_dataset_save",
            "nir_dataset_attach",
            "nir_model_list",
            "nir_model_get",
            "nir_model_promote",
            "nir_model_attach",
        )
    } == {
        "nir_dataset_list": "deerflow.community.nir.dataset_tools:nir_dataset_list_tool",
        "nir_dataset_get": "deerflow.community.nir.dataset_tools:nir_dataset_get_tool",
        "nir_dataset_history": "deerflow.community.nir.dataset_tools:nir_dataset_history_tool",
        "nir_dataset_save": "deerflow.community.nir.dataset_tools:nir_dataset_save_tool",
        "nir_dataset_attach": "deerflow.community.nir.dataset_tools:nir_dataset_attach_tool",
        "nir_model_list": "deerflow.community.nir.model_tools:nir_model_list_tool",
        "nir_model_get": "deerflow.community.nir.model_tools:nir_model_get_tool",
        "nir_model_promote": "deerflow.community.nir.model_tools:nir_model_promote_tool",
        "nir_model_attach": "deerflow.community.nir.model_tools:nir_model_attach_tool",
    }
