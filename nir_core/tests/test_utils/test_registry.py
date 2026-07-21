"""Tests for ModelRegistry CRUD operations (uses tmp_path fixture)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nir_core.utils.registry import ModelRegistry, RegistryCorruptionError


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(str(tmp_path / "registry.json"))


def test_register_returns_version_tag(registry: ModelRegistry):
    ver = registry.register(
        model_id="corn_pls",
        method="pls",
        metrics={"R2": 0.9, "RPD": 4.0, "RMSE": 0.1},
        preprocessing_steps=[{"method": "snv", "params": {}}],
        data_hash="abc123",
        model_path="artifacts/corn_pls_v1.joblib",
    )
    assert ver.startswith("corn_pls-v")


def test_load_latest_after_register(registry: ModelRegistry):
    registry.register(
        model_id="m1",
        method="pls",
        metrics={"RPD": 3.0},
        preprocessing_steps=[],
        data_hash="h1",
        model_path="p1",
    )
    registry.register(
        model_id="m1",
        method="pls",
        metrics={"RPD": 4.0},
        preprocessing_steps=[],
        data_hash="h2",
        model_path="p2",
    )
    latest = registry.load_latest("m1")
    assert latest is not None
    assert latest["metrics"]["RPD"] == 4.0
    assert latest["model_path"] == "p2"


def test_load_version(registry: ModelRegistry):
    ver = registry.register(
        model_id="m2",
        method="pls",
        metrics={"RPD": 3.5},
        preprocessing_steps=[],
        data_hash="h",
        model_path="p",
    )
    rec = registry.load_version("m2", ver)
    assert rec is not None
    assert rec["version"] == ver
    assert registry.load_version("m2", "nonexistent") is None


def test_list_versions_order(registry: ModelRegistry):
    for i in range(3):
        registry.register(
            model_id="m3",
            method="pls",
            metrics={"RPD": float(i)},
            preprocessing_steps=[],
            data_hash="h",
            model_path=f"p{i}",
        )
    versions = registry.list_versions("m3")
    assert len(versions) == 3
    # Order should be oldest first (registration order).
    rpds = [v["metrics"]["RPD"] for v in versions]
    assert rpds == [0.0, 1.0, 2.0]


def test_list_versions_empty(registry: ModelRegistry):
    assert registry.list_versions("nope") == []


def test_load_latest_missing_model(registry: ModelRegistry):
    assert registry.load_latest("absent") is None


def test_compare(registry: ModelRegistry):
    registry.register(
        model_id="a",
        method="pls",
        metrics={"RPD": 3.0, "R2": 0.85},
        preprocessing_steps=[],
        data_hash="h",
        model_path="pa",
    )
    registry.register(
        model_id="b",
        method="svr",
        metrics={"RPD": 4.0, "R2": 0.90},
        preprocessing_steps=[],
        data_hash="h",
        model_path="pb",
    )
    cmp = registry.compare(["a", "b", "missing"])
    assert cmp["a"]["metrics"]["RPD"] == 3.0
    assert cmp["b"]["metrics"]["RPD"] == 4.0
    assert cmp["missing"] is None


def test_best_by_metric(registry: ModelRegistry):
    for rpd in [2.0, 5.0, 3.0]:
        registry.register(
            model_id="m4",
            method="pls",
            metrics={"RPD": rpd},
            preprocessing_steps=[],
            data_hash="h",
            model_path=f"p{rpd}",
        )
    best = registry.best("m4", metric="RPD")
    assert best is not None
    assert best["metrics"]["RPD"] == 5.0


def test_best_missing_model(registry: ModelRegistry):
    assert registry.best("absent") is None


def test_file_not_exists_graceful(tmp_path: Path):
    reg = ModelRegistry(str(tmp_path / "never.json"))
    assert reg.load_latest("x") is None
    assert reg.list_versions("x") == []
    assert reg.best("x") is None
    cmp = reg.compare(["x"])
    assert cmp == {"x": None}


def test_persists_to_disk(registry: ModelRegistry, tmp_path: Path):
    registry.register(
        model_id="persist",
        method="pls",
        metrics={"RPD": 3.0},
        preprocessing_steps=[{"method": "snv"}],
        data_hash="h",
        model_path="p",
    )
    p = Path(registry.registry_path)
    assert p.exists()
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    assert "persist" in raw
    assert len(raw["persist"]) == 1


def test_wavelength_indices_stored(registry: ModelRegistry):
    registry.register(
        model_id="m5",
        method="pls",
        metrics={"RPD": 3.0},
        preprocessing_steps=[],
        data_hash="h",
        model_path="p",
        wavelength_indices=[0, 5, 10],
    )
    rec = registry.load_latest("m5")
    assert rec["wavelength_indices"] == [0, 5, 10]


def test_corrupt_registry_fails_closed(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text("{not-json", encoding="utf-8")
    registry = ModelRegistry(str(path))

    with pytest.raises(RegistryCorruptionError, match="Cannot read model registry"):
        registry.list_versions("anything")


def test_registry_stores_separate_training_and_artifact_hashes(registry: ModelRegistry):
    registry.register(
        model_id="traceable",
        method="pls",
        metrics={"RPD": 3.0},
        preprocessing_steps=[],
        data_hash="training-sha256",
        artifact_hash="artifact-sha256",
        model_path="model.pkl",
    )
    record = registry.load_latest("traceable")
    assert record["training_data_hash"] == "training-sha256"
    assert record["artifact_sha256"] == "artifact-sha256"


def test_concurrent_registrations_are_not_lost(registry: ModelRegistry):
    def register(index: int) -> str:
        return registry.register(
            model_id="concurrent",
            method="pls",
            metrics={"RPD": float(index)},
            preprocessing_steps=[],
            data_hash=f"training-{index}",
            artifact_hash=f"artifact-{index}",
            model_path=f"model-{index}.pkl",
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        versions = list(executor.map(register, range(12)))

    records = registry.list_versions("concurrent")
    assert len(records) == 12
    assert len(set(versions)) == 12
