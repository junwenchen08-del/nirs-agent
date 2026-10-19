"""Legacy comparison must not predict or rank by the final holdout."""

import json

import numpy as np


def test_comparison_never_predicts_test_and_ranks_validation(tmp_path, monkeypatch):
    from nir_core.model.evaluation import split_dataset
    from nir_core.models import SpectralData

    from deerflow.community.nir import modeling

    rng = np.random.default_rng(89)
    X = rng.normal(size=(60, 8))
    y = 2 * X[:, 0] + rng.normal(scale=0.1, size=60)
    (_, _), (_, _), (test, _) = split_dataset(X, y, test_ratio=0.2, val_ratio=0.1, random_state=42)
    source = tmp_path / "source.npz"
    np.savez(source, X=X, y=y)
    monkeypatch.setattr(modeling, "_resolve", lambda *a, **k: str(source))
    monkeypatch.setattr(modeling, "_resolve_writable_dir", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr("nir_core.io.loaders.auto_detect_and_load", lambda path: SpectralData(X=X, y=y))
    seen = []
    original = modeling._train_one_model

    def train(*args, **kwargs):
        model, n, cv, predict = original(*args, **kwargs)

        def guarded(model, matrix):
            seen.append(matrix)
            assert not any(np.array_equal(row, holdout) for row in matrix for holdout in test)
            return predict(model, matrix)

        return model, n, cv, guarded

    monkeypatch.setattr(modeling, "_train_one_model", train)
    result = json.loads(modeling.nir_compare_tool.func(runtime=None, data_path="source", pipelines="[[], []]", method="pls"))
    assert result.get("status") != "error", result
    assert result["best"]["evaluation_scope"] == "validation"
    assert "RMSEP" not in result["best"]
    all_metrics = json.loads((tmp_path / "all_metrics.json").read_text(encoding="utf-8"))
    assert all("test" not in item and item["evaluation_scope"] == "validation" for item in all_metrics)
    assert "验证" in (tmp_path / "comparison_summary.md").read_text(encoding="utf-8")
