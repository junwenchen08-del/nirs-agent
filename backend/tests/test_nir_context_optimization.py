"""Regression tests for bounded-context NIR collection analysis."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from deerflow.community.nir._report import _build_report

REPO_ROOT = Path(__file__).resolve().parents[2]


def _metrics(*, subset: str) -> dict:
    return {
        "status": "ok",
        "method": "pls",
        "n_components": 8,
        "preprocessing": "SNV -> autoscale",
        "R2_val": 0.91 if subset == "R562" else 0.89,
        "RPD": 3.4,
        "RMSEP": 0.42,
        "grade": "excellent",
        "passed": True,
        "action": "accept",
        "report": f"/mnt/user-data/outputs/nir_collection/{subset}/report.md",
        "model": f"/mnt/user-data/outputs/nir_collection/{subset}/model.pkl",
        "metrics": f"/mnt/user-data/outputs/nir_collection/{subset}/metrics.json",
        "plots": {
            "raw_spectra": f"/mnt/user-data/outputs/nir_collection/{subset}/raw_spectra.png",
        },
    }


def test_nir_report_references_plot_files_without_base64_payloads() -> None:
    data = SimpleNamespace(
        X=np.zeros((12, 5)),
        y=np.linspace(1.0, 2.0, 12),
        wv=np.linspace(900.0, 1700.0, 5),
    )
    metrics = {
        "n_samples": 12,
        "method": "pls",
        "n_components": 2,
        "R2_val": 0.91,
        "RPD": 3.2,
        "RMSEP": 0.4,
    }
    quality = {
        "grade": "excellent",
        "passed": True,
        "action": "accept",
        "thresholds_used": {"min_r2": 0.8, "min_rpd": 3.0, "domain": "default"},
    }

    report = _build_report(
        data,
        metrics,
        quality,
        SimpleNamespace(description=lambda: "SNV"),
        raw_spectra_b64="A" * 100_000,
        predicted_vs_reference_b64="B" * 100_000,
        residuals_b64="C" * 100_000,
        cv_curve_b64="D" * 100_000,
    )

    assert "data:image" not in report
    assert "raw_spectra.png" in report
    assert "predicted_vs_reference.png" in report
    assert len(report) < 10_000


def test_nir_skill_uses_compact_metrics_instead_of_full_report_reads() -> None:
    text = (REPO_ROOT / "skills/custom/nir-coordinator/SKILL.md").read_text(encoding="utf-8")

    assert "禁止完整读取 `report.md`" in text
    assert "优先使用建模工具返回的 JSON" in text
    assert "用 `read_file` 读取 `report` 路径" not in text


def test_runtime_context_retention_is_token_bounded() -> None:
    config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))

    assert config["summarization"]["keep"]["type"] == "tokens"
    assert config["summarization"]["keep"]["value"] <= 8_000
    assert config["summarization"]["trim_tokens_to_summarize"] <= 12_000
    assert config["sandbox"]["read_file_output_max_chars"] <= 12_000


def test_collection_tool_is_registered_in_runtime_config_and_facades() -> None:
    from deerflow.community.nir import nir_analyze_collection_tool as package_tool
    from deerflow.community.nir.tools import nir_analyze_collection_tool as facade_tool

    config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in config["tools"] if item["name"] == "nir_analyze_collection")
    assert entry["use"] == "deerflow.community.nir.modeling:nir_analyze_collection_tool"
    assert package_tool is facade_tool


def test_collection_tool_analyzes_all_mat_subsets_in_one_compact_result(tmp_path: Path) -> None:
    from deerflow.community.nir.modeling import nir_analyze_collection_tool

    source = tmp_path / "collection.mat"
    source.write_bytes(b"MATLAB placeholder")
    output_dir = tmp_path / "outputs" / "nir_collection"
    calls: list[str] = []

    def fake_analyze(**kwargs) -> str:
        subset = kwargs["subset"]
        assert kwargs["method"] == "auto"
        assert kwargs["wavelength_selection"] == "auto"
        calls.append(subset)
        return json.dumps(_metrics(subset=subset), ensure_ascii=False)

    def resolve(_runtime, path: str, *, read_only: bool) -> str:
        assert read_only is True
        assert path == "/mnt/user-data/uploads/collection.mat"
        return str(source)

    with (
        patch("deerflow.community.nir.modeling._resolve", side_effect=resolve),
        patch("deerflow.community.nir.modeling._resolve_writable_dir", return_value=str(output_dir)),
        patch(
            "deerflow.community.nir.modeling.inspect_file",
            return_value=json.dumps({"format": "mat", "available_subsets": ["R562", "R568"]}),
        ),
        patch("deerflow.community.nir.modeling._analyze_collection_item", side_effect=fake_analyze),
    ):
        result = nir_analyze_collection_tool.func(
            runtime=MagicMock(),
            data_path="/mnt/user-data/uploads/collection.mat",
            subsets="auto",
            auto_preprocess=True,
            output_dir="/mnt/user-data/outputs/nir_collection",
            domain="pharma",
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["subset_count"] == 2
    assert calls == ["R562", "R568"]
    assert payload["primary_subset"] == "R562"
    assert payload["model_path"].endswith("/R562/model.pkl")
    assert payload["report"].endswith("/collection_summary.md")
    assert (output_dir / "collection_summary.md").is_file()
    assert "report_content" not in result
    assert len(result) < 12_000
