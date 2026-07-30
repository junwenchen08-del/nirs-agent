"""Mandatory science and literature-evidence gate regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def _approved_workflow(model_path: str, metrics_path: str) -> dict:
    from deerflow.community.nir.workflow import start_workflow, transition_workflow

    state = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/workspace/calibration.npz",
        analyte="target",
        unit="a.u.",
        domain="default",
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=True,
        grade="A",
        model_path=model_path,
        metrics_path=metrics_path,
    )
    return transition_workflow(state, action="approve", notes="test approval")


def test_training_rejects_conflicting_duplicate_spectra(tmp_path: Path) -> None:
    from deerflow.community.nir.modeling import nir_train_model_tool

    rng = np.random.RandomState(31)
    X = rng.normal(size=(18, 6))
    y = np.linspace(1.0, 4.0, 18)
    X[7] = X[2]
    y[7] = y[2] + 3.0
    input_path = tmp_path / "calibration.npz"
    np.savez(input_path, X=X, y=y, wv=np.linspace(900.0, 1500.0, 6))

    with patch(
        "deerflow.community.nir.modeling._resolve",
        return_value=str(input_path),
    ):
        payload = json.loads(
            nir_train_model_tool.func(
                runtime=MagicMock(),
                input_path="/mnt/user-data/workspace/calibration.npz",
            )
        )

    assert "error" in payload
    assert "conflicting reference values" in payload["error"]


def test_registration_rejects_metrics_without_science_evidence(
    tmp_path: Path,
) -> None:
    from deerflow.community.nir.modeling import nir_register_model_tool

    model = tmp_path / "model.pkl"
    metrics = tmp_path / "metrics.json"
    registry = tmp_path / "registry.json"
    model.write_bytes(b"placeholder")
    metrics.write_text(
        json.dumps(
            {
                "training_data_hash": "a" * 64,
                "quality": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    virtual_model = "/mnt/user-data/outputs/model.pkl"
    virtual_metrics = "/mnt/user-data/outputs/metrics.json"
    virtual_registry = "/mnt/user-data/outputs/registry.json"
    paths = {
        virtual_model: str(model),
        virtual_metrics: str(metrics),
        virtual_registry: str(registry),
    }

    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        payload = json.loads(
            nir_register_model_tool.func(
                runtime=SimpleNamespace(
                    state={
                        "nir_workflow": _approved_workflow(
                            virtual_model,
                            virtual_metrics,
                        )
                    }
                ),
                model_id="unsafe-model",
                model_path=virtual_model,
                metrics_path=virtual_metrics,
                registry_path=virtual_registry,
            )
        )

    assert "error" in payload
    assert "scientific_validation is missing" in payload["error"]
    assert not registry.exists()


def test_registration_rejects_metrics_modified_after_model_binding(
    tmp_path: Path,
) -> None:
    from deerflow.community.nir._common import (
        _bind_model_metrics,
        _write_trusted_model_artifact,
    )
    from deerflow.community.nir.modeling import nir_register_model_tool

    model = tmp_path / "model.pkl"
    metrics = tmp_path / "metrics.json"
    registry = tmp_path / "registry.json"
    training_hash = "b" * 64
    valid_metrics = {
        "training_data_hash": training_hash,
        "scientific_validation": {
            "passed": True,
            "dataset": {"passed": True},
            "partition_separation": {"passed": True},
        },
        "reproducibility": {
            "schema_version": 1,
            "random_state": 42,
            "input_sha256": training_hash,
            "protocol": "test",
        },
        "quality": {"passed": True},
        "method": "pls",
    }
    _write_trusted_model_artifact({"model": "placeholder"}, str(model))
    metrics.write_text(json.dumps(valid_metrics), encoding="utf-8")
    _bind_model_metrics(
        str(model),
        str(metrics),
        training_data_hash=training_hash,
    )
    valid_metrics["method"] = "tampered"
    metrics.write_text(json.dumps(valid_metrics), encoding="utf-8")

    virtual_model = "/mnt/user-data/outputs/model.pkl"
    virtual_metrics = "/mnt/user-data/outputs/metrics.json"
    virtual_registry = "/mnt/user-data/outputs/registry.json"
    paths = {
        virtual_model: str(model),
        virtual_metrics: str(metrics),
        virtual_registry: str(registry),
    }
    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        payload = json.loads(
            nir_register_model_tool.func(
                runtime=SimpleNamespace(
                    state={
                        "nir_workflow": _approved_workflow(
                            virtual_model,
                            virtual_metrics,
                        )
                    }
                ),
                model_id="tampered-model",
                model_path=virtual_model,
                metrics_path=virtual_metrics,
                registry_path=virtual_registry,
            )
        )

    assert "error" in payload
    assert "exact provenance record" in payload["error"]
    assert not registry.exists()


def test_knowledge_decision_returns_citations_and_readiness(
    monkeypatch,
) -> None:
    from nir_core.knowledge.base import Chunk, SearchResult

    from deerflow.community.nir import knowledge

    def result(evidence_id: str, doc_id: str) -> SearchResult:
        return SearchResult(
            chunk=Chunk(
                id=evidence_id,
                content="A decision-relevant passage.",
                source=f"{doc_id}.pdf",
                metadata={
                    "doc_id": doc_id,
                    "title": f"Paper {doc_id}",
                    "authors": json.dumps(["Researcher"]),
                    "year": 2024,
                    "doi": f"10.1000/{doc_id}",
                    "page_start": 5,
                    "quality_tier": "B",
                    "review_status": "published",
                },
            ),
            score=0.8,
        )

    class Decision:
        results = (result("ev-1", "doc-1"), result("ev-2", "doc-2"))

        @staticmethod
        def diagnostics() -> dict:
            return {
                "abstained": False,
                "reason": "strong_score",
                "candidate_count": 2,
                "result_count": 2,
            }

    class Retriever:
        @staticmethod
        def search_with_diagnostics(query: str, top_k: int):  # noqa: ARG004
            return Decision()

    import nir_core.knowledge.config

    monkeypatch.setattr(
        nir_core.knowledge.config,
        "get_retriever",
        lambda: Retriever(),
    )
    monkeypatch.setattr(knowledge, "_knowledge_search_mode", None)

    payload = json.loads(
        knowledge.nir_search_knowledge_tool.func(
            runtime=MagicMock(),
            query="Should SNV replace MSC for this matrix?",
            purpose="decision",
        )
    )

    assessment = payload["evidence_assessment"]
    assert assessment["decision_allowed"] is True
    assert assessment["independent_document_count"] == 2
    assert assessment["citations"][0]["citation_marker"] == "[KB:ev-1]"
    assert assessment["usage_contract"]["must_report_conflicting_findings"] is True


def test_knowledge_http_timeout_is_configurable_and_defaults_safely(
    monkeypatch,
) -> None:
    from deerflow.community.nir import knowledge

    monkeypatch.delenv("NIR_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS", raising=False)
    assert knowledge._knowledge_http_timeout_seconds() == 60.0

    monkeypatch.setenv("NIR_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS", "90")
    assert knowledge._knowledge_http_timeout_seconds() == 90.0

    monkeypatch.setenv("NIR_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS", "invalid")
    assert knowledge._knowledge_http_timeout_seconds() == 60.0

    monkeypatch.setenv("NIR_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS", "0")
    assert knowledge._knowledge_http_timeout_seconds() == 60.0


def test_http_knowledge_search_uses_configured_timeout(monkeypatch) -> None:
    import urllib.request

    from deerflow.community.nir import knowledge

    observed: dict[str, object] = {}

    class Response:
        @staticmethod
        def read() -> bytes:
            return b'{"results": [], "count": 0, "query": "validation"}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
            return None

    def fake_urlopen(request, timeout):  # noqa: ANN001
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setenv("NIR_KNOWLEDGE_URL", "http://knowledge.test:8089")
    monkeypatch.setenv("NIR_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS", "75")
    monkeypatch.setattr(knowledge, "_knowledge_http_url", None)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    payload = json.loads(
        knowledge._search_knowledge_via_http(
            "validation",
            top_k=5,
            purpose="answer",
        )
    )

    assert payload["count"] == 0
    assert observed == {
        "url": "http://knowledge.test:8089/search",
        "timeout": 75.0,
    }
