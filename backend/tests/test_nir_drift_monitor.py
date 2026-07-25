"""State-machine contracts for continuous NIR drift alerting."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from deerflow.community.nir._drift_monitor import (
    DriftAlertPolicy,
    observe_prediction_drift,
)
from deerflow.community.nir._prediction_audit import verify_prediction_audit


def _observe(
    state_path: Path,
    alerts_path: Path,
    score: float,
    *,
    model_sha256: str = "a" * 64,
) -> dict:
    return observe_prediction_drift(
        state_path=state_path,
        alerts_path=alerts_path,
        model_sha256=model_sha256,
        model_path="/mnt/user-data/outputs/model.pkl",
        data_sha256="b" * 64,
        drift_score=score,
        n_samples=20,
        flagged_count=round(score * 20),
        attribution={"thread_id": "thread-1", "run_id": "run-1"},
        policy=DriftAlertPolicy(
            alert_threshold=0.5,
            consecutive_alert_batches=2,
            recovery_threshold=0.1,
            consecutive_recovery_batches=2,
        ),
    )


def test_continuous_drift_alerts_only_on_state_transitions(tmp_path: Path) -> None:
    state_path = tmp_path / "prediction-drift-state.json"
    alerts_path = tmp_path / "prediction-drift-alerts.jsonl"

    first = _observe(state_path, alerts_path, 0.7)
    entered = _observe(state_path, alerts_path, 0.8)
    repeated = _observe(state_path, alerts_path, 0.9)
    recovering = _observe(state_path, alerts_path, 0.05)
    recovered = _observe(state_path, alerts_path, 0.0)

    assert first["state"] == "watch"
    assert first["alert_emitted"] is False
    assert entered["state"] == "alert"
    assert entered["transition"] == "alert_started"
    assert entered["alert_emitted"] is True
    assert repeated["state"] == "alert"
    assert repeated["alert_emitted"] is False
    assert recovering["state"] == "recovering"
    assert recovered["state"] == "normal"
    assert recovered["transition"] == "alert_recovered"

    alerts = [json.loads(line) for line in alerts_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in alerts] == [
        "drift_alert_started",
        "drift_alert_recovered",
    ]
    assert alerts[0]["severity"] == "critical"
    assert alerts[0]["recommended_action"] == "review_model_and_pause_automatic_decisions"
    assert verify_prediction_audit(alerts_path)["event_count"] == 2


def test_continuous_drift_state_is_model_scoped_and_atomic(tmp_path: Path) -> None:
    state_path = tmp_path / "prediction-drift-state.json"
    alerts_path = tmp_path / "prediction-drift-alerts.jsonl"

    def observe(index: int) -> None:
        _observe(
            state_path,
            alerts_path,
            0.0,
            model_sha256=("a" if index % 2 == 0 else "c") * 64,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(observe, range(40)))

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state["models"]) == {"a" * 64, "c" * 64}
    assert state["models"]["a" * 64]["total_batches"] == 20
    assert state["models"]["c" * 64]["total_batches"] == 20
    assert len(state["models"]["a" * 64]["recent_scores"]) == 20


def test_concurrent_drift_batches_emit_one_transition_alert(tmp_path: Path) -> None:
    state_path = tmp_path / "prediction-drift-state.json"
    alerts_path = tmp_path / "prediction-drift-alerts.jsonl"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda _index: _observe(state_path, alerts_path, 0.9),
                range(20),
            )
        )

    alerts = [json.loads(line) for line in alerts_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in alerts] == ["drift_alert_started"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["models"]["a" * 64]["total_batches"] == 20
    assert state["models"]["a" * 64]["alert_active"] is True


def test_drift_alert_policy_reads_validated_environment(monkeypatch) -> None:
    monkeypatch.setenv("NIR_DRIFT_ALERT_THRESHOLD", "0.4")
    monkeypatch.setenv("NIR_DRIFT_ALERT_CONSECUTIVE_BATCHES", "4")
    monkeypatch.setenv("NIR_DRIFT_RECOVERY_THRESHOLD", "0.08")
    monkeypatch.setenv("NIR_DRIFT_RECOVERY_CONSECUTIVE_BATCHES", "3")

    policy = DriftAlertPolicy.from_environment()

    assert policy == DriftAlertPolicy(
        alert_threshold=0.4,
        consecutive_alert_batches=4,
        recovery_threshold=0.08,
        consecutive_recovery_batches=3,
    )
