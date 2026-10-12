"""A locked holdout is physically absent from model-search execution."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.process import ProcessStore


def partitions():
    rng = np.random.default_rng(905)
    X = rng.normal(size=(60, 8))
    y = 3 * X[:, 0] - X[:, 1] + rng.normal(scale=0.03, size=60)
    return dict(X_cal=X[:36], y_cal=y[:36], X_val=X[36:48], y_val=y[36:48], X_test=X[48:], y_test=y[48:], wv=np.arange(8, dtype=float))


def store(tmp_path):
    return SearchSessionStore(ProcessStore(tmp_path / "private"))


@pytest.mark.parametrize("family", ["pls", "ridge", "svr", "et"])
def test_search_never_opens_holdout_and_test_target_changes_are_irrelevant(tmp_path, monkeypatch, family):
    first, second = store(tmp_path / "a"), store(tmp_path / "b")
    data = partitions()
    a = first.prepare("project", data, cv_folds=3)
    data["y_test"] = data["y_test"] + 10000
    b = second.prepare("project", data, cv_folds=3)
    original = np.load
    opened = []

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        assert "holdout" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy)
    ra = first.search(a["session_id"], "call-a", method=family, steps=[], max_components=3)
    rb = second.search(b["session_id"], "call-b", method=family, steps=[], max_components=3)
    assert ra["metrics"] == rb["metrics"]
    assert ra["parameters"] == rb["parameters"]
    assert ra["metrics"]["test_status"] == "sealed"
    assert opened and all("search.npz" in name for name in opened)


def test_mutating_already_locked_test_targets_cannot_change_search(tmp_path):
    service = store(tmp_path)
    session_id = service.prepare("project", partitions(), cv_folds=3)["session_id"]
    before = service.search(session_id, "before", method="pls", steps=[], max_components=3)
    holdout_path = service._dir(session_id) / "holdout.npz"
    with np.load(holdout_path, allow_pickle=False) as archive:
        X_test, y_test = archive["X_test"], archive["y_test"]
    np.savez(holdout_path, X_test=X_test, y_test=y_test + 10000)
    after = service.search(session_id, "after", method="pls", steps=[], max_components=3)
    assert before["metrics"] == after["metrics"]
    assert before["parameters"] == after["parameters"]
    service.freeze(session_id, after["experiment_id"])
    with pytest.raises(ValueError, match="integrity"):
        service.finalize(session_id)  # Locked Test tampering must fail closed.


def test_freeze_is_required_and_test_is_consumed_once(tmp_path, monkeypatch):
    service = store(tmp_path)
    session = service.prepare("project", partitions(), cv_folds=3)["session_id"]
    with pytest.raises(ValueError, match="frozen"):
        service.finalize(session)
    result = service.search(session, "one", method="pls", steps=[], max_components=3)
    service.freeze(session, result["experiment_id"])
    final = service.finalize(session)
    assert final["metrics"]["test"]["scope"] == "final_test"
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("A second finalization must not load data"))
    assert service.finalize(session) == final
    with pytest.raises(ValueError, match="closed"):
        service.search(session, "two", method="ridge", steps=[])


def test_prepare_and_search_replays_do_not_overwrite_history(tmp_path):
    service = store(tmp_path)
    marker = tmp_path / "old-report.md"
    marker.write_text("historical report", encoding="utf-8")
    session = service.prepare("project", partitions(), cv_folds=3)
    assert service.prepare("project", partitions(), cv_folds=3) == session
    changed = partitions()
    changed["y_test"] += 1000
    with pytest.raises(ValueError, match="locked"):
        service.prepare("project", changed, cv_folds=3)
    first = service.search(session["session_id"], "one", method="pls", steps=[], max_components=2)
    assert service.search(session["session_id"], "one", method="pls", steps=[], max_components=2) == first
    with pytest.raises(ValueError, match="replay"):
        service.search(session["session_id"], "one", method="ridge", steps=[])
    assert marker.read_text(encoding="utf-8") == "historical report"
    assert "test" not in first["metrics"]
    assert "RMSEP" not in json.dumps(first)


def test_failed_final_evaluation_does_not_reopen_search(tmp_path, monkeypatch):
    service = store(tmp_path)
    session = service.prepare("project", partitions(), cv_folds=3)["session_id"]
    result = service.search(session, "one", method="pls", steps=[], max_components=2)
    service.freeze(session, result["experiment_id"])
    monkeypatch.setattr(np, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("IO failure")))
    with pytest.raises(RuntimeError, match="IO failure"):
        service.finalize(session)
    with pytest.raises(ValueError, match="consumed"):
        service.finalize(session)
    with pytest.raises(ValueError, match="closed"):
        service.search(session, "two", method="pls", steps=[])


def test_concurrent_finalization_reads_holdout_at_most_once(tmp_path, monkeypatch):
    service = store(tmp_path)
    session_id = service.prepare("project", partitions(), cv_folds=3)["session_id"]
    selected = service.search(session_id, "one", method="pls", steps=[], max_components=2)
    service.freeze(session_id, selected["experiment_id"])
    reads = []
    original = np.load

    def spy(path, *args, **kwargs):
        if "holdout.npz" in str(path):
            reads.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy)

    def finish():
        try:
            return service.finalize(session_id)
        except ValueError as error:
            assert "consumed" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: finish(), range(2)))
    assert any(result is not None for result in results)
    assert len(reads) == 1


def test_hard_fit_budget_rejects_more_fits_without_test_access(tmp_path, monkeypatch):
    service = store(tmp_path)
    session_id = service.prepare("project", partitions(), cv_folds=3, max_fits=2)["session_id"]
    original = np.load

    def spy(path, *args, **kwargs):
        assert "holdout" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy)
    with pytest.raises(ValueError, match="fit budget"):
        service.search(session_id, "one", method="pls", steps=[], max_components=2)
    with service.process._connection() as db:
        assert service._session(db, session_id)["fits"] == 2
        row = db.execute("SELECT status,spec FROM search_experiments").fetchone()
        assert row["status"] == "failed" and json.loads(row["spec"])["method"] == "pls"
    with pytest.raises(ValueError, match="consumed"):
        service.search(session_id, "one", method="pls", steps=[], max_components=2)
