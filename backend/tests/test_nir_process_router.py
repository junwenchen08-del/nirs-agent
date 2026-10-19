from unittest.mock import AsyncMock

from _router_auth_helpers import make_authed_test_app
from fastapi.testclient import TestClient

from app.gateway.routers import nir_process
from deerflow.community.nir.process import ProcessStore
from deerflow.config.paths import Paths


def test_process_api_enforces_ownership_versions_and_etag(tmp_path, monkeypatch):
    paths = Paths(tmp_path)
    monkeypatch.setattr(nir_process, "get_paths", lambda: paths)
    monkeypatch.setattr(nir_process, "get_current_user", AsyncMock(return_value="alice"))
    store = ProcessStore(paths.thread_dir("thread", user_id="alice") / "nir-process")
    attempt = store.begin("project", "call", "nir_analyze")
    store.publish(attempt, [{"step_key": "audit", "facts": {"n_samples": 2}, "charts": [{"type": "distribution", "x": [1], "y": [2], "n_total": 2, "n_returned": 1, "sampling": "full_population_histogram_v1"}]}])
    app = make_authed_test_app()
    app.include_router(nir_process.router)
    client = TestClient(app)
    base = f"/api/threads/thread/nir-process/runs/{attempt['run_id']}"
    response = client.get(base)
    assert response.status_code == 200
    assert client.get(base, headers={"If-None-Match": response.headers["etag"]}).status_code == 304
    step_id = response.json()["attempts"][0]["steps"][0]["step_execution_id"]
    step = client.get(f"{base}/steps/{step_id}").json()
    ref = step["charts"][0]
    assert client.get(f"{base}/charts/{ref['chart_id']}?version={ref['version']}").status_code == 200
    assert client.get(f"{base}/charts/{ref['chart_id']}?version=wrong").status_code == 409
    assert client.get(f"{base}/charts/chart_nonexistent?version={ref['version']}").status_code == 404
    assert client.get(base.replace("thread/nir-process", "other-thread/nir-process")).status_code == 404
    assert client.get(f"{base}/steps/{step_id}/candidates?offset=-1").status_code == 422
    app.state.thread_store.check_access = AsyncMock(return_value=False)
    assert client.get(base).status_code == 404
    assert client.get(f"{base}/steps/{step_id}").status_code == 404
    assert client.get(f"{base}/charts/{ref['chart_id']}?version={ref['version']}").status_code == 404


def test_empty_old_thread_has_no_invented_process_and_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(nir_process, "get_paths", lambda: Paths(tmp_path))
    monkeypatch.setattr(nir_process, "get_current_user", AsyncMock(return_value="alice"))
    app = make_authed_test_app()
    app.include_router(nir_process.router)
    client = TestClient(app)
    assert client.get("/api/threads/legacy/nir-process/runs").json()["data"] == []
    assert not (tmp_path / "users/alice/threads/legacy/nir-process").exists()
    monkeypatch.setattr(nir_process, "get_current_user", AsyncMock(return_value=None))
    assert client.get("/api/threads/legacy/nir-process/runs").status_code == 401
