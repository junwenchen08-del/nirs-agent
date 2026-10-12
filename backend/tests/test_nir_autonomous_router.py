from unittest.mock import AsyncMock

from _router_auth_helpers import make_authed_test_app
from fastapi.testclient import TestClient
from test_nir_delivery_freeze import stopped_session

from app.gateway.routers import nir_process
from deerflow.config.paths import Paths


def test_cal_spectra_route_is_read_only_and_owner_session_experiment_bound(tmp_path, monkeypatch):
    from unittest.mock import patch

    from deerflow.community.nir.autonomous_view import AutonomousView
    from deerflow.community.nir.process import ProcessStore

    paths = Paths(tmp_path)
    root = paths.thread_dir("thread", user_id="alice") / "nir-process"
    # Seed the trusted executor in the exact owner/thread location. The read
    # route cannot create a session or recompute old charts.
    service, ctx, result, _ = stopped_session(tmp_path / "seed")
    root.mkdir(parents=True)
    (root / "index.sqlite3").write_bytes(service.search.process.db.read_bytes())
    sid, eid = ctx["session_id"], result["experiment_id"]
    destination = root / "sessions" / sid
    destination.mkdir(parents=True)
    (destination / (eid + ".spectra.json")).write_bytes((service.search._dir(sid) / (eid + ".spectra.json")).read_bytes())
    monkeypatch.setattr(nir_process, "get_paths", lambda: paths)
    monkeypatch.setattr(nir_process, "get_current_user", AsyncMock(return_value="alice"))
    app = make_authed_test_app()
    app.include_router(nir_process.router)
    client = TestClient(app)
    route = f"/api/threads/thread/nir-process/autonomous/{sid}/spectra/{eid}"
    before = (root / "index.sqlite3").read_bytes()
    with patch("numpy.load", side_effect=AssertionError("GET cannot read Test or Cal")), patch("joblib.load", side_effect=AssertionError("GET cannot load state")):
        assert client.get(route).json() == AutonomousView(ProcessStore(root)).spectra(sid, eid)
        assert client.get(route).json()["scope"] == "calibration"
        assert client.get(route.replace(eid, "exp_wrong")).status_code == 404
    assert (root / "index.sqlite3").read_bytes() == before
    app.state.thread_store.check_access.return_value = False
    assert client.get(route).status_code == 404


def test_owner_thread_session_bound_read_only_routes(tmp_path, monkeypatch):
    service, ctx, _result, stop = stopped_session(tmp_path / "seed")
    paths = Paths(tmp_path)
    monkeypatch.setattr(nir_process, "get_paths", lambda: paths)
    monkeypatch.setattr(nir_process, "get_current_user", AsyncMock(return_value="alice"))
    destination = paths.thread_dir("thread", user_id="alice") / "nir-process"
    destination.mkdir(parents=True)
    (destination / "index.sqlite3").write_bytes(service.search.process.db.read_bytes())
    before = (destination / "index.sqlite3").read_bytes()
    app = make_authed_test_app()
    app.include_router(nir_process.router)
    client = TestClient(app)
    base = f"/api/threads/thread/nir-process/autonomous/{ctx['session_id']}"
    result = client.get(base)
    assert result.status_code == 200
    assert client.get(base, headers={"If-None-Match": result.headers["etag"]}).status_code == 304
    assert client.get(base + "/decisions?offset=0&limit=1").json()["total"] == 2
    assert client.get(base + "/decisions/" + stop["decision_id"]).status_code == 200
    assert client.get(base + "/charts/final_test").status_code == 404
    assert client.get(base.replace(ctx["session_id"], "search_unknown")).status_code == 404
    assert client.get(base + "/decisions?offset=-1").status_code == 422
    assert client.get(base + "/decisions?limit=21").status_code == 422
    assert client.get(base, headers={"If-None-Match": '"stale"'}).status_code == 200
    assert (destination / "index.sqlite3").read_bytes() == before
    app.state.thread_store.check_access = AsyncMock(return_value=False)
    assert client.get(base).status_code == 404
    assert client.get(base + "/decisions/" + stop["decision_id"]).status_code == 404


def test_empty_legacy_autonomous_read_does_not_create_storage(tmp_path, monkeypatch):
    paths = Paths(tmp_path)
    monkeypatch.setattr(nir_process, "get_paths", lambda: paths)
    monkeypatch.setattr(nir_process, "get_current_user", AsyncMock(return_value="alice"))
    app = make_authed_test_app()
    app.include_router(nir_process.router)
    client = TestClient(app)
    assert client.get("/api/threads/legacy/nir-process/autonomous").json()["data"] == []
    assert not (paths.thread_dir("legacy", user_id="alice") / "nir-process").exists()
