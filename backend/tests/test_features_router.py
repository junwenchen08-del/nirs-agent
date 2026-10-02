from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.deps import get_config
from app.gateway.routers import features


def _app_with_config(*, agents_api_enabled: bool, nir_library_enabled: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(features.router)
    fake_config = SimpleNamespace(
        agents_api=SimpleNamespace(enabled=agents_api_enabled),
        nir_library=SimpleNamespace(enabled=nir_library_enabled),
    )
    app.dependency_overrides[get_config] = lambda: fake_config
    return app


def test_features_reports_agents_api_enabled() -> None:
    with TestClient(_app_with_config(agents_api_enabled=True, nir_library_enabled=False)) as client:
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json() == {
        "agents_api": {"enabled": True},
        "nir_library": {"enabled": False},
    }


def test_features_reports_agents_api_disabled() -> None:
    with TestClient(_app_with_config(agents_api_enabled=False, nir_library_enabled=True)) as client:
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json() == {
        "agents_api": {"enabled": False},
        "nir_library": {"enabled": True},
    }
