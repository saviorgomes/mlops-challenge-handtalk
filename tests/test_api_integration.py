import sys
import types

from fastapi.testclient import TestClient


if "tensorflow" not in sys.modules:
    tf_stub = types.SimpleNamespace(
        saved_model=types.SimpleNamespace(load=lambda *args, **kwargs: object()),
        constant=lambda value: value,
    )
    sys.modules["tensorflow"] = tf_stub

import inference_api.main as main_module
from inference_api.main import app
from inference_api.metrics import metrics


def _client() -> TestClient:
    """Cria um cliente sem acionar o lifespan da aplicação."""
    return TestClient(app, raise_server_exceptions=False)


def _reset_metrics() -> None:
    """Zera os contadores compartilhados entre os testes."""
    with metrics._lock:
        metrics.requests_total = 0
        metrics.errors_total = 0
        metrics.translations_total = 0


class _FakeManager:
    """Fake simples para exercitar reload/predict sem carregar TensorFlow."""

    def __init__(self) -> None:
        self.artifacts_dir = "artifacts"
        self.default_run_id = "fake-run"
        self._lock = None
        self._translator = None
        self._run_id = None

    def current_run_id(self):
        return self._run_id

    def is_loaded(self) -> bool:
        return self._run_id is not None

    def load(self, run_id=None):
        self._run_id = run_id or self.default_run_id
        self._translator = object()
        return self._run_id

    def translate(self, text: str):
        if not self._run_id:
            raise FileNotFoundError("modelo não carregado")
        return f"translated:{text}", self._run_id


class _FailingReloadManager(_FakeManager):
    """Fake que simula falha de recarga de modelo."""

    def load(self, run_id=None):
        raise FileNotFoundError("run_id inválido")


def test_reload_predict_and_metrics_flow(monkeypatch):
    """Valida o fluxo integrado de reload, predict e metrics."""
    _reset_metrics()
    fake_manager = _FakeManager()
    monkeypatch.setattr(main_module, "manager", fake_manager)

    client = _client()

    reload_response = client.post("/reload", json={"run_id": "run-test", "artifacts_dir": None})
    assert reload_response.status_code == 200
    assert reload_response.json() == {"status": "reloaded", "run_id": "run-test"}

    predict_response = client.post("/predict", json={"text": "ola mundo"})
    assert predict_response.status_code == 200
    body = predict_response.json()
    assert body["translation"] == "translated:ola mundo"
    assert body["run_id"] == "run-test"
    assert body["latency_ms"] >= 0

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert metrics_response.json() == {
        "requests_total": 1,
        "errors_total": 0,
        "translations_total": 1,
    }


def test_reload_invalid_run_id_returns_500(monkeypatch):
    """Valida erro de recarga quando o modelo informado não existe."""
    _reset_metrics()
    monkeypatch.setattr(main_module, "manager", _FailingReloadManager())

    client = _client()
    response = client.post("/reload", json={"run_id": "run-invalido", "artifacts_dir": None})

    assert response.status_code == 500
    assert "run_id inválido" in response.json()["detail"]


def test_predict_without_loaded_model_increments_error_metrics(monkeypatch):
    """Valida o fluxo integrado de erro em /predict sem modelo carregado."""
    _reset_metrics()
    monkeypatch.setattr(main_module, "manager", _FakeManager())

    client = _client()
    before = client.get("/metrics").json()

    response = client.post("/predict", json={"text": "teste sem modelo"})
    after = client.get("/metrics").json()

    assert response.status_code == 503
    assert "modelo não carregado" in response.json()["detail"]
    assert after["requests_total"] == before["requests_total"] + 1
    assert after["errors_total"] == before["errors_total"] + 1
    assert after["translations_total"] == before["translations_total"]
