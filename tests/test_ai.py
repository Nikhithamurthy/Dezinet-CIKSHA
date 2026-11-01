import math

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from tests.conftest import HEADERS


def test_status(client: TestClient) -> None:
    body = client.get("/v1/status").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_missing_api_key_returns_structured_401(client: TestClient) -> None:
    response = client.post("/v1/infer", json={"prompt": "hi"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_infer(client: TestClient) -> None:
    response = client.post("/v1/infer", json={"prompt": "hello world"}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["completion_tokens"] > 0
    assert "X-Process-Time-Ms" in response.headers


def test_validation_errors_are_clear(client: TestClient) -> None:
    blank = client.post("/v1/infer", json={"prompt": "   "}, headers=HEADERS)
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "validation_error"
    assert blank.json()["error"]["details"][0]["field"] == "prompt"

    bad_model = client.post(
        "/v1/infer", json={"prompt": "hi", "model": "bad model!"}, headers=HEADERS
    )
    assert bad_model.status_code == 422


def test_embeddings_are_unit_vectors(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"texts": ["a", "b"]}, headers=HEADERS)
    body = response.json()
    assert len(body["embeddings"]) == 2
    assert math.isclose(sum(x * x for x in body["embeddings"][0]), 1.0, rel_tol=1e-6)


def test_sse_stream(client: TestClient) -> None:
    with client.stream("POST", "/v1/stream", json={"prompt": "hi"}, headers=HEADERS) as r:
        body = "".join(r.iter_text())
    assert "event: token" in body
    assert "event: done" in body


def test_rate_limit(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(rate_limit_per_minute=2)
    try:
        codes = [
            client.post("/v1/infer", json={"prompt": "hi"}, headers=HEADERS).status_code
            for _ in range(3)
        ]
    finally:
        app.dependency_overrides.clear()
    assert codes == [200, 200, 429]
