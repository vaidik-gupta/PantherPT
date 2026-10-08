from uuid import UUID

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import Field, ValidationError

from src.services.inference.auth import require_api_key
from src.services.inference.schemas import InferenceRequest
from src.services.inference.server import add_request_id_header, app, lifespan


API_KEY = "test-inference-key"
AUTH_HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", API_KEY)
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/health", None),
        ("GET", "/ready", None),
        ("GET", "/models", None),
        ("POST", "/infer", {"prompt": "Hello"}),
    ],
)
def test_endpoints_are_explicitly_unimplemented(client, method, path, payload):
    response = client.request(method, path, json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"]
    assert UUID(response.headers["X-Request-ID"]).version == 4


@pytest.mark.parametrize(
    "payload",
    [{}, {"prompt": ""}, {"prompt": 123}, {"prompt": "Hello", "model": ""}],
)
def test_inference_rejects_invalid_requests(client, payload):
    response = client.post("/infer", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 422
    assert UUID(response.headers["X-Request-ID"]).version == 4


def test_request_supports_subclassing():
    class GenerationRequest(InferenceRequest):
        max_new_tokens: int = Field(default=32, gt=0)

    request = GenerationRequest(prompt="Hello", model="gpt2", max_new_tokens=8)
    assert request.prompt == "Hello"
    assert request.model == "gpt2"
    assert request.max_new_tokens == 8
    assert InferenceRequest(prompt="Hello").model is None

    with pytest.raises(ValidationError):
        GenerationRequest(prompt="Hello", max_new_tokens=0)


def test_openapi_exposes_endpoint_shells_and_request_schema():
    schema = app.openapi()
    assert set(schema["paths"]) == {"/health", "/ready", "/models", "/infer"}
    operation = schema["paths"]["/infer"]["post"]
    assert "501" in operation["responses"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/InferenceRequest"
    }
    assert schema["components"]["securitySchemes"]["InferenceAPIKey"] == {
        "type": "apiKey",
        "description": "API key configured through INFERENCE_API_KEY.",
        "in": "header",
        "name": "X-API-Key",
    }
    for path in schema["paths"].values():
        for endpoint in path.values():
            assert endpoint["security"] == [{"InferenceAPIKey": []}]
            assert "401" in endpoint["responses"]


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/health", None),
        ("GET", "/ready", None),
        ("GET", "/models", None),
        ("POST", "/infer", {"prompt": "Hello"}),
    ],
)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-API-Key": ""},
        {"X-API-Key": "wrong-key"},
        {"X-API-Key": API_KEY + "-suffix"},
        {"X-API-Key": " " + API_KEY},
        {"X-API-Key": b"\xff"},
        [("X-API-Key", API_KEY), ("X-API-Key", "wrong-key")],
        [("X-API-Key", "wrong-key"), ("X-API-Key", API_KEY)],
    ],
)
def test_all_endpoints_require_api_key(client, method, path, payload, headers):
    response = client.request(method, path, json=payload, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing or invalid API key."}
    assert response.headers["WWW-Authenticate"] == "APIKey"
    assert API_KEY not in response.text
    assert "X-Request-ID" not in response.headers


def test_api_key_header_is_case_insensitive(client):
    response = client.get("/models", headers={"x-api-key": API_KEY})

    assert response.status_code == 501


def test_api_key_in_query_is_not_accepted(client):
    response = client.get("/models", params={"api_key": API_KEY})

    assert response.status_code == 401


def test_authentication_precedes_request_validation(client):
    response = client.post("/infer", json={})

    assert response.status_code == 401


def test_configured_key_is_loaded_at_each_startup(monkeypatch):
    for key in ("first-configured-key", "rotated-configured-key"):
        monkeypatch.setenv("INFERENCE_API_KEY", key)
        with TestClient(app) as client:
            assert client.get("/models", headers={"X-API-Key": key}).status_code == 501
            assert client.get("/models", headers=AUTH_HEADERS).status_code == 401


@pytest.mark.parametrize("api_key", [None, "", " \t\n"])
def test_startup_requires_configured_key(monkeypatch, api_key):
    if api_key is None:
        monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("INFERENCE_API_KEY", api_key)

    with pytest.raises(RuntimeError, match="INFERENCE_API_KEY must be set"):
        with TestClient(app):
            pass


def test_api_key_is_removed_on_shutdown(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", API_KEY)
    with TestClient(app):
        assert app.state.inference_api_key == API_KEY.encode("utf-8")

    assert not hasattr(app.state, "inference_api_key")


def test_documentation_is_public(client):
    for path in ("/docs", "/openapi.json"):
        response = client.get(path)
        assert response.status_code == 200
        assert "X-Request-ID" not in response.headers


def test_request_ids_are_unique_and_ignore_client_supplied_ids(client):
    supplied_id = "12345678-1234-4234-8234-123456789abc"
    request_ids = {
        client.get(
            "/models", headers={**AUTH_HEADERS, "X-Request-ID": supplied_id}
        ).headers["X-Request-ID"]
        for _ in range(5)
    }

    assert len(request_ids) == 5
    assert supplied_id not in request_ids
    assert all(UUID(request_id).version == 4 for request_id in request_ids)


def test_uuid_is_not_generated_when_authentication_fails(client, monkeypatch):
    def unexpected_uuid():
        pytest.fail("UUID must not be generated before successful authentication.")

    monkeypatch.setattr("src.services.inference.auth.uuid4", unexpected_uuid)
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"X-API-Key": "wrong"}).status_code == 401


def test_endpoint_receives_same_uuid_as_response_header(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", API_KEY)
    test_app = FastAPI(
        lifespan=lifespan, dependencies=[Depends(require_api_key)]
    )
    test_app.middleware("http")(add_request_id_header)

    @test_app.get("/request-id")
    async def request_id(request: Request):
        assert isinstance(request.state.request_id, UUID)
        return {"request_id": str(request.state.request_id)}

    with TestClient(test_app) as client:
        response = client.get("/request-id", headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert UUID(response.headers["X-Request-ID"]).version == 4
