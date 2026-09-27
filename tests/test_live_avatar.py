"""Avatar session boundaries and credential handling."""

import requests
from fastapi.testclient import TestClient

from backend import main
from backend.config import settings
from backend.infra import live_avatar


def configure(monkeypatch):
    """Supply fake credentials without contacting the provider."""
    monkeypatch.setattr(settings, "LIVEAVATAR_API_KEY", "test-secret")
    monkeypatch.setattr(settings, "LIVEAVATAR_AVATAR_ID", "test-avatar")
    monkeypatch.setattr(settings, "LIVEAVATAR_VOICE_ID", "test-voice")


def test_unconfigured_avatar_keeps_chat_available(monkeypatch):
    monkeypatch.setattr(settings, "LIVEAVATAR_API_KEY", "")
    client = TestClient(main.app)
    assert client.get("/api/avatar/config").json()["configured"] is False
    response = client.post("/api/avatar/session")
    assert response.json() == {"status": "disabled"}
    assert response.headers["cache-control"] == "no-store"


def test_avatar_uses_restricted_mode_and_never_returns_key(monkeypatch):
    configure(monkeypatch)

    def create(url, **kwargs):
        assert url == "https://api.liveavatar.com/v1/sessions/token"
        payload = kwargs["json"]
        assert payload["mode"] == "FULL"
        assert "context_id" not in payload["avatar_persona"]
        assert payload["max_session_duration"] == 120
        assert kwargs["headers"]["X-API-KEY"] == "test-secret"

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": {"session_token": "ephemeral-token"}}

        return Response()

    monkeypatch.setattr(requests, "post", create)
    result = TestClient(main.app).post("/api/avatar/session")
    assert result.json() == {"status": "ready", "session_token": "ephemeral-token"}
    assert "test-secret" not in result.text


def test_avatar_network_failure_is_recoverable(monkeypatch):
    configure(monkeypatch)

    def unavailable(*args, **kwargs):
        raise requests.Timeout("Private upstream diagnostic")

    monkeypatch.setattr(requests, "post", unavailable)
    assert live_avatar.create_session() == {"status": "unavailable"}


def test_avatar_rejects_malformed_token_response(monkeypatch):
    configure(monkeypatch)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": None}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    assert live_avatar.create_session() == {"status": "unavailable"}
