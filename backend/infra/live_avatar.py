"""Server-side access to restricted LiveAvatar conversation sessions."""

import requests

from backend.config import settings


def is_configured() -> bool:
    """Require the key, avatar and voice before creating a session."""
    return all((settings.LIVEAVATAR_API_KEY, settings.LIVEAVATAR_AVATAR_ID, settings.LIVEAVATAR_VOICE_ID))


def create_session() -> dict:
    """Create an ephemeral token without exposing the account API key."""
    if not is_configured():
        return {"status": "disabled"}
    try:
        response = requests.post(
            "https://api.liveavatar.com/v1/sessions/token",
            headers={"X-API-KEY": settings.LIVEAVATAR_API_KEY},
            json={
                "mode": "FULL",
                "avatar_id": settings.LIVEAVATAR_AVATAR_ID,
                "is_sandbox": settings.LIVEAVATAR_SANDBOX,
                "max_session_duration": 120,
                "avatar_persona": {
                    "voice_id": settings.LIVEAVATAR_VOICE_ID,
                    "language": "en",
                },
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        token = data.get("session_token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token.strip():
            return {"status": "unavailable"}
        return {"status": "ready", "session_token": token}
    except (requests.RequestException, ValueError):
        return {"status": "unavailable"}
