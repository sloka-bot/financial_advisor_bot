"""Server-side access to restricted LiveAvatar conversation sessions."""

import logging

import requests

from backend.config import settings

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """Require the key, avatar and voice before creating a session."""
    return all((settings.LIVEAVATAR_API_KEY, settings.LIVEAVATAR_AVATAR_ID, settings.LIVEAVATAR_VOICE_ID))


def create_session() -> dict:
    """Create an ephemeral token without exposing the account API key."""
    if not is_configured():
        missing = [
            name
            for name, value in (
                ("LIVEAVATAR_API_KEY", settings.LIVEAVATAR_API_KEY),
                ("LIVEAVATAR_AVATAR_ID", settings.LIVEAVATAR_AVATAR_ID),
                ("LIVEAVATAR_VOICE_ID", settings.LIVEAVATAR_VOICE_ID),
            )
            if not value
        ]
        logger.warning("LiveAvatar disabled - missing .env values: %s", ", ".join(missing))
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
        if response.status_code != 200:
            # Include the provider's error details when session creation fails.
            logger.warning(
                "LiveAvatar token request failed: HTTP %s - %s",
                response.status_code,
                response.text[:300],
            )
            return {"status": "unavailable"}
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        token = data.get("session_token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token.strip():
            logger.warning("LiveAvatar response carried no session_token: %s", str(payload)[:300])
            return {"status": "unavailable"}
        return {"status": "ready", "session_token": token}
    except requests.RequestException as exc:
        logger.warning("LiveAvatar request error: %s", exc)
        return {"status": "unavailable"}
    except ValueError as exc:
        logger.warning("LiveAvatar returned invalid JSON: %s", exc)
        return {"status": "unavailable"}
