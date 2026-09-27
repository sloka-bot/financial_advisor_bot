"""HTTP routes for media."""

from fastapi import APIRouter, Response

from backend.config import settings
from backend.infra import live_avatar

router = APIRouter()


@router.get("/api/avatar/config")
def avatar_config():
    """Expose connection readiness without exposing credentials."""
    return {"configured": live_avatar.is_configured(), "sandbox": settings.LIVEAVATAR_SANDBOX}


@router.post("/api/avatar/session")
def avatar_session(response: Response):
    """Return a short-lived token for the browser video session."""
    response.headers["Cache-Control"] = "no-store"
    return live_avatar.create_session()
