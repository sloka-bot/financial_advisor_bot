"""
main.py - FastAPI application entry point for the Financial Advisor Bot backend.

The backend exposes a REST API consumed by the vanilla HTML/JS frontend.
All ML training runs in a background thread so the API stays responsive
while the pipeline executes. Pipeline progress is polled by the frontend
via /api/pipeline-status.

Author: Sloka Mudunuru
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routers import (
    analysis,
    backtests,
    chat,
    evaluation,
    media,
    portfolios,
    system,
    users,
)
from backend.api.routers.imports import router as import_router
from backend.data.contracts import json_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
Path("data/logs").mkdir(parents=True, exist_ok=True)
_file_log = logging.FileHandler("data/logs/app.log")
_file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logging.getLogger().addHandler(_file_log)


class SafeJSONResponse(JSONResponse):
    """Serialize numerical results with non-finite values represented as null."""

    def render(self, content):
        """Sanitise numerical values before encoding the JSON response."""
        return super().render(json_safe(content))


app = FastAPI(title="Financial Advisor Bot API", version="1.1.0", default_response_class=SafeJSONResponse)


@app.middleware("http")
async def local_origin_only(request: Request, call_next):
    """Reject browser requests originating outside the local application."""
    origin = request.headers.get("origin")
    if origin and origin not in {"http://localhost:8000", "http://127.0.0.1:8000"}:
        return JSONResponse({"detail": "Open the app at http://localhost:8000/app/"}, status_code=403)
    return await call_next(request)


# Serve the local frontend from /app so API calls use the same origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


app.include_router(import_router)
app.include_router(system.router)
app.include_router(users.router)
app.include_router(portfolios.router)
app.include_router(backtests.router)
app.include_router(analysis.router)
app.include_router(chat.router)
app.include_router(media.router)
app.include_router(evaluation.router)

app.mount(
    "/app", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="frontend"
)
