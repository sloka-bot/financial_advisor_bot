"""Read-only holdings-file preview endpoint."""

from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from openpyxl.utils.exceptions import InvalidFileException

from backend.api.schemas import ImportPortfolioRequest

router = APIRouter()


@router.post("/api/import-preview")
async def import_preview(request: Request, filename: str):
    """Validate a spreadsheet without modifying the user's saved holdings."""
    import io

    payload = await request.body()
    if len(payload) > 2_000_000:
        raise HTTPException(413, "The file must be smaller than 2 MB")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(422, "Use a CSV or XLSX file with ticker, shares and price columns")
    try:
        stream = io.BytesIO(payload)
        frame = pd.read_csv(stream) if suffix == ".csv" else pd.read_excel(stream, engine="openpyxl")
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if not {"ticker", "shares", "price"}.issubset(frame.columns):
            raise ValueError("Required columns: ticker, shares, price")
        if len(frame) == 0 or len(frame) > 500:
            raise ValueError("Provide between 1 and 500 holdings")
        parsed = ImportPortfolioRequest(user_id="", holdings=frame[["ticker", "shares", "price"]].to_dict("records"))
        holdings = [h.model_dump() for h in parsed.holdings]
        if len({h["ticker"] for h in holdings}) != len(holdings):
            raise ValueError("Combine duplicate ticker rows before importing")
        return {"holdings": holdings}
    except (ValueError, OSError, KeyError, TypeError, BadZipFile, InvalidFileException, SyntaxError) as exc:
        raise HTTPException(422, "Invalid holdings file: " + str(exc)) from exc
