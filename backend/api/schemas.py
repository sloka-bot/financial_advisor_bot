"""Validated request bodies for the HTTP API."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VALID_PROFILES = {"conservative", "moderate", "aggressive"}


def _normalize_risk_profile(v: str) -> str:
    """Validate/normalise a risk profile to one of the three supported tiers."""
    v = (v or "moderate").strip().lower()
    if v not in VALID_PROFILES:
        raise ValueError(f"risk_profile must be one of {list(VALID_PROFILES)}")
    return v


class RunRequest(BaseModel):
    """Validate the scope, risk profile and budget for a training run."""

    risk_profile: str = "moderate"
    budget: float = Field(10000.0, gt=0, allow_inf_nan=False, description="must be a positive amount")
    top_n: int = Field(10, ge=1, le=100)
    scope: str = Field("sample", pattern="^(sample|all)$")
    user_id: str = ""

    @field_validator("risk_profile")
    @classmethod
    def validate_risk_profile(cls, v):
        return _normalize_risk_profile(v)


class ChatRequest(BaseModel):
    """Carry the user question and optional explanation context."""

    message: str
    mode: str = "normal"
    portfolio: dict | None = None
    context: str | None = None
    user_id: str = "dev-user"


class ProfileRequest(BaseModel):
    """Validate the initial investor profile and available budget."""

    name: str = ""
    risk_profile: str = "moderate"
    investment_horizon: str = "5-10 years"
    goal: str = "growth"
    budget: float = Field(10000.0, gt=0, allow_inf_nan=False)
    monthly_contribution: float = Field(0.0, ge=0, allow_inf_nan=False)

    @field_validator("risk_profile")
    @classmethod
    def validate_risk_profile(cls, v):
        return _normalize_risk_profile(v)


class ImportHoldingItem(BaseModel):
    """Validate one positive holding and its average purchase price."""

    ticker: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,14}$")
    shares: float = Field(gt=0, allow_inf_nan=False)
    price: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value):
        """Normalise ticker case and the provider-specific class separator."""
        return value.upper().replace(".", "-")


class ImportPortfolioRequest(BaseModel):
    """Carry validated holdings for a saved user portfolio."""

    user_id: str
    holdings: list[ImportHoldingItem]


class ApprovalRequest(BaseModel):
    """Identify a saved recommendation and its owning user."""

    user_id: str
    rec_id: int


class BuyRequest(BaseModel):
    """Request a whole-share addition to a saved holding."""

    ticker: str = Field(min_length=1, max_length=12, pattern=r"^[A-Za-z0-9.^-]+$")
    shares: int = Field(gt=0, strict=True)


class SellRequest(BaseModel):
    """Identify a held ticker to sell from the saved portfolio."""

    ticker: str


class ProfileUpdateRequest(BaseModel):
    """Validate only the profile fields supplied for an update."""

    name: str | None = None
    risk_profile: str | None = None
    investment_horizon: str | None = None
    goal: str | None = None
    budget: float | None = Field(None, gt=0, allow_inf_nan=False)
    monthly_contribution: float | None = Field(None, ge=0, allow_inf_nan=False)

    @field_validator("risk_profile")
    @classmethod
    def validate_profile(cls, value):
        """Validate a supplied risk tier while preserving an omitted value."""
        return _normalize_risk_profile(value) if value is not None else value


class PortfolioRequest(BaseModel):
    """Validate risk profile and budget for portfolio and recommendation requests."""

    risk_profile: str = "moderate"
    budget: float = Field(10000.0, gt=0, allow_inf_nan=False)
    top_n: int = Field(10, ge=1, le=100)
    user_id: str = ""

    @field_validator("risk_profile")
    @classmethod
    def validate_risk_profile(cls, v):
        return _normalize_risk_profile(v)


Ticker = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,14}$")]


class BacktestRequest(BaseModel):
    """Validate selected tickers and starting capital for the historical simulator."""

    tickers: list[Ticker] = Field(min_length=1, max_length=100)
    capital: float = Field(10000, gt=0, allow_inf_nan=False)
    portfolio_mode: bool = False


class BacktestPredictRequest(BaseModel):
    """Validate one ticker, a historical start date and an observation count."""

    ticker: Ticker
    start_date: date
    window: int = Field(60, ge=5, le=3000)


class BacktestRangeRequest(BaseModel):
    """Validate a dated evaluation window and supported forecast horizon."""

    tickers: list[Ticker] = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date
    horizon: Literal[1, 5, 21] = 21
    feature_set: Literal["technical", "both"] = "both"

    @model_validator(mode="after")
    def ordered_dates(self):
        """Require the evaluation end to be on or after its start."""
        if self.end_date < self.start_date:
            raise ValueError("End date must not precede start date")
        return self
