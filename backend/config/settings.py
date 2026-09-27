"""
Central runtime configuration for the Financial Advisor Bot pipeline.

This module is the single source of truth for every tunable that affects
reproducibility: prediction horizons, the leakage embargo, cleaning thresholds
and risk tiers. Nothing here should be duplicated as a literal elsewhere in the
codebase - import from here instead, so that changing a value changes it
everywhere and every experiment stays comparable.
"""

import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Prediction horizons (trading days)
# ---------------------------------------------------------------------------
# Three horizons:
#   1  day  - immediate predictive value / cost sensitivity
#   5  days - ~1 trading week persistence
#   21 days - ~1 trading month persistence; the PRIMARY portfolio horizon.
# These are chosen a-priori, before looking at any test results.
HORIZONS = [1, 5, 21]
PRIMARY_HORIZON = 21

# PREDICTION_HORIZON is the primary horizon. Code that needs every horizon
# iterates over HORIZONS instead.
PREDICTION_HORIZON = PRIMARY_HORIZON

# ---------------------------------------------------------------------------
# Leakage controls
# ---------------------------------------------------------------------------
# Purge H sessions so the last retained training label ends strictly before
# the first validation/test session, including equality at the boundary.


def embargo_for(horizon: int) -> int:
    """Number of training rows to drop before a split boundary for `horizon`."""
    return max(0, int(horizon))


# ---------------------------------------------------------------------------
# History window
# ---------------------------------------------------------------------------
# Earliest date of price history to download. FNSPID news coverage begins in
# 1999 and runs through 2023; 2010 gives a long common window with liquid
# S&P 500 names while avoiding the thinnest early-2000s coverage.
HISTORY_START = "2010-01-01"

# ---------------------------------------------------------------------------
# Data cleaning thresholds
# ---------------------------------------------------------------------------
CLEANING = {
    # Forward-fill only, and only tiny gaps (a couple of missing prints inside
    # an otherwise active series). Never backward-fill: that invents a price a
    # trader could not have known. Longer gaps are treated as suspensions and
    # the window is excluded rather than filled.
    "max_ffill_days": 2,
    # A day whose |return| exceeds this is flagged (not removed) for manual
    # checking against corporate actions. Genuine crashes/earnings gaps are real
    # signals and are preserved.
    "extreme_move": 0.40,
    # Rolling robust outlier detection: flag a return whose deviation from the
    # rolling median exceeds `mad_k` * rolling MAD. Flags only - never deletes.
    # window=63 is ~one trading quarter; mad_k=8 is deliberately conservative so
    # only extreme prints are flagged (a normal MAD multiplier of 3 would flag
    # ordinary volatility on daily equity returns).
    "outlier_window": 63,
    "outlier_mad_k": 8.0,
    # Prices below this (in the security's own currency) are treated as penny
    # stocks and flagged; the eligibility filter can exclude them.
    "penny_price": 5.0,
    # A one-day jump in the UNADJUSTED close close to a common split ratio,
    # not matched by a same-day move in the adjusted close, is a likely split
    # that was not applied. Flagged for audit.
    "split_ratios": (2.0, 3.0, 1.5, 4.0, 7.0, 10.0),
    "split_tol": 0.05,
}

# Per-profile portfolio constraints that enter the optimisation directly (bounds
# and the risk-aversion penalty), not just the explanation. Rationale: a
# conservative book caps any single position at 15% and holds >=20% cash with a
# high risk-aversion penalty (8.0), an aggressive book allows 40% concentration,
# 0% cash floor and a low penalty (1.0). The gap between tiers is what makes the
# three profiles produce visibly different allocations (verified in
# tests/test_profile_enforcement.py).
RISK_CONSTRAINTS = {
    "conservative": {"max_weight": 0.15, "min_cash": 0.20, "risk_aversion": 8.0},
    "moderate": {"max_weight": 0.25, "min_cash": 0.05, "risk_aversion": 3.0},
    "aggressive": {"max_weight": 0.40, "min_cash": 0.00, "risk_aversion": 1.0},
}

# Compact, cross-stock-transferable market features the RL agent observes (no model signal is included).
RL_STATE_FEATURES = ["momentum_21d", "volatility", "rsi", "volume_ratio", "close_to_sma20"]

# Transaction cost assumption (per unit turnover) used consistently by the
# backtester, the RL reward and the portfolio evaluation so results are matched.
TX_COST = 0.001  # 10 bps per unit of weight turnover

# Annual risk-free rate for Sharpe / excess-return. Centralised here (not
# hard-coded in the optimiser) so every experiment shares one assumption.
RF_ANNUAL = 0.02
TRADING_DAYS = 252  # trading days per year (annualisation factor)

# Named numerical tolerances (single source; avoids scattered 1e-8/1e-9 literals).
STD_EPS = 1e-9  # floor added to a standard deviation before dividing (Sharpe etc.)
WEIGHT_TOL = 1e-9  # portfolio-weight comparison tolerance
CASH_TOL = 0.01  # currency reconciliation tolerance (one cent)

# Bump whenever any indicator FORMULA in feature_engineer changes (not just the
# column set). Recorded in each model artifact so a model trained under an old
# feature pipeline is rejected on load rather than scored against new features.
FEATURE_PIPELINE_VERSION = 1

# ---------------------------------------------------------------------------
# External services - single source of truth for Ollama and HeyGen
# ---------------------------------------------------------------------------
# The advisor chat (Ollama) and the avatar video (HeyGen) read their endpoints
# and keys from here, which in turn read from environment variables so nothing
# secret is committed. Copy .env.example to .env and fill these in. Every module
# that talks to Ollama or HeyGen must import from here - never hard-code a URL,
# model name or key elsewhere.


load_dotenv()

# -- Ollama (local LLM used by the advisor) --
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


# LiveAvatar credentials are separate from the rendered-video service.
LIVEAVATAR_API_KEY = os.environ.get("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_AVATAR_ID = os.environ.get("LIVEAVATAR_AVATAR_ID", "")
LIVEAVATAR_VOICE_ID = os.environ.get("LIVEAVATAR_VOICE_ID", "")
LIVEAVATAR_SANDBOX = os.environ.get("LIVEAVATAR_SANDBOX", "true").lower() == "true"
