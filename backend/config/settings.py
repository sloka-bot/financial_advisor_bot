"""Shared horizons, cleaning thresholds, risk limits and service configuration."""

import os

from dotenv import load_dotenv

# Evaluate 1-, 5- and 21-session horizons; portfolios use 21 sessions.
HORIZONS = [1, 5, 21]
PRIMARY_HORIZON = 21

# Use the primary horizon for portfolio forecasts.
PREDICTION_HORIZON = PRIMARY_HORIZON


def embargo_for(horizon: int) -> int:
    """Number of training rows to drop before a split boundary for `horizon`."""
    return max(0, int(horizon))


# Start history in 2010 to retain a common price and news window.
HISTORY_START = "2010-01-01"

# Data quality thresholds.
CLEANING = {
    # Forward-fill short analytical gaps and exclude longer missing intervals.
    "max_ffill_days": 2,
    # Flag extreme returns for review without removing observed market moves.
    "extreme_move": 0.40,
    # Flag deviations beyond eight rolling MADs over a 63-session window.
    "outlier_window": 63,
    "outlier_mad_k": 8.0,
    # Flag prices below the minimum price threshold in the security's currency.
    "penny_price": 5.0,
    # Detect raw-price split ratios absent from adjusted returns.
    "split_ratios": (2.0, 3.0, 1.5, 4.0, 7.0, 10.0),
    "split_tol": 0.05,
}

# Apply profile-specific position caps, cash floors and risk penalties.
RISK_CONSTRAINTS = {
    "conservative": {"max_weight": 0.15, "min_cash": 0.20, "risk_aversion": 8.0},
    "moderate": {"max_weight": 0.25, "min_cash": 0.05, "risk_aversion": 3.0},
    "aggressive": {"max_weight": 0.40, "min_cash": 0.00, "risk_aversion": 1.0},
}

# Compact, cross-stock-transferable market features the RL agent observes (no model signal is included).
RL_STATE_FEATURES = ["momentum_21d", "volatility", "rsi", "volume_ratio", "close_to_sma20"]

# Charge proportional costs on portfolio turnover.
TX_COST = 0.001  # 10 bps per unit of weight turnover

# Use one annual risk-free rate for excess-return calculations.
RF_ANNUAL = 0.02
TRADING_DAYS = 252  # trading days per year (annualisation factor)

# Named numerical tolerances.
STD_EPS = 1e-9  # floor added to a standard deviation before dividing (Sharpe etc.)
WEIGHT_TOL = 1e-9  # portfolio-weight comparison tolerance
CASH_TOL = 0.01  # currency reconciliation tolerance (one cent)

# Version indicator formulas to reject incompatible saved feature schemas.
FEATURE_PIPELINE_VERSION = 1

# Load service endpoints and credentials from environment variables.


load_dotenv()

# Ollama settings.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


# LiveAvatar credentials are separate from the rendered-video service.
LIVEAVATAR_API_KEY = os.environ.get("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_AVATAR_ID = os.environ.get("LIVEAVATAR_AVATAR_ID", "")
LIVEAVATAR_VOICE_ID = os.environ.get("LIVEAVATAR_VOICE_ID", "")
LIVEAVATAR_SANDBOX = os.environ.get("LIVEAVATAR_SANDBOX", "true").lower() == "true"
