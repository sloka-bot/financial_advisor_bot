"""
Portfolio constructor.

Given a ranked list of stocks, this module:
  1. Selects the top-N BUY candidates above the profile threshold
  2. Allocates budget to whole shares only (can't buy 0.3 of a stock)
  3. Optimises weights to maximise the Sharpe ratio via scipy.optimize
  4. Computes VaR and CVaR using a parametric (Gaussian) assumption

Portfolio optimisation:
  The scipy SLSQP solver maximises (w'μ - rf) / σ_p subject to:
    - weights sum to 1
    - each weight ∈ [0, MAX_WEIGHT]

  We use a diagonal covariance matrix (zero correlation assumption).
  Full covariance matrices on short histories are notoriously noisy —
  Ledoit & Wolf (2004) showed that even shrinkage estimators underperform
  the diagonal assumption when fewer than 500 observations are available.

Risk metrics:
  VaR (Value-at-Risk): the loss not exceeded with probability p.
  CVaR (Conditional VaR / Expected Shortfall): the expected loss in the
  (1-p) worst cases. CVaR is a coherent risk measure (Artzner et al., 1999)
  and provides a more complete picture of tail risk than VaR alone.
"""
import logging

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm

logger = logging.getLogger(__name__)

CONFIDENCE  = 0.95    # 95th percentile for VaR and CVaR
RF_RATE     = 0.02    # assumed annual risk-free rate (US T-bill proxy)
MAX_WEIGHT  = 0.30    # cap any single stock at 30% to prevent concentration

# Score thresholds for BUY/HOLD/SELL by risk profile
THRESHOLDS = {
    'conservative': {'buy': 80, 'sell': 30},
    'moderate':     {'buy': 70, 'sell': 35},
    'aggressive':   {'buy': 60, 'sell': 30},
}


class PortfolioConstructor:

    def construct(self, ranked_df: pd.DataFrame, master_data: dict,
                  budget: float, risk_profile: str, top_n: int = 10) -> dict:
        """
        Build a portfolio from the top-ranked BUY signals.

        Returns a dictionary with holdings (ticker, shares, price, weights),
        risk metrics (Sharpe, VaR, CVaR), and cash remaining.
        """
        thresh = THRESHOLDS.get(risk_profile, THRESHOLDS['moderate'])

        # Filter to BUY-signal stocks; the remaining budget is kept as cash
        buy_candidates = ranked_df[ranked_df['composite_score'] >= thresh['buy']].head(top_n)

        if buy_candidates.empty:
            logger.warning(f'No BUY signals above threshold {thresh["buy"]} for {risk_profile}')
            return self._empty_response(budget, risk_profile)

        # Look up the latest price for each candidate
        prices = {}
        for ticker in buy_candidates['ticker']:
            df = master_data.get(ticker)
            if df is not None and not df.empty:
                prices[ticker] = float(df['close'].iloc[-1])

        valid = buy_candidates[buy_candidates['ticker'].isin(prices)].copy()
        if valid.empty:
            return self._empty_response(budget, risk_profile)

        tickers = valid['ticker'].tolist()

        # Per-ticker expected returns and volatilities for the optimiser
        exp_returns = {}
        vols        = {}
        for ticker in tickers:
            row  = valid[valid['ticker'] == ticker].iloc[0]
            df   = master_data[ticker]
            pred = float(row.get('predicted_return', 0)) / 100   # convert from % to decimal
            vol  = float(df['volatility'].iloc[-1]) if 'volatility' in df.columns else 0.02
            exp_returns[ticker] = pred * 252                        # annualised return
            vols[ticker]        = vol * np.sqrt(252)                # annualised volatility

        # Optimise weights for maximum Sharpe ratio
        weights = self._max_sharpe_weights(tickers, exp_returns, vols)

        # Convert weights to whole-share allocations
        holdings     = []
        total_cost   = 0.0
        for ticker, w in zip(tickers, weights):
            price      = prices[ticker]
            allocation = budget * w
            shares     = max(1, int(allocation / price))   # at least 1 share if selected
            cost       = shares * price
            total_cost += cost

            holdings.append({
                'ticker':           ticker,
                'signal':           'BUY',
                'composite_score':  float(valid[valid['ticker']==ticker]['composite_score'].iloc[0]),
                'predicted_return': float(valid[valid['ticker']==ticker]['predicted_return'].iloc[0]),
                'sentiment':        str(valid[valid['ticker']==ticker]['sentiment'].iloc[0]),
                'shares':           shares,
                'price':            round(price, 2),
                'total_cost':       round(cost, 2),
                'weight_pct':       round(w * 100, 2),
            })

        # Portfolio-level expected return and risk
        w_arr        = np.array([h['weight_pct']/100 for h in holdings])
        mu_arr       = np.array([exp_returns[h['ticker']] for h in holdings])
        sigma_arr    = np.array([vols[h['ticker']]        for h in holdings])

        port_ret  = float(np.dot(w_arr, mu_arr))
        port_vol  = float(np.sqrt(np.dot(w_arr**2, sigma_arr**2)))   # diagonal cov
        sharpe    = (port_ret - RF_RATE) / (port_vol + 1e-8)

        # Parametric VaR and CVaR (assumes normally distributed daily P&L)
        daily_vol = port_vol / np.sqrt(252)
        z         = norm.ppf(CONFIDENCE)
        var_1d    = float(daily_vol * z)                                           # loss in fraction
        cvar_1d   = float(daily_vol * norm.pdf(z) / (1 - CONFIDENCE))             # expected shortfall

        expected_pct_return = round(port_ret * 100, 4)

        return {
            'portfolio': {
                'holdings':                  holdings,
                'n_positions':               len(holdings),
                'total_invested':            round(total_cost, 2),
                'cash_remaining':            round(max(0, budget - total_cost), 2),
                'risk_profile':              risk_profile,
                'expected_portfolio_return': expected_pct_return,
                'risk_metrics': {
                    'annualized_sharpe':    round(sharpe, 4),
                    'portfolio_volatility': round(port_vol, 6),
                    'var_95_1day':          round(var_1d, 6),
                    'cvar_95_1day':         round(cvar_1d, 6),
                },
            }
        }

    def _max_sharpe_weights(self, tickers, mu, sigma) -> np.ndarray:
        """
        SLSQP optimisation to maximise Sharpe ratio.
        The objective is negated because scipy minimises by default.
        """
        n = len(tickers)
        if n == 1:
            return np.array([1.0])

        mu_arr    = np.array([mu[t]    for t in tickers])
        sigma_arr = np.array([sigma[t] for t in tickers])

        def neg_sharpe(w):
            ret = np.dot(w, mu_arr)
            vol = np.sqrt(np.dot(w**2, sigma_arr**2))
            return -(ret - RF_RATE) / (vol + 1e-8)

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds      = [(0, MAX_WEIGHT)] * n
        x0          = np.full(n, 1.0 / n)   # equal-weight starting point

        result = opt.minimize(neg_sharpe, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'ftol': 1e-9, 'maxiter': 500})

        w = result.x if result.success else x0
        w = np.maximum(w, 0)
        return w / w.sum()

    def _empty_response(self, budget, risk_profile):
        return {
            'portfolio': {
                'holdings': [], 'n_positions': 0,
                'total_invested': 0.0, 'cash_remaining': budget,
                'risk_profile': risk_profile, 'expected_portfolio_return': 0.0,
                'risk_metrics': {'annualized_sharpe': 0, 'var_95_1day': 0, 'cvar_95_1day': 0},
            }
        }
