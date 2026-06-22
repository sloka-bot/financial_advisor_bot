"""
Walk-forward backtester.

Tests the XGBoost signal on the final 20% of each ticker's history —
data the model never saw during training (true out-of-sample evaluation).

Strategy:
  - Go long (buy) when the model predicts a positive next-day return.
  - Stay flat (hold cash) otherwise.
  - No short selling, no leverage.

Metrics computed:
  Sharpe ratio      — annualised (ret - rf) / σ  (Sharpe, 1966)
  Max drawdown      — worst peak-to-trough decline during the test period
  Calmar ratio      — annualised return / max drawdown
  Direction accuracy — fraction of days where predicted sign == actual sign
  Information coefficient (IC) — Spearman rank correlation of predictions
                                  with actual returns; a signal-quality measure
                                  used by systematic hedge funds
  Excess return     — strategy total return minus buy-and-hold benchmark
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

OUT_OF_SAMPLE_FRACTION = 0.20   # last 20% of data used for testing
RF_RATE_DAILY          = 0.02 / 252


class Backtester:

    def run(self, ticker: str, df: pd.DataFrame, model, capital: float = 10_000) -> dict:
        """
        Run a single-ticker backtest and return performance metrics plus
        equity curves for charting.
        """
        if df is None or len(df) < 50:
            return {'error': f'Not enough data for {ticker}'}

        # Split: the model was trained on the first 80%; we test on the rest.
        split   = int(len(df) * (1 - OUT_OF_SAMPLE_FRACTION))
        test_df = df.iloc[split:].copy()

        if len(test_df) < 20:
            return {'error': f'Test period too short for {ticker} ({len(test_df)} rows)'}

        # Generate predictions day by day to avoid any forward-looking bias.
        # Each prediction uses only the row for that day.
        preds   = []
        actuals = []

        for i in range(len(test_df) - 1):
            window = test_df.iloc[:i+1]           # history up to current day
            pred   = model.predict_ticker(window)
            actual = float(test_df['daily_return'].iloc[i+1])
            preds.append(pred if pred is not None else 0.0)
            actuals.append(actual)

        preds   = np.array(preds)
        actuals = np.array(actuals)

        # Strategy P&L: capture the actual return only on days we were long.
        # Being flat earns the risk-free rate (modelled as zero for simplicity).
        long_days    = preds > 0
        strat_rets   = np.where(long_days, actuals, 0.0)
        bench_rets   = actuals                    # buy-and-hold benchmark

        # Build equity curves: cumulative product of (1 + daily_return)
        strat_curve  = capital * np.cumprod(1 + strat_rets)
        bench_curve  = capital * np.cumprod(1 + bench_rets)

        strat_total  = (strat_curve[-1] / capital) - 1
        bench_total  = (bench_curve[-1] / capital) - 1

        # Annualise Sharpe: (mean daily return - rf) / std × √252
        sharpe = float(
            ((strat_rets.mean() - RF_RATE_DAILY) / (strat_rets.std() + 1e-8)) * np.sqrt(252)
        )

        # Max drawdown: largest percentage decline from any rolling peak
        roll_max = np.maximum.accumulate(strat_curve)
        drawdown = (strat_curve - roll_max) / (roll_max + 1e-8)
        max_dd   = float(drawdown.min())

        # Direction accuracy: fraction of predictions with the correct sign
        dir_acc  = float(np.mean(np.sign(preds) == np.sign(actuals)))

        # Information coefficient: Spearman correlation between predicted and
        # actual returns. A value of 0.05 is considered meaningful by practitioners.
        ic = float(spearmanr(preds, actuals).correlation) if len(preds) > 2 else 0.0

        # Date labels for the equity curves
        dates = [str(test_df.index[i].date()) if hasattr(test_df.index[i], 'date') else str(test_df.index[i])
                 for i in range(1, len(test_df))]
        dates = dates[:len(strat_curve)]

        equity_curve    = [{'date': d, 'value': round(v, 2)} for d, v in zip(dates, strat_curve.tolist())]
        benchmark_curve = [{'date': d, 'value': round(v, 2)} for d, v in zip(dates, bench_curve.tolist())]

        start_date = dates[0]  if dates else ''
        end_date   = dates[-1] if dates else ''

        return {
            'ticker':           ticker,
            'initial_capital':  capital,
            'test_period':      f'{start_date} → {end_date}',
            'n_total_days':     len(actuals),
            'n_long_days':      int(long_days.sum()),
            'equity_curve':     equity_curve,
            'benchmark_curve':  benchmark_curve,
            'metrics': {
                'sharpe_ratio':           round(sharpe,    4),
                'total_return':           round(strat_total, 4),
                'benchmark_total_return': round(bench_total, 4),
                'excess_return':          round(strat_total - bench_total, 4),
                'max_drawdown':           round(max_dd,    4),
                'calmar_ratio':           round(-strat_total / (max_dd or -1e-8), 4),
                'direction_accuracy':     round(dir_acc,   4),
                'information_coefficient': round(ic,       4),
            }
        }

    def run_portfolio(self, tickers: list, master_data: dict,
                      model, capital: float = 10_000) -> dict:
        """
        Run individual backtests for each ticker and return a portfolio summary.
        """
        results = {}
        for ticker in tickers:
            df = master_data.get(ticker)
            if df is not None:
                results[ticker] = self.run(ticker, df, model, capital)

        valid   = {t: r for t, r in results.items() if 'metrics' in r}
        sharpes = [r['metrics']['sharpe_ratio']         for r in valid.values()]
        dirs    = [r['metrics']['direction_accuracy']   for r in valid.values()]

        best_ticker = max(valid.keys(),
                          key=lambda t: valid[t]['metrics']['total_return']) if valid else None

        return {
            'individual_metrics': {t: r['metrics'] for t, r in valid.items()},
            'portfolio_summary': {
                'avg_sharpe':         round(np.mean(sharpes), 4) if sharpes else 0,
                'avg_direction_acc':  round(np.mean(dirs),    4) if dirs    else 0,
                'best_ticker':        best_ticker,
                'n_backtested':       len(valid),
            },
            # return first ticker's curve for chart display
            'equity_curve':    valid[tickers[0]]['equity_curve']    if tickers and tickers[0] in valid else [],
            'benchmark_curve': valid[tickers[0]]['benchmark_curve'] if tickers and tickers[0] in valid else [],
        }
