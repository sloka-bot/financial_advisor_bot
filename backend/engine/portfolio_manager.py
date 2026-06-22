"""
Portfolio Manager — the brain of the advisory platform.

Responsibilities:
  - Track holdings, cash, and allocation weights for each user
  - Detect allocation drift (when a position deviates >5% from target)
  - Generate rebalancing suggestions with human-readable reasons
  - Calculate confidence scores for each recommendation
  - Apply approved recommendations to the portfolio state
"""
import logging
import math
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.05   # flag for rebalancing when weight drifts >5% from target


class PortfolioManager:

    # ── Portfolio construction ─────────────────────────────────────────────

    def build_initial_portfolio(self, ranked_df, master_data: dict,
                                budget: float, risk_profile: str, top_n: int = 8) -> dict:
        """
        Create the initial portfolio for a new user.
        Returns holdings list and pending recommendations.
        """
        from backend.engine.portfolio import PortfolioConstructor
        from backend.engine.recommender import RecommendationEngine

        rc   = RecommendationEngine()
        pc   = PortfolioConstructor()

        result    = rc.recommend(ranked_df, risk_profile=risk_profile, top_n=top_n)
        portfolio = pc.construct(ranked_df, master_data, budget, risk_profile, top_n)

        holdings = portfolio.get('portfolio', {}).get('holdings', [])
        cash     = portfolio.get('portfolio', {}).get('cash_remaining', 0)

        # enrich holdings with confidence scores
        for h in holdings:
            ticker = h['ticker']
            df     = master_data.get(ticker)
            h['confidence'] = self.confidence_score(
                ticker, df,
                h.get('predicted_return', 0) / 100 if h.get('predicted_return') else 0,
            )

        # build recommendation objects for the approval queue
        recs = []
        for signal in result.get('all_signals', []):
            if signal['signal'] == 'BUY':
                df         = master_data.get(signal['ticker'])
                confidence = self.confidence_score(signal['ticker'], df,
                                                   signal.get('predicted_return', 0) / 100)
                recs.append({
                    'action':     'BUY',
                    'ticker':     signal['ticker'],
                    'signal':     'BUY',
                    'score':      signal.get('composite_score', 50),
                    'confidence': confidence['overall'],
                    'factors':    confidence['factors'],
                    'reason':     self._buy_reason(signal, confidence),
                    'predicted_return': signal.get('predicted_return', 0),
                })

        return {
            'holdings': holdings,
            'cash':     cash,
            'recommendations': recs,
            'portfolio_summary': portfolio.get('portfolio', {}),
        }

    # ── Drift detection ────────────────────────────────────────────────────

    def detect_drift(self, holdings: list, master_data: dict) -> list:
        """
        Check each holding against its target weight.
        Target is equal-weight (1/n) unless the holding has a stored target.

        Returns a list of holdings that need rebalancing.
        """
        if not holdings:
            return []

        # update current prices
        total_value = 0.0
        enriched    = []
        for h in holdings:
            ticker  = h['ticker']
            df      = master_data.get(ticker)
            price   = float(df.iloc[-1]['close']) if df is not None and not df.empty else h.get('price', 0)
            value   = h.get('shares', 0) * price
            total_value += value
            enriched.append({**h, 'current_price': price, 'current_value': value})

        if total_value == 0:
            return []

        n_holdings     = len(enriched)
        target_weight  = 1.0 / n_holdings

        drifted = []
        for h in enriched:
            current_weight = h['current_value'] / total_value
            drift          = current_weight - target_weight
            h['current_weight_pct'] = round(current_weight * 100, 2)
            h['target_weight_pct']  = round(target_weight * 100, 2)
            h['drift_pct']          = round(drift * 100, 2)

            if abs(drift) >= DRIFT_THRESHOLD:
                h['rebalance_action'] = 'REDUCE' if drift > 0 else 'INCREASE'
                drifted.append(h)

        return drifted

    # ── Rebalancing suggestions ────────────────────────────────────────────

    def suggest_rebalance(self, holdings: list, master_data: dict,
                          ranked_df=None, risk_profile='moderate') -> list:
        """
        Generate human-readable rebalancing suggestions for drifted holdings.
        """
        drifted = self.detect_drift(holdings, master_data)
        if not drifted:
            return []

        suggestions = []
        for h in drifted:
            ticker  = h['ticker']
            action  = h['rebalance_action']
            drift   = h['drift_pct']
            df      = master_data.get(ticker)
            latest  = df.iloc[-1] if df is not None and not df.empty else None

            # reason explains the drift
            if action == 'REDUCE':
                reason = (f'{ticker} has grown to {h["current_weight_pct"]:.1f}% of the portfolio '
                          f'(target {h["target_weight_pct"]:.1f}%, drift +{drift:.1f}%). '
                          f'Reducing exposure manages concentration risk.')
            else:
                reason = (f'{ticker} has fallen to {h["current_weight_pct"]:.1f}% of the portfolio '
                          f'(target {h["target_weight_pct"]:.1f}%, drift {drift:.1f}%). '
                          f'Adding shares restores target allocation.')

            confidence = self.confidence_score(ticker, df, 0)

            suggestions.append({
                'action':           'REBALANCE',
                'sub_action':       action,
                'ticker':           ticker,
                'current_weight':   h['current_weight_pct'],
                'target_weight':    h['target_weight_pct'],
                'drift':            drift,
                'reason':           reason,
                'confidence':       confidence['overall'],
                'factors':          confidence['factors'],
            })

        return suggestions

    # ── Confidence score ───────────────────────────────────────────────────

    def confidence_score(self, ticker: str, df, predicted_return: float) -> dict:
        """
        Compute a composite confidence score [0–100] for a recommendation.

        Four factors, each scored 0–100:
          model_signal   — how strongly XGBoost/LSTM predicts positive return
          sentiment      — FinBERT sentiment strength
          momentum       — price momentum and RSI headroom
          trend_strength — ADX value (trend clarity)
        """
        factors = {
            'model_signal':   50,
            'sentiment':      50,
            'momentum':       50,
            'trend_strength': 50,
        }

        if df is not None and not df.empty:
            latest = df.iloc[-1]

            # model signal: |predicted_return| scaled, direction matters
            pred_score = min(100, abs(predicted_return) * 5000)
            if predicted_return > 0:
                factors['model_signal'] = int(50 + pred_score / 2)
            elif predicted_return < 0:
                factors['model_signal'] = int(50 - pred_score / 2)

            # sentiment: |sent_score| * 100, direction-adjusted
            sent = float(latest.get('sent_score', 0) or 0)
            factors['sentiment'] = max(0, min(100, int(50 + sent * 100)))

            # momentum + RSI headroom
            mom  = float(latest.get('momentum_10d', 0) or 0)
            rsi  = float(latest.get('rsi', 50) or 50)
            rsi_room = (100 - rsi) / 100   # 0 near overbought, 1 near oversold
            factors['momentum'] = max(0, min(100, int(50 + mom * 1000 * rsi_room)))

            # trend strength from ADX
            adx = float(latest.get('adx', 0) or 0)
            factors['trend_strength'] = min(100, int(adx * 2.5))

        overall = int(
            factors['model_signal']   * 0.40 +
            factors['sentiment']      * 0.25 +
            factors['momentum']       * 0.20 +
            factors['trend_strength'] * 0.15
        )

        return {'overall': overall, 'factors': factors}

    # ── Apply approved recommendation ──────────────────────────────────────

    def apply_recommendation(self, rec: dict, current_holdings: list,
                             current_cash: float, master_data: dict) -> dict:
        """
        Execute an approved recommendation against the user's portfolio.
        Returns updated (holdings, cash).
        """
        ticker  = rec.get('ticker')
        action  = rec.get('action', rec.get('sub_action', 'BUY'))
        df      = master_data.get(ticker)
        price   = float(df.iloc[-1]['close']) if df is not None and not df.empty else 0

        if price == 0:
            return {'holdings': current_holdings, 'cash': current_cash,
                    'error': f'No price data for {ticker}'}

        holdings = [dict(h) for h in current_holdings]

        if action in ('BUY', 'INCREASE'):
            # allocate 10% of remaining cash to the position
            spend  = current_cash * 0.10
            shares = max(1, int(spend / price))
            cost   = shares * price

            if cost > current_cash:
                return {'holdings': holdings, 'cash': current_cash,
                        'error': 'Insufficient cash'}

            # update existing holding or add new one
            existing = next((h for h in holdings if h['ticker'] == ticker), None)
            if existing:
                old_cost         = existing['shares'] * existing['price']
                existing['shares'] += shares
                existing['price']  = round((old_cost + cost) / existing['shares'], 4)
                existing['total_cost'] = round(existing['shares'] * existing['price'], 2)
            else:
                holdings.append({
                    'ticker': ticker, 'shares': shares, 'price': round(price, 2),
                    'total_cost': round(cost, 2), 'signal': 'BUY',
                    'predicted_return': rec.get('predicted_return', 0),
                    'sentiment': 'neutral', 'weight_pct': 0,
                })
            current_cash -= cost

        elif action in ('SELL', 'REDUCE'):
            existing = next((h for h in holdings if h['ticker'] == ticker), None)
            if not existing:
                return {'holdings': holdings, 'cash': current_cash,
                        'error': f'{ticker} not in portfolio'}
            # sell 50% of the position
            sell_shares  = max(1, existing['shares'] // 2)
            proceeds     = sell_shares * price
            existing['shares'] -= sell_shares
            existing['total_cost'] = round(existing['shares'] * existing['price'], 2)
            current_cash += proceeds
            # remove if fully sold
            holdings = [h for h in holdings if h['shares'] > 0]

        # recalculate weights
        total = sum(h['shares'] * price for h in holdings) + current_cash
        for h in holdings:
            df_h   = master_data.get(h['ticker'])
            p      = float(df_h.iloc[-1]['close']) if df_h is not None and not df_h.empty else h['price']
            h['weight_pct'] = round(h['shares'] * p / total * 100, 2) if total > 0 else 0

        return {'holdings': holdings, 'cash': round(current_cash, 2)}

    # ── Reason generation ─────────────────────────────────────────────────

    def _buy_reason(self, signal: dict, confidence: dict) -> str:
        ticker  = signal.get('ticker', '')
        ret     = signal.get('predicted_return', 0)
        sent    = signal.get('sentiment_label', 'neutral')
        score   = signal.get('composite_score', 50)
        factors = confidence.get('factors', {})

        reasons = []
        if factors.get('model_signal', 50) > 65:
            reasons.append(f'model forecasts +{ret:.2f}% tomorrow')
        if factors.get('sentiment', 50) > 65:
            reasons.append(f'{sent} news sentiment')
        if factors.get('momentum', 50) > 65:
            reasons.append('bullish momentum with RSI headroom')
        if factors.get('trend_strength', 50) > 65:
            reasons.append('strong directional trend (ADX elevated)')

        if not reasons:
            reasons = [f'composite score {score:.0f}/100']

        return f'{ticker} rated BUY: {", ".join(reasons)}.'
