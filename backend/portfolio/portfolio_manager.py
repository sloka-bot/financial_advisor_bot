"""Calculate holding drift, recommendation support and fee-adjusted transactions."""

import logging
import math

from backend.config.settings import RISK_CONSTRAINTS, TX_COST
from backend.data.contracts import executable_price, freshness

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.05  # rebalance when weight drifts over 5% from target


class PortfolioManager:
    # Rebalancing suggestions.

    """Calculate holding evidence, risk metrics and rebalance suggestions."""

    def suggest_rebalance(
        self, holdings: list, master_data: dict, ranked_df=None, risk_profile="moderate", cash: float = 0.0
    ) -> list:
        """Compare whole-portfolio holding weights with constrained historical allocation targets."""
        if not holdings:
            return []
        from backend.portfolio.allocation import build_markowitz_portfolio

        # Value the current book and derive stock weights.
        priced, invested = [], 0.0
        for h in holdings:
            df = master_data.get(h["ticker"])
            price = executable_price(df) or h.get("price", 0)
            val = h.get("shares", 0) * price
            invested += val
            priced.append({**h, "current_price": price, "current_value": val})
        total_value = invested + max(0.0, float(cash))
        if total_value <= 0:
            return []

        # Build historical targets using total portfolio value, including cash.
        tickers = [h["ticker"] for h in holdings]
        port = build_markowitz_portfolio(master_data, tickers, total_value, risk_profile, current_holdings=holdings)
        # Skip rebalancing when optimisation is unavailable.
        if not port.get("portfolio", {}).get("available", True):
            logger.warning("Rebalance skipped: portfolio optimisation unavailable")
            return []
        target = {h["ticker"]: h["weight_pct"] / 100.0 for h in port.get("portfolio", {}).get("holdings", [])}

        suggestions = []
        for h in priced:
            ticker = h["ticker"]
            df = master_data.get(ticker)
            # Skip holdings without usable history instead of treating them as liquidation targets.
            if df is None or df.empty or executable_price(df) is None:
                continue
            cur_w = h["current_value"] / total_value
            tgt_w = target.get(ticker, 0.0)  # zero means the optimiser exits the position
            drift = cur_w - tgt_w
            if abs(drift) < DRIFT_THRESHOLD:
                continue
            action = "REDUCE" if drift > 0 else "INCREASE"
            df = master_data.get(ticker)
            if action == "REDUCE":
                reason = (
                    f"{ticker} is {cur_w * 100:.1f}% of holdings vs a {tgt_w * 100:.1f}% target "
                    f"(drift +{drift * 100:.1f}%). Trimming restores the target allocation."
                )
            else:
                reason = (
                    f"{ticker} is {cur_w * 100:.1f}% of holdings vs a {tgt_w * 100:.1f}% target "
                    f"(drift {drift * 100:.1f}%). Adding restores the target allocation."
                )
            confidence = self.confidence_score(ticker, df, 0)
            suggestions.append(
                {
                    "action": "REBALANCE",
                    "sub_action": action,
                    "ticker": ticker,
                    "current_weight": round(cur_w * 100, 2),
                    "target_weight": round(tgt_w * 100, 2),
                    "drift": round(drift * 100, 2),
                    "reason": reason,
                    "confidence": confidence["overall"],
                    "factors": confidence["factors"],
                }
            )
        return suggestions

    # Confidence score.

    def confidence_score(self, ticker: str, df, predicted_return: float, prob_up: float = None) -> dict:
        """Combine model signal, sentiment, momentum and trend into a 0-100 support score."""
        factors = {
            "model_signal": 50,
            "sentiment": 50,
            "momentum": 50,
            "trend_strength": 50,
        }

        if df is not None and not df.empty:
            latest = df.iloc[-1]

            # Combine available return and probability signals into the support score.
            pred_score = min(100, abs(predicted_return) * 5000)
            if predicted_return > 0:
                lstm_signal = int(50 + pred_score / 2)
            elif predicted_return < 0:
                lstm_signal = int(50 - pred_score / 2)
            else:
                lstm_signal = 50
            if prob_up is not None:
                xgb_signal = max(0, min(100, int(float(prob_up) * 100)))
                factors["model_signal"] = int(round((lstm_signal + xgb_signal) / 2))
            else:
                factors["model_signal"] = lstm_signal

            # Sentiment component, direction adjusted.
            sent = float(latest.get("sent_score", 0) or 0)
            factors["sentiment"] = max(0, min(100, int(50 + sent * 100)))

            # Momentum and RSI headroom.
            mom = float(latest.get("momentum_10d", 0) or 0)
            rsi = float(latest.get("rsi", 50) or 50)
            rsi_room = (100 - rsi) / 100  # 0 near overbought, 1 near oversold
            factors["momentum"] = max(0, min(100, int(50 + mom * 1000 * rsi_room)))

            # Trend strength from ADX.
            adx = float(latest.get("adx", 0) or 0)
            factors["trend_strength"] = min(100, int(adx * 2.5))

        overall = int(
            factors["model_signal"] * 0.40
            + factors["sentiment"] * 0.25
            + factors["momentum"] * 0.20
            + factors["trend_strength"] * 0.15
        )

        return {"overall": overall, "factors": factors}

    # Apply approved recommendation.

    def apply_recommendation(
        self,
        rec: dict,
        current_holdings: list,
        current_cash: float,
        master_data: dict,
        risk_profile: str = "moderate",
        tx_cost: float = TX_COST,
    ) -> dict:
        """Apply approved trades with fees, position limits and minimum cash constraints."""
        cons = RISK_CONSTRAINTS.get(risk_profile, RISK_CONSTRAINTS["moderate"])
        max_w, min_cash_frac = cons["max_weight"], cons["min_cash"]

        ticker = rec.get("ticker")
        action = rec.get("action", rec.get("sub_action", "BUY"))
        # A REBALANCE suggestion carries its direction in sub_action.
        if action == "REBALANCE":
            action = rec.get("sub_action", "BUY")

        def _price(tk):
            d = master_data.get(tk)
            return executable_price(d) or 0.0

        price = _price(ticker)
        if price <= 0:
            return {"holdings": current_holdings, "cash": current_cash, "error": f"No price data for {ticker}"}

        holdings = [dict(h) for h in current_holdings]
        # Value each holding at its current execution price and include cash.
        prices = {h["ticker"]: (_price(h["ticker"]) or h.get("price", 0.0)) for h in holdings}
        prices[ticker] = price
        # Require a current execution price for every holding before assessing risk limits.
        stale = [
            h["ticker"]
            for h in holdings
            if (executable_price(master_data.get(h["ticker"])) or 0.0) <= 0
            or not freshness(master_data.get(h["ticker"]))["fresh"]
        ]
        if stale:
            return {
                "holdings": current_holdings,
                "cash": current_cash,
                "error": f"Cannot execute: current price unavailable for holding(s) {stale}",
            }
        total_value = sum(h["shares"] * prices.get(h["ticker"], 0.0) for h in holdings) + current_cash
        if total_value <= 0:
            return {"holdings": holdings, "cash": round(current_cash, 2), "error": "Non-positive portfolio value"}

        existing = next((h for h in holdings if h["ticker"] == ticker), None)
        cur_val = (existing["shares"] * price) if existing else 0.0

        # Use explicit target weights when supplied; otherwise apply the default trade fraction.
        tgt_w = rec.get("target_weight")
        target_frac = min(float(tgt_w) / 100.0, max_w) if tgt_w is not None else None
        if target_frac is not None:
            trade_val = target_frac * total_value - cur_val
            if abs(trade_val) < price:  # already within one share of target
                return {
                    "holdings": holdings,
                    "cash": round(current_cash, 2),
                    "note": f"{ticker} already within one share of its {tgt_w}% target",
                }
            action = "BUY" if trade_val > 0 else "SELL"

        if action in ("BUY", "INCREASE"):
            cash_floor = min_cash_frac * total_value  # keep minimum cash
            spendable = max(0.0, current_cash - cash_floor)
            if target_frac is not None:
                budget = min(max(0.0, target_frac * total_value - cur_val), spendable)  # to target
            else:
                room_wt = max(0.0, max_w * total_value - cur_val)  # maximum position weight
                budget = (
                    min(room_wt, spendable)
                    if rec.get("shares") is not None
                    else min(0.10 * total_value, room_wt, spendable)
                )
            shares = int(budget / (price * (1.0 + tx_cost)))  # leave room for the fee
            requested = rec.get("shares")
            if requested is not None:
                if requested > shares:
                    return {"error": "Requested shares exceed available cash or profile limits"}
                shares = requested
            elif rec.get("max_shares") is not None:
                shares = min(shares, int(rec["max_shares"]))
            if shares < 1:
                return {
                    "holdings": holdings,
                    "cash": round(current_cash, 2),
                    "error": (
                        f"BUY of {ticker} skipped: {risk_profile} profile caps "
                        f"(max {max_w:.0%} per position, {min_cash_frac:.0%} cash floor) "
                        "leave no room to add"
                    ),
                }
            cost = shares * price
            fee = cost * tx_cost
            if cost + fee > current_cash + 1e-6:
                return {"holdings": holdings, "cash": round(current_cash, 2), "error": "Insufficient cash after fees"}
            if existing:
                old_cost = existing["shares"] * existing["price"]
                existing["shares"] += shares
                existing["price"] = round((old_cost + cost) / existing["shares"], 4)
                existing["total_cost"] = round(existing["shares"] * existing["price"], 2)
            else:
                holdings.append(
                    {
                        "ticker": ticker,
                        "shares": shares,
                        "price": round(price, 2),
                        "total_cost": round(cost, 2),
                        "signal": "BUY",
                        "predicted_return": rec.get("predicted_return", 0),
                        "sentiment": "neutral",
                        "weight_pct": 0,
                    }
                )
            current_cash -= cost + fee

        elif action in ("SELL", "REDUCE"):
            if not existing:
                return {"holdings": holdings, "cash": round(current_cash, 2), "error": f"{ticker} not in portfolio"}
            if rec.get("sell_all"):
                sell_shares = existing["shares"]
            elif target_frac is not None:
                # Sell enough to reach the target weight, at least one share.
                over_val = cur_val - target_frac * total_value
                sell_shares = min(existing["shares"], max(1, math.ceil(over_val / price)))
            else:
                sell_shares = min(existing["shares"], max(1, existing["shares"] // 2))  # default 50% sell
            proceeds = sell_shares * price
            fee = proceeds * tx_cost
            existing["shares"] -= sell_shares
            existing["total_cost"] = round(existing["shares"] * existing["price"], 2)
            current_cash += proceeds - fee
            holdings = [h for h in holdings if h["shares"] > 0]
        else:
            return {"holdings": holdings, "cash": round(current_cash, 2), "error": f"Unknown action {action}"}

        # Recompute weights from executed holdings and cash.
        prices = {h["ticker"]: (_price(h["ticker"]) or h.get("price", 0.0)) for h in holdings}
        total = sum(h["shares"] * prices[h["ticker"]] for h in holdings) + current_cash
        for h in holdings:
            h["weight_pct"] = round(h["shares"] * prices[h["ticker"]] / total * 100, 2) if total > 0 else 0

        max_pos_pct = max((h["weight_pct"] for h in holdings), default=0.0)
        cash_pct = round(current_cash / total * 100, 2) if total > 0 else 100.0
        return {
            "holdings": holdings,
            "cash": round(current_cash, 2),
            "fees": round(fee, 6),
            "ticker": ticker,
            "proceeds": round(proceeds - fee, 2) if action in ("SELL", "REDUCE") else None,
            "max_position_pct": max_pos_pct,
            "cash_pct": cash_pct,
            "respects_profile": bool(max_pos_pct <= max_w * 100 + 0.5 and cash_pct >= min_cash_frac * 100 - 0.5),
        }

    # Reason generation.

    def _buy_reason(self, signal: dict, confidence: dict) -> str:
        ticker = signal.get("ticker", "")
        ret = signal.get("predicted_return", 0)
        sent = signal.get("sentiment_label", "neutral")
        score = signal.get("composite_score", 50)
        factors = confidence.get("factors", {})

        reasons = []
        if factors.get("model_signal", 50) > 65:
            reasons.append(f"model forecasts {ret:+.2f}% over the next ~21 trading days")
        if factors.get("sentiment", 50) > 65:
            reasons.append(f"{sent} news sentiment")
        if factors.get("momentum", 50) > 65:
            reasons.append("bullish momentum with RSI headroom")
        if factors.get("trend_strength", 50) > 65:
            reasons.append("strong directional trend (ADX elevated)")

        if not reasons:
            reasons = [f"composite score {score:.0f}/100"]

        return f"{ticker} rated BUY: {', '.join(reasons)}."
