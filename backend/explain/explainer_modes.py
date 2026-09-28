"""Render beginner and technical explanations and validate their numerical claims."""

import re

ACTIONS = {"BUY", "REDUCE", "SELL", "HOLD", "NO_CHANGE", "INSUFFICIENT_EVIDENCE"}


def make_facts(
    ticker,
    action,
    *,
    horizon_days,
    forecast_return_pct=None,
    confidence_pct=None,
    current_weight_pct=None,
    target_weight_pct=None,
    qty_change=None,
    est_cost=None,
    price=None,
    price_date=None,
    news_cutoff=None,
    model_version=None,
    drivers=None,
    portfolio_reason=None,
    realized_backtest_return_pct=None,
    reliability=None,
):
    """Build the single canonical fact set shared by both explanation modes."""
    assert action in ACTIONS, f"unknown action {action}"
    return {
        "ticker": ticker,
        "action": action,
        "horizon_days": horizon_days,
        "forecast_return_pct": forecast_return_pct,  # forecast
        "confidence_pct": confidence_pct,
        "current_weight_pct": current_weight_pct,
        "target_weight_pct": target_weight_pct,
        "qty_change": qty_change,
        "est_cost": est_cost,
        "price": price,
        "price_date": price_date,  # source
        "news_cutoff": news_cutoff,  # source
        "model_version": model_version,  # source
        "drivers": drivers or [],  # list of name, direction and value
        "portfolio_reason": portfolio_reason,
        "realized_backtest_return_pct": realized_backtest_return_pct,  # realised history
        "reliability": reliability or {},
    }


def _fact_numbers(facts):
    """Collect numeric values supplied by the explanation facts."""
    vals = []
    for k in (
        "forecast_return_pct",
        "confidence_pct",
        "current_weight_pct",
        "target_weight_pct",
        "qty_change",
        "est_cost",
        "price",
        "realized_backtest_return_pct",
    ):
        if facts.get(k) is not None:
            vals.append(float(facts[k]))
    for d in facts.get("drivers", []):
        if isinstance(d, dict) and d.get("value") is not None:
            try:
                vals.append(float(d["value"]))
            except (TypeError, ValueError):
                pass
    for v in facts.get("reliability", {}).values():
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    # Include numeric values from the supplied portfolio reason.
    if facts.get("portfolio_reason"):
        vals += [float(x) for x in re.findall(r"-?\d+\.?\d*", facts["portfolio_reason"])]
    return vals


def render(facts, mode="technical"):
    """Render the supplied facts in beginner or technical language."""
    t = facts["ticker"]
    act = facts["action"]
    h = facts["horizon_days"]
    fr = facts.get("forecast_return_pct")
    conf = facts.get("confidence_pct")

    if act == "INSUFFICIENT_EVIDENCE":
        if mode == "beginner":
            return (
                f"We're not making a call on {t} right now. There isn't enough reliable "
                "information to be confident, so the safe choice is to do nothing."
            )
        return (
            f"{t}: INSUFFICIENT_EVIDENCE - reliability below threshold "
            "(news/freshness/similar-case accuracy). No position change recommended."
        )
    if act in ("NO_CHANGE", "HOLD"):
        if mode == "beginner":
            return (
                f"No change needed for {t}. Your current amount already lines up with the plan, "
                "so there's nothing to buy or sell today."
            )
        return (
            f"{t}: HOLD - current weight {facts.get('current_weight_pct')}% is within tolerance "
            f"of target {facts.get('target_weight_pct')}%. No trade."
        )

    # Actionable BUY, REDUCE or SELL.
    driver_txt = ", ".join(f"{d['name']} ({d['direction']})" for d in facts.get("drivers", [])[:3])
    fore = f"{fr:+.2f}%" if fr is not None else "n/a"
    if mode == "beginner":
        parts = [f"We suggest you {act.lower()} {t}."]
        if fr is not None:
            parts.append(
                f"Our model expects about a {fore} move over the next {h} trading days - "
                "this is a forecast, not a guarantee, so it can be wrong."
            )
        if conf is not None:
            parts.append(f"Confidence is around {conf:.0f}%.")
        if facts.get("portfolio_reason"):
            parts.append(facts["portfolio_reason"])
        if facts.get("realized_backtest_return_pct") is not None:
            parts.append(
                "In past tests, similar calls actually returned "
                f"{facts['realized_backtest_return_pct']:+.2f}% on average (this already happened)."
            )
        return " ".join(parts)

    # Technical mode.
    parts = [f"{t}: {act}"]
    if fr is not None:
        parts.append(f"forecast(H={h}d)={fore}")
    if conf is not None:
        parts.append(f"conf={conf:.0f}%")
    if facts.get("current_weight_pct") is not None and facts.get("target_weight_pct") is not None:
        parts.append(f"weight {facts['current_weight_pct']}%→{facts['target_weight_pct']}%")
    if facts.get("qty_change") is not None:
        parts.append(f"Δqty={facts['qty_change']}")
    if facts.get("est_cost") is not None:
        parts.append(f"est_cost=${facts['est_cost']}")
    if driver_txt:
        parts.append(f"drivers[{driver_txt}]")
    if facts.get("realized_backtest_return_pct") is not None:
        parts.append(f"realised(backtest)={facts['realized_backtest_return_pct']:+.2f}%")
    prov = [x for x in (facts.get("price_date"), facts.get("news_cutoff"), facts.get("model_version")) if x]
    line = " | ".join(parts)
    if prov:
        line += f"  [provenance: {', '.join(map(str, prov))}]"
    return line


def _extract_claim_numbers(text):
    """Extract numerical claims after removing dates and provenance metadata."""
    text = re.sub(r"\[provenance:[^\]]*\]", "", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)  # ISO dates
    text = re.sub(r"[A-Za-z][A-Za-z_]*\d+", "", text)  # version tokens e.g. xgb_v3
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def explanation_numbers_supported(facts):
    """Check that both explanation modes use supported numeric values."""
    canonical = {round(v, 2) for v in _fact_numbers(facts)} | {round(float(facts["horizon_days"]), 2)}
    b = {round(x, 2) for x in _extract_claim_numbers(render(facts, "beginner"))}
    tch = {round(x, 2) for x in _extract_claim_numbers(render(facts, "technical"))}

    def within(vals):
        """Check that each generated number matches an allowed value within tolerance."""
        return all(any(abs(v - a) <= 0.011 + abs(a) * 0.02 for a in canonical) for v in vals)

    return within(b) and within(tch)


def validate_numeric_claims(text, facts, tol=0.011):
    """Return numeric support status and unsupported values; field and unit checks are separate."""
    allowed = _fact_numbers(facts) + [float(facts["horizon_days"])]
    bad = []
    for n in _extract_claim_numbers(text):
        if not any(abs(n - a) <= tol + abs(a) * 0.02 for a in allowed):
            bad.append(n)
    return (len(bad) == 0, bad)


# Reject phrases promising guaranteed or risk-free outcomes.
UNSUPPORTED_CLAIM_PATTERNS = [
    r"guarantee",
    r"risk[\s-]?free",
    r"no risk",
    r"cannot lose",
    r"can'?t lose",
    r"zero risk",
    r"sure thing",
    r"certain to (?:gain|profit|rise|win|make)",
    r"will definitely",
    r"100%\s*(?:safe|return|profit|gain)",
    r"never lose",
]


def _dollar_claims(text):
    t = re.sub(r"\[provenance:[^\]]*\]", "", text)
    return [float(x) for x in re.findall(r"\$\s*(-?\d+\.?\d*)", t)]


def _percent_claims(text):
    t = re.sub(r"\[provenance:[^\]]*\]", "", text)
    return [float(x) for x in re.findall(r"(-?\d+\.?\d*)\s*%", t)]


def _dollar_facts(facts):
    return [float(facts[k]) for k in ("price", "est_cost") if facts.get(k) is not None]


def _percent_facts(facts):
    vals = [
        float(facts[k])
        for k in (
            "forecast_return_pct",
            "confidence_pct",
            "current_weight_pct",
            "target_weight_pct",
            "realized_backtest_return_pct",
        )
        if facts.get(k) is not None
    ]
    for v in facts.get("reliability", {}).values():
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    return vals


def validate_explanation(text, facts, tol=0.011):
    """Return validity and issues for numeric values, units, horizons and guarantee wording."""
    issues = []
    low = text.lower()
    for pat in UNSUPPORTED_CLAIM_PATTERNS:
        if re.search(pat, low):
            issues.append(f"banned phrase /{pat}/")

    def _match(v, allowed):
        return any(abs(v - a) <= tol + abs(a) * 0.02 for a in allowed)

    d_allowed = _dollar_facts(facts)
    for v in _dollar_claims(text):
        if not _match(v, d_allowed):
            issues.append(f"$ amount not a price/cost fact: {v}")
    p_allowed = _percent_facts(facts)
    for v in _percent_claims(text):
        if not _match(v, p_allowed):
            issues.append(f"% figure not a supported fact: {v}")
    h = float(facts.get("horizon_days") or 0)
    for m in re.findall(r"(\d+)\s*(?:trading\s*)?days?\b", low):
        if abs(float(m) - h) > 0.5:
            issues.append(f"wrong horizon {m}d vs {h:.0f}d")
    ok_num, bad = validate_numeric_claims(text, facts, tol=tol)
    if not ok_num:
        issues.append(f"unsupported numbers {bad}")
    return (len(issues) == 0, issues)
