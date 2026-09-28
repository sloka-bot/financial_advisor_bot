"""Regression tests for saved state, missing models and service failures."""

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import main, runtime
from backend.api.routers import chat as chat_router
from backend.api.schemas import ChatRequest
from backend.data.contracts import executable_price
from backend.store import user_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(user_store, "STORE", tmp_path / "users.json")
    monkeypatch.setattr(runtime.fusion, "features_dir", tmp_path)
    monkeypatch.setattr(runtime.xgb_model, "is_trained", lambda: False)
    monkeypatch.setattr(runtime.lstm_model, "is_trained", lambda: False)
    return TestClient(main.app)


def test_partial_profile_update_preserves_other_values(client):
    assert client.post("/api/user/test", json={"name": "Student", "budget": 50000}).status_code == 200
    response = client.put("/api/user/test", json={"risk_profile": "conservative"})
    assert response.status_code == 200
    profile = client.get("/api/user/test").json()
    assert profile["name"] == "Student"
    assert profile["budget"] == 50000


def test_chat_without_models_or_ollama_does_not_crash(client, monkeypatch):
    client.post("/api/user/test", json={"budget": 10000})

    def offline(*args, **kwargs):
        raise requests.ConnectionError("Offline")

    monkeypatch.setattr(requests, "post", offline)
    for message in ("What is my portfolio?", "Explain risk", "What should I buy?"):
        response = client.post("/api/chat", json={"message": message, "user_id": "test"})
        assert response.status_code == 200
        assert response.json()["source"] == "template"
    prompt = chat_router.build_chat_prompt(ChatRequest(message="Is return 999?", user_id="test"))
    assert not chat_router._reply_numbers_supported("Your return is 999%.", prompt)
    isolated = '<verified_data>{"price": 3}</verified_data>'
    assert not chat_router._reply_numbers_supported("The price is $300.", isolated)
    assert not chat_router._reply_numbers_supported("Sharpe is 987.", isolated)
    assert not chat_router._reply_numbers_supported("The price is $3,000.", isolated)
    assert chat_router._reply_numbers_supported("The price is $3.00.", isolated)
    from backend.explain.explainer import Explainer

    assert not Explainer._validate_numeric_claims("The price is $300.", "price: 3")[0]
    assert not Explainer._validate_numeric_claims("Sharpe is 987.", "price: 3")[0]


def test_no_models_returns_actionable_error(client):
    response = client.post("/api/portfolio", json={"budget": 10000})
    assert response.status_code == 400
    assert "not trained" in response.json()["detail"].lower()


def test_saved_holdings_render_without_model_prices(client):
    client.post("/api/user/test", json={"budget": 10000})
    response = client.post(
        "/api/user/test/import-portfolio",
        json={"user_id": "test", "holdings": [{"ticker": "AAPL", "shares": 0.25, "price": 100}]},
    )
    assert response.status_code == 200
    saved = client.get("/api/user/test/portfolio").json()
    assert saved["holdings"][0]["valuation_estimated"]
    assert np.isfinite(saved["holdings"][0]["weight_pct"])


def test_corrupt_store_remains_refused_on_second_read(tmp_path, monkeypatch):
    path = tmp_path / "users.json"
    path.write_text("{broken")
    monkeypatch.setattr(user_store, "STORE", path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="corrupt"):
            user_store.UserStore().get("test")
    assert path.read_text() == "{broken"


def test_execution_price_uses_raw_close():
    assert executable_price(pd.DataFrame({"close": [100]})) is None
    assert executable_price(pd.DataFrame({"close": [100], "exec_close": [np.nan], "close_unadj": [120]})) == 120


def test_pipeline_can_prepare_predictions_before_done(client, monkeypatch):
    monkeypatch.setitem(runtime.pipeline_state, "step", "recommendations")
    predictions, data = runtime.collect_predictions([], pipeline_internal=True)
    assert predictions == {} and data == {}
    with pytest.raises(HTTPException):
        runtime.collect_predictions([])


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_file_import_preview_validates_without_saving(client, extension):
    import io

    frame = pd.DataFrame({"ticker": ["AAPL"], "shares": [0.25], "price": [100]})
    stream = io.BytesIO()
    if extension == "csv":
        payload = frame.to_csv(index=False).encode()
    else:
        frame.to_excel(stream, index=False)
        payload = stream.getvalue()
    response = client.post("/api/import-preview", params={"filename": "holdings." + extension}, content=payload)
    assert response.status_code == 200
    assert response.json()["holdings"][0]["shares"] == 0.25
    assert not user_store.STORE.exists()


def test_file_import_rejects_duplicate_and_invalid_values(client):
    for payload in (b"ticker,shares,price\nAAPL,1,100\nAAPL,1,100", b"ticker,shares,price\nAAPL,-1,100"):
        response = client.post("/api/import-preview?filename=holdings.csv", content=payload)
        assert response.status_code == 422


def test_invalid_excel_returns_validation_error(client):
    response = client.post("/api/import-preview?filename=holdings.xlsx", content=b"not a workbook")
    assert response.status_code == 422


def test_missing_return_model_keeps_recommendations_serializable(client, monkeypatch):
    from backend.explain.explainer import Explainer
    from backend.prediction.ranker import StockRanker
    from backend.prediction.recommender import RecommendationEngine

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    data = {
        "AAPL": pd.DataFrame(
            {
                "close": np.linspace(100, 150, 300),
                "exec_close": np.linspace(100, 150, 300),
                "volume": 1000000,
                "volatility": 0.02,
                "rsi": 50,
                "momentum_10d": 0.01,
                "sent_score": 0,
                "sent_news_count": 0,
            },
            index=dates,
        )
    }
    predictions = {"AAPL": {"prob_up": 0.7, "expected_return": None, "horizon": 21}}
    monkeypatch.setattr(runtime.xgb_model, "is_trained", lambda: True)
    monkeypatch.setattr(runtime, "collect_predictions", lambda *args, **kwargs: (predictions, data))
    monkeypatch.setitem(runtime.pipeline_state, "processed_tickers", ["AAPL"])
    monkeypatch.setattr(runtime.explainer, "explain_batch", lambda *args: None)
    for endpoint in ("/api/recommend", "/api/portfolio"):
        response = client.post(endpoint, json={"budget": 10000})
        assert response.status_code == 200, response.text
    rec = RecommendationEngine().recommend(StockRanker().rank(predictions, data, "moderate"))["all_signals"][0]
    assert rec["predicted_return"] is None
    assert isinstance(Explainer()._template(rec), str)


@pytest.mark.parametrize("budget", [0, -10, "invalid"])
def test_risk_preview_rejects_invalid_budget(client, budget):
    response = client.post("/api/risk-preview", json={"budget": budget})
    assert response.status_code == 422


def test_risk_preview_uses_raw_execution_prices_without_forecasts(client, monkeypatch, tmp_path):
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    frame = pd.DataFrame(
        {"close": np.linspace(10, 20, 300), "exec_close": np.linspace(100, 200, 300), "volume": 1_000_000},
        index=dates,
    )
    frame.to_csv(tmp_path / "TEST_master.csv")
    monkeypatch.setitem(runtime.pipeline_state, "processed_tickers", ["TEST"])
    monkeypatch.setitem(runtime.pipeline_state, "step", "idle")
    response = client.post("/api/risk-preview", json={"budget": 10000, "risk_profile": "aggressive"})
    assert response.status_code == 200
    result = response.json()
    assert result["trained"]
    assert result["holdings"]
    holding = result["holdings"][0]
    assert holding["price"] == 200
    assert holding["weight_pct"] <= 40
    assert holding["total_cost"] == holding["shares"] * 200


def test_invalid_workbook_xml_returns_validation_error(client):
    import io
    import zipfile

    stream = io.BytesIO()
    pd.DataFrame({"ticker": ["AAPL"], "shares": [1], "price": [100]}).to_excel(stream, index=False)
    damaged = io.BytesIO()
    with zipfile.ZipFile(stream) as source, zipfile.ZipFile(damaged, "w") as target:
        for item in source.infolist():
            target.writestr(item, b"<broken" if item.filename == "xl/workbook.xml" else source.read(item.filename))
    response = client.post("/api/import-preview?filename=holdings.xlsx", content=damaged.getvalue())
    assert response.status_code == 422


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/api/backtest", {"tickers": [], "capital": 1000}),
        ("/api/backtest", {"tickers": ["AAPL"], "capital": -1}),
        ("/api/backtest-predict", {"ticker": "AAPL", "start_date": "not-a-date"}),
        ("/api/backtest-range", {"tickers": ["AAPL"], "start_date": "2024-02-01", "end_date": "2024-01-01"}),
        (
            "/api/backtest-range",
            {"tickers": ["AAPL"], "start_date": "2024-01-01", "end_date": "2024-02-01", "horizon": 0},
        ),
    ],
)
def test_backtest_invalid_inputs_are_validation_errors(client, endpoint, payload):
    assert client.post(endpoint, json=payload).status_code == 422


def test_chat_model_question_explains_models_without_ollama(client, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("Offline")

    monkeypatch.setattr(requests, "post", offline)
    response = client.post("/api/chat", json={"message": "How does the model work?", "user_id": "test"})
    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "XGBoost" in reply and "LSTM" in reply and "21 trading sessions" in reply
    assert "can be wrong" in reply


def test_repeated_profile_creation_does_not_reset_saved_account(client):
    client.post("/api/user/test", json={"budget": 10000, "risk_profile": "conservative"})
    runtime.user_store.update_portfolio("test", [{"ticker": "AAPL", "shares": 2, "price": 100}], 9800)
    runtime.user_store.add_recommendations("test", [{"ticker": "MSFT", "action": "BUY"}])
    before = runtime.user_store.get("test")
    response = client.post("/api/user/test", json={"budget": 500})
    assert response.status_code == 200
    assert runtime.user_store.get("test") == before


def test_invalid_import_preserves_saved_holdings(client):
    client.post("/api/user/test", json={"budget": 10000})
    runtime.user_store.update_portfolio("test", [{"ticker": "AAPL", "shares": 2, "price": 100}], 9800)
    before = runtime.user_store.get("test")
    response = client.post(
        "/api/user/test/import-portfolio",
        json={"user_id": "test", "holdings": [{"ticker": "MSFT", "shares": 1000, "price": 100}]},
    )
    assert response.status_code == 400
    assert runtime.user_store.get("test") == before


def test_recommendation_queue_keeps_all_eligible_signals_without_promoting_hold(client, monkeypatch):
    client.post("/api/user/test", json={"budget": 10000})
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    signals = [{"ticker": t, "signal": "BUY", "composite_score": 100 - i} for i, t in enumerate(tickers)]
    signals += [dict(signals[0]), {"ticker": "HOLD", "signal": "HOLD"}]
    monkeypatch.setattr(runtime.universe_builder, "members", lambda: tickers + ["HOLD"])
    monkeypatch.setattr(runtime, "collect_predictions", lambda *a, **k: ({t: {} for t in tickers}, {}))
    monkeypatch.setattr(runtime.ranker, "rank", lambda *a: None)
    monkeypatch.setattr(runtime.recommender, "recommend", lambda *a, **k: {"all_signals": signals})
    monkeypatch.setattr(
        runtime.portfolio_manager,
        "apply_recommendation",
        lambda rec, *a, **k: {"error": "unaffordable"} if rec["ticker"] == "GGG" else {"respects_profile": True},
    )
    monkeypatch.setattr(runtime.portfolio_manager, "confidence_score", lambda *a, **k: {"overall": 50, "factors": {}})
    monkeypatch.setattr(runtime.portfolio_manager, "_buy_reason", lambda *a: "Forecast meets threshold")
    monkeypatch.setattr(runtime.drift_monitor, "log_recommendation", lambda **k: None)
    runtime.queue_recommendations("test", tickers + ["HOLD"], "moderate", 10000)
    pending = runtime.user_store.get("test")["pending_recommendations"]
    assert [r["ticker"] for r in pending] == tickers[:6]
    runtime.user_store.reject_recommendation("test", pending[0]["id"])
    remaining = runtime.user_store.get("test")["pending_recommendations"]
    assert [r["ticker"] for r in remaining[:3]] == tickers[1:4]


def _review_prices(monkeypatch):
    frame = pd.DataFrame({"exec_close": [10.0], "close": [10.0]}, index=[pd.Timestamp.today().normalize()])
    monkeypatch.setattr(runtime.fusion, "load_master", lambda ticker: frame)
    return frame


def test_manual_buy_and_sell_charge_fees_and_preserve_cash_after_exit(client, monkeypatch):
    _review_prices(monkeypatch)
    client.post("/api/user/test", json={"budget": 1000})
    runtime.user_store.update_portfolio("test", [{"ticker": "AAA", "shares": 10, "price": 10, "total_cost": 100}], 900)
    bought = client.post("/api/user/test/buy", json={"ticker": "AAA", "shares": 2})
    assert bought.status_code == 200
    assert bought.json()["cash"] == pytest.approx(879.98)
    assert bought.json()["holdings"][0]["shares"] == 12
    sold = client.post("/api/user/test/sell", json={"ticker": "AAA"})
    assert sold.status_code == 200
    assert sold.json()["proceeds"] == pytest.approx(119.88)
    assert sold.json()["cash"] == pytest.approx(999.86)
    client.put("/api/user/test", json={"budget": 1000})
    assert runtime.user_store.get("test")["portfolio"]["cash"] == pytest.approx(999.86)


def test_manual_trade_rejects_missing_price_and_excess_quantity(client, monkeypatch):
    client.post("/api/user/test", json={"budget": 1000})
    runtime.user_store.update_portfolio("test", [{"ticker": "AAA", "shares": 10, "price": 10, "total_cost": 100}], 900)
    before = runtime.user_store.get("test")
    assert client.post("/api/user/test/sell", json={"ticker": "AAA"}).status_code == 409
    _review_prices(monkeypatch)
    assert client.post("/api/user/test/buy", json={"ticker": "AAA", "shares": 100}).status_code == 409
    assert runtime.user_store.get("test") == before


def test_build_reconciles_fees_and_checks_whole_book(client, monkeypatch):
    from backend.api.routers import users

    frame = _review_prices(monkeypatch)
    client.post("/api/user/test", json={"budget": 1000})
    runtime.user_store.update_portfolio("test", [{"ticker": "AAA", "shares": 24, "price": 10, "total_cost": 240}], 760)
    monkeypatch.setitem(runtime.pipeline_state, "processed_tickers", ["AAA", "BBB"])
    monkeypatch.setattr(runtime, "collect_predictions", lambda *a: ({}, {"AAA": frame, "BBB": frame}))
    monkeypatch.setattr(
        users,
        "build_markowitz_portfolio",
        lambda *a, **k: {
            "portfolio": {
                "holdings": [{"ticker": "AAA", "shares": 10}, {"ticker": "BBB", "shares": 5}],
                "available": True,
            }
        },
    )
    res = client.post("/api/user/test/build").json()
    assert res["built"] == 1
    assert res["cash"] == pytest.approx(709.95)
    assert sum(h["shares"] * 10 for h in res["holdings"]) + res["fees"] + res["cash"] == pytest.approx(1000)
    assert max(h["shares"] * 10 / (1000 - res["fees"]) for h in res["holdings"]) <= 0.25


def test_build_does_not_overwrite_concurrent_portfolio_change(client, monkeypatch):
    from backend.api.routers import users

    frame = _review_prices(monkeypatch)
    client.post("/api/user/test", json={"budget": 1000})
    monkeypatch.setitem(runtime.pipeline_state, "processed_tickers", ["AAA"])
    monkeypatch.setattr(runtime, "collect_predictions", lambda *a: ({}, {"AAA": frame}))

    def changed(*a, **k):
        runtime.user_store.update_portfolio(
            "test", [{"ticker": "BBB", "shares": 1, "price": 10, "total_cost": 10}], 990
        )
        return {"portfolio": {"holdings": [{"ticker": "AAA", "shares": 1}]}}

    monkeypatch.setattr(users, "build_markowitz_portfolio", changed)
    assert client.post("/api/user/test/build").status_code == 409
    assert runtime.user_store.get("test")["portfolio"]["holdings"][0]["ticker"] == "BBB"


def test_generate_rotates_visible_candidates_without_converting_hold(client, monkeypatch):
    client.post("/api/user/test", json={"budget": 1000})
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "HOLD"]
    signals = [{"ticker": t, "signal": "BUY", "composite_score": 100 - i} for i, t in enumerate(tickers[:-1])]
    signals.append({"ticker": "HOLD", "signal": "HOLD", "composite_score": 99})
    monkeypatch.setattr(runtime.universe_builder, "members", lambda: tickers)
    monkeypatch.setattr(runtime, "collect_predictions", lambda *a, **k: ({t: {} for t in tickers}, {}))
    monkeypatch.setattr(runtime.ranker, "rank", lambda *a: None)
    monkeypatch.setattr(runtime.recommender, "recommend", lambda *a, **k: {"all_signals": signals})
    monkeypatch.setattr(runtime.portfolio_manager, "apply_recommendation", lambda *a, **k: {"respects_profile": True})
    monkeypatch.setattr(runtime.portfolio_manager, "confidence_score", lambda *a, **k: {"overall": 50, "factors": {}})
    monkeypatch.setattr(runtime.portfolio_manager, "_buy_reason", lambda *a: "Meets thresholds")
    monkeypatch.setattr(runtime.drift_monitor, "log_recommendation", lambda **k: None)
    runtime.queue_recommendations("test", tickers, "moderate", 1000)
    first = runtime.user_store.get("test")["pending_recommendations"]
    assert len(first) == 6
    runtime.queue_recommendations("test", tickers, "moderate", 1000)
    second = runtime.user_store.get("test")["pending_recommendations"]
    assert [r["ticker"] for r in second] == ["DDD", "EEE", "FFF"]
