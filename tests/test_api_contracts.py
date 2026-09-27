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
    assert not chat_router._numbers_grounded("Your return is 999%.", prompt)


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


def test_execution_price_never_uses_adjusted_close():
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
