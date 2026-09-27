"""Malformed sentiment must not prevent technical features from being used."""

import pandas as pd
import pytest

from backend.data.fusion import SENT_NUMERIC, FeatureFusion


@pytest.mark.parametrize("future_only", [False, True])
def test_unrecognized_sentiment_uses_neutral_defaults(tmp_path, future_only):
    features = tmp_path / "features"
    sentiment = tmp_path / "sentiment"
    features.mkdir()
    sentiment.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=30)
    pd.DataFrame({"close": range(100, 130)}, index=dates).to_csv(features / "TEST.csv")
    news_date = dates[-1] + pd.Timedelta(days=10) if future_only else dates[0]
    pd.DataFrame({"unexpected_column": ["bad input"]}, index=[news_date]).to_csv(sentiment / "TEST.csv")

    result = FeatureFusion(features, sentiment).fuse_ticker("TEST", save=False)

    assert len(result) == 30
    assert result[SENT_NUMERIC].eq(0).all().all()
    assert result["sent_no_news"].eq(1).all()
    assert result["sent_label"].eq("neutral").all()
    assert result["target_return"].iloc[0] == pytest.approx(0.21)


def test_short_price_history_skips_unsupported_indicators(tmp_path):
    from backend.data.feature_engineer import FeatureEngineer

    frame = pd.DataFrame(
        {"close": [100], "high": [101], "low": [99], "volume": [1000]}, index=pd.to_datetime(["2024-01-02"])
    )
    assert FeatureEngineer(features_dir=tmp_path).generate("NEW", frame, save=False) is None
