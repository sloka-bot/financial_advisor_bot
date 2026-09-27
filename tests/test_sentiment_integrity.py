"""News timing and inference failures must not create fabricated sentiment evidence."""

import json

import pandas as pd
import pytest

from backend.news.sentiment_analyzer import SentimentAnalyzer
from scripts.build_finbert_input import build_inputs


def test_historical_input_preserves_unknown_publication_time(tmp_path):
    raw, news = tmp_path / "raw", tmp_path / "news"
    raw.mkdir()
    pd.DataFrame(
        {"date": ["2023-06-01", "2023-06-02T20:15:00Z"], "title": ["First headline", "Second headline"]}
    ).to_csv(raw / "TEST.csv", index=False)
    result = build_inputs(raw, news, tmp_path / "input.jsonl")
    articles = json.loads((news / "TEST.json").read_text())["articles"]
    assert result["output_articles"] == 2
    assert articles[0]["published_at"] == "2023-06-01"
    assert articles[1]["published_at"] == "2023-06-02T20:15:00Z"


def test_failed_inference_does_not_write_neutral_scores(tmp_path):
    news = tmp_path / "news"
    news.mkdir()
    path = news / "TEST.json"
    original = json.dumps({"articles": [{"title": "Company reports growing sales", "published_at": "2023-06-01"}]})
    path.write_text(original)
    analyzer = SentimentAnalyzer(news_dir=news, sentiment_dir=tmp_path / "scores")

    def failed(batch):
        raise RuntimeError("Inference unavailable")

    analyzer._pipe = failed
    result = analyzer.analyze_universe(["TEST"])
    assert result["failed"] == ["TEST"]
    assert path.read_text() == original
    assert not (tmp_path / "scores" / "TEST.csv").exists()


def test_sentiment_cache_requires_matching_text(tmp_path):
    news = tmp_path / "news"
    news.mkdir()
    path = news / "TEST.json"
    path.write_text(
        json.dumps({"articles": [{"title": "Company reports growing sales", "published_at": "2023-06-01"}]})
    )
    analyzer = SentimentAnalyzer(news_dir=news, sentiment_dir=tmp_path / "scores")
    calls = []

    def classify(batch):
        calls.append(batch)
        return [{"label": "positive", "score": 0.8} for _ in batch]

    analyzer._pipe = classify
    analyzer.analyze_ticker("TEST")
    analyzer.analyze_ticker("TEST")
    assert len(calls) == 1
    saved = json.loads(path.read_text())
    saved["articles"][0]["title"] = "Company reports falling sales"
    path.write_text(json.dumps(saved))
    analyzer.analyze_ticker("TEST")
    assert len(calls) == 2


def test_incomplete_classification_batch_is_rejected(tmp_path):
    analyzer = SentimentAnalyzer(news_dir=tmp_path, sentiment_dir=tmp_path / "scores")
    analyzer._pipe = lambda batch: []
    with pytest.raises(ValueError, match="incomplete"):
        analyzer._classify(["Some news"])


class _FakeTokenizerPipe:
    """Test double for a transformers pipeline: classifies and exposes a tokenizer
    so truncation recording can be exercised without downloading FinBERT."""

    def __init__(self, n_tokens=10, label="neutral", score=0.9):
        self._n = n_tokens
        self._label = label
        self._score = score

    def __call__(self, batch):
        return [{"label": self._label, "score": self._score} for _ in batch]

    def tokenizer(self, batch, truncation=False):
        return {"input_ids": [list(range(self._n)) for _ in batch]}


def test_no_news_day_is_distinct_from_neutral_news_day(tmp_path):
    # Hard invariant (#12): a day with neutral articles and a day with NO news must
    # never collapse to the same representation.
    news = tmp_path / "news"
    news.mkdir()
    (news / "HASN.json").write_text(
        json.dumps({"articles": [{"title": "Company holds steady", "published_at": "2023-06-01T14:00:00Z"}]})
    )
    analyzer = SentimentAnalyzer(news_dir=news, sentiment_dir=tmp_path / "scores")
    analyzer._pipe = lambda batch: [{"label": "neutral", "score": 0.9} for _ in batch]

    neutral_news = analyzer.analyze_ticker("HASN")
    no_news = analyzer.analyze_ticker("MISSING")  # no file -> neutral fallback

    assert (neutral_news["sent_no_news"] == 0).all()
    assert (neutral_news["sent_news_count"] >= 1).all()
    assert (no_news["sent_no_news"] == 1).all()
    assert (no_news["sent_news_count"] == 0).all()
    assert int(neutral_news["sent_no_news"].iloc[0]) != int(no_news["sent_no_news"].iloc[0])


def test_finbert_truncation_is_recorded(tmp_path):
    # #11: inputs exceeding max_length are flagged and token counts saved.
    news = tmp_path / "news"
    news.mkdir()
    (news / "LONG.json").write_text(
        json.dumps({"articles": [{"title": "x", "content": "y", "published_at": "2023-06-01T14:00:00Z"}]})
    )
    analyzer = SentimentAnalyzer(news_dir=news, sentiment_dir=tmp_path / "scores", max_length=512)
    analyzer._pipe = _FakeTokenizerPipe(n_tokens=600)
    result = analyzer.analyze_universe(["LONG"])
    saved = json.loads((news / "LONG.json").read_text())["articles"][0]
    assert saved["sentiment_truncated"] is True
    assert saved["sentiment_token_count"] == 600
    assert result.get("truncated_fraction") == 1.0


def test_decision_date_before_close_is_same_session():
    # 10:00 ET (before the 16:00 close) -> same day's session
    assert SentimentAnalyzer._decision_date("2023-06-01T14:00:00Z").date() == pd.Timestamp("2023-06-01").date()


def test_decision_date_after_close_is_next_session():
    # 19:30 ET (after close) -> next session
    assert SentimentAnalyzer._decision_date("2023-06-01T23:30:00Z").date() == pd.Timestamp("2023-06-02").date()


def test_decision_date_weekend_maps_to_next_trading_session():
    # Saturday -> following Monday
    assert SentimentAnalyzer._decision_date("2023-06-03T12:00:00Z").date() == pd.Timestamp("2023-06-05").date()


def test_decision_date_holiday_skips_to_next_session():
    # US market holiday (Independence Day) -> next session
    assert SentimentAnalyzer._decision_date("2023-07-04T12:00:00Z").date() == pd.Timestamp("2023-07-05").date()


def test_decision_date_date_only_delays_at_least_a_day():
    # Date-only publications have unknown intraday timing: usable no earlier than
    # the following day's session.
    d = SentimentAnalyzer._decision_date("2023-06-01")
    assert d is not None and d.date() >= pd.Timestamp("2023-06-02").date()


def test_decision_date_respects_early_close():
    # Day after Thanksgiving has a 13:00 ET early close.
    before = SentimentAnalyzer._decision_date("2023-11-24T17:00:00Z")  # 12:00 ET, before
    after = SentimentAnalyzer._decision_date("2023-11-24T19:00:00Z")  # 14:00 ET, after
    assert before.date() == pd.Timestamp("2023-11-24").date()
    assert after.date() == pd.Timestamp("2023-11-27").date()
