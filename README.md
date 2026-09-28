# Financial Advisor Bot

An educational decision-support tool for beginner investors over the US S&P 500.
It combines direction/return prediction, news sentiment, risk-profiled portfolio
construction, and plain-language explanations behind a single web interface. It
does not execute trades or manage real money, and every recommendation is meant
to be independently verified.

Forecasting and portfolio optimisation have separate inputs: the
optimiser sets weights from historical prices and the user's risk profile only,
and the model forecasts are shown as display metadata, not fed into allocation.
This keeps portfolio evaluation independent of forecast quality.

---

## System requirements

| Requirement | Version / notes                              |
|-------------|----------------------------------------------|
| Python      | 3.11 (developed on Apple M2, macOS 14+)      |
| RAM         | 8 GB minimum (FinBERT loads ~2 GB)           |
| Disk        | ~15 GB free (full price/news data ~9 GB, venv ~1.5 GB, model weights + headroom) |

---

## Installation

```bash
cd financial_advisor_bot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The optional natural-language advisor uses a local Ollama model:

```bash
brew install ollama
ollama pull llama3.2
ollama serve
```

Ollama runs locally and reads
`OLLAMA_URL` and `OLLAMA_MODEL` from `.env` (defaults `http://localhost:11434`
and `llama3.2`). If Ollama is not running, the advisor falls back to a
deterministic template, so the app still works.

---

## Running the application

Start the servers, then open the app in a browser:

```bash
# Terminal 1 - Ollama server (optional; powers the Chatbot advice text and the avatar speech)
ollama serve

# Terminal 2 - backend API (also serves the frontend)
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Then open the app (use this URL)
open http://localhost:8000/app/
```

Only the Chatbot/avatar need Ollama; every other tab works without it. The app
must be opened at `http://localhost:8000/app/` - opening the HTML file directly
(`file://`) is blocked by the backend's local-origin check.

---

## First-time use

The trained models are already saved in `models/`, so no training is needed before
using the app. Start the backend as shown above and open the app at
`http://localhost:8000/app/` in a browser. Opening `frontend/index.html` directly
does not work. Create a risk profile and budget, or import holdings, then review
proposals on the Portfolio page. Approving a proposal updates the local portfolio
only and does not place a brokerage order.

## Data

The price, feature and news data (about 7 GB) is not included in the repository
because of its size, so the demonstration runs from the author's machine. The
trained models in `models/` and the saved evaluation results in
`data/experiments/` are included, so the reported figures can be inspected
without the raw data. The steps below recreate the data from scratch. They need
an internet connection and take several hours on the full universe; all commands
run from the project root with the virtual environment active.

```bash
source venv/bin/activate

# 1. Historical news (optional). Download the FNSPID news file (about 23 GB):
mkdir -p data/fnspid
curl -L -o data/fnspid/nasdaq_exteral_data.csv \
  "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv?download=true"
python scripts/extract_fnspid_news.py            # per-ticker S&P 500 news
python scripts/build_finbert_input.py            # deduplicated articles per ticker
python scripts/score_historical_sentiment.py     # FinBERT scores (downloads the model on first run)

# 2. Prices, cleaning, technical features and fused datasets:
python scripts/rebuild_data.py
```

Step 1 can be skipped for a quicker setup; sentiment features then default to
neutral values. Step 2 downloads prices from Yahoo Finance and the historical
S&P 500 membership list, and writes the datasets the app and training scripts
read. Running `scripts/rebuild_data.py` again later downloads only new sessions;
run it after the US market close so the latest session has an observed closing
price, then use Generate new recommendations on the Portfolio page.

## Regenerating models

Optional. The saved models were produced with the commands below, which only need
to be run again to rebuild them from scratch. Training uses data up to
2023-12-31 (the news-covered window used in the report) and the point-in-time
S&P 500 membership filter.

```bash
# Primary 21-day models (deployed to models/xgboost and models/lstm):
./venv/bin/python scripts/retrain_models.py              # XGBoost + LSTM
./venv/bin/python scripts/retrain_models.py --skip-lstm  # XGBoost only (fast)
./venv/bin/python scripts/retrain_lstm.py                # LSTM only (batch 256, early stopping)

# 1-day and 5-day side models for the Stock Predictor horizon selector,
# saved under models/xgboost/h{1,5} and models/lstm/h{1,5}:
./venv/bin/python scripts/train_horizons.py
./venv/bin/python scripts/train_horizons.py --limit 5 --epochs 2   # fast smoke test
```

The 21-day model is the primary deployed artifact used by the recommendation
engine; the 1/5-day models are used only by the predictor's horizon selector and
never touch the 21-day model registry. Deployed LSTM training uses up to 50 epochs, early-stop patience 10 and batch
size 256. The saved evaluation run used its separate 15-epoch budget.

## Reproducing the experiments

The forecasting and portfolio results in the report are produced by the research
scripts below. Existing datasets and trained artifacts can be reused; run
`./venv/bin/python -m pip check` first to confirm the environment.

```bash
# Forecasting: experimental regressors and rules,
# then the deployed architectures on a separate chronological split
./venv/bin/python -u scripts/run_experiments.py --sp500-only --start 2010-01-01 --end 2023-12-31
./venv/bin/python -u scripts/evaluate_deployed.py --sp500-only --with-lstm --start 2010-01-01 --end 2023-12-31

# Portfolio: PPO vs Markowitz on historical-price inputs (three seeds)
./venv/bin/python -u scripts/train_ppo_oos.py --sp500-only --seeds 3 --start 2010-01-01 --end 2023-12-31

# Explanations and checks
./venv/bin/python scripts/export_explanations.py
./venv/bin/python scripts/verify_portfolio.py
```

`scripts/run_advisor.py` runs the end-to-end demo (download, features, news
scoring, one recommendation) into a throwaway session directory; it never writes
the deployed models, which come only from the training scripts above. LSTM
comparisons use the same sequence-eligible test rows for every baseline, and
exclusion counts are recorded. The default model seed is 42; PPO uses seeds 0, 1
and 2. Corrected reruns are kept distinct from earlier results and are not a
newly unseen test set. Local credentials, saved profiles, logs and caches are
excluded from distribution.

## Application tabs

| Tab | Purpose |
|---|---|
| Dashboard | Saved holdings, cash and valuation changes |
| Portfolio | Review recommendations and approve local changes |
| Stock Predictor | 1/5/21-day forecast with a BUY/HOLD/SELL signal and confidence, technical-indicator charts (SMA, Bollinger, RSI, MACD, volume), and a separately labelled historical comparison |
| Chatbot | Written advice, optional Ollama and live avatar conversation |
| Tools | Compound interest, Kelly, risk/reward and simulation calculators |

## Live avatar conversation

The avatar repeats the reply already shown in chat. Spoken questions return
through the same advisor endpoint. No separate avatar language model generates
financial advice. Text chat remains available without video credentials.

Add these values to the local `.env` file and restart the backend:

```text
LIVEAVATAR_API_KEY=
LIVEAVATAR_AVATAR_ID=
LIVEAVATAR_VOICE_ID=
LIVEAVATAR_SANDBOX=true
```

LiveAvatar is HeyGen's real-time (streaming) avatar product - the browser SDK is
`@heygen/liveavatar-web-sdk` - and is separate from HeyGen's rendered-video API.
Get the API key, avatar ID and voice ID from the LiveAvatar dashboard
(https://www.liveavatar.com/). Set `LIVEAVATAR_SANDBOX=false` to use your own
avatar; `true` uses the provider sandbox. The integration limits sessions
to 120 seconds. In Chatbot, select Start conversation, then Enable microphone
only when speech input is wanted. Leaving Chatbot ends the session. Audio and
reply text are sent to LiveAvatar while connected; account keys remain on the server.
The local tests use mocked provider responses and do not establish live service availability.

Provider reference: https://docs.liveavatar.com/docs/full-mode/configuration

The browser SDK is bundled locally with its licence notices. To rebuild it:

```bash
cd frontend
npm ci
npm run build:avatar
```

## Verification

```bash
./venv/bin/python -m pip check
./venv/bin/python -m pytest -q
node tests/frontend_contracts.cjs
node tests/avatar_contracts.cjs
./venv/bin/python scripts/verify_portfolio.py
```

## Architecture

```
frontend/              Vanilla HTML/JS
  index.html           Application shell and navigation
  js/core/             state, API requests, UI helpers and charts
  js/pages/            dashboard, portfolio, predictor, chatbot and tools
  js/components/       onboarding and live avatar conversation
  js/vendor/           locally bundled third-party libraries

backend/               FastAPI service
  main.py              Application setup and router registration
  runtime.py           Shared services and pipeline state
  api/                 Validated request schemas and focused HTTP routers
  services/            Training workflow and saved-portfolio valuation
  config/              settings.py - central tunables (horizons, costs, risk)
  data/                downloader, cleaner, feature_engineer, fusion
  news/                news_collector, sentiment_analyzer (FinBERT)
  prediction/          xgboost_model, lstm_model, ranker, recommender,
                       regime_detector
  portfolio/           allocation, markowitz, portfolio_manager, rl_env,
                       rl_agent
  evaluation/          experiments, backtester
  explain/             explainer, explainer_modes
  infra/               model_registry, drift_monitor, live_avatar
  store/               user_store
  universe/            universe_builder, sp500_membership

scripts/               retrain_models.py, retrain_lstm.py, train_horizons.py,
                       run_experiments.py, train_ppo_oos.py, run_advisor.py, ...
tests/                 pytest suite (isolated from the production model dir)

models/                Local weights and versioned training metadata
  xgboost/  lstm/  rl/
data/                  Price data, features, sentiment, results
  raw/  processed/  features/ (per-ticker *.csv and *_master.csv)
  sentiment/  news/  universe/  experiments/  audit/
```

Point-in-time S&P 500 membership comes from the fja05680/sp500 components history
(cached under `data/universe/`), so the historical universe reflects the index as
it stood on each date rather than today's constituents.

---

## Models

| Model     | Role                                                  | Reference                        |
|-----------|-------------------------------------------------------|----------------------------------|
| XGBoost   | Direction classifier, calibrated (1/5/21-day; 21-day primary) | Chen & Guestrin (2016)           |
| LSTM      | Return regressor, 30-day window (1/5/21-day; 21-day primary)   | Hochreiter & Schmidhuber (1997)  |
| FinBERT   | News sentiment features                               | Araci (2019); Malo et al. (2014) |
| Markowitz | Live mean-variance allocation (historical inputs)     | Markowitz (1952)                 |
| PPO (SB3) | Offline experimental allocation comparator            | Schulman et al. (2017)           |

The live `/api/portfolio` path uses the historical Markowitz optimiser. **PPO is
an offline experimental comparator evaluated against Markowitz in
`scripts/train_ppo_oos.py`; it is not part of the operational advisor.**

Sentiment has two distinct pipelines: historical training and evaluation use the
FNSPID news dataset, while live inference scores recent Yahoo Finance headlines.

Validation is leakage-controlled: expanding-window walk-forward folds split by
calendar date, the scaler is fit on training rows only, and an embargo drops
training rows whose forward label reaches into the validation window. XGBoost
probabilities are calibrated with isotonic regression on a separate held-out
block; how closely a stated confidence matches observed accuracy is measured on
that block rather than assumed.

---

## Evaluation status

Under leakage-controlled walk-forward validation, 21-day directional accuracy is
close to the non-machine-learning baselines (roughly 0.49–0.51 balanced accuracy
in the saved experiments), so the current evidence does not show a large
predictive edge. This is reported accurately; the value of the system is in the
integrated, explainable pipeline and the risk-profiled allocation rather than in
a headline accuracy figure. Reproduce the experiment matrix with
`python scripts/run_experiments.py --sp500-only`.

---

## Known limitations

- Directional prediction over 21 sessions is close to the evaluated baselines;
  these configurations do not establish a reliable predictive advantage.
- Historical sentiment features come from FNSPID; live sentiment reads only
  recent headlines, so the two are different data sources.
- The PPO agent is trained offline and is an experimental comparator, not part of
  the live portfolio path.
- Portfolio comparisons use a common-history subset, which can introduce selection
  bias. Forecast experiments apply historical membership, but unavailable delisted
  histories can still limit coverage.
