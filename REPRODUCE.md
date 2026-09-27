# Reproducibility guide

Commands run from the project root using Python 3.11. Existing datasets and
trained artifacts can be reused; deleting caches is not a prerequisite.

## Environment

```bash
python3.11 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pip check
./venv/bin/python -m pip freeze > requirements.lock.txt
```

Optional chat and video features require the local configuration described in
README.md. Credentials belong in the local `.env` file.

## Data and deployment models

For a new dataset or installation without trained models:

```bash
./venv/bin/python scripts/run_advisor.py
```

This interactive workflow downloads prices, builds features, scores available
news and trains deployment models. The historical-news dataset is a separate
input; current headlines do not replace historical news coverage. The stored
membership history controls point-in-time stock eligibility in the prediction
training and evaluation paths.

Artifacts are stored in `models/xgboost/` and `models/lstm/`. Their training
metadata records the actual fitted sample and validation results. Full-history
deployment weights are not used to claim independent held-out performance.

## Forecasting evaluation

```bash
./venv/bin/python -u scripts/run_experiments.py --sp500-only --deep --start 2010-01-01 --end 2023-12-31
./venv/bin/python -u scripts/evaluate_deployed.py --sp500-only --with-lstm --start 2010-01-01 --end 2023-12-31
```

The first command evaluates experimental regressors, rules, GRU, ensemble and
HMM selection. The second evaluates the deployed model architectures with a
separate chronological split. LSTM comparisons use the same sequence-eligible
test observations for all baselines. Exclusion counts are recorded.

## Portfolio evaluation

```bash
./venv/bin/python -u scripts/train_ppo_oos.py --sp500-only --seeds 3 --start 2010-01-01 --end 2023-12-31
```

This comparison uses historical-price inputs, not model forecasts. Current
limitations include whole-period universe selection, a common-history calendar,
a short shared test period, and different rebalancing frequencies for PPO and
Markowitz. These results are exploratory and do not establish unbiased
outperformance. The equal-weight top-K strategy is not a whole-universe benchmark.

## Explanations and checks

```bash
./venv/bin/python scripts/export_explanations.py
./venv/bin/python -m pytest -q
node tests/frontend_contracts.cjs
./venv/bin/python scripts/verify_portfolio.py
```

SHAP plots describe raw classifier margins before calibration. They are not
causal explanations or independent performance tests. Automated tests isolate
model writes in temporary directories.

## Run provenance and distribution

Retain dataset windows, parameters, package versions, code revision, uncommitted
changes, generated timestamps and model metadata with each result set. Default
model seeds are 42; PPO runs use seeds 0, 1 and 2. Corrected reruns remain distinct
from earlier results and do not constitute a newly unseen test set.

Local credentials, saved user profiles, runtime logs, environments and caches
are excluded from distributable archives. Existing tracked files require a
separate repository review: adding an ignore rule does not remove their history.
