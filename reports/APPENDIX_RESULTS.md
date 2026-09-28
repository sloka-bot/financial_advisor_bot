# Appendix - Actual Results

_Auto-generated 2026-09-27T05:17:30 from the result JSONs. Do not edit by hand; re-run `scripts/make_appendix_tables.py` after any retrain._

## A. Provenance

| File | generated_at | schema |
|---|---|---|
| deployed_evaluation.json | 2026-09-27T11:57:39.904985 | 3 |
| deployed_evaluation_h5.json | 2026-09-27T12:46:42.095669 | 3 |
| deployed_evaluation_h1.json | 2026-09-27T12:48:35.912958 | 3 |
| experiment_results.json | 2026-09-27T12:50:28.124943 | 3 |
| portfolio_results.json | 2026-09-27T12:53:37.001478 | 3 |

## B. Deployed classifier by horizon (out-of-sample test)

| Horizon | Model | Balanced acc | ROC-AUC | ROC-AUC - 0.5 | n_test |
|---|---|---|---|---|---|
| 1 | majority_class | 0.5 | - | - | 381143 |
| 1 | logistic_technical | 0.5049 | 0.5083 | 0.0083 | 381143 |
| 1 | xgb_technical | 0.5033 | 0.5093 | 0.0093 | 381143 |
| 1 | xgb_technical_sentiment | 0.5037 | 0.5088 | 0.0088 | 381143 |
| 5 | majority_class | 0.5 | - | - | 380559 |
| 5 | logistic_technical | 0.4999 | 0.4982 | -0.0018 | 380559 |
| 5 | xgb_technical | 0.5002 | 0.5020 | 0.002 | 380559 |
| 5 | xgb_technical_sentiment | 0.4999 | 0.5029 | 0.0029 | 380559 |
| 21 | majority_class | 0.5 | - | - | 378236 |
| 21 | logistic_technical | 0.4989 | 0.5072 | 0.0072 | 378236 |
| 21 | xgb_technical | 0.4989 | 0.5025 | 0.0025 | 378236 |
| 21 | xgb_technical_sentiment | 0.4997 | 0.5032 | 0.0032 | 378236 |

_Balanced accuracy 0.50 = no separation; ROC-AUC near 0.50 = little discrimination. Note the 1-day model shows slightly more lift than 21-day, consistent with a short-lived news effect._

## C. Probability calibration (21-day, held-out block)

| Metric | Raw | Isotonic-calibrated |
|---|---|---|
| Brier score | 0.25071 | 0.25062 |
| ECE (10-bin) | 0.03593 | 0.05205 |

## D. Sentiment contribution (paired, date-block bootstrap, 21-day)

| Metric | Mean diff | 95% CI low | 95% CI high | Significant |
|---|---|---|---|---|
| Directional-hit improvement | 0.000801 | -0.000264 | 0.001735 | False |

## E. Return regression (21-day, matched test rows)

| Model | MAE | RMSE | Directional acc |
|---|---|---|---|
| zero_return | 0.06864 | 0.091894 | 0.0005 |
| dev_hist_mean | 0.068225 | 0.091144 | 0.547 |
| xgb_regressor | 0.069022 | 0.092196 | 0.5275 |
| lstm_actual | 0.068124 | 0.090659 | 0.5472 |

## F. Research regressors by horizon (test set: balanced dir-acc, IC, net return)

| Horizon | Features | Bal dir-acc | IC | MAE | Trading net % |
|---|---|---|---|---|---|
| 1 | Technical | 0.5035 | 0.0112 | 0.014314 | -41.22 |
| 1 | Sentiment | 0.5005 | 0.0006 | 0.014278 | -61.81 |
| 1 | Technical+Sentiment | 0.5034 | 0.0106 | 0.014316 | -28.37 |
| 5 | Technical | 0.4988 | 0.0091 | 0.03282 | 82.74 |
| 5 | Sentiment | 0.5003 | 0.0002 | 0.03257 | 115.49 |
| 5 | Technical+Sentiment | 0.4992 | 0.0104 | 0.032801 | 63.46 |
| 21 | Technical | 0.4952 | 0.0234 | 0.0691 | 120.2 |
| 21 | Sentiment | 0.4999 | 0.0008 | 0.068274 | 68.35 |
| 21 | Technical+Sentiment | 0.4956 | 0.0226 | 0.069056 | 102.46 |

_Trading net % is a top-K long strategy on the research regressor; it is not the deployed advisor and carries the stated benchmarking limitations._

## G. Portfolio comparison (21-day horizon, shared split 2023-08-22 to 2023-12-21)

| Strategy | Net return % | Sharpe | Max DD % | Avg turnover | Rebalances |
|---|---|---|---|---|---|
| Equal-weight top-K | 2.37 | 0.402 | -7.34 | 1.702 | 4 |
| Markowitz | 10.41 | 1.458 | -6.37 | 0.6196 | 4 |
| Markowitz (cost-aware) | 10.36 | 1.453 | -6.37 | 0.6171 | 4 |
| Markowitz (smart rebalance) | 10.36 | 1.887 | -3.11 | 0.3345 | 4 |
| PPO (mean of seeds) | 10.91 | 1.861 | 9.71 | - | - |

**Stated limitations:** Common-history universe excludes assets without full overlapping history; report this selection bias.; historical_markowitz first screens to the top-K trailing-positive-return names (plus retained holdings), so it reflects a return-momentum selection stage, not mean-variance optimisation over the full eligible universe.; Markowitz rebalances every horizon; PPO can rebalance daily. Gross buy-and-hold excludes transaction costs.; PPO daily risk metrics and horizon-sampled Markowitz metrics are not directly comparable.; Research uses fractional weights; the live allocator rounds to whole shares.; Equal-weight top-K is unconstrained and is not a risk-profile-matched comparator.

## H. Deployed model training metadata

| Model | n_samples | metric | config | train date range |
|---|---|---|---|---|
| XGBoost H1 | 1319716 | cv_auc=0.5092 | fixed XGB hyperparams | 2010-10-18..2023-12-29 |
| XGBoost H5 | 1317524 | cv_auc=0.5041 | fixed XGB hyperparams | 2010-10-18..2023-12-29 |
| XGBoost H21 (primary) | 1320264 | cv_auc=0.5052 | fixed XGB hyperparams | 2010-10-18..2023-12-29 |
| LSTM H1 | 1305267 | best_epoch=2/15 | ep_max=50, batch=256, lr=0.001, pat=10 | 2010-10-18..2023-12-29 |
| LSTM H5 | 1301045 | best_epoch=2/12 | ep_max=50, batch=256, lr=0.001, pat=10 | 2010-10-18..2023-12-29 |
| LSTM H21 (primary) | 899153 | best_epoch=14/15 | ep_max=15, batch=256, lr=0.001, pat=10 | 2010-10-18..2020-07-21 |
