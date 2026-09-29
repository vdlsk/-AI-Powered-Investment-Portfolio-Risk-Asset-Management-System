# Data Leakage Audit — FinRisk

## 1. What is data leakage?

Data leakage happens when information that would not actually have been
available at the time a decision was made is allowed to influence that
decision during model development or evaluation. The model then looks
accurate during testing but fails in live use, because live use never has
access to that "future" information.

## 2. Why financial time-series data is especially vulnerable

- Data is sequential and autocorrelated, so a randomly-shuffled train/test
  split silently mixes future and past observations next to each other.
- Rolling/lookback features are easy to compute "centered" by mistake
  (`.rolling(20).mean()` without shifting), which uses future values.
- Targets are inherently forward-looking ("will risk exceed X in the next 5
  days?"), so it is easy to accidentally leave a forward-looking column in
  the feature set used to *predict* that target.
- Preprocessing statistics (mean/std for scaling, feature-selection
  correlations, hyperparameter choices) computed on the full dataset
  implicitly "see" the test period.
- Backtests that optimize a portfolio using the entire historical window and
  then report that portfolio's historical performance are circular: the
  weights were chosen *knowing* how the whole period played out.

## 3. Every possible leakage source in this project, and how it is prevented

| Source | Prevention |
|---|---|
| Target leakage | Targets (`future_5d_*`) are computed in a separate step from features (`src/features/feature_pipeline.py`), never merged into the feature table itself. Automated check: `test_feature_columns_contain_no_future_or_target_leakage`. |
| Look-ahead bias in features | All rolling features in `feature_pipeline.py` are trailing windows (`.rolling(window, min_periods=window)`) ending at date `t`, never centered. Automated check: `test_rolling_features_are_trailing_not_centered`. |
| Survivorship bias | Documented as a known limitation in `README.md` §20 — not silently ignored. |
| Train/test contamination | Chronological split only (`train_end` / `validation_end` in `config.yaml`); `train_test_split(shuffle=True)` is never used on the primary time-series model. Automated check: `test_chronological_split_is_strictly_ordered`. |
| Temporal leakage | Feature date == prediction date by construction; every feature uses only rows up to and including its own date. Automated check: `test_feature_dates_never_exceed_prediction_date`. |
| Preprocessing/scaling leakage | Scalers/imputers are fit only on the training split, then applied (`.transform`, never re-`.fit`) to validation/test. Automated check: `test_scaler_fit_only_on_training_data`. |
| Rolling-window leakage | See "look-ahead bias in features" above; also verified by perturbing a single future price and confirming past features are unchanged (`test_future_data_does_not_change_past_features`). |
| Feature-selection leakage | Feature selection (Phase 2 ML pipeline) is performed inside the training fold / cross-validation only, never on the full dataset. |
| Hyperparameter-selection leakage | Tuning uses TRAIN+VALIDATION only, via time-aware cross-validation; the TEST set is touched exactly once, for final evaluation. |
| Portfolio-construction leakage | `optimization.py` is estimation-window-agnostic; the *caller* must pass mean/cov estimated only from data through time `t` when backtesting (documented explicitly in the module docstring and README §64). A current, non-backtested "what should I hold today" analysis using full history is allowed but must be labeled as such in the UI. |
| Use of future prices | All price/feature access happens through `clean_market_prices` / `market_features` filtered by date; no downstream code queries "next row" by index. |
| Future information entering historical decisions | Execution timing convention: signal generated at close of day `t` → trade assumed executed at `t+1` (`config.yaml: backtest.execution_lag_days`). |

## 4. How train/validation/test are separated

Chronological, non-overlapping windows defined in `config/config.yaml`:

```text
TRAIN:      earliest date  -> ml.train_end
VALIDATION: ml.train_end+1 -> ml.validation_end
TEST:       ml.validation_end+1 -> latest date
```

The TEST window is not touched for any modeling decision (feature selection,
hyperparameters, thresholds) until final evaluation.

## 5. How rolling features are constructed

Every rolling feature is a trailing window: `series.rolling(window,
min_periods=window)`, which by pandas semantics only includes the current
row and the `window - 1` rows before it — never rows after it. Where a
feature must reference "yesterday's" value explicitly (rather than today's),
an explicit `.shift(1)` is used and commented.

## 6. How target variables are created

Targets are built as a separate step from features, using data strictly
*after* the feature date (`t+1 … t+horizon`), per the diagram in README §16.
Target construction never writes into `market_features`.

## 7. How scaling is performed

`sklearn.Pipeline` fits the scaler only on `X_train`; `X_val`/`X_test` are
transformed with the already-fitted scaler. See `test_scaler_fit_only_on_training_data`.

## 8. How hyperparameter tuning is performed

Time-aware cross-validation (walk-forward / expanding window) restricted to
TRAIN+VALIDATION. The TEST window is evaluated exactly once, after the final
model and hyperparameters are frozen.

## 9. How walk-forward validation works

Expanding-window: train on `[start, y]`, validate on `[y+1, y+1+step]`, then
grow the training window and repeat, always moving forward in time (README
§18). A purge/embargo gap (`ml.purge_days`, `ml.embargo_days` in config) is
applied between the end of a training fold's features and the start of the
next validation fold, because a target spanning `t+1…t+5` can otherwise
overlap the following fold's training features.

## 10. How backtesting avoids look-ahead bias

At each backtest step: use only data available through `t` → generate a
prediction/decision → apply the risk-management rule → advance to `t+1` and
observe the outcome. No later data is available to the code executing step
`t`. Execution is assumed to happen at `t+1`, not at the same close used to
generate the signal (README §63).

## 11. How regime detection avoids future information

For backtesting purposes, regime models are refit using only information
available at each point in time (an expanding window), matching the
walk-forward pattern used for the risk-prediction model. Any regime model
fit on the *entire* historical dataset for exploratory visualization is
explicitly labeled `is_exploratory = TRUE` in the `market_regimes` table and
documented as unsuitable for causal backtesting (README §32).

## 12. How portfolio optimization avoids using future data

See row "Portfolio-construction leakage" above and `optimization.py`'s
module docstring.

## 13. How the automated tests detect leakage

`tests/test_no_leakage.py` implements, at minimum:

1. Perturbing future prices and asserting past features are unchanged.
2. Confirming scaler parameters differ (and must be chosen) based on
   train-only data, not train+test.
3. Confirming feature dates never exceed the prediction date.
4. Scanning feature column names for forbidden substrings (`future`,
   `target`, `label`).
5. Confirming chronological splits have zero index overlap and are strictly
   time-ordered.
6. Confirming rolling features are trailing, not centered, by perturbing a
   single future observation and checking earlier feature rows don't move.

These tests are run in CI (or manually via `pytest tests/ -v`) before any
model result from this project is reported or shown on the dashboard.
