# FinRisk — AI-Powered Investment Portfolio Risk Analytics, Optimization & Risk Management Platform

> **Disclaimer:** This application is for educational and analytical purposes only.
> Risk estimates and portfolio recommendations are model-based and do not guarantee
> future performance. Nothing in this project constitutes investment advice.

## 1. Project Overview

FinRisk is an end-to-end, production-style project that takes raw historical market
data all the way to a Streamlit dashboard that shows portfolio risk, VaR/CVaR,
Monte Carlo simulation, stress testing, market regimes, an ML-based risk-prediction
model, portfolio optimization, and rebalancing recommendations.

The project is built so that **an MSc student can understand every step**, while
still being technically rigorous enough to show a Quant/Risk/Data Science
interviewer: strict time-aware methodology, no data leakage, reproducible
pipelines, and a normalized MySQL schema.

## 2. Problem Statement

Retail and academic portfolio tools often either (a) oversimplify risk down to a
single volatility number, or (b) silently leak future information into
"predictive" models, producing results that look great in a backtest and fail in
production. FinRisk is built to avoid (b) by construction: every feature, split,
scaler, threshold, and backtest step is explicitly time-ordered.

## 3. Financial / Business Objective

Given a user-selected basket of assets, FinRisk should let a user:

- understand historical and current portfolio risk (volatility, VaR, CVaR, drawdown)
- see how the portfolio would behave under stress scenarios and Monte Carlo paths
- see which market regime we are likely in
- get a machine-learning-based estimate of near-term risk
- get a mathematically optimized target allocation (min-variance / max-Sharpe / target-return)
- get a concrete, cost-aware rebalancing recommendation from current → target weights

## 4. Architecture

```text
Streamlit UI  (app/)
     ↓
Data Service   (src/data/service.py)   -- the ONLY path the dashboard reads through
     ↓
Data Access Layer             (src/data/repository.py)
     ↓
MySQL (finrisk)
```

**Ingestion is a separate, explicit step**, not something the dashboard
triggers implicitly:

```text
python -m src.data.ingestion --tickers ... --start ...   (MarketDataProvider -> raw_market_prices)
```

This separation was not always enforced — see `docs/FORENSIC_AUDIT.md`
(Issue 1) for a real bug where the dashboard called `MarketDataProvider`
(Yahoo Finance) directly, bypassing MySQL entirely. The dashboard now
requires an explicit sidebar choice ("MySQL historical data" vs.
"Synthetic demo data") before loading anything, with no automatic,
silent switch between the two (`docs/FORENSIC_AUDIT.md`, Issue 2).

Raw data is never overwritten:

```text
RAW DATA → VALIDATION → CLEAN DATA → FEATURE DATA → MODEL DATA
raw_market_prices → clean_market_prices → market_features → model input
```

## 5. Data Sources

Market data is fetched behind a `DataProvider` abstraction
(`src/data/ingestion.py`) so the underlying API can be swapped without touching
downstream code:

```text
DataProvider
    ├── MarketDataProvider   (OHLCV, adjusted close)
    ├── MacroDataProvider    (optional macro indicators)
    └── VolatilityDataProvider (optional, e.g. VIX)
```

The default implementation uses `yfinance` for OHLCV data. No credentials are
hard-coded; anything sensitive comes from `.env` (see `.env.example`).

## 6. Database Schema

See `database/schema.sql` for full DDL. Summary of tables and why each exists:

| Table | Purpose |
|---|---|
| `assets` | Master list of tickers/instruments with metadata |
| `raw_market_prices` | Untouched OHLCV as pulled from the provider (audit trail) |
| `clean_market_prices` | Validated/cleaned prices used for all downstream calculation |
| `market_features` | Leakage-safe engineered features per asset/date |
| `portfolio` | A named portfolio definition (base currency, creation date) |
| `portfolio_holdings` | Current/point-in-time weights per asset in a portfolio |
| `portfolio_transactions` | Log of buy/sell/rebalance actions (paper, not executed) |
| `risk_metrics` | Computed risk metrics (VaR, CVaR, vol, Sharpe, etc.) per portfolio/date |
| `market_regimes` | Detected regime label per date |
| `stress_scenarios` | Named hypothetical shock definitions and results |
| `risk_predictions` | ML model output (probability, class) per portfolio/date |
| `model_runs` | Reproducibility record for every trained model |
| `backtest_results` | Walk-forward backtest performance per experiment |
| `rebalance_recommendations` | Current vs target weights and suggested trades |

Raw and clean prices are kept in separate tables so a bug in cleaning can always
be diagnosed against the untouched source data.

## 7. Feature Engineering, Selection & Leakage Prevention

See `docs/DATA_LEAKAGE_AUDIT.md` for the full leakage write-up, and
`docs/PREPROCESSING_AND_FEATURE_SELECTION.md` for the reasoning behind
every preprocessing/feature-selection/transformation decision (imputation
strategy, scaling choice, why returns not price levels, iterative VIF-based
collinearity removal, Yeo-Johnson transformation, data-sufficiency
accounting). In short:

> For a prediction made at time `t`, only information available at or
> before `t` may be used as a feature. Every rolling/lagging feature is
> built with a trailing window so day `t`'s feature never contains day
> `t`'s own future-looking information. Targets look forward
> (`t+1 … t+5`); features never do.

The engineered feature set (`src/features/feature_pipeline.py`) includes 5
return horizons, 4 volatility horizons plus a volatility-regime-shift ratio
(`vol_ratio_20_60`), momentum, RSI, drawdown, a momentum×volatility
interaction term, and optional market beta/correlation — deliberately
broad and overlapping. `src/models/feature_selection.py` then filters this
down (variance filter → iterative VIF-based collinearity removal →
optional model-importance ranking), fit on TRAIN only in every case.

## 8. Risk Methodology

Implemented in `src/risk/`:

- **VaR**: historical, parametric (Gaussian), Monte Carlo (`var.py`, `monte_carlo.py`)
- **CVaR / Expected Shortfall**: conditional mean loss beyond VaR (`cvar.py`)
- **Stress testing**: scenario shocks applied to current holdings (`stress_testing.py`)
- **Risk contribution**: marginal & component contribution to portfolio variance (`risk_contribution.py`)
- **Core metrics**: Sharpe, Sortino, Calmar, max drawdown, beta (`metrics.py`)

Monte Carlo (`monte_carlo.py`) supports three resampling modes: i.i.d.
empirical bootstrap, a Normal-distribution fit, and a **moving-block
bootstrap** (`distribution="block"`) that resamples contiguous chunks of
real historical days rather than independent single days — this preserves
the volatility clustering that notebook 03's own EDA demonstrates is
present in real return data (and that i.i.d. resampling destroys by
construction). `tests/test_monte_carlo.py` empirically proves the block
bootstrap preserves autocorrelation of squared returns that the i.i.d.
mode does not, on the same input data.

All confidence levels, horizons, and the risk-free rate are configuration values
(`config/config.yaml`), never hard-coded inside functions.

## 9. Portfolio Optimization

Implemented in `src/portfolio/optimization.py` using `scipy.optimize`:
minimum-variance, maximum-Sharpe, and target-return portfolios, with long-only
and max-weight constraints.

## 10. ML Methodology

Implemented in `src/models/`; full rationale in
`docs/PREPROCESSING_AND_FEATURE_SELECTION.md` and
`docs/MODEL_SELECTION_AND_EVALUATION.md`:

- **Targets** (`targets.py`): forward-looking loss/volatility event labels, built as a strictly separate step from features so a target column can never accidentally end up in a feature set.
- **Chronological split + purge/embargo** (`preprocessing.py`): TRAIN/VALIDATION/TEST are time-ordered, non-overlapping windows from `config.yaml`, with purge and embargo gaps removed around each boundary — **never a random shuffle** — so a forward-looking target can never straddle two splits.
- **Feature selection** (`feature_selection.py`): variance filter, then collinearity removal — either a fast pairwise-correlation pass or the more rigorous `iterative_vif_removal` (repeatedly drops the highest-VIF feature and recomputes from scratch), used in the notebooks and dashboard. Random Forest importance and mutual information provide a second-opinion ranking. Everything fit on TRAIN only.
- **Transformation**: an optional Yeo-Johnson power transform for skewed features (volatility-based features are right-skewed by construction), applied automatically for Logistic Regression only — tree models are invariant to monotonic transforms, so it's skipped for them.
- **Training** (`train.py`): Logistic Regression, Random Forest, and HistGradientBoostingClassifier, each wrapped in a `Pipeline(preprocessing -> model)` so the scaler/imputer/transform is only ever fit on the active split. Class imbalance is handled uniformly via `class_weight="balanced"`, supported natively by all three estimators. `time_series_cv_tune` uses `TimeSeriesSplit` and must only be called on TRAIN+VALIDATION.
- **Model selection methodology**: train all three baselines → compare on VALIDATION by ROC-AUC → tune only the top 2 → re-compare the tuned pair → select the final winner → evaluate that winner on TEST exactly once (see notebooks 11 and 13).
- **Evaluation** (`evaluate.py`): ROC-AUC/PR-AUC, recall/false-negative-rate on the risk class (not just accuracy, since risk events are rare), and a calibration table.
- **Model registry** (`registry.py`): every saved model bundles the full inference pipeline with an `ExperimentRecord` (hyperparameters, seed, exact date ranges) for reproducibility.
- **Market regimes** (`regime.py`): KMeans over trailing return/volatility features, with two explicit modes — `fit_exploratory_regimes` (full-history, visualization only) and `fit_walk_forward_regimes` (expanding-window, backtest-safe).

## 11. Backtesting

Implemented in `src/backtesting/engine.py`: a walk-forward loop where a
pluggable `target_weight_fn` receives only data through the current date,
trades execute `execution_lag_days` later (default: next day), and
`transaction_cost_bps` is charged on every rebalance's turnover.
`make_periodic_reoptimization_strategy` wraps any `optimization.py` function
into a valid backtest strategy using only trailing in-sample data.

`src/backtesting/risk_managed_strategy.py` connects the ML risk-prediction
pipeline directly into this engine: `make_risk_managed_strategy` retrains a
full classifier **from scratch at every rebalance date, using only data
available at that date** (true walk-forward retraining, not train-once),
and de-risks toward the lowest-volatility asset when predicted loss
probability is HIGH. Notebook 12 runs this alongside an equal-weight
benchmark and a periodically-reoptimized strategy and reports the honest,
run-specific verdict — connecting what were previously two disconnected
pipeline halves (ML model development and backtesting) into one actual
tested hypothesis.

## 12. Results

See `notebooks/` for the full narrative walkthrough (01–14, all pre-executed
with outputs saved). Headline points from a representative run:

- **Leakage tests**: 29/29 automated tests pass (`pytest tests/ -v`), including 6 dedicated leakage checks that actively perturb future data and confirm past outputs don't change.
- **Risk metrics**: historical, parametric, and Monte Carlo VaR/CVaR agree closely on roughly-Normal synthetic data and diverge on fat-tailed real data — see notebook 06.
- **ML honesty check**: on a short/synthetic history, the demo risk-prediction model correctly shows near-0.50 ROC-AUC and can show 0% recall on the rare risk-event class despite >95% accuracy — proof the pipeline isn't inflating results via leakage (notebook 13). Real, longer-history data would need independent validation before any live use.
- **Backtesting**: a periodically-reoptimized minimum-variance strategy is compared against an equal-weight benchmark net of transaction costs, with execution assumed one day after the signal (notebook 12).

## 13. Dashboard

A 9-page Streamlit app in `app/`:

| Page | What it shows |
|---|---|
| Home (Dashboard) | Key metrics, portfolio value over time, allocation pie, correlation heatmap |
| Portfolio Analysis | Cumulative growth vs. individual assets, drawdown, rolling volatility |
| Risk Analytics | Historical/parametric VaR, CVaR, return distribution, per-asset risk contribution |
| Stress Testing | Predefined + custom hypothetical shock scenarios |
| Monte Carlo | Simulated future paths, terminal-value distribution, percentile outcomes |
| Market Regime | KMeans-detected low/medium/high-volatility regimes over portfolio history |
| ML Risk Prediction | Trains a demo model on leakage-safe features; shows held-out test metrics and a current risk classification |
| Portfolio Optimization | Efficient frontier, minimum-variance and maximum-Sharpe portfolios |
| Rebalancing | Current vs. target allocation, BUY/SELL plan with estimated transaction costs |

The sidebar (shared across all pages) lets you set tickers, start date, and
portfolio value; each page also lets you edit the current allocation.

**Data availability:** the app tries to fetch real data via `yfinance`
first. If that fails (no internet access, rate limiting, delisted ticker),
it automatically falls back to reproducible synthetic price series and
shows a visible warning banner — every page still works end to end either
way, but only real data should be used to draw real conclusions.

```bash
streamlit run app/app.py
```

## 14. Installation

```bash
git clone <this-repo>
cd finrisk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your local MySQL credentials
```

## 15. MySQL Setup

```bash
docker compose up -d          # starts local MySQL on localhost:3306
mysql -h 127.0.0.1 -P 3306 -u root -p < database/schema.sql
```

Or point `.env` at an existing local MySQL instance.

This schema has been verified against a real, running MySQL 8.0 server
(not just reviewed as text): all 14 tables load without error, foreign
keys reject orphaned rows, unique constraints block duplicate
`(asset_id, date)` rows, and `ON DELETE CASCADE` correctly removes
dependent `raw_market_prices`/`market_features` rows when an asset is
deleted. See `tests/test_database_integration.py` to re-run these checks
yourself against your own instance.

## 16. Environment Variables

See `.env.example`:

```text
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=finrisk
```

## 17. Running the Pipeline

```bash
python -m src.data.ingestion --tickers AAPL MSFT GLD BND --start 2015-01-01
python -m src.features.feature_pipeline
```

## 18. Running Tests

```bash
pytest tests/ -v
```

`tests/test_no_leakage.py` is the most important file in this repository — it
must pass before any model result is trusted. `test_targets_and_splits.py`,
`test_backtesting.py`, `test_risk_managed_strategy.py`, and
`test_monte_carlo.py` extend the same leakage/correctness guarantees to the
ML target construction, chronological splitting, walk-forward backtest
engine, the ML-driven risk-managed strategy, and the Monte Carlo block
bootstrap.

`tests/test_database_integration.py` is different from every other test
file here: it runs against a **real, running MySQL database** (loading the
actual `database/schema.sql`, not a mock), and verifies foreign key
enforcement, unique constraints, CASCADE deletes, and the NaN→SQL NULL
conversion pymysql requires but pandas doesn't. It auto-skips (not fails)
if no MySQL instance is reachable, matching standard practice for
infrastructure-dependent integration tests — see that file's docstring for
local setup instructions. 98 tests total, all passing as of this commit
(84 unit/leakage tests that need no infrastructure + 14 live-database
integration tests, which run against an isolated `finrisk_test` database
auto-provisioned by the test suite itself — never the application's real
database; see `docs/SECOND_FORENSIC_AUDIT.md` Issue C3).

## 19. Running Streamlit

```bash
streamlit run app/app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).
The sidebar first asks you to choose a **data source**:
- **MySQL historical data** — requires you to have already run the
  ingestion pipeline (§17) for the tickers you want to analyze. If MySQL
  has no data for a requested ticker, the app shows a clear error and
  stops — it will not silently substitute synthetic data.
- **Synthetic demo data** — generates reproducible fake price paths for
  demo purposes, with no MySQL setup required. Chosen explicitly, never
  automatic.

Use the rest of the sidebar to set tickers/dates, and the page-specific
controls to set allocations, confidence levels, simulation counts, etc.

## 19.5. Troubleshooting

**If you're on an older clone of this repo and see
`xgboost._c_api.XGBoostError: Library (libxgboost.dylib) could not be
loaded`** (macOS): that error came from an earlier revision of this
project that used XGBoost, whose compiled native library needs the OpenMP
runtime that macOS doesn't ship by default. This project no longer depends
on XGBoost — the gradient-boosting model is now
`sklearn.ensemble.HistGradientBoostingClassifier`, which ships inside
scikit-learn itself with no separate native library to install or fail to
load. Pull the latest version of this repo and the error should disappear
entirely; there is nothing to install to fix it. See
`docs/MODEL_SELECTION_AND_EVALUATION.md` §1.3 for the full rationale
behind this choice.

## 20. Limitations

- Survivorship bias: if the data provider only returns currently-listed tickers,
  historical constituents that were delisted are missing. This is **not**
  corrected in Phase 1 and is documented, not hidden.
- Adjusted-close corporate-action handling depends on the upstream data
  provider's accuracy.
- VaR/CVaR are model-based estimates, not guarantees of maximum loss.

## 21. Future Improvements

- Point-in-time index constituent data to remove survivorship bias.
- Shrinkage/EWMA covariance estimators as an alternative to sample covariance.
- SHAP-based interpretability for the risk-prediction model (Phase 2).

## 21.5. Forensic Audit

This project has undergone two deliberate forensic audit passes.
`docs/FORENSIC_AUDIT.md` (first pass) documents real bugs found and fixed,
including a critical backtest timing bug, a dashboard architecture bug
that bypassed MySQL entirely, and a non-reproducible synthetic-data seed.
`docs/SECOND_FORENSIC_AUDIT.md` (second pass) found — among other things —
that the first pass's dashboard fix was never extended to the notebooks
(which still call Yahoo Finance directly), a real mathematical error in
parametric VaR's horizon scaling, a stale prediction bug in the ML
risk-managed strategy, and a test suite that was silently deleting real
data from the developer's own MySQL database. The 4 CRITICAL issues and
9 of 12 HIGH issues from the second pass are fixed and verified (see
`docs/SECOND_AUDIT_SUMMARY.md`); the remaining 3 HIGH, 6 MEDIUM, and 6 LOW/
enhancement items from that pass remain open and documented, not yet
addressed — stated plainly rather than implied to be resolved.

## 22. Disclaimer

This application is for educational and analytical purposes only. Risk estimates
and portfolio recommendations are model-based and do not guarantee future
performance. This is not investment advice.
