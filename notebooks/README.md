# Notebooks

Run in order — each builds on context from the previous one. All notebooks
use `notebooks/nb_utils.fetch_price_panel`, which tries real Yahoo Finance
data first and automatically falls back to reproducible synthetic data if
the network is unavailable (you'll see an `[info] Live data unavailable...`
message when that happens — every notebook still runs to completion either
way).

| # | Notebook | Covers |
|---|---|---|
| 01 | `01_data_collection_and_storage.ipynb` | `DataProvider` abstraction, raw data summary |
| 02 | `02_data_cleaning_and_validation.ipynb` | Validation checks, cleaning decisions, injected data errors |
| 03 | `03_exploratory_data_analysis.ipynb` | Return/volatility/correlation characteristics |
| 04 | `04_feature_engineering.ipynb` | The full leakage-safe feature set |
| 05 | `05_leakage_audit_demonstration.ipynb` | Live proof of every leakage-prevention guarantee |
| 06 | `06_risk_metrics_var_cvar.ipynb` | Historical/parametric VaR, CVaR, comparison |
| 07 | `07_monte_carlo_simulation.ipynb` | Empirical bootstrap vs. Normal Monte Carlo |
| 08 | `08_stress_testing.ipynb` | Predefined + custom shock scenarios |
| 09 | `09_market_regime_detection.ipynb` | Exploratory vs. walk-forward regime detection |
| 10 | `10_portfolio_optimization.ipynb` | Efficient frontier, min-variance, max-Sharpe |
| 11 | `11_ml_model_development.ipynb` | Target/feature construction, chronological split, model training |
| 12 | `12_walk_forward_backtesting.ipynb` | Equal-weight vs. reoptimized vs. ML risk-managed (walk-forward retrained) strategy, realistic costs |
| 13 | `13_model_evaluation_and_results.ipynb` | Held-out test metrics, calibration, honest discussion of results |
| 14 | `14_final_summary_and_conclusions.ipynb` | Project recap, limitations, future improvements |

## Running

```bash
pip install -r ../requirements.txt jupyter nbconvert
jupyter notebook   # or: jupyter lab
```

To re-run and re-save all notebooks headlessly:

```bash
for nb in *.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

## A note on notebook 11/13 results

With a short history and/or synthetic demo data, the ML model's test-set
ROC-AUC sits close to 0.50 and recall on the (rare) risk-event class can be
0% even with high overall accuracy. **This is the correct, expected
outcome** — there is no real predictive signal in random-walk synthetic
prices, and accuracy alone is a misleading metric when the positive class
is rare. This is exactly why `classification_report` always reports
recall/false-negative-rate/ROC-AUC alongside accuracy (see
`src/models/evaluate.py`), and why the leakage tests (notebook 05) matter
more than any single accuracy number: a leaky pipeline can produce an
impressive-looking score on ANY data, which is a red flag, not reassurance.
