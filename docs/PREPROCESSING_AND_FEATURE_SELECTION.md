# Preprocessing, Feature Selection, and Data Sufficiency — Rationale

This document explains **why** each preprocessing and feature-selection
decision was made, not just what the code does. See
`docs/DATA_LEAKAGE_AUDIT.md` for the leakage-prevention rules that constrain
every decision below.

## 1. Why EDA comes before anything else

`notebooks/03_exploratory_data_analysis.ipynb` exists to answer four
questions *before* a single feature or model is built, because the answers
determine which modeling choices are even valid:

| Question | Why it matters | Where it's answered |
|---|---|---|
| Are returns fat-tailed (excess kurtosis)? | If yes, parametric (Gaussian) VaR will understate tail risk — historical/Monte Carlo VaR become the more trustworthy methods (`src/risk/var.py`, `monte_carlo.py`). | Notebook 03, skew/kurtosis table |
| Does volatility cluster over time? | If yes, a single full-sample volatility number is misleading — trailing/rolling volatility features are necessary, not optional (`src/features/volatility.py`). | Notebook 03, rolling-vol chart |
| Are returns stationary? | Most classical statistical assumptions (constant mean/variance) require stationarity; price *levels* are almost never stationary but *returns* usually are. This justifies modeling returns, not raw prices. | Notebook 03, Augmented Dickey-Fuller test |
| How much history is actually available, and is it enough? | Determines whether a 60-day or 252-day rolling feature can even be computed, and whether there's enough history to form a TRAIN/VALIDATION/TEST split with purge/embargo gaps and still have a usable amount of data in each. | Notebook 03, data sufficiency section; `config.yaml: data.min_history_days` |

### Stationarity testing (Augmented Dickey-Fuller)

Notebook 03 runs an ADF test (`statsmodels.tsa.stattools.adfuller`) on both
price levels and daily returns for each asset. The expected (and observed)
result: price levels fail to reject the unit-root null hypothesis (i.e.
they behave like a random walk, non-stationary), while returns reject it
comfortably. This is the formal justification for why every feature and
target in this project is built from **returns**, not raw price levels —
using non-stationary price levels directly as model inputs would let a
model "cheat" by learning the general upward price trend of the training
period, which will not generalize to unseen data with a different trend.

## 2. Preprocessing decisions and why

### 2.1 Missing value imputation: median, not mean or a placeholder

`src/models/preprocessing.py::build_preprocessing_pipeline` uses
`SimpleImputer(strategy="median")`. Reasoning:

- Financial return/volatility features are right-skewed and can contain
  outliers (a single crash day skews vol_60d for weeks) — the **median**
  is robust to those outliers, while the **mean** would be pulled toward
  them.
- A constant placeholder (e.g. 0) would be interpretable as "no return" or
  "no volatility", which is a real, misleading value, not a missingness
  marker in this feature space.
- The imputer is fit on TRAIN only and applied (`.transform`) unchanged to
  validation/test, per the leakage rule in `docs/DATA_LEAKAGE_AUDIT.md`.

### 2.2 Scaling: StandardScaler, and only after imputation

`Pipeline(imputer -> StandardScaler)`. Reasoning:

- Logistic Regression's coefficients and regularization penalty are scale
  -sensitive — an unscaled feature with a larger numeric range would
  dominate the L2 penalty for no principled reason. StandardScaler
  (zero mean, unit variance) removes this artifact.
- Tree-based models (Random Forest, HistGradientBoostingClassifier) are scale-invariant and don't
  strictly need this step, but running the same pipeline for every model
  keeps the comparison between models fair (identical inputs) and the code
  simpler (one pipeline definition, not one per model family).
- Scaling is always fit on TRAIN only (`test_scaler_fit_only_on_training_data`
  in `tests/test_no_leakage.py` is the automated guardrail for this).

### 2.3 Why returns-based features, not price-based features

Beyond the stationarity argument above: a price-based feature like "price
= 142.30" carries no information a model can generalize from (that exact
price will never recur, and its meaning changes completely across a stock
split or a different asset). A *return* ("price is 3% above its 20-day
average") is scale-free and comparable across assets, time periods, and
even across a stock split — which is exactly why every feature in
`src/features/feature_pipeline.py` is expressed as a return, a ratio, or a
volatility, never as a raw price level.

### 2.4 Class imbalance handling (preprocessing-adjacent, but really a
training-time decision)

Risk events (the ML target's positive class — see `src/models/targets.py`)
are rare by construction: a >3% loss over the next 5 days doesn't happen on
most days. Left uncorrected, all three models would learn to always predict
"no risk" and still score >95% accuracy while catching zero real events —
exactly the failure mode notebook 13 demonstrates. This is corrected
explicitly at training time (`src/models/train.py`) via
`class_weight="balanced"`, supported natively by all three estimators used
in this project (`LogisticRegression`, `RandomForestClassifier`, and
`HistGradientBoostingClassifier`) — sklearn reweights each class inversely
proportional to its frequency, computed fresh from whatever TRAIN fold is
passed in, so the correction automatically adapts if the class balance
shifts across a different asset universe, date range, or CV fold. (An
earlier revision of this project used XGBoost for the third model, which
has no `class_weight` parameter and needed a manually-computed
`scale_pos_weight` substitute instead — switching to
`HistGradientBoostingClassifier`, which gained native `class_weight`
support in scikit-learn 1.4, removed that special-casing entirely; see
`docs/MODEL_SELECTION_AND_EVALUATION.md` §1.3 for the full story, which
also includes a real native-library failure that motivated the switch.)

## 3. Feature selection: what, why, and in what order

`src/models/feature_selection.py` was a genuine gap addressed in this pass
— the original feature pipeline intentionally builds a broad, overlapping
feature set (5 return horizons, 4 volatility horizons, momentum, RSI,
drawdown, beta, correlation, volume z-score) so nothing useful is discarded
too early, but that same breadth creates real problems if fed to a model
unfiltered:

| Problem | Selection step that addresses it |
|---|---|
| Near-constant features carry no information and can destabilize some solvers | `drop_low_variance_features` (variance threshold) |
| Highly correlated features (e.g. `vol_20d` vs `vol_60d` in a calm period) inflate coefficient variance in linear models and split importance credit in tree models | `drop_correlated_features` (single-pass pairwise correlation filter, threshold 0.90) — a fast, explainable first cut |
| Multi-feature redundancy that a single pairwise correlation check can miss (a feature reconstructible as a *combination* of several others, without any single pairwise correlation exceeding the threshold) | `iterative_vif_removal` — repeatedly drops the single highest-VIF feature and **recomputes VIF from scratch** after each removal, since multicollinearity is a property of the whole feature set, not a fixed number computed once; this is the mechanism actually used in notebooks 11/13 and the ML dashboard page, in place of the single-pass correlation filter |
| Diagnosing *how bad* multicollinearity is, beyond a single correlation number | `variance_inflation_factors` (VIF; the formal diagnostic that `iterative_vif_removal` is built around — VIF > 10 is the threshold used throughout this project) |
| Ranking which of the surviving features are actually predictive, non-linearly | `select_by_model_importance` (Random Forest impurity importance) and `mutual_information_ranking` (captures non-linear relationships a Pearson correlation would miss) as a cross-check |

`select_features()` chains the simpler pipeline (variance filter → pairwise
correlation filter → optional top-N by importance) for quick, exploratory
use; `iterative_vif_removal` is the more rigorous alternative used in the
notebooks and dashboard, since VIF-based removal is worth its extra compute
cost on a feature set this important. On the shipped demo data, this
concretely catches `momentum_20d` at a VIF of ~52 (a clear multicollinearity
red flag against the return-horizon features) and removes it — see
notebook 11, section 3, for the live before/after VIF table. Every step is
fit on `X_train`/`y_train` only; the resulting column list is then applied
unchanged to validation and test (`X_val[selected_cols]`,
`X_test[selected_cols]`) — re-running selection on validation/test would be
"feature-selection leakage" (see `docs/DATA_LEAKAGE_AUDIT.md`'s leakage
table), since it would let the evaluation period's own statistics influence
which features the model is allowed to see.

`tests/test_feature_selection.py` includes dedicated leakage and
correctness tests, including a regression test for a real bug caught during
development: an early version of `iterative_vif_removal` treated an
*infinite* VIF (perfect collinearity) as "good enough to stop" instead of
"definitely needs removing" — `test_iterative_vif_removal_handles_perfect_collinearity`
guards against that exact mistake recurring.

### 3.1 Distribution transformation (Yeo-Johnson), and why it's conditional

Volatility-based features (`vol_5d/10d/20d/60d`, and `momentum_vol_interaction`
which inherits `vol_20d`'s skew) are right-skewed by construction — a
standard deviation can't be negative, so the distribution has a hard floor
at zero and a long right tail. `src/models/preprocessing.py::build_preprocessing_pipeline`
supports an optional `PowerTransformer(method="yeo-johnson")` step, inserted
between imputation and scaling:

- **Why Yeo-Johnson specifically, not Box-Cox**: Box-Cox (the classical
  power transform) requires strictly positive input. Several features here
  (returns, momentum, `rolling_drawdown`) can be zero or negative, which
  rules it out. Yeo-Johnson is the standard extension that supports the
  full real line while pursuing the same goal (making a skewed distribution
  closer to Normal).
- **Why it's conditional, not applied to every model**: tree-based models
  (Random Forest, HistGradientBoostingClassifier) choose split points based on the *ordering* of
  values, which a monotonic transform like Yeo-Johnson never changes — so
  it adds computation for zero benefit there. It's reserved for Logistic
  Regression (`train_model`'s `use_power_transform` parameter defaults to
  `True` only for that model — see `MODELS_THAT_BENEFIT_FROM_POWER_TRANSFORM`
  in `src/models/train.py`), where an additive, own-scale fit is genuinely
  distorted by heavy skew.
- **Leakage rule applies here too**: like the imputer and scaler, the
  `PowerTransformer` is fit on `X_train` only inside the same
  `Pipeline(preprocessing -> model)` object, then applied via `.transform()`
  to validation/test.

## 4. Data sufficiency: how much history is actually needed?

Before training anything, the pipeline needs enough history to satisfy
three independent constraints simultaneously:

1. **Feature warm-up**: the longest rolling window used is `rolling_drawdown`
   at 252 trading days (`FEATURE_SPEC` in `feature_pipeline.py`) — the
   first 252 rows of any asset's history cannot produce a valid drawdown
   feature and must be dropped, not imputed with a placeholder that would
   misrepresent "insufficient history" as "zero drawdown".
2. **Target horizon**: `build_loss_event_target` looks `target_horizon_days`
   (default 5) into the future, so the *last* 5 rows of history can never
   have a valid target and must also be dropped (see
   `src/models/targets.py`'s docstring).
3. **Split + purge/embargo overhead**: `chronological_split` removes
   `purge_days` from the end of an earlier split and `embargo_days` from
   the start of the next one, at *each* of the two split boundaries (see
   `docs/DATA_LEAKAGE_AUDIT.md` §9) — this overhead comes out of usable
   data, not on top of it.

`config.yaml: data.min_history_days = 252` documents the minimum in code;
notebooks 03 and 11 both print the actual row counts surviving each stage
(raw rows → after feature warm-up → after target construction → after
purge/embargo split) so a genuinely too-short history is visible as a
small/empty split rather than a silent, misleading result. A rare-event
target additionally needs enough *positive* examples in each split to
compute a meaningful ROC-AUC/recall at all — notebook 13 prints the class
balance per split for exactly this reason, and both the ML dashboard page
and the evaluation notebook fail loudly (`st.warning` / a printed message)
rather than silently reporting a meaningless metric when a split has too
few or only one class present.
