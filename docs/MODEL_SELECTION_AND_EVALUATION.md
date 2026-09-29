# Model Selection and Evaluation Metrics — Rationale

## 1. Why these three models, in this order

`src/models/train.py` implements Logistic Regression, Random Forest, and
HistGradientBoostingClassifier — deliberately in increasing order of
complexity, so each model has to justify its added cost against the
simpler one before it.

### 1.1 Logistic Regression — the baseline

- **Why include it at all**: it's the simplest model that can output a
  calibrated probability for a binary target. If a Random Forest or
  gradient-boosted model can't beat it by a meaningful margin on held-out
  data, the added
  complexity (harder to interpret, more hyperparameters, slower to
  retrain) isn't earning its keep — this is the standard "beat the
  baseline" discipline, not a formality.
- **Why it's genuinely useful beyond being a baseline**: every coefficient
  is a signed, additive effect on the log-odds of a risk event. A
  stakeholder can be told "higher 20-day volatility increases estimated
  risk, holding other features fixed" in a way that's much harder to say
  responsibly about a Random Forest.
- **Its known limitation**: it assumes a linear (additive, in log-odds
  space) relationship between features and outcome. Financial risk is
  often non-linear (e.g. risk might jump sharply only once volatility
  crosses a threshold, not increase smoothly) — this is exactly what the
  next two models are included to capture.
- **`class_weight="balanced"`**: risk events are rare (see
  `src/models/targets.py`); without this, the model would learn to predict
  "no risk" for everyone and still score deceptively well on raw accuracy.

### 1.2 Random Forest — non-linear effects and interactions, no scaling required

- **Why it's the second model, not the first**: it captures non-linear
  relationships and feature interactions (e.g. "high volatility AND
  negative momentum" mattering more than either alone) that Logistic
  Regression structurally cannot represent.
- **Why it's useful beyond raw accuracy**: it needs no feature scaling,
  is relatively robust to outliers and irrelevant features, and its
  impurity-based feature importances are the mechanism
  `src/models/feature_selection.py::select_by_model_importance` uses to
  rank which engineered features actually matter — a Random Forest is used
  for this (rather than the final production model) specifically because
  it's fast to fit and doesn't require the preprocessing pipeline that
  Logistic Regression needs.
- **Its known limitation**: many correlated, deep trees can overfit
  training-period quirks that don't generalize; `max_depth=6` and
  `n_estimators=300` are deliberately conservative rather than tuned for
  maximum training-set accuracy, to reduce this risk in a genuinely
  small-sample setting.

### 1.3 HistGradientBoostingClassifier — the strongest tabular-data performer, with no external native-library risk

- **Why it's included as the "advanced" model**: gradient-boosted trees
  are consistently among the strongest performers on structured/tabular
  data in practice (as opposed to unstructured data like images or text,
  where deep learning dominates) — this project's features are exactly
  that: a table of engineered numeric ratios and rolling statistics.
- **What it adds over Random Forest**: boosting fits each new tree to the
  *residual error* of the ensemble so far, rather than averaging
  independent trees — this typically produces sharper, better-calibrated
  probability estimates, which matters directly for the LOW/MEDIUM/HIGH
  threshold-based classification in `select_risk_thresholds`.
- **Why `HistGradientBoostingClassifier` and not XGBoost**: this project
  originally used XGBoost here, and that choice was reverted after hitting
  a real, concrete failure: `XGBoostError: Library (libxgboost.dylib)
  could not be loaded` on macOS, because XGBoost is a separately-compiled
  C++ library with its own binary wheel per platform, and it needs the
  OpenMP runtime (`libomp`) that macOS doesn't ship by default. That
  failure crashed importing `src/models/train.py` entirely — breaking
  Logistic Regression and Random Forest too, even though neither one has
  anything to do with gradient boosting. `HistGradientBoostingClassifier`
  implements the same algorithmic family (histogram-based gradient
  boosting over binned feature values, the same core idea as
  XGBoost/LightGBM) but ships inside scikit-learn itself, so there is no
  separate native library to install, version-match, or fail to load on
  any platform. Removing an entire class of environment-dependent failure
  is worth more here than the small, inconsistent accuracy differences
  typically seen between the two implementations at this data scale.
- **Class-imbalance handling**: unlike XGBoost (which has no
  `class_weight` parameter and needs a manually-computed
  `scale_pos_weight` substitute), `HistGradientBoostingClassifier`
  supports `class_weight="balanced"` directly (added in scikit-learn 1.4)
  — the exact same mechanism used for Logistic Regression and Random
  Forest, so no model-specific special-casing is needed anywhere in
  `train.py`.
- **Its known limitation**: more hyperparameters to tune than Random
  Forest, more prone to overfitting a small dataset if left untuned, and
  its probability outputs need the calibration check in
  `evaluate.py::calibration_table` more than Logistic Regression's do —
  the same limitations gradient boosting has regardless of implementation.

### 1.4 What was deliberately NOT included, and why

- **Deep learning (LSTM/Transformer)**: this project's history length
  (thousands of rows, not millions) and feature count (~15) are far below
  what deep learning needs to outperform gradient boosting on tabular
  data — adding it would trade interpretability and simplicity for
  complexity with no expected accuracy benefit at this data scale.
- **SVM**: doesn't naturally output calibrated probabilities (needs Platt
  scaling as an extra step), doesn't scale well past a few thousand rows,
  and offers no clear advantage over the three models already chosen for
  this feature set.
- **XGBoost / LightGBM**: see §1.3 above — deliberately avoided in favor
  of scikit-learn's built-in equivalent specifically to eliminate external
  native-library dependency risk, after that risk caused a real failure
  during this project's development.

## 1.5 Model selection methodology: baseline → top 2 → tune → re-compare → select

Rather than tuning all three models (wasteful) or picking one model
up-front by intuition (unprincipled), notebooks 11 and 13 follow a
deliberate four-step process:

1. **Train all three candidates with default hyperparameters** and compare
   them on the VALIDATION split (never TEST) by ROC-AUC — the
   threshold-independent ranking metric (see §2 below for why).
2. **Select the top 2 performers** by validation ROC-AUC. Tuning a
   clearly-third-place model wastes compute for a result unlikely to
   change the final choice.
3. **Hyperparameter-tune only those two**, using `time_series_cv_tune`
   (`TimeSeriesSplit`-based, restricted to TRAIN+VALIDATION — the TEST
   split is still untouched at this point).
4. **Re-compare the tuned top 2 on VALIDATION one more time** and select
   the final winner. Only the winner is then evaluated on TEST, exactly
   once, in notebook 13's final section — preserving TEST as a genuine
   held-out check rather than something used to choose between candidates.

On the shipped demo run, this process is fully visible in the notebook
output: baseline validation ROC-AUC of {hist_gradient_boosting: 0.631,
random_forest: ~0.6, logistic_regression: lower} selects
HistGradientBoostingClassifier and Random Forest as the top 2; after
tuning, HistGradientBoostingClassifier remains ahead (ROC-AUC 0.656) and
is selected as the final model — then notebook 13 reproduces this exact
process and reports its one-time TEST evaluation. The specific winner will
vary run to run with different data/seeds; what's fixed is the *process*
by which it's chosen.

## 2. Why these evaluation metrics, and why accuracy alone is actively misleading here

`src/models/evaluate.py::classification_report` deliberately reports more
than accuracy, because **risk events are a rare positive class by
construction** (see `src/models/targets.py`) — a model that always predicts
"no risk" can score >95% accuracy while catching zero real risk events,
which is precisely the failure mode this project exists to avoid, not a
hypothetical edge case (notebook 13 shows this exact number on real output:
96.4% accuracy, 0% recall, 0 true positives).

| Metric | What it actually measures | Why it's included |
|---|---|---|
| **Accuracy** | Fraction of all predictions correct | Reported for completeness/familiarity, but explicitly *not* trusted alone — with a ~2-5% positive rate, a trivial always-negative model scores >95% while being useless. |
| **Precision** | Of predicted risk events, how many were real | Answers "if the dashboard flags HIGH risk, how often is it right" — controls false-alarm fatigue. |
| **Recall** (risk-event recall) | Of real risk events, how many were caught | The single most important metric for this application: missing a real risk event (a false negative) is the costly failure mode — a missed warning before a real loss — while a false alarm merely causes unnecessary caution. |
| **False negative rate** | Of real risk events, how many were MISSED | The direct complement of recall, surfaced separately because "recall" as a word is sometimes skimmed past — a 95% false-negative-rate is a much harder number to miss in a report. |
| **ROC-AUC** | Threshold-independent ranking quality (probability a random positive is ranked above a random negative) | Doesn't depend on picking a specific probability cutoff, which matters because the LOW/MEDIUM/HIGH thresholds themselves are chosen separately (`select_risk_thresholds`, on TRAIN+VALIDATION only) — ROC-AUC evaluates the model before that later, somewhat arbitrary choice is even made. |
| **PR-AUC** (average precision) | Like ROC-AUC, but focused on precision/recall trade-off | More informative than ROC-AUC specifically under class imbalance, since ROC-AUC can look deceptively good even when precision on the rare positive class is poor (a well-known property of ROC curves with skewed classes). |
| **Calibration table** | Whether a predicted 30% probability actually corresponds to a ~30% observed event rate | Matters because the dashboard shows a raw probability number to the user ("Estimated probability of a >3% loss: 12.4%") — an uncalibrated model could show confident-looking numbers that don't mean what they claim to mean. |

### Why thresholds (LOW/MEDIUM/HIGH) are chosen on TRAIN+VALIDATION only

`select_risk_thresholds` explicitly requires TRAIN+VALIDATION predictions,
never TEST — choosing a cutoff using the same data it's later evaluated
against would let the evaluation period's own outcomes leak into a
decision that's supposed to be made in advance (the same "hyperparameter
/threshold selection leakage" pattern documented in
`docs/DATA_LEAKAGE_AUDIT.md`).

### Honest interpretation of this project's own demo results

On the short/synthetic history used in the shipped demo, the model's
ROC-AUC sits close to 0.50 (no better than random ranking) and recall on
the rare risk-event class can be 0%. **This is the correct, expected
outcome for random-walk synthetic data with no real signal to find** — see
`notebooks/13_model_evaluation_and_results.ipynb` for the full discussion.
A leakage-free pipeline that finds no signal in random data is a *sign the
pipeline is trustworthy*, not a failure; a pipeline that finds strong
"signal" in random data is the actual red flag, and would indicate a
leakage bug elsewhere in the pipeline.
