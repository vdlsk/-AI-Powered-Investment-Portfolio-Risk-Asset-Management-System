"""
Model training for the ML risk-prediction task.

WHY THESE THREE MODELS (full rationale in docs/MODEL_SELECTION_AND_EVALUATION.md):

  * Logistic Regression -- the baseline. Linear, fast, and fully
    interpretable (a coefficient is a signed, additive effect on the
    log-odds of a risk event). If a more complex model can't beat this,
    the added complexity isn't earning its keep.
  * Random Forest -- captures non-linear effects and feature interactions
    (e.g. "high volatility AND negative momentum" mattering more than
    either alone) without requiring feature scaling, and gives a
    model-based feature-importance ranking used by
    `src/models/feature_selection.py`.
  * HistGradientBoostingClassifier -- scikit-learn's native histogram-based
    gradient boosting implementation (the same algorithmic family as
    XGBoost/LightGBM: boosted trees over binned feature values). Chosen
    over XGBoost deliberately: it ships as part of scikit-learn itself,
    so it has no separate compiled native library to install or fail to
    load -- unlike XGBoost, which is a distinct C++ library with its own
    binary wheel per platform and a real-world failure mode this project
    hit directly (`XGBoostError: Library (libxgboost.dylib) could not be
    loaded` on macOS without the OpenMP runtime installed). Removing that
    entire class of environment-dependent failure is worth more here than
    the small, inconsistent accuracy differences typically seen between
    the two implementations on tabular data this size. It also supports
    `class_weight="balanced"` directly (see CLASS IMBALANCE below),
    which is simpler than XGBoost's `scale_pos_weight` workaround.

CLASS IMBALANCE: risk events are rare by construction (loss events are a
small minority class -- see `src/models/targets.py`). All three models
correct for this explicitly rather than silently optimizing for the
majority class, via `class_weight="balanced"` (sklearn reweights each
class inversely proportional to its frequency) -- supported natively by
all three estimators used here, so no per-model special-casing is needed.

Hyperparameter tuning uses TimeSeriesSplit cross-validation restricted to
TRAIN+VALIDATION only -- the TEST set from preprocessing.chronological_split
must never be passed into this module's tuning functions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.config import CONFIG
from src.models.preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)

RANDOM_SEED = CONFIG["general"]["random_seed"]


def _build_model_registry(y_train: pd.Series | None = None) -> dict:
    """Build the model registry.

    `y_train` is accepted (and unused) for backward-compatible call-site
    parity with earlier revisions of this function -- it existed
    previously to compute XGBoost's `scale_pos_weight` fresh per fold,
    which is no longer needed since every model here supports
    `class_weight="balanced"` directly. Kept as a parameter rather than
    removed so `train_model` and `time_series_cv_tune` don't need to
    change their call sites.
    """
    return {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED, class_weight="balanced"
        ),
    }


# Kept for backward-compatible introspection (e.g. "what models exist").
MODEL_REGISTRY = _build_model_registry()


@dataclass
class TrainedModel:
    model_name: str
    pipeline: Pipeline
    feature_names: list[str]
    best_params: dict[str, Any] = field(default_factory=dict)


# Models that are invariant to monotonic per-feature transforms (all
# tree-based models here) get no benefit from a power transform -- it's
# reserved for models that fit an additive effect in the feature's own
# scale, where skew genuinely distorts the fit.
MODELS_THAT_BENEFIT_FROM_POWER_TRANSFORM = {"logistic_regression"}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    numeric_features: list[str] | None = None,
    use_power_transform: bool | None = None,
) -> TrainedModel:
    """Fit a single named model inside a Pipeline(preprocessing -> model).

    The preprocessing step is fit ONLY on X_train inside this call (via
    Pipeline.fit), which is what makes this the leakage-safe way to train --
    never call `preprocessing.fit_transform` separately on a combined
    train+val+test frame before this function.

    `use_power_transform`: if left as `None` (the default), applies a
    Yeo-Johnson power transform automatically for models that actually
    benefit from it (currently just Logistic Regression -- see
    `build_preprocessing_pipeline`'s docstring for the full rationale) and
    skips it for scale-invariant tree models. Pass `True`/`False` explicitly
    to override this default.
    """
    registry = _build_model_registry(y_train)
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(registry)}")

    if use_power_transform is None:
        use_power_transform = model_name in MODELS_THAT_BENEFIT_FROM_POWER_TRANSFORM

    numeric_features = numeric_features or list(X_train.columns)
    preprocessing = build_preprocessing_pipeline(numeric_features, use_power_transform=use_power_transform)
    estimator = registry[model_name]()

    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", estimator)])
    pipeline.fit(X_train, y_train)

    return TrainedModel(model_name=model_name, pipeline=pipeline, feature_names=numeric_features)


def time_series_cv_tune(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    param_grid: dict[str, list[Any]],
    n_splits: int = 5,
    scoring: str = "roc_auc",
) -> tuple[dict[str, Any], float]:
    """Time-aware hyperparameter tuning using sklearn's TimeSeriesSplit.

    IMPORTANT: `X`, `y` passed here must already be restricted to
    TRAIN+VALIDATION data (see README §23 -- "never tune on the test set").
    TimeSeriesSplit itself guarantees every CV fold's validation indices
    come strictly after its training indices, preserving chronological
    order within the tuning process too.

    With a rare positive class (see docs/MODEL_SELECTION_AND_EVALUATION.md
    §2), an early CV fold can easily end up with a validation slice that
    contains zero positive examples, making ROC-AUC undefined for that
    fold -- this is EXPECTED with small/imbalanced data, not a bug. Those
    folds are skipped (not counted as 0 or penalized), and a fold count is
    returned alongside the result so the caller can see how many folds
    actually contributed. If NO fold ever produces a usable score for a
    given parameter combination, this raises rather than silently
    returning an empty `{}`/`-inf` result that looks like tuning "worked"
    when it didn't.
    """
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import roc_auc_score

    numeric_features = list(X.columns)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_score = -np.inf
    best_params: dict[str, Any] = {}
    best_n_folds_used = 0
    any_param_had_valid_fold = False

    for params in ParameterGrid(param_grid):
        fold_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_tr.nunique() < 2 or y_val.nunique() < 2:
                # Expected with rare-event targets on early/small folds --
                # skip this fold rather than fabricate a score for it.
                continue

            preprocessing = build_preprocessing_pipeline(numeric_features)
            fold_registry = _build_model_registry(y_tr)
            estimator = fold_registry[model_name]()
            estimator.set_params(**params)
            pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", estimator)])
            pipeline.fit(X_tr, y_tr)

            if hasattr(pipeline, "predict_proba"):
                y_score = pipeline.predict_proba(X_val)[:, 1]
            else:
                y_score = pipeline.decision_function(X_val)

            try:
                fold_scores.append(roc_auc_score(y_val, y_score))
            except ValueError:
                continue  # a fold with a single class can't compute ROC-AUC

        if fold_scores:
            any_param_had_valid_fold = True
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_n_folds_used = len(fold_scores)

    if not any_param_had_valid_fold:
        raise ValueError(
            f"time_series_cv_tune could not compute {scoring} for ANY "
            f"parameter combination across {n_splits} folds -- every fold's "
            f"train or validation slice had only one class present. This "
            f"typically means the positive class is too rare for this "
            f"amount of data/n_splits. Options: reduce n_splits, use a "
            f"larger TRAIN+VALIDATION window, lower ml.loss_threshold in "
            f"config.yaml to make the positive class less rare, or skip "
            f"tuning and use the untuned default hyperparameters."
        )

    logger.info(
        "Best params for %s: %s (CV %s=%.4f, from %d usable fold(s))",
        model_name, best_params, scoring, best_score, best_n_folds_used,
    )
    return best_params, best_score


def select_risk_thresholds(
    predicted_probabilities: pd.Series,
    actual_outcomes: pd.Series,
    medium_level: float | None = None,
    high_level: float | None = None,
) -> dict[str, float]:
    """Convert predicted probabilities into LOW/MEDIUM/HIGH cut points,
    tied to an ABSOLUTE, outcome-calibrated criterion.

    WHY THIS CHANGED (see docs/SECOND_FORENSIC_AUDIT.md, Issue H4): the
    original version picked cut points as quantiles of the model's OWN
    predicted-probability distribution -- "HIGH" meant "the top ~34% of
    whatever probabilities this particular model happened to output,"
    a purely RELATIVE, self-referential statement. If a model's
    predictions all clustered between 4% and 8% (plausible for a rare-
    event classifier), the top third of that narrow range would still be
    labeled "HIGH" even though every value is objectively a low
    probability -- not a defensible basis for a label a user is meant to
    act on.

    This version instead finds, for each level, the LOWEST probability
    threshold such that among all predictions AT OR ABOVE it, the ACTUAL
    observed positive rate (precision at that cutoff, walking down the
    sorted-by-confidence list) reaches `medium_level`/`high_level`. e.g.
    with `high_level=0.20`: "HIGH" means "historically, at least 20% of
    predictions this confident or higher were actually followed by a real
    loss event" -- an absolute, outcome-tied statement, not a ranking
    statement.

    MUST be called using TRAIN+VALIDATION predictions AND outcomes only
    (never TEST) -- see docs/DATA_LEAKAGE_AUDIT.md.
    """
    medium_level = CONFIG["risk"]["risk_class_thresholds"]["medium"] if medium_level is None else medium_level
    high_level = CONFIG["risk"]["risk_class_thresholds"]["high"] if high_level is None else high_level

    df = pd.DataFrame(
        {"proba": predicted_probabilities.values, "outcome": actual_outcomes.values}
    ).sort_values("proba", ascending=False).reset_index(drop=True)
    # At row i (0-indexed), this is the precision achieved if the model
    # predicted "positive" for exactly the (i+1) most confident cases --
    # i.e. the actual observed positive rate among predictions >= this
    # row's probability. This is precision as a function of threshold,
    # computed directly rather than via a fixed-width calibration table.
    df["precision_at_or_above"] = df["outcome"].expanding().mean()

    def _find_cutoff(level: float) -> float:
        eligible = df[df["precision_at_or_above"] >= level]
        if eligible.empty:
            # Even the single most confident prediction doesn't reach
            # this precision level -- there is no threshold that
            # satisfies the request on this data. Fall back to the
            # highest observed probability (the narrowest, most
            # conservative possible cutoff) rather than silently
            # returning an arbitrary value.
            return float(df["proba"].iloc[0]) if len(df) else float("nan")
        # The LOWEST probability among eligible rows is the widest
        # threshold that still achieves >= level precision.
        return float(eligible["proba"].iloc[-1])

    return {"medium": _find_cutoff(medium_level), "high": _find_cutoff(high_level)}


def classify_risk(probability: float, thresholds: dict[str, float]) -> str:
    if probability >= thresholds["high"]:
        return "HIGH"
    if probability >= thresholds["medium"]:
        return "MEDIUM"
    return "LOW"
