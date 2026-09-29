"""
Feature selection.

WHY this module exists (it was a real gap before this pass): the feature
pipeline (`src/features/feature_pipeline.py`) intentionally builds a broad,
overlapping feature set (ret_1d/5d/10d/20d/60d, vol_5d/10d/20d/60d, ...) so
nothing useful is thrown away too early. But many of those features are
highly correlated with each other (e.g. `vol_20d` and `vol_60d` move
together), which:

  1. Adds noise/variance to linear models (multicollinearity inflates
     coefficient variance and makes them unstable/hard to interpret).
  2. Wastes model capacity and slows training for no accuracy benefit.
  3. Makes feature importances misleading -- correlated features "split"
     credit for the same underlying signal, making each look individually
     weaker than it is.

LEAKAGE RULE (same principle as preprocessing.py): every function here
must be fit/decided using TRAIN data only, then the same selected feature
list is applied unchanged to validation/test. Computing correlations or
importances on the full dataset (including validation/test) and then
selecting features would leak information about the evaluation period into
the modeling decision -- a documented failure mode in
docs/DATA_LEAKAGE_AUDIT.md ("feature-selection leakage").
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

logger = logging.getLogger(__name__)


def drop_low_variance_features(
    X_train: pd.DataFrame, threshold: float = 1e-6
) -> list[str]:
    """Drop near-constant features (e.g. a feature that is almost always
    the same value carries ~no information and can destabilize some
    models). Fit on X_train only; returns the list of columns to KEEP.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train.fillna(X_train.median()))
    kept = X_train.columns[selector.get_support()].tolist()
    dropped = sorted(set(X_train.columns) - set(kept))
    if dropped:
        logger.info("Dropping %d near-constant feature(s): %s", len(dropped), dropped)
    return kept


def drop_correlated_features(
    X_train: pd.DataFrame, threshold: float = 0.90
) -> list[str]:
    """Greedy correlation filter: for every pair of features with
    |correlation| > threshold (computed on X_train only), drop the second
    one encountered. This is a simple, explainable heuristic -- not optimal
    (it doesn't consider three-way relationships), but transparent and
    stable, which matters more than marginal optimality for a feature set
    this small (~15 features).

    Why 0.90 as a default: high enough that genuinely distinct signals
    (e.g. short vs. long momentum) survive, low enough to catch clear
    redundancy (e.g. vol_20d vs vol_60d in a calm, low-regime-change period).
    Treat it as a config-level assumption, not a universal constant --
    revisit if the feature set changes.
    """
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated_with = upper.index[upper[col] > threshold].tolist()
        for other in correlated_with:
            if other not in to_drop:
                to_drop.add(col)
                break

    kept = [c for c in X_train.columns if c not in to_drop]
    if to_drop:
        logger.info("Dropping %d correlated feature(s) (|corr|>%.2f): %s", len(to_drop), threshold, sorted(to_drop))
    return kept


def variance_inflation_factors(X_train: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor per feature -- a formal multicollinearity
    diagnostic (VIF > ~10 is a common rule-of-thumb red flag). This is
    reported for interpretability/diagnosis; `drop_correlated_features`
    above is the actual selection mechanism used in the pipeline, since VIF
    itself doesn't say *which* variable to drop from a correlated group.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_filled = X_train.fillna(X_train.median()).values
    vifs = {
        col: variance_inflation_factor(X_filled, i)
        for i, col in enumerate(X_train.columns)
    }
    return pd.Series(vifs, name="VIF").sort_values(ascending=False)


def select_by_model_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int | None = None,
    importance_threshold: float | None = None,
    random_seed: int = 42,
) -> pd.Series:
    """Fit a Random Forest on X_train/y_train ONLY and rank features by
    impurity-based importance. Random Forest is used here (rather than the
    final production model) because it needs no scaling, handles
    non-linear/interaction effects natively, and gives a fast, robust
    importance ranking usable regardless of which model is chosen later.

    Returns the full importance Series (sorted descending) for
    inspection; use `top_n` or `importance_threshold` to get just the
    selected feature names via `.index[:top_n]` / thresholding.
    """
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=random_seed, class_weight="balanced"
    )
    model.fit(X_train.fillna(X_train.median()), y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns, name="importance")
    importances = importances.sort_values(ascending=False)

    if top_n is not None:
        return importances.iloc[:top_n]
    if importance_threshold is not None:
        return importances[importances >= importance_threshold]
    return importances


def mutual_information_ranking(X_train: pd.DataFrame, y_train: pd.Series, random_seed: int = 42) -> pd.Series:
    """Mutual information between each feature and the target, computed on
    TRAIN only. Unlike a linear correlation, mutual information captures
    non-linear relationships too (e.g. a U-shaped relationship between
    volatility and risk that a Pearson correlation would score near zero).
    Useful as a second opinion alongside Random Forest importances --
    features that rank highly on both are the most trustworthy signals.
    """
    mi = mutual_info_classif(
        X_train.fillna(X_train.median()), y_train, random_state=random_seed
    )
    return pd.Series(mi, index=X_train.columns, name="mutual_info").sort_values(ascending=False)


def iterative_vif_removal(
    X_train: pd.DataFrame, vif_threshold: float = 10.0, max_iterations: int = 50
) -> list[str]:
    """Iteratively remove the single feature with the HIGHEST VIF, one at a
    time, recomputing VIF after each removal, until every remaining
    feature's VIF is below `vif_threshold` (a common rule-of-thumb cutoff
    for "problematic" multicollinearity).

    Why iterative rather than the single-pass pairwise filter in
    `drop_correlated_features`: pairwise correlation only catches
    two-way redundancy (feature A vs feature B). VIF captures a feature's
    R^2 against ALL other features combined, so it can flag a feature that
    is only moderately correlated with any single other feature but is
    still near-perfectly explainable as a *combination* of several others
    (e.g. `ma_ratio_20_60` may be reconstructible from `momentum_20d` and
    `vol_ratio_20_60` together even if no single pairwise correlation with
    either exceeds the 0.90 pairwise-filter threshold). Recomputing VIF
    after each removal is necessary because removing one collinear feature
    changes every other feature's VIF (multicollinearity is a property of
    the whole feature SET, not of a fixed pairwise number computed once).

    Fit on X_train only, per the leakage rule in
    docs/DATA_LEAKAGE_AUDIT.md -- the returned column list is then applied
    unchanged to validation/test.
    """
    remaining = list(X_train.columns)
    for _ in range(max_iterations):
        if len(remaining) <= 1:
            break
        vifs = variance_inflation_factors(X_train[remaining])
        worst_feature, worst_vif = vifs.index[0], vifs.iloc[0]
        # Stop once the worst remaining VIF is finite AND at/below the
        # threshold. An infinite VIF (perfect multicollinearity) must
        # trigger removal, not an early stop.
        if np.isfinite(worst_vif) and worst_vif <= vif_threshold:
            break
        logger.info("Removing '%s' (VIF=%.1f > threshold=%.1f)", worst_feature, worst_vif, vif_threshold)
        remaining.remove(worst_feature)
    return remaining


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    correlation_threshold: float = 0.90,
    variance_threshold: float = 1e-6,
    top_n_by_importance: int | None = None,
    random_seed: int = 42,
) -> list[str]:
    """End-to-end selection pipeline, applied in this order:

    1. Drop near-constant features (variance filter).
    2. Drop redundant, highly-correlated features (correlation filter).
    3. Optionally keep only the top-N remaining features by Random Forest
       importance.

    Every step uses X_train/y_train ONLY. The caller must apply the
    resulting column list unchanged to validation/test
    (`X_val[selected_cols]`, `X_test[selected_cols]`) -- never re-run
    selection on those splits.
    """
    step1 = drop_low_variance_features(X_train, threshold=variance_threshold)
    step2 = drop_correlated_features(X_train[step1], threshold=correlation_threshold)

    if top_n_by_importance is not None and len(step2) > top_n_by_importance:
        importances = select_by_model_importance(
            X_train[step2], y_train, top_n=top_n_by_importance, random_seed=random_seed
        )
        return importances.index.tolist()

    return step2
