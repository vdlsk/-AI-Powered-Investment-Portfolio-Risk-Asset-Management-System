"""
Risk-managed backtest strategy: connects the ML risk-prediction pipeline
(src/models/) directly to the walk-forward backtest engine
(src/backtesting/engine.py).

WHY THIS MODULE EXISTS: before this pass, the ML risk classifier
(notebooks 11/13) and the walk-forward backtest (notebook 12) were two
independently-tested but never-combined pieces of the project. That meant
the single most interesting question this project could answer -- "does
using the ML risk signal to manage exposure actually improve risk-adjusted
returns, net of the extra retraining cost and complexity?" -- was never
actually asked. This module asks it.

TRUE walk-forward, not train-once-and-freeze: `make_risk_managed_strategy`
returns a `target_weight_fn` that RETRAINS the risk classifier at every
call, using ONLY `historical_returns` (which the backtest engine already
guarantees is truncated to data available through the current rebalance
date -- see `run_walk_forward_backtest`'s docstring). This is more
expensive than a fixed model but is the only way to avoid a much subtler
leakage bug: a model trained once on the full backtest period and then
"used" at every historical rebalance date would have implicitly seen the
entire future relative to early rebalance dates.

DEFENSIVE REALLOCATION DESIGN (see docs/SECOND_FORENSIC_AUDIT.md, Issue
H9, for the three original gaps this addresses):

  - H9a (configurable safe asset): the "safe asset" a HIGH-risk signal
    reallocates toward is, by default, the lowest-historical-volatility
    asset in the WHOLE universe -- a defensible but crude heuristic
    (lowest past volatility is not guaranteed to mean defensive in a
    genuine crisis; correlations often rise exactly when it matters).
    Pass `defensive_assets` to restrict the search to a specific,
    explicitly-chosen subset (e.g. bonds only) instead.
  - H9b (weight-constraint validation): the freed weight is now capped at
    `max_weight` on the chosen safe asset, with any remainder distributed
    proportionally across the other assets (each also respecting
    `max_weight`) rather than dumped unconditionally onto one position --
    a defensive move must not itself create an over-concentrated
    portfolio.
  - H9c (real exposure reduction, not just reallocation): pass
    `allow_cash_allocation=True` to make `defensive_scale` a genuine
    reduction in total invested weight (weights sum to `defensive_scale`,
    not 1.0 -- the remainder is implicitly uninvested/"cash", earning
    zero return) instead of always reallocating 100% of the freed weight
    into the safe asset. Default is `False` for backward compatibility
    with earlier notebook/dashboard results.

EXCEPTION HANDLING (Issue H10): only `ValueError` and `numpy.linalg.
LinAlgError` -- the enumerable, expected failure modes from degenerate
feature matrices at early/thin backtest dates -- are caught and converted
to a base-weights fallback. Any other exception type (a genuine bug
introduced elsewhere in the training pipeline) propagates, rather than
being silently swallowed for the rest of a multi-year backtest run.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.features.feature_pipeline import build_features_for_asset
from src.models.feature_selection import drop_low_variance_features, iterative_vif_removal
from src.models.targets import build_loss_event_target
from src.models.train import train_model

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = 300  # below this, fall back to the base strategy rather than train on too little data
MIN_POSITIVE_EXAMPLES = 5  # below this, a classifier's probability estimate is not trustworthy

# The enumerable, EXPECTED failure modes from training on a degenerate
# feature matrix (e.g. too few varying rows at an early backtest date).
# Deliberately narrow -- see Issue H10 above -- so a genuine bug
# introduced elsewhere is never silently converted into "use base
# weights" for the rest of a long backtest run.
EXPECTED_TRAINING_FAILURE_TYPES = (ValueError, np.linalg.LinAlgError)


def _apply_defensive_reallocation(
    base_weights: pd.Series,
    safe_asset: str,
    defensive_scale: float,
    max_weight: float,
    allow_cash_allocation: bool,
) -> pd.Series:
    """Scale every position by `defensive_scale`, then either (cash mode)
    leave the freed weight uninvested, or (reallocation mode) place it
    into `safe_asset` up to `max_weight`, spilling any remainder
    proportionally across the other assets (each also capped at
    `max_weight`). This is the fix for Issues H9b/H9c: the freed weight
    can no longer push a single asset's weight past the same concentration
    limit every other strategy in this project is held to.
    """
    defensive_weights = base_weights * defensive_scale
    freed_weight = base_weights.sum() - defensive_weights.sum()

    if allow_cash_allocation:
        # Genuine exposure reduction: freed_weight is simply not
        # reinvested. Total invested weight is now < 1.0.
        return defensive_weights

    # Reallocation mode: cap the safe asset's addition at max_weight.
    current_safe_weight = defensive_weights.get(safe_asset, 0.0)
    room = max(0.0, max_weight - current_safe_weight)
    add_to_safe = min(freed_weight, room)
    defensive_weights[safe_asset] = current_safe_weight + add_to_safe
    remaining = freed_weight - add_to_safe

    if remaining > 1e-9:
        # Spill the remainder proportionally across the other assets,
        # each also respecting max_weight (single proportional pass --
        # documented simplification: with typical defensive_scale/
        # max_weight/asset-count combinations this fully absorbs the
        # remainder; in a pathological case where every other asset is
        # already at its cap too, the tiny leftover is placed on the safe
        # asset as a last resort rather than silently discarded, which
        # would break the weights-sum-to-1 invariant the backtest engine
        # relies on).
        others = [a for a in defensive_weights.index if a != safe_asset]
        total_other = defensive_weights[others].sum()
        original_remaining = remaining  # fixed total to distribute -- see note below
        if total_other > 0:
            # IMPORTANT: `share` for each asset is computed against
            # `original_remaining` (fixed once, before the loop), not a
            # running/shrinking `remaining` -- using a shrinking value
            # here was a real bug caught during testing: it under-allocated
            # every asset after the first one in `others`, leaving a
            # genuine shortfall that then got dumped entirely onto
            # `safe_asset` via the "last resort" step below, silently
            # breaking the max_weight cap this whole function exists to
            # enforce (Issue H9b).
            distributed = 0.0
            for asset in others:
                share = original_remaining * (defensive_weights[asset] / total_other)
                room_a = max(0.0, max_weight - defensive_weights[asset])
                add = min(share, room_a)
                defensive_weights[asset] += add
                distributed += add
            remaining = original_remaining - distributed
        if remaining > 1e-9:
            defensive_weights[safe_asset] += remaining

    return defensive_weights


def make_risk_managed_strategy(
    base_weight_fn,
    horizon_days: int | None = None,
    loss_threshold: float | None = None,
    high_risk_threshold: float | None = None,
    defensive_scale: float = 0.5,
    model_name: str = "logistic_regression",
    random_seed: int = 42,
    defensive_assets: list[str] | None = None,
    allow_cash_allocation: bool = False,
    max_weight: float | None = None,
    stats: dict | None = None,
):
    """Build a target_weight_fn that scales down exposure when a freshly
    (walk-forward) retrained risk classifier estimates elevated near-term
    loss probability, and defers to `base_weight_fn` otherwise.

    `defensive_assets`: restrict the "safe asset" search to this subset of
    tickers (e.g. `["BND"]`); `None` (default) searches the whole universe
    by trailing volatility -- see this module's docstring, Issue H9a.

    `allow_cash_allocation`: if `True`, a HIGH-risk signal genuinely
    reduces total invested weight rather than reallocating 100% of it --
    see Issue H9c.

    `max_weight`: concentration cap applied to the defensive reallocation
    itself (defaults to `config.yaml`'s `portfolio.max_weight`) -- see
    Issue H9b.

    `stats`: optional dict; if provided, this call increments counters
    (`insufficient_history`, `insufficient_labeled_data`,
    `training_failed_expected`, `latest_row_nan`, `low_risk`,
    `high_risk_triggered`) so a caller can see WHY each call did or didn't
    trigger the defensive branch, rather than that information being
    visible only in DEBUG-level logs easy to miss across a long backtest.

    Falls back to `base_weight_fn` UNCHANGED (not a zero-risk assumption)
    whenever there isn't enough historical data to train and validate a
    classifier at all -- an early backtest date with only a few months of
    history should not pretend to have a risk estimate it cannot actually
    compute.
    """
    # Explicit `is None` checks, not `param or default` -- the latter
    # silently discards a caller-provided falsy-but-valid value like
    # `high_risk_threshold=0.0` or `loss_threshold=0.0` (both legitimate
    # edge-case configurations, e.g. "always treat any predicted risk as
    # HIGH"), since `0.0` is falsy in Python. This was a real bug caught
    # during testing of this exact function: passing
    # `high_risk_threshold=0.0` to force the defensive branch for a test
    # was silently ignored in favor of the config default.
    horizon_days = CONFIG["ml"]["target_horizon_days"] if horizon_days is None else horizon_days
    loss_threshold = CONFIG["ml"]["loss_threshold"] if loss_threshold is None else loss_threshold
    high_risk_threshold = (
        CONFIG["risk"]["risk_class_thresholds"]["high"] if high_risk_threshold is None else high_risk_threshold
    )
    max_weight = CONFIG["portfolio"]["max_weight"] if max_weight is None else max_weight

    def _record(key: str) -> None:
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    def strategy(as_of_date: pd.Timestamp, historical_returns: pd.DataFrame) -> pd.Series:
        base_weights = base_weight_fn(as_of_date, historical_returns)

        if len(historical_returns) < MIN_TRAINING_ROWS:
            logger.debug("Risk-managed strategy: insufficient history (%d rows) at %s, using base weights", len(historical_returns), as_of_date)
            _record("insufficient_history")
            return base_weights

        # Build a single portfolio-level return series (using the BASE
        # strategy's weights as the risk-monitoring proxy) and derive
        # leakage-safe features + a forward loss-event target on it. Using
        # base_weights here (rather than, say, an equal-weighted proxy)
        # keeps the risk model's notion of "the portfolio" consistent with
        # what's actually being risk-managed.
        proxy_returns = historical_returns.mul(
            base_weights.reindex(historical_returns.columns).fillna(0.0), axis=1
        ).sum(axis=1)
        wealth = (1 + proxy_returns).cumprod()
        prices_df = pd.DataFrame({"date": wealth.index, "adj_close": wealth.values, "volume": 1})

        feats = build_features_for_asset(prices_df).drop(
            columns=["beta_60d", "corr_market_60d", "volume_zscore_20d"]
        )
        target = build_loss_event_target(
            proxy_returns.reset_index(drop=True), horizon_days=horizon_days, loss_threshold=loss_threshold
        )
        df = feats.copy()
        df["target"] = target.values

        # IMPORTANT (see docs/SECOND_FORENSIC_AUDIT.md, Issue C1): the
        # target is forward-looking, so its last `horizon_days` rows are
        # always NaN (there's no future data yet to compute them from --
        # see src/models/targets.py::forward_return). `df.dropna()` is
        # correct and necessary for the TRAINING frame (a row with an
        # undefined target cannot be trained on), but taking "the latest
        # row" AFTER that dropna gives a row from `horizon_days` trading
        # days ago, not the actual latest available date -- silently
        # making every "current risk" prediction stale by the target
        # horizon. The fix: capture the true latest feature row from the
        # UNFILTERED `feats` frame, before the target-driven dropna, and
        # use the dropna'd frame ONLY for training.
        train_df = df.dropna()

        if len(train_df) < MIN_TRAINING_ROWS or train_df["target"].sum() < MIN_POSITIVE_EXAMPLES or train_df["target"].nunique() < 2:
            logger.debug(
                "Risk-managed strategy: insufficient labeled data at %s (%d usable rows, %d positive), using base weights",
                as_of_date, len(train_df), train_df["target"].sum() if len(train_df) else 0,
            )
            _record("insufficient_labeled_data")
            return base_weights

        feature_cols = [c for c in train_df.columns if c not in ("date", "ticker", "target")]
        X, y = train_df[feature_cols], train_df["target"].astype(int)

        # Collinearity removal on this call's TRAIN data only -- same rule
        # as everywhere else in this project (see
        # docs/PREPROCESSING_AND_FEATURE_SELECTION.md §3). Only the
        # enumerable, EXPECTED failure modes (see EXPECTED_TRAINING_FAILURE_TYPES
        # above) are caught here -- a genuine bug elsewhere propagates
        # instead of being silently converted into "use base weights" for
        # the rest of a long backtest run (Issue H10).
        try:
            after_variance = drop_low_variance_features(X)
            selected = iterative_vif_removal(X[after_variance], vif_threshold=10.0)
            X = X[selected]
            trained = train_model(X, y, model_name, numeric_features=selected)
        except EXPECTED_TRAINING_FAILURE_TYPES as e:
            logger.warning("Risk-managed strategy: training failed at %s (%s: %s), using base weights", as_of_date, type(e).__name__, e)
            _record("training_failed_expected")
            return base_weights

        # The prediction input for "today" comes from the FULL, unfiltered
        # `feats` frame's actual last row (same date as `as_of_date`),
        # never from `train_df` -- this is the actual fix for Issue C1.
        latest_row = feats[selected].iloc[[-1]]
        if latest_row.isna().any(axis=None):
            # The true latest row can itself contain NaN in a feature that
            # needs a longer warm-up window than is currently available
            # (e.g. very early in a backtest) -- fall back rather than
            # feed NaN into predict_proba.
            logger.debug("Risk-managed strategy: latest feature row has NaN at %s, using base weights", as_of_date)
            _record("latest_row_nan")
            return base_weights
        risk_probability = float(trained.pipeline.predict_proba(latest_row)[:, 1][0])

        if risk_probability < high_risk_threshold:
            _record("low_risk")
            return base_weights

        _record("high_risk_triggered")

        candidate_universe = defensive_assets or list(historical_returns.columns)
        candidate_universe = [a for a in candidate_universe if a in historical_returns.columns]
        if not candidate_universe:
            candidate_universe = list(historical_returns.columns)
        safe_asset = historical_returns[candidate_universe].std().idxmin()

        defensive_weights = _apply_defensive_reallocation(
            base_weights, safe_asset, defensive_scale, max_weight, allow_cash_allocation,
        )
        logger.debug(
            "Risk-managed strategy: HIGH risk (p=%.2f) at %s -> defensive reallocation toward %s (cash_mode=%s)",
            risk_probability, as_of_date, safe_asset, allow_cash_allocation,
        )
        return defensive_weights

    return strategy
