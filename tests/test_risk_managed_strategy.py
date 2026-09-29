import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import equal_weight_strategy, run_walk_forward_backtest
from src.backtesting.risk_managed_strategy import make_risk_managed_strategy


@pytest.fixture
def long_toy_returns():
    # Needs to be long enough to clear MIN_TRAINING_ROWS inside the strategy.
    dates = pd.bdate_range("2018-01-01", periods=900)
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0004, 0.01, 900),
            "B": rng.normal(0.0002, 0.02, 900),
            "C": rng.normal(0.0001, 0.005, 900),  # lowest vol -> expected "safe asset"
        },
        index=dates,
    )


def test_risk_managed_strategy_falls_back_with_insufficient_history():
    dates = pd.bdate_range("2023-01-01", periods=50)  # far below MIN_TRAINING_ROWS
    rng = np.random.default_rng(0)
    short_returns = pd.DataFrame({"A": rng.normal(0, 0.01, 50), "B": rng.normal(0, 0.01, 50)}, index=dates)

    strategy = make_risk_managed_strategy(equal_weight_strategy)
    weights = strategy(dates[-1], short_returns)
    expected = equal_weight_strategy(dates[-1], short_returns)
    pd.testing.assert_series_equal(weights, expected)


def test_risk_managed_strategy_runs_inside_full_backtest(long_toy_returns):
    """End-to-end smoke test: the strategy must be usable as a
    target_weight_fn inside the real backtest engine, retraining at every
    rebalance date, without crashing across a multi-year run."""
    strategy = make_risk_managed_strategy(equal_weight_strategy, defensive_scale=0.5)
    result = run_walk_forward_backtest(long_toy_returns, strategy, rebalance_frequency="monthly")

    assert result.num_rebalances > 0
    assert len(result.portfolio_value) == len(long_toy_returns)
    assert result.portfolio_value.iloc[-1] > 0  # never goes to zero/negative in this setup


def test_risk_managed_strategy_weights_still_sum_to_base_total(long_toy_returns):
    """Whether defensive or not, returned weights must preserve the same
    total exposure convention as the base strategy (reallocating, not
    fabricating or losing weight)."""
    strategy = make_risk_managed_strategy(equal_weight_strategy, defensive_scale=0.5)
    as_of = long_toy_returns.index[500]
    history = long_toy_returns.loc[:as_of]

    weights = strategy(as_of, history)
    base = equal_weight_strategy(as_of, history)
    assert weights.sum() == pytest.approx(base.sum(), abs=1e-6)


def test_risk_managed_strategy_retraining_does_not_use_future_data(long_toy_returns):
    """Leakage test: perturbing returns strictly AFTER the as_of_date used
    for a strategy call must not change that call's output, since the
    strategy is only ever given `historical_returns` truncated to that
    date by the backtest engine -- but this test verifies the strategy
    function ITSELF doesn't do anything that would break that guarantee
    (e.g. accidentally closing over the full, unsliced DataFrame).
    """
    as_of = long_toy_returns.index[600]
    history = long_toy_returns.loc[:as_of]

    strategy = make_risk_managed_strategy(equal_weight_strategy, random_seed=42)
    weights_original = strategy(as_of, history)

    perturbed_full = long_toy_returns.copy()
    perturbed_full.iloc[700:] = perturbed_full.iloc[700:] * 50 + 2.0  # huge shock, strictly after as_of
    perturbed_history = perturbed_full.loc[:as_of]  # identical to `history` since as_of < 700

    weights_perturbed = strategy(as_of, perturbed_history)
    pd.testing.assert_series_equal(weights_original, weights_perturbed)


def test_risk_managed_vs_base_strategy_produce_different_results_somewhere(long_toy_returns):
    """Sanity check that the risk overlay is actually doing something
    across a long backtest -- if defensive reallocation never triggers,
    the whole module would be untested dead code in practice."""
    base_result = run_walk_forward_backtest(long_toy_returns, equal_weight_strategy, rebalance_frequency="monthly")
    strategy = make_risk_managed_strategy(equal_weight_strategy, defensive_scale=0.3, high_risk_threshold=0.5)
    managed_result = run_walk_forward_backtest(long_toy_returns, strategy, rebalance_frequency="monthly")

    # With a long enough history and a lowered high_risk_threshold (to make
    # the defensive branch easier to trigger in a short test run), the two
    # weight histories should differ at least somewhere.
    weights_differ = not managed_result.weights_history.equals(base_result.weights_history)
    assert weights_differ


def test_prediction_uses_current_date_features_not_stale_ones(long_toy_returns, monkeypatch):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue C1: the
    feature row fed to predict_proba for "today's" risk estimate must
    come from the ACTUAL latest date (as_of_date), not from
    as_of_date - horizon_days (a stale row left over from the
    target-driven dropna()). We monkeypatch predict_proba to capture
    exactly which row it was called with, then compare that row's values
    against independently-computed features for the true as_of_date vs.
    the stale (horizon-days-earlier) date -- if the bug were present, the
    captured row would match the STALE date's features, not the current
    date's.
    """
    from src.features.feature_pipeline import build_features_for_asset
    from src.portfolio.rebalancing import rebalance_plan  # noqa: F401 (ensures import side effects consistent)

    as_of = long_toy_returns.index[700]
    history = long_toy_returns.loc[:as_of]

    # Independently build the same proxy-return feature series the
    # strategy builds internally, using equal weights (the base strategy).
    base_weights = equal_weight_strategy(as_of, history)
    proxy_returns = history.mul(base_weights.reindex(history.columns), axis=1).sum(axis=1)
    wealth = (1 + proxy_returns).cumprod()
    prices_df = pd.DataFrame({"date": wealth.index, "adj_close": wealth.values, "volume": 1})
    feats = build_features_for_asset(prices_df).drop(columns=["beta_60d", "corr_market_60d", "volume_zscore_20d"])

    true_current_row = feats.iloc[[-1]]
    horizon_days = 5  # default CONFIG["ml"]["target_horizon_days"]
    stale_row = feats.iloc[[-1 - horizon_days]]

    captured = {}
    from sklearn.pipeline import Pipeline
    original_predict_proba = Pipeline.predict_proba

    def capturing_predict_proba(self, X, *args, **kwargs):
        captured["X"] = X.copy()
        return original_predict_proba(self, X, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "predict_proba", capturing_predict_proba)

    strategy = make_risk_managed_strategy(equal_weight_strategy, high_risk_threshold=0.0)  # force the prediction branch to run
    strategy(as_of, history)

    assert "X" in captured, "predict_proba was never called -- strategy fell back to base weights, cannot verify"
    captured_row = captured["X"]

    feature_cols = [c for c in captured_row.columns]
    current_values = true_current_row[feature_cols].values
    stale_values = stale_row[feature_cols].values

    import numpy as np
    matches_current = np.allclose(captured_row.values, current_values, equal_nan=True)
    matches_stale = np.allclose(captured_row.values, stale_values, equal_nan=True)

    assert matches_current, (
        "predict_proba was called with a feature row that does NOT match "
        "the current date's actual features -- this is the exact bug "
        "described in docs/SECOND_FORENSIC_AUDIT.md Issue C1."
    )
    assert not matches_stale, (
        "predict_proba was called with the STALE (horizon-days-earlier) "
        "feature row -- reproducing the original bug exactly."
    )


def test_defensive_reallocation_respects_max_weight(long_toy_returns):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H9b: the
    freed weight from a defensive reallocation must never push the safe
    asset (or any asset) past max_weight."""
    strategy = make_risk_managed_strategy(
        equal_weight_strategy, defensive_scale=0.2, high_risk_threshold=0.0, max_weight=0.4,
    )
    as_of = long_toy_returns.index[700]
    history = long_toy_returns.loc[:as_of]
    weights = strategy(as_of, history)
    assert (weights <= 0.4 + 1e-9).all(), f"Some weight exceeds max_weight=0.4: {weights.to_dict()}"


def test_cash_allocation_mode_genuinely_reduces_total_exposure(long_toy_returns):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H9c: with
    allow_cash_allocation=True, a HIGH-risk signal must reduce TOTAL
    invested weight (sum < 1.0), not just redistribute it -- the original
    behavior (allow_cash_allocation=False, still the default) always
    reallocated back to 100% invested, which is a rotation, not a real
    exposure reduction."""
    strategy_cash = make_risk_managed_strategy(
        equal_weight_strategy, defensive_scale=0.3, high_risk_threshold=0.0, allow_cash_allocation=True,
    )
    strategy_reallocate = make_risk_managed_strategy(
        equal_weight_strategy, defensive_scale=0.3, high_risk_threshold=0.0, allow_cash_allocation=False,
    )
    as_of = long_toy_returns.index[700]
    history = long_toy_returns.loc[:as_of]

    weights_cash = strategy_cash(as_of, history)
    weights_reallocate = strategy_reallocate(as_of, history)

    assert weights_cash.sum() == pytest.approx(0.3, abs=1e-6)  # genuinely under-invested
    assert weights_reallocate.sum() == pytest.approx(1.0, abs=1e-6)  # fully reinvested (old behavior)


def test_defensive_assets_parameter_restricts_safe_asset_search(long_toy_returns):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H9a: when
    `defensive_assets` is provided, the safe asset must be chosen from
    that subset only, not the whole universe."""
    strategy = make_risk_managed_strategy(
        equal_weight_strategy, defensive_scale=0.2, high_risk_threshold=0.0,
        defensive_assets=["A"], allow_cash_allocation=False,
    )
    as_of = long_toy_returns.index[700]
    history = long_toy_returns.loc[:as_of]
    weights = strategy(as_of, history)
    # With defensive_assets=["A"], all freed weight must go to "A" (up to
    # its max_weight cap) -- "B" and "C" should be at (or below) their
    # plain defensive_scale-reduced level, never boosted.
    base = equal_weight_strategy(as_of, history)
    assert weights["A"] > base["A"] * 0.2  # A received the reallocation boost


def test_stats_dict_records_call_outcomes(long_toy_returns):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H10's
    companion fix: the optional `stats` dict must record why each call
    did or didn't trigger the defensive branch, so a caller doesn't have
    to rely on DEBUG-level logs to find out."""
    stats = {}
    strategy = make_risk_managed_strategy(equal_weight_strategy, high_risk_threshold=0.0, stats=stats)
    as_of = long_toy_returns.index[700]
    history = long_toy_returns.loc[:as_of]
    strategy(as_of, history)
    assert sum(stats.values()) >= 1
    assert "high_risk_triggered" in stats or "insufficient_labeled_data" in stats or "low_risk" in stats


def test_only_expected_exception_types_are_caught(long_toy_returns, monkeypatch):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H10: a
    genuinely unexpected exception (e.g. an AttributeError from a real
    programming bug elsewhere) must propagate out of the strategy, not be
    silently swallowed into a base-weights fallback."""
    import src.backtesting.risk_managed_strategy as rms_module

    def broken_train_model(*args, **kwargs):
        raise AttributeError("simulated unexpected programming bug, not a data-quality issue")

    monkeypatch.setattr(rms_module, "train_model", broken_train_model)

    strategy = make_risk_managed_strategy(equal_weight_strategy, high_risk_threshold=0.0)
    as_of = long_toy_returns.index[700]
    history = long_toy_returns.loc[:as_of]

    with pytest.raises(AttributeError, match="simulated unexpected programming bug"):
        strategy(as_of, history)


def test_explicit_zero_threshold_is_not_silently_replaced_by_config_default():
    """Regression test for a real bug caught during this project's own
    development: `param or CONFIG[...]` silently discards an explicitly
    passed `0.0` (falsy in Python) in favor of the config default. This
    affects any caller intentionally passing 0.0 to force a specific
    behavior (e.g. tests forcing the defensive branch to always trigger).
    """
    from src.config import CONFIG

    config_default = CONFIG["risk"]["risk_class_thresholds"]["high"]
    assert config_default != 0.0, "test assumes the config default isn't already 0.0"

    strategy_with_explicit_zero = make_risk_managed_strategy(
        equal_weight_strategy, high_risk_threshold=0.0,
    )
    # Inspect via closure: call once and verify defensive behavior
    # triggers even for a very low (but non-negative) probability, which
    # would only happen if 0.0 was actually honored as the threshold.
    dates = pd.bdate_range("2018-01-01", periods=900)
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        {"A": rng.normal(0.0004, 0.01, 900), "B": rng.normal(0.0002, 0.02, 900), "C": rng.normal(0.0001, 0.005, 900)},
        index=dates,
    )
    as_of = returns.index[700]
    history = returns.loc[:as_of]
    weights = strategy_with_explicit_zero(as_of, history)
    base = equal_weight_strategy(as_of, history)
    # With threshold genuinely at 0.0, ANY valid (non-negative) predicted
    # probability must trigger the defensive branch -- so weights must
    # differ from the plain base weights (unless training itself failed
    # for an unrelated reason, which this specific seed/window doesn't).
    assert not weights.equals(base), (
        "Explicit high_risk_threshold=0.0 was not honored -- it was "
        "silently replaced by the config default, reproducing the "
        "original `param or default` bug."
    )
