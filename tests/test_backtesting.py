import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import run_walk_forward_backtest, equal_weight_strategy


@pytest.fixture
def toy_returns():
    dates = pd.bdate_range("2022-01-01", periods=260)
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.01, 260),
            "B": rng.normal(0.0003, 0.015, 260),
        },
        index=dates,
    )


def test_backtest_runs_and_produces_expected_shapes(toy_returns):
    result = run_walk_forward_backtest(
        toy_returns, equal_weight_strategy, rebalance_frequency="monthly"
    )
    assert len(result.portfolio_value) == len(toy_returns)
    assert result.num_rebalances > 0
    assert "sharpe_ratio" in result.performance


def test_backtest_transaction_costs_reduce_returns(toy_returns):
    result_with_cost = run_walk_forward_backtest(
        toy_returns, equal_weight_strategy, rebalance_frequency="monthly", transaction_cost_bps=50
    )
    result_no_cost = run_walk_forward_backtest(
        toy_returns, equal_weight_strategy, rebalance_frequency="monthly", transaction_cost_bps=0
    )
    assert result_with_cost.transaction_costs.sum() > 0
    assert result_no_cost.transaction_costs.sum() == 0
    # Higher costs should never produce a HIGHER final portfolio value than
    # the zero-cost version, all else equal.
    assert result_with_cost.portfolio_value.iloc[-1] <= result_no_cost.portfolio_value.iloc[-1]


def test_backtest_signal_uses_only_past_data(toy_returns):
    """Perturbing returns far in the future must not change portfolio value
    on dates strictly before the perturbation, proving the engine doesn't
    peek ahead when generating signals or applying weights."""
    cutoff = 150

    def suspicious_strategy(as_of_date, historical_returns):
        # A deliberately leaky-looking strategy that only ever sees
        # `historical_returns` (<= as_of_date) -- if the engine is correct,
        # this still can't see post-cutoff perturbations before they happen.
        return equal_weight_strategy(as_of_date, historical_returns)

    result_original = run_walk_forward_backtest(toy_returns, suspicious_strategy, rebalance_frequency="monthly")

    perturbed = toy_returns.copy()
    perturbed.iloc[cutoff + 5 :] = perturbed.iloc[cutoff + 5 :] * 10 + 0.5

    result_perturbed = run_walk_forward_backtest(perturbed, suspicious_strategy, rebalance_frequency="monthly")

    pre_cutoff_original = result_original.portfolio_value.iloc[:cutoff]
    pre_cutoff_perturbed = result_perturbed.portfolio_value.iloc[:cutoff]
    pd.testing.assert_series_equal(pre_cutoff_original, pre_cutoff_perturbed)


def test_execution_lag_of_two_days_delays_weight_change():
    """Regression test for a real bug (docs/FORENSIC_AUDIT.md, Issue 5):
    with execution_lag_days=2, a trade signaled on day t must NOT affect
    the portfolio's realized return until day t+2 -- the old code updated
    `current_weights` immediately at signal time, so day t+1 already
    earned returns under the new weights, one day too early. This test
    fails against that old code and passes against the fix.
    """
    dates = pd.bdate_range("2023-01-01", periods=10)
    # Asset A always returns 0%, asset B always returns 5% -- so switching
    # 100% from A to B produces an unmistakable, easy-to-detect jump in
    # daily portfolio return exactly when the new weights take effect.
    returns = pd.DataFrame({"A": [0.0] * 10, "B": [0.05] * 10}, index=dates)

    def flip_to_all_B_on_first_call(as_of_date, historical_returns):
        return pd.Series({"A": 0.0, "B": 1.0})

    result = run_walk_forward_backtest(
        returns, flip_to_all_B_on_first_call,
        rebalance_frequency="daily", execution_lag_days=2, transaction_cost_bps=0,
    )

    # Signal is generated on day 0 (first day, i=0). With lag=2, the trade
    # must execute at day index 2, meaning day index 1's return must still
    # reflect the OLD (all-zero) weights -- i.e. zero return -- and only
    # day index 2 onward should show the 5% return from asset B.
    day_1_return = result.gross_returns.iloc[1]
    day_2_return = result.gross_returns.iloc[2]

    assert day_1_return == pytest.approx(0.0), (
        "Day 1 must still earn the OLD weights' return (0%) -- the new "
        "weights should not take effect until day 2 (signal day + lag)."
    )
    assert day_2_return == pytest.approx(0.05), (
        "Day 2 (signal day + execution_lag_days) must earn the NEW "
        "weights' return (5%) -- this is when execution actually happens."
    )


def test_transaction_cost_charged_at_execution_not_signal_date():
    """The transaction cost for a queued trade must appear in
    `transaction_costs` on the EXECUTION date, not the signal date --
    otherwise a cost computed against the signal-date portfolio value
    could differ from the true cost at the (later) execution-date value.
    """
    dates = pd.bdate_range("2023-01-01", periods=10)
    returns = pd.DataFrame({"A": [0.0] * 10, "B": [0.05] * 10}, index=dates)

    def flip_to_all_B(as_of_date, historical_returns):
        return pd.Series({"A": 0.0, "B": 1.0})

    result = run_walk_forward_backtest(
        returns, flip_to_all_B,
        rebalance_frequency="daily", execution_lag_days=3, transaction_cost_bps=50,
    )

    signal_date = dates[0]
    execution_date = dates[3]

    assert result.transaction_costs.loc[signal_date] == 0.0
    assert result.transaction_costs.loc[execution_date] > 0.0


def test_weights_history_labels_match_actual_execution_date():
    """`weights_history` must show the new weights starting exactly at the
    execution date -- not one day earlier, not one day later."""
    dates = pd.bdate_range("2023-01-01", periods=10)
    returns = pd.DataFrame({"A": [0.0] * 10, "B": [0.0] * 10}, index=dates)

    def flip_to_all_B(as_of_date, historical_returns):
        return pd.Series({"A": 0.0, "B": 1.0})

    result = run_walk_forward_backtest(
        returns, flip_to_all_B,
        rebalance_frequency="daily", execution_lag_days=2, transaction_cost_bps=0,
    )

    execution_date = dates[2]
    day_before = dates[1]

    assert result.weights_history.loc[day_before, "B"] == pytest.approx(0.0)
    assert result.weights_history.loc[execution_date, "B"] == pytest.approx(1.0)
