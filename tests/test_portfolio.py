import numpy as np
import pandas as pd
import pytest

from src.portfolio.analytics import portfolio_returns
from src.portfolio.optimization import (
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    target_return_portfolio,
)
from src.portfolio.rebalancing import rebalance_plan


@pytest.fixture
def toy_returns():
    dates = pd.bdate_range("2023-01-01", periods=100)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.01, 100),
            "B": rng.normal(0.0005, 0.02, 100),
            "C": rng.normal(0.0003, 0.005, 100),
        },
        index=dates,
    )


def test_portfolio_returns_weighted_correctly(toy_returns):
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    result = portfolio_returns(toy_returns, weights)
    manual = 0.5 * toy_returns["A"] + 0.3 * toy_returns["B"] + 0.2 * toy_returns["C"]
    pd.testing.assert_series_equal(result, manual, check_names=False)


def test_portfolio_returns_rejects_weights_not_summing_to_one(toy_returns):
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.3})  # sums to 1.1
    with pytest.raises(ValueError):
        portfolio_returns(toy_returns, weights)


def test_minimum_variance_weights_sum_to_one_and_respect_bounds(toy_returns):
    mean_returns = toy_returns.mean()
    cov = toy_returns.cov()
    result = minimum_variance_portfolio(mean_returns, cov, max_weight=0.6, min_weight=0.0)
    assert result.success
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)
    assert (result.weights >= -1e-6).all()
    assert (result.weights <= 0.6 + 1e-6).all()


def test_maximum_sharpe_portfolio_is_feasible(toy_returns):
    mean_returns = toy_returns.mean()
    cov = toy_returns.cov()
    result = maximum_sharpe_portfolio(mean_returns, cov, risk_free_rate=0.0001, max_weight=0.6)
    assert result.success
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)


def test_target_return_portfolio_meets_or_exceeds_target(toy_returns):
    mean_returns = toy_returns.mean()
    cov = toy_returns.cov()
    target = mean_returns.mean()
    result = target_return_portfolio(mean_returns, cov, target_return=target, max_weight=0.6)
    assert result.success
    assert result.expected_return >= target - 1e-6


def test_rebalance_plan_classifies_actions_correctly():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.3, "B": 0.7})
    plan = rebalance_plan(current, target, portfolio_value=10_000)
    assert plan.loc["A", "action"] == "SELL"
    assert plan.loc["B", "action"] == "BUY"
    assert plan["estimated_transaction_cost"].sum() > 0


def test_random_feasible_portfolios_respects_weight_bounds(toy_returns):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H7:
    random_feasible_portfolios must actually respect max_weight/min_weight,
    not just sum(weights)=1 and weights>=0 (plain Dirichlet)."""
    from src.portfolio.optimization import random_feasible_portfolios

    mean_returns = toy_returns.mean()
    cov = toy_returns.cov()
    max_weight = 0.5  # tight enough that plain Dirichlet(1,1,1) would frequently violate this

    result = random_feasible_portfolios(
        mean_returns, cov, n_portfolios=500, max_weight=max_weight, min_weight=0.0, random_seed=1,
    )
    assert len(result) > 0
    # We can't directly see the underlying weights from the returned
    # DataFrame (only aggregated return/vol/sharpe), so instead verify
    # indirectly: no expected_return should exceed what's achievable
    # under max_weight (the same feasible-return-range helper used by
    # efficient_frontier), which would only be possible if some sampled
    # weight vector violated the constraint.
    from src.portfolio.optimization import _feasible_return_range
    lo, hi = _feasible_return_range(mean_returns, max_weight, 0.0)
    assert result["expected_return"].max() <= hi + 1e-6
    assert result["expected_return"].min() >= lo - 1e-6


def test_random_feasible_portfolios_raises_on_infeasible_constraints(toy_returns):
    from src.portfolio.optimization import random_feasible_portfolios

    mean_returns = toy_returns.mean()
    cov = toy_returns.cov()
    with pytest.raises(ValueError, match="Infeasible constraints"):
        random_feasible_portfolios(mean_returns, cov, max_weight=0.2, n_portfolios=10)  # 0.2*3=0.6 < 1.0


def test_efficient_frontier_uses_genuinely_feasible_target_range(toy_returns):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H8: the
    frontier's target-return sweep must be the actual achievable range
    under max_weight, not the raw min/max of individual asset returns."""
    from src.portfolio.optimization import efficient_frontier, _feasible_return_range

    mean_returns = toy_returns.mean()
    cov = toy_returns.cov()
    max_weight = 0.5

    lo, hi = _feasible_return_range(mean_returns, max_weight, 0.0)
    # The naive (wrong) range would have used mean_returns.min()/.max().
    naive_lo, naive_hi = mean_returns.min(), mean_returns.max()
    assert hi < naive_hi  # max_weight makes the true ceiling strictly lower

    frontier = efficient_frontier(mean_returns, cov, n_points=20, max_weight=max_weight, min_weight=0.0)
    # Nearly every requested point should be feasible now (the old
    # behavior wasted a chunk of points near the infeasible naive_hi).
    assert len(frontier) >= 18  # allow for a couple of solver edge-case misses at the boundary
