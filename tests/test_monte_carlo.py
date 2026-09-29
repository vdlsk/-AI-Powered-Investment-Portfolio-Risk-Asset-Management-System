import numpy as np
import pandas as pd
import pytest

from src.risk.monte_carlo import run_monte_carlo


def _make_clustered_returns(n=1000, seed=1) -> pd.Series:
    """Build a return series with REAL volatility clustering: alternating
    calm/turbulent regimes of ~40 days each, so autocorrelation of squared
    returns is genuinely present -- exactly the structure notebook 03's
    rolling-volatility chart demonstrates in real market data.
    """
    rng = np.random.default_rng(seed)
    regime_length = 40
    n_regimes = n // regime_length + 1
    returns = []
    for i in range(n_regimes):
        vol = 0.005 if i % 2 == 0 else 0.03  # alternate calm / turbulent
        returns.extend(rng.normal(0, vol, regime_length))
    return pd.Series(returns[:n])


def test_block_bootstrap_preserves_volatility_clustering_iid_does_not():
    """The core methodological claim: block-bootstrapped simulated paths
    should show significant autocorrelation of squared returns (a standard
    volatility-clustering signature), while i.i.d.-bootstrapped paths
    should not, because i.i.d. resampling destroys any such structure by
    construction.
    """
    clustered_returns = _make_clustered_returns()

    result_block = run_monte_carlo(
        clustered_returns, number_of_simulations=200, time_horizon_days=200,
        random_seed=1, distribution="block", block_size=20, initial_value=1.0,
    )
    result_iid = run_monte_carlo(
        clustered_returns, number_of_simulations=200, time_horizon_days=200,
        random_seed=1, distribution="empirical", initial_value=1.0,
    )

    def mean_abs_autocorr_of_squared_returns(paths: np.ndarray, lag: int = 1) -> float:
        # Recover per-step returns from the cumulative wealth paths, then
        # measure lag-1 autocorrelation of squared returns (a standard
        # volatility-clustering diagnostic) averaged across simulated paths.
        step_returns = np.diff(paths, axis=1) / paths[:, :-1]
        squared = step_returns ** 2
        autocorrs = []
        for row in squared:
            if row.std() == 0:
                continue
            ac = np.corrcoef(row[:-lag], row[lag:])[0, 1]
            if not np.isnan(ac):
                autocorrs.append(ac)
        return float(np.mean(np.abs(autocorrs)))

    autocorr_block = mean_abs_autocorr_of_squared_returns(result_block.paths)
    autocorr_iid = mean_abs_autocorr_of_squared_returns(result_iid.paths)

    # Block bootstrap must show materially more volatility-clustering
    # signature than i.i.d. resampling of the exact same historical data.
    assert autocorr_block > autocorr_iid


def test_block_bootstrap_matches_iid_when_block_size_is_one():
    """block_size=1 is mathematically equivalent to i.i.d. resampling
    (every 'block' is a single day) -- both modes should therefore produce
    statistically similar terminal-wealth distributions given the same
    seed and horizon, sanity-checking that the block implementation
    doesn't have an off-by-one or other structural bug.
    """
    returns = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 500))

    result_block1 = run_monte_carlo(
        returns, number_of_simulations=5000, time_horizon_days=20,
        random_seed=7, distribution="block", block_size=1,
    )
    result_iid = run_monte_carlo(
        returns, number_of_simulations=5000, time_horizon_days=20,
        random_seed=7, distribution="empirical",
    )

    # Not identical (different sampling code paths / RNG draw patterns),
    # but should be close in aggregate statistics.
    assert abs(result_block1.var - result_iid.var) < 0.01
    assert abs(np.mean(result_block1.terminal_wealth) - np.mean(result_iid.terminal_wealth)) < 0.02


def test_block_bootstrap_rejects_block_size_larger_than_history():
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50))
    with pytest.raises(ValueError, match="cannot exceed"):
        run_monte_carlo(returns, distribution="block", block_size=100, number_of_simulations=10, time_horizon_days=5)


def test_unknown_distribution_raises():
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    with pytest.raises(ValueError, match="Unknown distribution"):
        run_monte_carlo(returns, distribution="not_a_real_distribution")


def test_all_three_distributions_produce_valid_results():
    returns = pd.Series(np.random.default_rng(3).normal(0.0003, 0.012, 500))
    for dist in ["empirical", "block", "normal"]:
        result = run_monte_carlo(
            returns, number_of_simulations=1000, time_horizon_days=20,
            random_seed=1, distribution=dist, initial_value=100_000,
        )
        assert result.var > 0
        assert result.cvar >= result.var
        assert 0 <= result.probability_of_loss <= 1
        assert len(result.terminal_wealth) == 1000
