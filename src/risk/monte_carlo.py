"""
Monte Carlo simulation of portfolio outcomes.

Three return-generation modes:
  - "empirical": i.i.d. bootstrap resampling from actual historical daily
    returns. Does not assume normality, captures real fat tails/skew, BUT
    assumes each simulated day is independent of the last -- which
    notebook 03's own volatility-clustering chart shows is false for real
    markets (calm periods and turbulent periods each persist in runs, not
    i.i.d.). i.i.d. resampling therefore understates the chance of several
    consecutive bad days in a row, exactly the sequence that produces the
    worst drawdowns.
  - "block": moving-block bootstrap -- resamples contiguous BLOCKS of
    `block_size` consecutive historical days (with replacement across
    blocks) instead of single independent days. This preserves
    within-block autocorrelation and volatility clustering, at the cost of
    a slightly less smooth resampling distribution. This is the
    methodologically preferred empirical mode; "empirical" (block_size=1)
    is kept as a simpler, faster default and for direct comparison.
  - "normal": draws from Normal(mu, sigma) fit to historical returns. Fastest
    and simplest, but understates tail risk if the true distribution is
    fat-tailed (documented limitation, same as parametric VaR) AND assumes
    i.i.d. days like the empirical mode.

All randomness is seeded (`random_seed` from config) so results are
reproducible run-to-run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    terminal_wealth: np.ndarray       # shape (n_simulations,)
    paths: np.ndarray                 # shape (n_simulations, time_horizon_days)
    var: float
    cvar: float
    probability_of_loss: float
    percentiles: dict[str, float]


def _sample_iid(historical_returns: np.ndarray, rng: np.random.Generator, n_sims: int, horizon: int) -> np.ndarray:
    return rng.choice(historical_returns, size=(n_sims, horizon), replace=True)


def _sample_block_bootstrap(
    historical_returns: np.ndarray, rng: np.random.Generator, n_sims: int, horizon: int, block_size: int
) -> np.ndarray:
    """Moving-block bootstrap: build each simulated path by concatenating
    randomly-chosen contiguous blocks of `block_size` real historical days
    (blocks may overlap with each other across draws and across paths, and
    the final block in a path is truncated if `horizon` isn't a multiple
    of `block_size`). This preserves the day-to-day autocorrelation and
    volatility-clustering structure WITHIN a block, which i.i.d. resampling
    destroys by construction.
    """
    n_hist = len(historical_returns)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if block_size > n_hist:
        raise ValueError(f"block_size ({block_size}) cannot exceed available history ({n_hist})")

    max_start = n_hist - block_size
    n_blocks_needed = int(np.ceil(horizon / block_size))

    paths = np.empty((n_sims, n_blocks_needed * block_size))
    for sim in range(n_sims):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        blocks = [historical_returns[s : s + block_size] for s in starts]
        paths[sim] = np.concatenate(blocks)
    return paths[:, :horizon]


def run_monte_carlo(
    historical_returns: pd.Series,
    number_of_simulations: int = 10000,
    time_horizon_days: int = 20,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    distribution: str = "empirical",
    initial_value: float = 1.0,
    block_size: int = 5,
) -> MonteCarloResult:
    """Simulate `number_of_simulations` possible future paths of portfolio
    value over `time_horizon_days`, using only `historical_returns` observed
    up to the simulation date (no future information).

    `block_size` (only used when `distribution="block"`): length of each
    resampled contiguous chunk, in trading days. Default of 5 (one trading
    week) is short enough to keep plenty of distinct blocks available even
    with a few years of history, while still being long enough to capture
    a few consecutive days of the SAME volatility regime -- see this
    module's docstring and `docs/PREPROCESSING_AND_FEATURE_SELECTION.md`
    for the volatility-clustering evidence this design responds to.
    """
    rng = np.random.default_rng(random_seed)
    historical_returns = historical_returns.dropna()
    hist_values = historical_returns.values

    if distribution == "empirical":
        sampled = _sample_iid(hist_values, rng, number_of_simulations, time_horizon_days)
    elif distribution == "block":
        sampled = _sample_block_bootstrap(hist_values, rng, number_of_simulations, time_horizon_days, block_size)
    elif distribution == "normal":
        mu = historical_returns.mean()
        sigma = historical_returns.std(ddof=1)
        sampled = rng.normal(mu, sigma, size=(number_of_simulations, time_horizon_days))
    else:
        raise ValueError(f"Unknown distribution '{distribution}', expected 'empirical', 'block', or 'normal'.")

    wealth_paths = initial_value * np.cumprod(1 + sampled, axis=1)
    terminal_wealth = wealth_paths[:, -1]

    losses = initial_value - terminal_wealth
    var_threshold = np.quantile(losses, confidence_level)
    tail_losses = losses[losses >= var_threshold]
    cvar = float(tail_losses.mean()) if len(tail_losses) else float("nan")

    probability_of_loss = float((terminal_wealth < initial_value).mean())

    percentiles = {
        f"p{p}": float(np.percentile(terminal_wealth, p)) for p in (1, 5, 25, 50, 75, 95, 99)
    }

    return MonteCarloResult(
        terminal_wealth=terminal_wealth,
        paths=wealth_paths,
        var=float(var_threshold),
        cvar=cvar,
        probability_of_loss=probability_of_loss,
        percentiles=percentiles,
    )
