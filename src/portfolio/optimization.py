"""
Portfolio optimization (Modern Portfolio Theory).

IMPORTANT (see README §64 / docs/DATA_LEAKAGE_AUDIT.md):
When this module is used for BACKTESTING, `mean_returns` and `cov_matrix`
passed in must be estimated using ONLY data available up to time t — never
the full historical dataset. This module itself is agnostic to that; the
caller is responsible for the correct estimation window. When used for a
current, exploratory "what should my portfolio look like today" analysis,
using the full available history is fine and should be clearly labeled as
such in the UI (not presented as a historical backtest).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import CONFIG
from src.risk.metrics import portfolio_expected_return, portfolio_variance, portfolio_volatility

MAX_WEIGHT = CONFIG["portfolio"]["max_weight"]
MIN_WEIGHT = CONFIG["portfolio"]["min_weight"]


@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str


_SOLVER_OPTIONS = {"maxiter": 1000, "ftol": 1e-12}


def _bounds(n_assets: int, max_weight: float, min_weight: float):
    return tuple((min_weight, max_weight) for _ in range(n_assets))


def _weights_sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _feasible_initial_weights(n: int, min_weight: float, max_weight: float) -> np.ndarray:
    """Equal-split initial guess, water-filled so it always satisfies both
    the per-asset bounds AND sums to 1 (an equal 1/n split can itself
    violate max_weight, e.g. n=3 and max_weight=0.30, which would hand the
    solver an infeasible starting point and make SLSQP fail immediately).
    """
    if max_weight * n < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible constraints: max_weight={max_weight} * n_assets={n} < 1.0"
        )
    w = np.full(n, min_weight, dtype=float)
    remaining = 1.0 - w.sum()
    capacity = max_weight - min_weight
    active = np.ones(n, dtype=bool)
    # distribute `remaining` equally among assets not yet at their cap,
    # repeating until nothing is left to distribute (simple water-filling).
    for _ in range(n):
        if remaining <= 1e-12 or not active.any():
            break
        share = remaining / active.sum()
        for i in np.where(active)[0]:
            room = capacity - (w[i] - min_weight)
            add = min(share, room)
            w[i] += add
            remaining -= add
            if add < share - 1e-12:
                active[i] = False
    return w


def minimum_variance_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    risk_free_rate: float = 0.0,
) -> OptimizationResult:
    n = len(mean_returns)
    x0 = _feasible_initial_weights(n, min_weight, max_weight)
    bounds = _bounds(n, max_weight, min_weight)
    constraints = [_weights_sum_to_one()]

    result = minimize(
        lambda w: portfolio_variance(w, cov_matrix.values) * 1e4,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SOLVER_OPTIONS,
    )
    return _to_result(result, mean_returns, cov_matrix, risk_free_rate)


def maximum_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
) -> OptimizationResult:
    n = len(mean_returns)
    x0 = _feasible_initial_weights(n, min_weight, max_weight)
    bounds = _bounds(n, max_weight, min_weight)
    constraints = [_weights_sum_to_one()]

    def neg_sharpe(w):
        ret = portfolio_expected_return(w, mean_returns.values)
        vol = portfolio_volatility(w, cov_matrix.values)
        if vol == 0:
            return 1e6
        return -(ret - risk_free_rate) / vol * 100  # scaled for solver conditioning

    result = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options=_SOLVER_OPTIONS,
    )
    return _to_result(result, mean_returns, cov_matrix, risk_free_rate)


def target_return_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: float,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    risk_free_rate: float = 0.0,
) -> OptimizationResult:
    n = len(mean_returns)
    x0 = _feasible_initial_weights(n, min_weight, max_weight)
    bounds = _bounds(n, max_weight, min_weight)
    constraints = [
        _weights_sum_to_one(),
        {"type": "ineq", "fun": lambda w: portfolio_expected_return(w, mean_returns.values) - target_return},
    ]

    result = minimize(
        lambda w: portfolio_variance(w, cov_matrix.values) * 1e4,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SOLVER_OPTIONS,
    )
    return _to_result(result, mean_returns, cov_matrix, risk_free_rate)


def _to_result(result, mean_returns, cov_matrix, risk_free_rate) -> OptimizationResult:
    weights = pd.Series(result.x, index=mean_returns.index)
    ret = portfolio_expected_return(weights.values, mean_returns.values)
    vol = portfolio_volatility(weights.values, cov_matrix.values)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return OptimizationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        success=bool(result.success),
        message=str(result.message),
    )


def _feasible_return_range(
    mean_returns: pd.Series, max_weight: float, min_weight: float
) -> tuple[float, float]:
    """The true achievable expected-return range under the weight bounds
    (long-only, `min_weight <= w_i <= max_weight`, `sum(w) = 1`), found by
    solving two linear programs: maximize and minimize `w @ mean_returns`
    subject to those exact constraints.

    WHY THIS EXISTS (see docs/SECOND_FORENSIC_AUDIT.md, Issue H8):
    `efficient_frontier` previously swept target returns between
    `mean_returns.min()` and `mean_returns.max()` -- the single lowest/
    highest INDIVIDUAL asset's own return, ignoring `max_weight` entirely.
    If `max_weight < 1.0`, no feasible portfolio can actually achieve
    `mean_returns.max()` (that would require 100% allocation to one
    asset), so a chunk of the requested sweep silently produced
    infeasible targets that got filtered out -- wasting part of the
    requested point count on grid positions that could never succeed.
    Solving for the true feasible range up front means (almost) every
    requested point is achievable.
    """
    from scipy.optimize import linprog

    n = len(mean_returns)
    bounds = [(min_weight, max_weight)] * n
    A_eq = [np.ones(n)]
    b_eq = [1.0]

    # maximize w @ mean_returns  <=>  minimize -(w @ mean_returns)
    res_max = linprog(c=-mean_returns.values, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    res_min = linprog(c=mean_returns.values, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res_max.success or not res_min.success:
        raise ValueError(
            f"Could not determine a feasible expected-return range for "
            f"max_weight={max_weight}, min_weight={min_weight} with "
            f"{n} assets -- the constraints may be infeasible "
            f"(e.g. max_weight * n_assets < 1.0)."
        )
    return float(res_min.fun), float(-res_max.fun)


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 40,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Trace the efficient frontier by solving target-return portfolios
    across a range of GENUINELY FEASIBLE target returns (see
    `_feasible_return_range` above -- this replaces the old
    `mean_returns.min()/.max()` range, which ignored `max_weight` and
    silently wasted part of the sweep on infeasible targets)."""
    lo, hi = _feasible_return_range(mean_returns, max_weight, min_weight)
    targets = np.linspace(lo, hi, n_points)
    rows = []
    for t in targets:
        res = target_return_portfolio(
            mean_returns, cov_matrix, t, max_weight, min_weight, risk_free_rate
        )
        if res.success:
            rows.append(
                {"target_return": t, "expected_return": res.expected_return,
                 "volatility": res.volatility, "sharpe_ratio": res.sharpe_ratio}
            )
    return pd.DataFrame(rows)


def random_feasible_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.0,
    random_seed: int = 42,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_WEIGHT,
) -> pd.DataFrame:
    """Random long-only portfolios for scatter-plotting the feasible region
    behind the efficient frontier (visualization only, not optimization).

    Uses REJECTION SAMPLING from the Dirichlet distribution: draws are
    discarded if any weight violates `max_weight`/`min_weight`, until
    `n_portfolios` valid samples are collected (or a generous attempt cap
    is hit). See docs/SECOND_FORENSIC_AUDIT.md, Issue H7: this function's
    own name and its use alongside the (genuinely constrained) optimizer
    outputs on the dashboard's efficient-frontier plot implied these
    points were achievable under the same constraints -- a plain
    Dirichlet(1,...,1) draw does not respect `max_weight` at all, and with
    few assets frequently produces samples where a single asset exceeds
    any reasonable concentration cap.
    """
    n_assets = len(mean_returns)
    if max_weight * n_assets < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible constraints: max_weight={max_weight} * n_assets={n_assets} < 1.0"
        )

    rng = np.random.default_rng(random_seed)
    accepted: list[np.ndarray] = []
    max_attempts = n_portfolios * 200  # generous cap so a very tight
    # (but technically feasible) constraint region doesn't hang forever
    attempts = 0
    batch_size = min(2000, max(n_portfolios, 100))

    while len(accepted) < n_portfolios and attempts < max_attempts:
        batch = rng.dirichlet(np.ones(n_assets), size=batch_size)
        attempts += len(batch)
        mask = (batch >= min_weight - 1e-9).all(axis=1) & (batch <= max_weight + 1e-9).all(axis=1)
        accepted.extend(batch[mask])

    if len(accepted) < n_portfolios:
        import logging

        logging.getLogger(__name__).warning(
            "random_feasible_portfolios: only found %d/%d samples satisfying "
            "min_weight=%.3f/max_weight=%.3f after %d attempts -- the "
            "feasible region may be small relative to the Dirichlet "
            "distribution's typical output.",
            len(accepted), n_portfolios, min_weight, max_weight, attempts,
        )

    if not accepted:
        return pd.DataFrame(columns=["expected_return", "volatility", "sharpe_ratio"])

    weights = np.array(accepted[:n_portfolios])
    rets = weights @ mean_returns.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix.values, weights))
    sharpes = (rets - risk_free_rate) / np.where(vols == 0, np.nan, vols)

    return pd.DataFrame({"expected_return": rets, "volatility": vols, "sharpe_ratio": sharpes})
