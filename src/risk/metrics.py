"""
Core financial risk/performance metrics.

For every metric: Definition -> Formula -> Implementation -> Interpretation
is documented in docs/METRICS.md as well as inline here. All annualization
uses `trading_days_per_year` from config/config.yaml (default 252) — never
hard-coded inline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG

TRADING_DAYS = CONFIG["general"]["trading_days_per_year"]


def portfolio_expected_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """E(Rp) = w^T mu"""
    return float(np.dot(weights, mean_returns))


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """sigma_p^2 = w^T Sigma w"""
    return float(weights.T @ cov_matrix @ weights)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """sigma_p = sqrt(w^T Sigma w)"""
    return float(np.sqrt(max(portfolio_variance(weights, cov_matrix), 0.0)))


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the whole series."""
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """CAGR-style annualization from a series of periodic (daily) simple returns."""
    n_periods = len(returns)
    if n_periods == 0:
        return float("nan")
    growth = (1 + returns).prod()
    years = n_periods / trading_days
    return float(growth ** (1 / years) - 1) if years > 0 else float("nan")


def annualized_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualized volatility = daily volatility * sqrt(trading_days)."""
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate_annual: float, trading_days: int = TRADING_DAYS
) -> float:
    """(Annualized return - risk-free rate) / annualized volatility."""
    ann_ret = annualized_return(returns, trading_days)
    ann_vol = annualized_volatility(returns, trading_days)
    if ann_vol == 0 or np.isnan(ann_vol):
        return float("nan")
    return (ann_ret - risk_free_rate_annual) / ann_vol


def sortino_ratio(
    returns: pd.Series, risk_free_rate_annual: float, trading_days: int = TRADING_DAYS
) -> float:
    """Like Sharpe, but the denominator only penalizes downside volatility
    (volatility of returns below zero), not upside swings.
    """
    ann_ret = annualized_return(returns, trading_days)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("nan")
    downside_dev = downside.std(ddof=1) * np.sqrt(trading_days)
    if downside_dev == 0 or np.isnan(downside_dev):
        return float("nan")
    return (ann_ret - risk_free_rate_annual) / downside_dev


def max_drawdown(cumulative_wealth: pd.Series) -> float:
    """Largest peak-to-trough decline in a cumulative wealth series.

    `cumulative_wealth` should be a running product of (1 + returns), i.e.
    portfolio value normalized to start at 1.0 (or actual portfolio value).
    """
    running_max = cumulative_wealth.cummax()
    drawdown = cumulative_wealth / running_max - 1
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualized return / abs(max drawdown). Rewards return per unit of the
    worst historical loss, rather than per unit of average volatility.
    """
    wealth = (1 + returns).cumprod()
    mdd = max_drawdown(wealth)
    ann_ret = annualized_return(returns, trading_days)
    if mdd == 0:
        return float("nan")
    return ann_ret / abs(mdd)


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Cov(asset, market) / Var(market), aligned on shared dates only."""
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    return float(cov / var) if var != 0 else float("nan")
