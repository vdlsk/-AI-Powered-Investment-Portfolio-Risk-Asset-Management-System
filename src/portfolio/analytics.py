"""
Portfolio analytics: turn (asset returns, weights) into portfolio-level
returns and a full performance/risk report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.risk import metrics as risk_metrics
from src.risk.cvar import historical_cvar
from src.risk.var import historical_var, parametric_var

RISK_FREE_RATE = CONFIG["risk"]["risk_free_rate_annual"]
CONFIDENCE_LEVEL = CONFIG["risk"]["confidence_level"]
VAR_HORIZON = CONFIG["risk"]["var_horizon_days"]


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted sum of aligned SIMPLE asset returns -> portfolio simple return
    per date. `asset_returns` columns must match `weights.index`.
    """
    weights = weights.reindex(asset_returns.columns).fillna(0.0)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.6f}")
    return asset_returns.mul(weights, axis=1).sum(axis=1)


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    return asset_returns.corr()


def performance_report(
    returns: pd.Series,
    risk_free_rate_annual: float = RISK_FREE_RATE,
    confidence_level: float = CONFIDENCE_LEVEL,
    horizon_days: int = VAR_HORIZON,
) -> dict[str, float]:
    """Single entry point that computes every headline metric for a
    portfolio return series. This is what backs the Streamlit "Dashboard"
    and "Risk Analytics" pages.
    """
    wealth = (1 + returns).cumprod()
    return {
        "cumulative_return": risk_metrics.cumulative_return(returns),
        "annualized_return": risk_metrics.annualized_return(returns),
        "annualized_volatility": risk_metrics.annualized_volatility(returns),
        "sharpe_ratio": risk_metrics.sharpe_ratio(returns, risk_free_rate_annual),
        "sortino_ratio": risk_metrics.sortino_ratio(returns, risk_free_rate_annual),
        "max_drawdown": risk_metrics.max_drawdown(wealth),
        "calmar_ratio": risk_metrics.calmar_ratio(returns),
        "historical_var": historical_var(returns, confidence_level, horizon_days),
        "parametric_var": parametric_var(returns, confidence_level, horizon_days),
        "cvar": historical_cvar(returns, confidence_level, horizon_days),
    }
