"""
Value at Risk (VaR).

VaR answers: "over the next `horizon` days, with `confidence_level`
probability, the portfolio will NOT lose more than X." It is a threshold,
not a worst-case bound — losses beyond VaR are exactly what CVaR (cvar.py)
measures.

Three approaches are implemented, each with its own assumptions/limitations
documented below. All three take a confidence_level and horizon (days) that
must come from config, never hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(
    returns: pd.Series, confidence_level: float = 0.95, horizon_days: int = 1
) -> float:
    """Historical VaR: the empirical quantile of past portfolio returns,
    scaled to the horizon via sqrt(time) (an approximation — see limitations).

    Assumptions: the historical return distribution is representative of the
    near future.
    Limitations: says nothing about losses more severe than the observed
    history; sqrt(time) scaling assumes i.i.d. returns, which real markets
    violate (volatility clusters).
    """
    if returns.empty:
        return float("nan")
    quantile = 1 - confidence_level
    daily_var = -returns.quantile(quantile)
    return float(daily_var * np.sqrt(horizon_days))


def parametric_var(
    returns: pd.Series, confidence_level: float = 0.95, horizon_days: int = 1
) -> float:
    """Parametric (Gaussian) VaR: assumes returns are normally distributed.

    Daily VaR = -(mu + z * sigma). For horizon_days > 1, under the same
    i.i.d.-Normal assumption, mu scales LINEARLY with time (mu_h = h*mu)
    while sigma scales with sqrt(time) (sigma_h = sqrt(h)*sigma) -- these
    are scaled separately, not as a combined expression multiplied by
    sqrt(h) (see docs/SECOND_FORENSIC_AUDIT.md, Issue C2 for a worked
    example of why that combined-then-scaled approach is mathematically
    wrong, not just an approximation: it silently used sqrt(h)*mu instead
    of h*mu, understating the multi-day drift contribution for any
    horizon_days > 1).

    Assumptions: returns are i.i.d. Normal(mu, sigma). This is a well-known
    simplification — real returns have fatter tails, so parametric VaR
    tends to UNDERSTATE tail risk relative to historical/Monte Carlo VaR.
    Limitations: does not capture skew/kurtosis; unreliable for very
    high confidence levels (e.g. 99.9%) or short, volatile histories; the
    i.i.d. assumption itself is an approximation (see historical_var's
    docstring) -- but the h-day scaling of mu and sigma below is the
    mathematically correct consequence of that assumption, not a further
    approximation on top of it.
    """
    if returns.empty:
        return float("nan")
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - confidence_level)
    mu_h = horizon_days * mu
    sigma_h = np.sqrt(horizon_days) * sigma
    return float(-(mu_h + z * sigma_h))


def var_summary(
    returns: pd.Series, confidence_level: float = 0.95, horizon_days: int = 1
) -> dict[str, float]:
    """Convenience wrapper returning both VaR estimates side by side, since
    disagreement between them is itself useful diagnostic information."""
    return {
        "historical_var": historical_var(returns, confidence_level, horizon_days),
        "parametric_var": parametric_var(returns, confidence_level, horizon_days),
        "confidence_level": confidence_level,
        "horizon_days": horizon_days,
    }
