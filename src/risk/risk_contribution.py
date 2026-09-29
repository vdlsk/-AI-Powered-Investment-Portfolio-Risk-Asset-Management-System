"""
Risk contribution: which assets actually drive portfolio risk.

Portfolio weights alone can be misleading — a 10% position in a volatile,
highly-correlated-with-everything-else asset can contribute far more to
total portfolio risk than a 30% position in a low-volatility, diversifying
asset. Risk contribution decomposes total portfolio volatility across
assets so this is visible.

Marginal Contribution to Risk (MCR_i):
    How much portfolio volatility changes for a small increase in asset i's
    weight, holding other weights fixed (in relative terms):
        MCR_i = (Sigma w)_i / sigma_p

Component Contribution to Risk (CCR_i):
    Asset i's actual share of total portfolio volatility:
        CCR_i = w_i * MCR_i
    By construction, sum_i CCR_i = sigma_p.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def marginal_contribution_to_risk(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    sigma_p = np.sqrt(weights.T @ cov_matrix @ weights)
    if sigma_p == 0:
        return np.zeros_like(weights)
    return (cov_matrix @ weights) / sigma_p


def component_contribution_to_risk(
    weights: np.ndarray, cov_matrix: np.ndarray, asset_names: list[str] | None = None
) -> pd.Series:
    mcr = marginal_contribution_to_risk(weights, cov_matrix)
    ccr = weights * mcr
    index = asset_names if asset_names is not None else range(len(weights))
    return pd.Series(ccr, index=index, name="component_contribution_to_risk")


def pct_contribution_to_risk(
    weights: np.ndarray, cov_matrix: np.ndarray, asset_names: list[str] | None = None
) -> pd.Series:
    """Component contribution as a percentage of total portfolio volatility,
    which is usually the most intuitive number to show on a dashboard."""
    ccr = component_contribution_to_risk(weights, cov_matrix, asset_names)
    sigma_p = ccr.sum()
    if sigma_p == 0:
        return ccr * 0
    return ccr / sigma_p
