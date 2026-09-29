"""
Conditional Value at Risk (CVaR), a.k.a. Expected Shortfall.

CVaR = expected loss, conditional on the loss already exceeding VaR.

Why CVaR captures tail behavior better than VaR:
VaR only tells you the threshold loss at a given confidence level — it says
nothing about HOW BAD things get beyond that threshold. Two portfolios can
have identical VaR but very different tail risk (one might lose exactly the
VaR amount in its worst 5% of days, the other might lose 5x that). CVaR
averages over the whole tail, so it reflects the severity of extreme losses,
not just their frequency.

LIMITATION -- horizon scaling (see docs/FORENSIC_AUDIT.md, Issue 6): like
`var.py`'s historical/parametric VaR, `historical_cvar` scales the 1-day
estimate by `sqrt(horizon_days)`. This assumes i.i.d. returns, which real
markets violate (volatility clusters -- see
`docs/PREPROCESSING_AND_FEATURE_SELECTION.md` and the empirical evidence
in `notebooks/03_exploratory_data_analysis.ipynb`). The approximation is
arguably weaker here than for VaR: CVaR is a tail-conditional MEAN of
empirical losses, and scaling a mean-of-a-subset by `sqrt(t)` has less
direct theoretical justification than scaling a single quantile does, even
under the same i.i.d. assumption. For horizons where this matters, prefer
`src/risk/monte_carlo.py` with `distribution="block"` (moving-block
bootstrap), which does not rely on this scaling shortcut and explicitly
preserves the autocorrelation/volatility-clustering structure this
approximation ignores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_cvar(
    returns: pd.Series, confidence_level: float = 0.95, horizon_days: int = 1
) -> float:
    """Empirical CVaR: mean of returns at or below the VaR quantile,
    scaled to `horizon_days` via sqrt(time) -- see this module's
    docstring for why that scaling is an approximation, not an identity.
    """
    if returns.empty:
        return float("nan")
    quantile = 1 - confidence_level
    var_threshold = returns.quantile(quantile)
    tail_losses = returns[returns <= var_threshold]
    if tail_losses.empty:
        return float("nan")
    daily_cvar = -tail_losses.mean()
    return float(daily_cvar * np.sqrt(horizon_days))
