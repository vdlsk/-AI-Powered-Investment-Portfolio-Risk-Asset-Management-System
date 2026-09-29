"""
Volatility calculations.

All rolling volatility features here are computed on RETURNS that have
already been shifted where needed by the caller (see feature_pipeline.py) —
this module itself just computes a plain rolling std; the leakage-safety
comes from *how* it is called (shift-then-roll), not from logic inside these
pure functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of returns over `window` trading days,
    using only the current and prior `window - 1` observations."""
    return returns.rolling(window=window, min_periods=window).std()


def annualize_volatility(daily_vol: float, trading_days_per_year: int = 252) -> float:
    return daily_vol * np.sqrt(trading_days_per_year)


def annualize_return(daily_return_mean: float, trading_days_per_year: int = 252) -> float:
    """Annualize a mean daily simple return via compounding."""
    return (1 + daily_return_mean) ** trading_days_per_year - 1
