"""
Return calculations.

Simple return:  R_t = P_t / P_(t-1) - 1
    Used for: portfolio-level aggregation (portfolio returns are a
    weighted average of SIMPLE asset returns, not log returns).

Log return:      r_t = ln(P_t / P_(t-1))
    Used for: statistical modelling / volatility estimation, because log
    returns are additive across time (r_t + r_(t+1) = cumulative log return),
    which makes rolling-window statistics much cleaner.

All functions operate on adjusted close prices (see cleaning.py) so that
splits/dividends don't masquerade as price moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """R_t = P_t / P_(t-1) - 1. First value is NaN (no P_(t-1))."""
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_(t-1)). First value is NaN."""
    return np.log(prices / prices.shift(1))


def multi_day_return(prices: pd.Series, days: int) -> pd.Series:
    """Simple return over `days` trading days, as of each date t:
    R = P_t / P_(t-days) - 1. Uses only data up to and including t.
    """
    return prices.pct_change(periods=days)


def align_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Given a wide price panel (index=date, columns=ticker), return a wide
    panel of simple returns aligned by date. Dates where any asset is
    missing are dropped so portfolio-level math never silently uses
    misaligned data.
    """
    returns = price_panel.pct_change()
    return returns.dropna(how="any")
