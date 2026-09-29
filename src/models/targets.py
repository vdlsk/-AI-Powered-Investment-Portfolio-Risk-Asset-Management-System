"""
Target variable construction for the ML risk-prediction model.

CRITICAL SEPARATION (README §14-15, docs/DATA_LEAKAGE_AUDIT.md):
The target is allowed to look into the future. Features (src/features/) are
NOT. This module is intentionally kept separate from src/features/ so the
two can never be accidentally merged into one "feature+target" table that
then gets fed wholesale into a model.

Target definition (config: ml.target_horizon_days, ml.loss_threshold):

    target_t = 1  if portfolio return over (t, t+horizon] <= -loss_threshold
    target_t = 0  otherwise

The forward return is computed using price data from t+1 ... t+horizon,
which by construction is unavailable at the time a prediction for date t
would actually be made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG

TARGET_HORIZON_DAYS = CONFIG["ml"]["target_horizon_days"]
LOSS_THRESHOLD = CONFIG["ml"]["loss_threshold"]


def forward_return(returns: pd.Series, horizon_days: int = TARGET_HORIZON_DAYS) -> pd.Series:
    """Forward (future) cumulative simple return over the next `horizon_days`,
    aligned to date t: forward_return[t] = product(1+returns[t+1..t+horizon]) - 1.

    The LAST `horizon_days` rows of the series will be NaN, because their
    future window extends beyond the available data -- these rows cannot
    be used for training/evaluation and must be dropped, not imputed.
    """
    # shift(-horizon_days) on a rolling product built from t+1 onward:
    # build rolling product of (1+r) over horizon days, then align it to
    # start at t+1 by shifting the rolling result backward by horizon_days.
    growth = (1 + returns).rolling(window=horizon_days).apply(np.prod, raw=True)
    forward = growth.shift(-horizon_days) - 1
    return forward


def forward_volatility(returns: pd.Series, horizon_days: int = TARGET_HORIZON_DAYS) -> pd.Series:
    """Realized volatility over the next `horizon_days`, aligned to date t."""
    vol = returns.rolling(window=horizon_days).std()
    return vol.shift(-horizon_days)


def build_loss_event_target(
    returns: pd.Series,
    horizon_days: int = TARGET_HORIZON_DAYS,
    loss_threshold: float = LOSS_THRESHOLD,
) -> pd.Series:
    """target = 1 if the forward `horizon_days` cumulative return <= -loss_threshold."""
    fwd_ret = forward_return(returns, horizon_days)
    target = (fwd_ret <= -abs(loss_threshold)).astype("Int64")
    target[fwd_ret.isna()] = pd.NA
    return target


def build_volatility_event_target(
    returns: pd.Series,
    horizon_days: int = TARGET_HORIZON_DAYS,
    vol_multiple: float = 1.5,
) -> pd.Series:
    """target = 1 if forward realized volatility exceeds `vol_multiple` times
    the trailing (historical, non-leaking) volatility over the same window
    length, ending at t.
    """
    trailing_vol = returns.rolling(window=horizon_days).std()
    fwd_vol = forward_volatility(returns, horizon_days)
    threshold = trailing_vol * vol_multiple
    target = (fwd_vol > threshold).astype("Int64")
    target[fwd_vol.isna() | threshold.isna()] = pd.NA
    return target
