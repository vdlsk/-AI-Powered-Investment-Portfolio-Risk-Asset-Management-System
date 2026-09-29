"""
Leakage-safe feature pipeline.

THE CENTRAL RULE (see docs/DATA_LEAKAGE_AUDIT.md for full discussion):

    A feature stored for date t must only use information available at or
    before the close of day t.

Every feature in FEATURE_SPEC below is defined with:
    name              - column name
    formula           - human-readable definition
    lookback_days     - how many past days of data it uses
    information_cutoff- "close of day t" for all features in this module
    leakage_risk      - what would go wrong if implemented naively

We intentionally do NOT use `.shift(1)` before computing e.g. `ret_1d`,
because `ret_1d` at date t is DEFINED as the return realized ON day t
(P_t / P_(t-1) - 1), which is fully known by the close of day t — that is
the correct "information available at t" feature, not a leak.

Where a feature genuinely needs the *previous* day's rolling stat (e.g. an
oscillator computed only from data strictly before t), that is called out
explicitly with `.shift(1)` and a comment. In this project's default design
every feature listed uses data <= t and is safe to use for a same-day
snapshot; the ML target (built in src/models, phase 2) then looks at
t+1 ... t+horizon, which is what actually needs to stay out of the feature
set. See test_no_leakage.py for the automated check.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.returns import log_returns, multi_day_return, simple_returns
from src.features.volatility import rolling_volatility

logger = logging.getLogger(__name__)

FEATURE_SPEC = {
    "ret_1d":  {"formula": "P_t/P_(t-1) - 1", "lookback_days": 1,  "leakage_risk": "none: uses only t and t-1"},
    "ret_5d":  {"formula": "P_t/P_(t-5) - 1", "lookback_days": 5,  "leakage_risk": "none"},
    "ret_10d": {"formula": "P_t/P_(t-10) - 1","lookback_days": 10, "leakage_risk": "none"},
    "ret_20d": {"formula": "P_t/P_(t-20) - 1","lookback_days": 20, "leakage_risk": "none"},
    "ret_60d": {"formula": "P_t/P_(t-60) - 1","lookback_days": 60, "leakage_risk": "none"},
    "vol_5d":  {"formula": "std(daily log returns, 5d window)",  "lookback_days": 5,  "leakage_risk": "none: rolling window ends at t"},
    "vol_10d": {"formula": "std(daily log returns, 10d window)", "lookback_days": 10, "leakage_risk": "none"},
    "vol_20d": {"formula": "std(daily log returns, 20d window)", "lookback_days": 20, "leakage_risk": "none"},
    "vol_60d": {"formula": "std(daily log returns, 60d window)", "lookback_days": 60, "leakage_risk": "none"},
    "vol_ratio_20_60": {"formula": "vol_20d / vol_60d", "lookback_days": 60, "leakage_risk": "none: ratio of two trailing windows, both ending at t"},
    "momentum_20d":   {"formula": "P_t / mean(P_(t-20..t)) - 1", "lookback_days": 20, "leakage_risk": "none"},
    "ma_ratio_20_60": {"formula": "MA20_t / MA60_t",             "lookback_days": 60, "leakage_risk": "none"},
    "rsi_14":         {"formula": "standard 14-day RSI",         "lookback_days": 14, "leakage_risk": "none"},
    "rolling_drawdown": {"formula": "P_t / rolling_max(P, 252d) - 1", "lookback_days": 252, "leakage_risk": "none: rolling max is trailing, not centered"},
    "momentum_vol_interaction": {"formula": "momentum_20d * vol_20d", "lookback_days": 20, "leakage_risk": "none: product of two features already using only data <= t"},
    "beta_60d":       {"formula": "cov(asset, market)/var(market), 60d window", "lookback_days": 60, "leakage_risk": "none"},
    "corr_market_60d": {"formula": "corr(asset returns, market returns), 60d window", "lookback_days": 60, "leakage_risk": "none"},
    "volume_zscore_20d": {"formula": "(vol_t - mean(vol,20d)) / std(vol,20d)", "lookback_days": 20, "leakage_risk": "none"},
}


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features_for_asset(
    prices: pd.DataFrame, market_prices: pd.Series | None = None
) -> pd.DataFrame:
    """Build all features for a single asset.

    `prices` must be sorted by date and contain columns ['date', 'adj_close', 'volume'].
    `market_prices` (optional) is a date-indexed Series of a market index's
    adj_close, used for beta/correlation features. It must be pre-aligned
    to the same trading calendar; no forward-looking alignment is performed
    here.
    """
    df = prices.sort_values("date").set_index("date").copy()
    px = df["adj_close"]
    vol = df["volume"]
    log_ret = log_returns(px)

    feats = pd.DataFrame(index=df.index)
    feats["ret_1d"] = simple_returns(px)
    feats["ret_5d"] = multi_day_return(px, 5)
    feats["ret_10d"] = multi_day_return(px, 10)
    feats["ret_20d"] = multi_day_return(px, 20)
    feats["ret_60d"] = multi_day_return(px, 60)

    feats["vol_5d"] = rolling_volatility(log_ret, 5)
    feats["vol_10d"] = rolling_volatility(log_ret, 10)
    feats["vol_20d"] = rolling_volatility(log_ret, 20)
    feats["vol_60d"] = rolling_volatility(log_ret, 60)
    # Volatility-regime-shift signal: ratio of short- to long-term trailing
    # volatility. >1 means recent volatility is elevated relative to the
    # longer baseline (a regime-shift-toward-risk signal); <1 means recent
    # calm relative to the baseline. Both terms are trailing windows ending
    # at t, so the ratio itself introduces no additional leakage.
    feats["vol_ratio_20_60"] = feats["vol_20d"] / feats["vol_60d"].replace(0, np.nan)

    ma20 = px.rolling(20, min_periods=20).mean()
    ma60 = px.rolling(60, min_periods=60).mean()
    feats["momentum_20d"] = px / ma20 - 1
    feats["ma_ratio_20_60"] = ma20 / ma60

    feats["rsi_14"] = _rsi(px, 14)

    rolling_max_252 = px.rolling(252, min_periods=1).max()
    feats["rolling_drawdown"] = px / rolling_max_252 - 1

    # Interaction term: momentum alone and volatility alone are each
    # individually weak predictors of stress; their PRODUCT captures the
    # "strong negative momentum during a high-volatility period" pattern
    # that a linear model (Logistic Regression) cannot construct on its own
    # from the two features separately, since it can only combine features
    # additively -- tree-based models (Random Forest/HistGradientBoostingClassifier) can discover
    # this interaction on their own via splits, but providing it explicitly
    # helps the linear baseline too and costs nothing extra to compute.
    feats["momentum_vol_interaction"] = feats["momentum_20d"] * feats["vol_20d"]

    mean_vol_20 = vol.rolling(20, min_periods=20).mean()
    std_vol_20 = vol.rolling(20, min_periods=20).std()
    feats["volume_zscore_20d"] = (vol - mean_vol_20) / std_vol_20.replace(0, np.nan)

    if market_prices is not None:
        market_ret = log_returns(market_prices).reindex(log_ret.index)
        cov = log_ret.rolling(60, min_periods=60).cov(market_ret)
        var_mkt = market_ret.rolling(60, min_periods=60).var()
        feats["beta_60d"] = cov / var_mkt.replace(0, np.nan)
        feats["corr_market_60d"] = log_ret.rolling(60, min_periods=60).corr(market_ret)
    else:
        feats["beta_60d"] = np.nan
        feats["corr_market_60d"] = np.nan

    feats = feats.reset_index()
    return feats


def build_features(
    price_panel: pd.DataFrame, market_ticker: str | None = None
) -> pd.DataFrame:
    """Build features for every ticker in a long-format price panel.

    `price_panel` columns: ['ticker', 'date', 'adj_close', 'volume'].
    Returns a long-format DataFrame: ['ticker', 'date', <feature columns>].
    """
    market_prices = None
    if market_ticker is not None and market_ticker in price_panel["ticker"].unique():
        market_prices = (
            price_panel[price_panel["ticker"] == market_ticker]
            .set_index("date")["adj_close"]
        )

    all_feats = []
    for ticker, g in price_panel.groupby("ticker"):
        f = build_features_for_asset(g, market_prices)
        f["ticker"] = ticker
        all_feats.append(f)

    result = pd.concat(all_feats, ignore_index=True)
    logger.info("Built features for %d tickers, %d rows", price_panel["ticker"].nunique(), len(result))
    return result
