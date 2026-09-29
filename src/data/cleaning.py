"""
Cleaning pipeline: raw -> clean.

Documented decisions (see also docs/DATA_LEAKAGE_AUDIT.md and README §9-10):

  * Duplicate (ticker, date) rows: keep the LAST occurrence (assumed to be the
    most recently ingested / corrected value from the provider).
  * Rows with missing close/adj_close: dropped — we cannot safely impute a
    fundamental price without inventing information.
  * Rows with non-positive prices or impossible OHLC relationships: dropped
    as data errors (a real market can produce huge moves, but never a
    negative price or High < Low).
  * Suspicious single-day returns (>25%) are NOT dropped — large real moves
    (e.g. crashes, earnings gaps) are legitimate and must remain in the
    series. They are only flagged in the validation report for manual review.
  * Small missing-volume values are forward-filled only within a ticker and
    only using PAST data (ffill), never backward-filled, to avoid leaking
    future information into a "clean" past row.
  * Adjusted close (adj_close) is the series used for all return calculations
    downstream (see README §10-11) because it accounts for splits/dividends;
    raw `close` is kept only for reference/display.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data.validation import validate_prices

logger = logging.getLogger(__name__)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a long-format OHLCV DataFrame.

    Input/output columns: ['ticker','date','open','high','low','close',
    'adj_close','volume'] (+ 'is_imputed' added on output).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    report = validate_prices(df)

    before = len(df)

    # 1) duplicates: keep last
    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )

    # 2) drop rows with missing fundamental prices
    df = df.dropna(subset=["close", "adj_close"])

    # 3) drop non-positive prices
    price_cols = ["open", "high", "low", "close", "adj_close"]
    df = df[(df[price_cols] > 0).all(axis=1)]

    # 4) drop impossible OHLC relationships
    valid_high = df["high"] >= df[["open", "close", "low"]].max(axis=1)
    valid_low = df["low"] <= df[["open", "close", "high"]].min(axis=1)
    df = df[valid_high & valid_low]

    # 5) invalid (negative) volume -> treat as missing, then forward-fill
    #    per ticker using only PAST values (never future).
    df.loc[df["volume"] < 0, "volume"] = pd.NA
    df["is_imputed"] = df["volume"].isna()
    df["volume"] = df.groupby("ticker")["volume"].ffill()
    # any still-missing volume at the very start of a series (no past value
    # to ffill from) is set to 0 rather than dropped, since volume is not
    # used for price/return calculations.
    df["volume"] = df["volume"].fillna(0)

    after = len(df)
    logger.info(
        "Cleaning: %d -> %d rows (%d dropped). See validation report for detail:\n%s",
        before, after, before - after, report.summary(),
    )

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
