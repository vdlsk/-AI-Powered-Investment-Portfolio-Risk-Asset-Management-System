"""
Data quality checks for raw market prices.

Philosophy: never silently delete anomalies. Every check returns a report
describing *what* looks wrong, and the caller (cleaning.py) decides whether
it's a data error to fix/drop or a legitimate market event to keep and flag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A single-day move larger than this is *flagged for review*, not deleted —
# real crashes (e.g. 2020-03-16, -12% on SPX) are legitimate and must survive.
SUSPICIOUS_RETURN_THRESHOLD = 0.25


@dataclass
class ValidationReport:
    duplicate_asset_date: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    negative_or_zero_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    impossible_ohlc: pd.DataFrame = field(default_factory=pd.DataFrame)
    invalid_volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    suspicious_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    non_monotonic_dates: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = []
        for field_name in self.__dataclass_fields__:
            n = len(getattr(self, field_name))
            if n:
                lines.append(f"{field_name}: {n} row(s)")
        return "\n".join(lines) if lines else "No data quality issues found."


def validate_prices(df: pd.DataFrame) -> ValidationReport:
    """Run all data quality checks on a long-format price DataFrame with
    columns ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'].
    """
    report = ValidationReport()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # duplicate (ticker, date)
    dup_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    report.duplicate_asset_date = df[dup_mask]

    # missing / non-positive prices
    price_cols = ["open", "high", "low", "close", "adj_close"]
    report.missing_prices = df[df[price_cols].isna().any(axis=1)]
    report.negative_or_zero_prices = df[(df[price_cols] <= 0).any(axis=1)]

    # impossible OHLC relationships: High >= max(O,C,L), Low <= min(O,C,H)
    valid_high = df["high"] >= df[["open", "close", "low"]].max(axis=1)
    valid_low = df["low"] <= df[["open", "close", "high"]].min(axis=1)
    report.impossible_ohlc = df[~(valid_high & valid_low)]

    # invalid volume
    report.invalid_volume = df[df["volume"] < 0]

    # suspicious single-day returns (per ticker), computed on adj_close
    df_sorted = df.sort_values(["ticker", "date"])
    df_sorted["_ret"] = df_sorted.groupby("ticker")["adj_close"].pct_change()
    report.suspicious_returns = df_sorted[df_sorted["_ret"].abs() > SUSPICIOUS_RETURN_THRESHOLD]

    # non-monotonic dates within a ticker (should never happen if source is sane)
    non_monotonic_rows = []
    for ticker, g in df.groupby("ticker"):
        g_sorted_idx = g["date"].values
        if not np.all(np.diff(g_sorted_idx.astype("datetime64[ns]").astype(np.int64)) >= 0):
            non_monotonic_rows.append(ticker)
    if non_monotonic_rows:
        report.non_monotonic_dates = df[df["ticker"].isin(non_monotonic_rows)]

    # missing trading dates: gaps larger than a long weekend/holiday run (>4 calendar days)
    missing_frames = []
    for ticker, g in df.sort_values("date").groupby("ticker"):
        gaps = g["date"].diff().dt.days
        gap_rows = g[gaps > 4]
        if not gap_rows.empty:
            missing_frames.append(gap_rows.assign(ticker=ticker))
    if missing_frames:
        report.missing_dates = pd.concat(missing_frames)

    logger.info("Validation summary:\n%s", report.summary())
    return report
