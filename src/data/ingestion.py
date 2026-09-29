"""
Data ingestion layer.

Defines a `DataProvider` abstraction so the rest of the codebase never talks
to a specific external API directly. Today only `MarketDataProvider` (backed
by yfinance) is implemented; `MacroDataProvider` and `VolatilityDataProvider`
are stubbed for future extension without breaking callers.

Usage:
    python -m src.data.ingestion --tickers AAPL MSFT GLD BND --start 2015-01-01
"""
from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from src.config import CONFIG, setup_logging
from src.data.repository import Repository

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Base interface every concrete data source must implement."""

    @abstractmethod
    def fetch(self, tickers: list[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
        """Return a long-format DataFrame with at least:
        ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        """
        raise NotImplementedError


class MarketDataProvider(DataProvider):
    """OHLCV + adjusted close via yfinance.

    Swappable: replace this class (same interface) to point at a different
    vendor without touching validation/cleaning/feature code downstream.
    """

    SOURCE_NAME = "yfinance"

    def fetch(self, tickers: list[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
        import yfinance as yf  # imported lazily so the rest of the package

        # doesn't require yfinance unless ingestion is actually run.
        logger.info("Fetching %s from %s to %s via yfinance", tickers, start, end or "today")
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
        )

        frames = []
        # yfinance returns a MultiIndex column frame when given >1 ticker,
        # and a flat frame when given exactly 1. Normalize both cases.
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker not in raw.columns.get_level_values(0):
                    logger.warning("No data returned for %s", ticker)
                    continue
                df = raw[ticker].copy()
                df["ticker"] = ticker
                frames.append(df)
        else:
            df = raw.copy()
            df["ticker"] = tickers[0]
            frames.append(df)

        combined = pd.concat(frames)
        combined = combined.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        combined["source"] = self.SOURCE_NAME
        return combined[
            ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]
        ]


class MacroDataProvider(DataProvider):
    """Placeholder for macroeconomic indicators (e.g. FRED). Not yet wired in."""

    def fetch(self, tickers: list[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
        raise NotImplementedError(
            "MacroDataProvider is a placeholder for a future data source. "
            "Implement fetch() against your chosen macro API and register "
            "credentials via .env (MACRO_API_KEY), never hard-coded."
        )


class VolatilityDataProvider(DataProvider):
    """Placeholder for a volatility index feed (e.g. VIX). Not yet wired in."""

    def fetch(self, tickers: list[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
        raise NotImplementedError(
            "VolatilityDataProvider is a placeholder for a future data source."
        )


def ingest(tickers: list[str], start: str, end: Optional[str] = None) -> int:
    """Fetch raw prices and persist them, unmodified, into raw_market_prices.

    Returns the number of rows written.
    """
    provider = MarketDataProvider()
    df = provider.fetch(tickers, start, end)
    if df.empty:
        logger.warning("No data fetched for tickers=%s", tickers)
        return 0

    repo = Repository()
    repo.upsert_assets(tickers)
    n = repo.write_raw_prices(df)
    logger.info("Wrote %d raw price rows", n)
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw market data into MySQL.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, defaults to today")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()
    ingest(args.tickers, args.start, args.end)
