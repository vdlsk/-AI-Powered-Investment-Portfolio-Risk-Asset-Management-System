"""
Data service: the ONLY path the Streamlit dashboard should use to read
market data for normal analytics.

WHY THIS FILE EXISTS (see docs/FORENSIC_AUDIT.md, Issue 1): earlier in
this project, the dashboard called `src.data.ingestion.MarketDataProvider`
(Yahoo Finance) directly, completely bypassing MySQL despite the
documented architecture claiming MySQL is the central data store. This
class is the fix: it is a thin, explicit wrapper around
`src.data.repository.Repository` that reads already-ingested, already-
cleaned data from `clean_market_prices`, and NEVER calls out to a live
data provider itself. Fetching fresh data from a provider is exclusively
the job of `src.data.ingestion.ingest()`, run as its own explicit step
(the CLI documented in README.md), not something dashboard page code
should trigger implicitly.

NO SILENT FALLBACK: if MySQL has no data for the requested tickers, this
raises `NoDataInDatabaseError` with an actionable message. It does not
generate synthetic data itself -- that choice belongs to the caller (the
dashboard), which must ask the user explicitly before ever showing
synthetic data (see app/components/state.py).
"""
from __future__ import annotations

import pandas as pd


class NoDataInDatabaseError(Exception):
    """Raised when MySQL has no usable data for the requested tickers.

    This is a distinct, catchable exception type specifically so callers
    (the dashboard) can show a clear, actionable message and stop --
    rather than the error being swallowed by a broad `except Exception`
    that might otherwise trigger an unwanted fallback.
    """


class DataService:
    """Single read path for the dashboard: Repository -> MySQL only."""

    def __init__(self, repository=None):
        # Imported lazily / accepted as a constructor argument so tests
        # can inject a fake repository without needing a real database.
        if repository is None:
            from src.data.repository import Repository

            repository = Repository()
        self.repository = repository

    def get_price_panel(self, tickers: list[str]) -> pd.DataFrame:
        """Return a wide adj_close price panel (index=date, columns=ticker)
        built exclusively from `clean_market_prices` in MySQL.

        Raises `NoDataInDatabaseError` if MySQL has no rows for ANY of the
        requested tickers, or if every requested ticker has no data at
        all. Partial coverage (some tickers have data, some don't) is
        allowed through, but the caller can inspect the returned panel's
        columns to see exactly which tickers actually had data -- no
        column is ever silently invented.
        """
        long_df = self.repository.read_clean_prices(tickers)
        if long_df.empty:
            raise NoDataInDatabaseError(
                f"No market data found in MySQL for tickers {tickers}. "
                f"Run the ingestion pipeline first, e.g.:\n"
                f"    python -m src.data.ingestion --tickers {' '.join(tickers)} --start 2015-01-01"
            )

        # MySQL DATE columns come back through pandas.read_sql as plain
        # `datetime.date` objects, not pandas Timestamps -- this silently
        # breaks any later comparison like `panel.index >= pd.Timestamp(x)`
        # (a real bug caught via AppTest during development: it raised
        # `Cannot compare Timestamp with datetime.date`). Normalizing to a
        # proper datetime64 index here, once, means every downstream
        # consumer of this panel can rely on standard pandas Timestamp
        # semantics without re-deriving this fix at each call site.
        long_df["date"] = pd.to_datetime(long_df["date"])

        panel = long_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        panel = panel.dropna(how="all")
        if panel.empty:
            raise NoDataInDatabaseError(
                f"MySQL returned rows for {tickers} but none had a usable adj_close value. "
                f"Check the clean_market_prices table for these tickers."
            )
        return panel

    def get_asset_metadata(self, tickers: list[str]) -> pd.DataFrame:
        """Return source/ingestion metadata so the dashboard can answer
        'where did this price come from, and when was it ingested' --
        see docs/FORENSIC_AUDIT.md Issue on data-source traceability.
        """
        from sqlalchemy import text

        placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
        query = f"""
            SELECT a.ticker, r.source, MIN(r.ingested_at) as first_ingested_at,
                   MAX(r.ingested_at) as last_ingested_at, COUNT(*) as row_count
            FROM raw_market_prices r
            JOIN assets a ON a.asset_id = r.asset_id
            WHERE a.ticker IN ({placeholders})
            GROUP BY a.ticker, r.source
        """
        params = {f"t{i}": t for i, t in enumerate(tickers)}
        with self.repository.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
