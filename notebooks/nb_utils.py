"""
Shared utility for the notebooks/ series. Fetches real market data via the
same MarketDataProvider used in production, and falls back to reproducible
synthetic price paths when network access isn't available (e.g. in a
sandboxed environment) -- so every notebook runs end to end regardless.

NOTE on scope (see docs/FORENSIC_AUDIT.md): the production Streamlit
dashboard was changed to require an explicit MySQL-vs-synthetic choice
with no silent fallback (see src/data/service.py). The notebooks in this
folder still use the simpler silent-fallback pattern below, printed
clearly to the notebook's own output so a reader always sees which mode
ran, because rewiring all 14 notebooks to require a pre-populated MySQL
instance is a larger, separate change than fits in this pass -- flagged
as an open item, not silently left inconsistent without explanation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data_loader import _stable_seed  # noqa: E402 -- single source of truth for the seed derivation


def synthetic_price_panel(tickers, start, end=None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    dates = pd.bdate_range(start_ts, end_ts)
    n = len(dates)
    data = {}
    for ticker in tickers:
        seed = _stable_seed(ticker)
        rng = np.random.default_rng(seed)
        annual_drift = rng.uniform(0.03, 0.12)
        annual_vol = rng.uniform(0.12, 0.35)
        daily_drift = annual_drift / 252
        daily_vol = annual_vol / np.sqrt(252)
        shocks = rng.normal(daily_drift, daily_vol, size=n)
        data[ticker] = 100 * np.cumprod(1 + shocks)
    return pd.DataFrame(data, index=dates)


def fetch_price_panel(tickers, start, end=None) -> tuple[pd.DataFrame, bool]:
    """Returns (wide adj_close panel, used_synthetic_flag)."""
    try:
        from src.data.ingestion import MarketDataProvider

        provider = MarketDataProvider()
        long_df = provider.fetch(list(tickers), start, end)
        if long_df.empty or long_df["adj_close"].isna().all():
            raise ValueError("empty result")
        panel = long_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        panel = panel.dropna(how="all")
        if panel.shape[0] < 30 or panel.isna().mean().max() > 0.5:
            raise ValueError("insufficient data")
        return panel.ffill().dropna(), False
    except Exception as e:
        print(f"[info] Live data unavailable ({e.__class__.__name__}); using synthetic demo data instead.")
        return synthetic_price_panel(tickers, start, end), True
