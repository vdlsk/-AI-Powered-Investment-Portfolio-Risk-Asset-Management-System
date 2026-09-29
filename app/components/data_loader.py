"""
Data loading helpers for the Streamlit app.

IMPORTANT (see docs/FORENSIC_AUDIT.md, Issues 1-2): this module used to
contain `load_price_panel`, which called
`src.data.ingestion.MarketDataProvider` (Yahoo Finance) directly and
silently fell back to synthetic data on any failure -- bypassing MySQL
entirely and switching data sources without the user's explicit consent.
That function has been removed. The dashboard now reads real data
exclusively through `src.data.service.DataService` (MySQL), and generates
synthetic data only when the user explicitly chooses that option in the
sidebar (see `app/components/state.py::render_sidebar`) -- never as an
automatic fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _stable_seed(ticker: str) -> int:
    """Deterministic integer seed for a ticker, stable across processes.

    Python's built-in `hash()` on strings is intentionally randomized
    per-process (PYTHONHASHSEED, on by default since Python 3.3, to
    prevent hash-flooding attacks) -- `hash("AAPL")` returns a DIFFERENT
    number every time a new Python process starts. Using it here would
    make the synthetic data change every time the app restarts, silently
    breaking the "reproducible demo data" guarantee this function claims.
    `hashlib.sha256` is stable across processes, platforms, and Python
    versions by design, which is what "reproducible" actually requires.
    """
    import hashlib

    digest = hashlib.sha256(ticker.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


def _synthetic_price_panel(tickers: tuple[str, ...], start: str, end: str | None = None) -> pd.DataFrame:
    """Reproducible geometric-Brownian-motion price panel, seeded per ticker
    so the same ticker always produces the same synthetic series across
    reruns AND across separate process invocations (see `_stable_seed`).

    Called ONLY when the user has explicitly chosen "Synthetic demo data"
    in the sidebar -- never automatically.
    """
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
        prices = 100 * np.cumprod(1 + shocks)
        data[ticker] = prices

    return pd.DataFrame(data, index=dates)


def compute_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    return price_panel.pct_change().dropna(how="all")
