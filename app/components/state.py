"""
Shared app state: default portfolio configuration, sidebar controls, and
small helpers reused across every page.

DATA SOURCE (see docs/FORENSIC_AUDIT.md, Issues 1-2): the dashboard reads
market data through exactly two explicit, user-chosen paths -- MySQL
(via src.data.service.DataService) or synthetic demo data (generated
directly, on request). There is no automatic, silent fallback from one to
the other anywhere in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data_loader import _synthetic_price_panel, compute_returns  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.data.service import DataService, NoDataInDatabaseError  # noqa: E402

DEFAULT_TICKERS = ["AAPL", "MSFT", "GLD", "BND", "SPY"]
DEFAULT_START = "2019-01-01"


def init_session_state() -> None:
    st.session_state.setdefault("tickers", DEFAULT_TICKERS.copy())
    st.session_state.setdefault("start_date", DEFAULT_START)
    st.session_state.setdefault("portfolio_value", 100_000.0)
    st.session_state.setdefault("weights", None)  # set once tickers are known
    st.session_state.setdefault("target_weights", None)  # set by Optimization page
    st.session_state.setdefault("data_source", None)  # user must choose explicitly


def render_sidebar() -> dict:
    """Render the shared sidebar (data source, ticker/date/value controls)
    and return the resolved configuration. Call this at the top of every
    page.
    """
    init_session_state()

    st.sidebar.header("Data source")
    data_source = st.sidebar.radio(
        "Choose where prices come from",
        options=["MySQL historical data", "Synthetic demo data"],
        index=0 if st.session_state["data_source"] != "synthetic" else 1,
        help=(
            "MySQL: reads real, previously-ingested prices from your local "
            "database (run `python -m src.data.ingestion ...` first). "
            "Synthetic: generates reproducible fake price paths for demo "
            "purposes only -- never real market data."
        ),
    )
    st.session_state["data_source"] = "mysql" if data_source.startswith("MySQL") else "synthetic"

    st.sidebar.header("Portfolio configuration")

    tickers_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value=", ".join(st.session_state["tickers"]),
        help="For MySQL mode: tickers must already be ingested. For synthetic mode: any label works.",
    )
    tickers = tuple(t.strip().upper() for t in tickers_input.split(",") if t.strip())
    st.session_state["tickers"] = list(tickers)

    start_date = st.sidebar.date_input(
        "Start date", value=pd.Timestamp(st.session_state["start_date"])
    )
    st.session_state["start_date"] = str(start_date)

    portfolio_value = st.sidebar.number_input(
        "Portfolio value ($)", min_value=1000.0, value=st.session_state["portfolio_value"], step=1000.0
    )
    st.session_state["portfolio_value"] = portfolio_value

    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ Educational tool only. Not investment advice. "
        "Risk estimates are model-based, not guarantees."
    )

    if len(tickers) < 2:
        st.sidebar.warning("Enter at least 2 tickers.")
        st.stop()

    if st.session_state["data_source"] == "mysql":
        st.info("📊 Data source: **MySQL** (real, previously-ingested prices).", icon="📊")
        try:
            price_panel = DataService().get_price_panel(list(tickers))
        except NoDataInDatabaseError as e:
            st.error(str(e), icon="🛑")
            st.stop()
        used_synthetic = False
        # Restrict to the requested start date; MySQL may hold more history.
        price_panel = price_panel[price_panel.index >= pd.Timestamp(start_date)]
        if price_panel.shape[0] < 30:
            st.error(
                f"Only {price_panel.shape[0]} rows of MySQL data found on/after {start_date} "
                f"for {list(tickers)}. Ingest more history or pick an earlier start date.",
                icon="🛑",
            )
            st.stop()
    else:
        st.warning(
            "🧪 Data source: **synthetic demo data** — reproducible fake price "
            "paths, chosen explicitly. This is NOT real market data.",
            icon="🧪",
        )
        price_panel = _synthetic_price_panel(tickers, str(start_date))
        used_synthetic = True

    returns = compute_returns(price_panel)

    if st.session_state["weights"] is None or set(st.session_state["weights"].index) != set(tickers):
        st.session_state["weights"] = pd.Series(1 / len(tickers), index=list(tickers))

    return {
        "tickers": list(tickers),
        "start_date": str(start_date),
        "portfolio_value": portfolio_value,
        "price_panel": price_panel,
        "returns": returns,
        "used_synthetic": used_synthetic,
        "weights": st.session_state["weights"],
    }


def weight_editor(tickers: list[str], key_prefix: str = "w") -> pd.Series:
    """Render one number input per ticker (as a percentage) and return a
    weights Series normalized to sum to 1.
    """
    current = st.session_state["weights"].reindex(tickers).fillna(1 / len(tickers))
    cols = st.columns(len(tickers))
    raw = {}
    for col, ticker in zip(cols, tickers):
        raw[ticker] = col.number_input(
            ticker, min_value=0.0, max_value=100.0,
            value=float(current[ticker] * 100), step=1.0,
            key=f"{key_prefix}_{ticker}",
        )
    total = sum(raw.values()) or 1.0
    weights = pd.Series({t: v / total for t, v in raw.items()})
    st.session_state["weights"] = weights
    if abs(sum(raw.values()) - 100.0) > 0.5:
        st.caption(f"Weights sum to {sum(raw.values()):.1f}% — normalized to 100% automatically.")
    return weights
