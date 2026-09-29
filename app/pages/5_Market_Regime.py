import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.state import render_sidebar, weight_editor
from src.features.feature_pipeline import build_features_for_asset
from src.models.regime import fit_exploratory_regimes
from src.portfolio.analytics import portfolio_returns

st.set_page_config(page_title="Market Regime", page_icon="🌊", layout="wide")
st.title("🌊 Market Regime Detection")
st.caption(
    "Unsupervised clustering (KMeans) over trailing 20-day return and "
    "volatility identifies low/medium/high-volatility regimes. This page "
    "fits on the FULL available history for a clear historical picture — "
    "it is **exploratory only** and not valid for backtesting a live "
    "strategy, since it uses information from the whole period at once."
)

cfg = render_sidebar()
tickers, returns = cfg["tickers"], cfg["returns"]

weights = weight_editor(tickers)
port_returns = portfolio_returns(returns, weights)

wealth = (1 + port_returns).cumprod()
prices_df = wealth.reset_index()
prices_df.columns = ["date", "adj_close"]
prices_df["volume"] = 1  # regime detection doesn't use volume

feats = build_features_for_asset(prices_df)

if feats.dropna(subset=["ret_20d", "vol_20d"]).shape[0] < 60:
    st.warning("Not enough history for reliable regime detection — pick an earlier start date.")
else:
    regimes = fit_exploratory_regimes(feats, n_regimes=3)
    regimes = regimes.set_index("date")

    color_map = {"low_vol": "seagreen", "medium_vol": "orange", "high_vol": "crimson"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, mode="lines", line=dict(color="black", width=1), name="Portfolio value"))

    for label, color in color_map.items():
        mask = regimes["regime_label"] == label
        dates = regimes.index[mask]
        if len(dates):
            fig.add_trace(
                go.Scatter(
                    x=dates, y=wealth.reindex(dates), mode="markers",
                    marker=dict(color=color, size=5), name=label.replace("_", " ").title(),
                )
            )
    fig.update_layout(title="Portfolio value colored by detected regime", height=450)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Time spent in each regime")
    counts = regimes["regime_label"].value_counts(normalize=True) * 100
    fig2 = go.Figure(data=[go.Bar(x=counts.index, y=counts.values)])
    fig2.update_layout(yaxis_title="% of days", height=300)
    st.plotly_chart(fig2, width='stretch')

    st.caption(
        "For a backtest-safe (expanding-window, no look-ahead) version of "
        "regime detection, see `src/models/regime.py::fit_walk_forward_regimes`."
    )
