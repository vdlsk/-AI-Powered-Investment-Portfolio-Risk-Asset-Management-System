"""
FinRisk — main Streamlit entry point.

Run with:
    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.state import render_sidebar  # noqa: E402
from src.portfolio.analytics import performance_report, portfolio_returns  # noqa: E402

st.set_page_config(page_title="FinRisk", page_icon="📈", layout="wide")

st.title("📈 FinRisk — Portfolio Risk Analytics Platform")
st.caption(
    "Educational tool. Not investment advice. Risk estimates are model-based "
    "and do not guarantee future performance."
)

cfg = render_sidebar()
tickers = cfg["tickers"]
returns = cfg["returns"]
weights = cfg["weights"]

port_returns = portfolio_returns(returns, weights)
report = performance_report(port_returns)
wealth = (1 + port_returns).cumprod() * cfg["portfolio_value"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cumulative return", f"{report['cumulative_return']*100:.1f}%")
col2.metric("Annualized volatility", f"{report['annualized_volatility']*100:.1f}%")
col3.metric("Sharpe ratio", f"{report['sharpe_ratio']:.2f}")
col4.metric("Max drawdown", f"{report['max_drawdown']*100:.1f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, mode="lines", name="Portfolio value"))
fig.update_layout(
    title="Portfolio value over time",
    xaxis_title="Date",
    yaxis_title="Value ($)",
    height=420,
)
st.plotly_chart(fig, width='stretch')

col5, col6 = st.columns(2)
with col5:
    pie = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values, hole=0.4)])
    pie.update_layout(title="Current allocation", height=380)
    st.plotly_chart(pie, width='stretch')

with col6:
    corr = returns[tickers].corr()
    heat = go.Figure(
        data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0)
    )
    heat.update_layout(title="Asset correlation", height=380)
    st.plotly_chart(heat, width='stretch')

st.divider()
st.subheader("Use the pages in the sidebar to explore:")
st.markdown(
    """
- **Portfolio Analysis** — returns, drawdowns, rolling volatility
- **Risk Analytics** — VaR, CVaR, risk contribution by asset
- **Stress Testing** — hypothetical shock scenarios
- **Monte Carlo** — simulated future portfolio paths
- **Market Regime** — detected low/medium/high volatility regimes
- **ML Risk Prediction** — model-based near-term risk classification
- **Portfolio Optimization** — efficient frontier, min-variance / max-Sharpe
- **Rebalancing** — current vs target allocation, suggested trades
"""
)
