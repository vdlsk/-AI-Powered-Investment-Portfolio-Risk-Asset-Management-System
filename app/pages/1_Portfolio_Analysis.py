import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.state import render_sidebar, weight_editor
from src.features.volatility import rolling_volatility
from src.portfolio.analytics import portfolio_returns

st.set_page_config(page_title="Portfolio Analysis", page_icon="💼", layout="wide")
st.title("💼 Portfolio Analysis")

cfg = render_sidebar()
tickers, returns = cfg["tickers"], cfg["returns"]

st.subheader("Allocation")
weights = weight_editor(tickers)

port_returns = portfolio_returns(returns, weights)
wealth = (1 + port_returns).cumprod()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=wealth.index, y=wealth.values, name="Portfolio (normalized to 1.0)"))
for t in tickers:
    asset_wealth = (1 + returns[t]).cumprod()
    fig1.add_trace(go.Scatter(x=asset_wealth.index, y=asset_wealth.values, name=t, opacity=0.5))
fig1.update_layout(title="Cumulative growth: portfolio vs individual assets", height=450)
st.plotly_chart(fig1, width='stretch')

running_max = wealth.cummax()
drawdown = wealth / running_max - 1
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy", name="Drawdown"))
fig2.update_layout(title="Portfolio drawdown", yaxis_tickformat=".0%", height=350)
st.plotly_chart(fig2, width='stretch')

col1, col2 = st.columns(2)
with col1:
    vol20 = rolling_volatility(port_returns, 20) * (252 ** 0.5)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=vol20.index, y=vol20.values, name="20d rolling ann. vol"))
    fig3.update_layout(title="Rolling annualized volatility (20d)", yaxis_tickformat=".0%", height=350)
    st.plotly_chart(fig3, width='stretch')

with col2:
    corr = returns[tickers].corr()
    heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
    heat.update_layout(title="Correlation matrix", height=350)
    st.plotly_chart(heat, width='stretch')

st.subheader("Recent daily returns")
st.dataframe(returns[tickers].tail(15).style.format("{:.2%}"), width='stretch')
