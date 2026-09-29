import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.state import render_sidebar
from src.config import CONFIG
from src.portfolio.optimization import (
    efficient_frontier,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    random_feasible_portfolios,
)

st.set_page_config(page_title="Portfolio Optimization", page_icon="🎯", layout="wide")
st.title("🎯 Portfolio Optimization")
st.caption(
    "Uses the FULL available history to estimate expected returns and "
    "covariance — appropriate for an exploratory 'what should I hold "
    "today' analysis, but not a valid backtest of historical performance "
    "(that would need out-of-sample, walk-forward estimation; see "
    "`src/backtesting/engine.py`)."
)

cfg = render_sidebar()
tickers, returns = cfg["tickers"], cfg["returns"]

max_weight = st.slider(
    "Max weight per asset", round(1 / len(tickers), 2), 1.0, CONFIG["portfolio"]["max_weight"], 0.05
)
risk_free_rate_daily = CONFIG["risk"]["risk_free_rate_annual"] / 252

mean_returns = returns[tickers].mean()
cov_matrix = returns[tickers].cov()

min_var = minimum_variance_portfolio(mean_returns, cov_matrix, max_weight=max_weight)
max_sharpe = maximum_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate_daily, max_weight=max_weight)
frontier = efficient_frontier(mean_returns, cov_matrix, max_weight=max_weight, risk_free_rate=risk_free_rate_daily)
random_portfolios = random_feasible_portfolios(mean_returns, cov_matrix, risk_free_rate=risk_free_rate_daily)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=random_portfolios["volatility"] * (252 ** 0.5) * 100,
        y=random_portfolios["expected_return"] * 252 * 100,
        mode="markers",
        marker=dict(size=4, color=random_portfolios["sharpe_ratio"], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
        name="Random long-only portfolios",
    )
)
if not frontier.empty:
    fig.add_trace(
        go.Scatter(
            x=frontier["volatility"] * (252 ** 0.5) * 100,
            y=frontier["expected_return"] * 252 * 100,
            mode="lines", line=dict(color="black", width=3), name="Efficient frontier",
        )
    )
fig.add_trace(
    go.Scatter(
        x=[min_var.volatility * (252 ** 0.5) * 100], y=[min_var.expected_return * 252 * 100],
        mode="markers", marker=dict(color="blue", size=14, symbol="star"), name="Min variance",
    )
)
fig.add_trace(
    go.Scatter(
        x=[max_sharpe.volatility * (252 ** 0.5) * 100], y=[max_sharpe.expected_return * 252 * 100],
        mode="markers", marker=dict(color="red", size=14, symbol="star"), name="Max Sharpe",
    )
)
fig.update_layout(
    title="Efficient frontier", xaxis_title="Annualized volatility (%)",
    yaxis_title="Annualized expected return (%)", height=550,
)
st.plotly_chart(fig, width='stretch')

col1, col2 = st.columns(2)
with col1:
    st.subheader("Minimum variance portfolio")
    st.metric("Expected return (ann.)", f"{min_var.expected_return*252*100:.1f}%")
    st.metric("Volatility (ann.)", f"{min_var.volatility*(252**0.5)*100:.1f}%")
    st.bar_chart(min_var.weights)

with col2:
    st.subheader("Maximum Sharpe portfolio")
    st.metric("Expected return (ann.)", f"{max_sharpe.expected_return*252*100:.1f}%")
    st.metric("Volatility (ann.)", f"{max_sharpe.volatility*(252**0.5)*100:.1f}%")
    st.metric("Sharpe ratio", f"{max_sharpe.sharpe_ratio*(252**0.5):.2f}")
    st.bar_chart(max_sharpe.weights)

st.divider()
choice = st.radio("Use as target allocation on the Rebalancing page:", ["Minimum variance", "Maximum Sharpe"], horizontal=True)
if st.button("Set as target weights"):
    st.session_state["target_weights"] = min_var.weights if choice == "Minimum variance" else max_sharpe.weights
    st.success(f"Target weights set from '{choice}' — go to the Rebalancing page to see the trade plan.")
