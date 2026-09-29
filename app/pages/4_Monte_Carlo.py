import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.state import render_sidebar, weight_editor
from src.config import CONFIG
from src.portfolio.analytics import portfolio_returns
from src.risk.monte_carlo import run_monte_carlo

st.set_page_config(page_title="Monte Carlo", page_icon="🎲", layout="wide")
st.title("🎲 Monte Carlo Simulation")
st.caption(
    "Simulates many possible future paths by resampling historical daily "
    "returns (or drawing from a fitted Normal distribution). This shows a "
    "*range* of plausible outcomes, not a forecast."
)

cfg = render_sidebar()
tickers, returns, portfolio_value = cfg["tickers"], cfg["returns"], cfg["portfolio_value"]

st.subheader("Allocation")
weights = weight_editor(tickers)
port_returns = portfolio_returns(returns, weights)

col1, col2, col3, col4 = st.columns(4)
n_sims = col1.slider("Simulations", 500, 20000, CONFIG["monte_carlo"]["number_of_simulations"], step=500)
horizon = col2.slider("Horizon (trading days)", 5, 252, CONFIG["monte_carlo"]["time_horizon_days"])
confidence = col3.slider("Confidence level", 0.90, 0.99, CONFIG["monte_carlo"]["confidence_level"], 0.01)
distribution = col4.selectbox(
    "Return model", ["block", "empirical", "normal"],
    help="'block' resamples contiguous historical chunks to preserve volatility clustering (recommended); "
         "'empirical' resamples single independent days; 'normal' assumes a Gaussian distribution.",
)
block_size = 5
if distribution == "block":
    block_size = st.slider("Block size (trading days)", 2, 20, 5, help="Length of each resampled historical chunk.")

result = run_monte_carlo(
    port_returns,
    number_of_simulations=n_sims,
    time_horizon_days=horizon,
    confidence_level=confidence,
    random_seed=CONFIG["monte_carlo"]["random_seed"],
    distribution=distribution,
    block_size=block_size,
    initial_value=portfolio_value,
)

col_a, col_b, col_c = st.columns(3)
col_a.metric(f"Simulated VaR ({confidence:.0%})", f"${result.var:,.0f}")
col_b.metric("Simulated CVaR", f"${result.cvar:,.0f}")
col_c.metric("Probability of any loss", f"{result.probability_of_loss*100:.1f}%")

st.divider()
st.subheader(f"Sample of simulated paths (over {horizon} trading days)")
fig = go.Figure()
sample_idx = np.random.default_rng(0).choice(result.paths.shape[0], size=min(200, result.paths.shape[0]), replace=False)
for i in sample_idx:
    fig.add_trace(go.Scatter(y=result.paths[i], mode="lines", line=dict(width=0.5), opacity=0.15, showlegend=False))
median_path = np.median(result.paths, axis=0)
fig.add_trace(go.Scatter(y=median_path, mode="lines", line=dict(width=3, color="black"), name="Median path"))
fig.update_layout(height=450, xaxis_title="Trading days ahead", yaxis_title="Portfolio value ($)")
st.plotly_chart(fig, width='stretch')

st.subheader("Distribution of terminal portfolio value")
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=result.terminal_wealth, nbinsx=60))
fig2.add_vline(x=portfolio_value, line_dash="dash", annotation_text="Starting value")
fig2.update_layout(height=350, xaxis_title="Terminal portfolio value ($)")
st.plotly_chart(fig2, width='stretch')

st.subheader("Percentile outcomes")
import pandas as pd
pct_df = pd.DataFrame([{f"P{p.replace('p','')}": f"${v:,.0f}" for p, v in result.percentiles.items()}])
st.dataframe(pct_df, width='stretch', hide_index=True)
