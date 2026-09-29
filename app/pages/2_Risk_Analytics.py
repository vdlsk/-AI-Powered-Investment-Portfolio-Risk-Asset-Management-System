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
from src.risk.cvar import historical_cvar
from src.risk.var import historical_var, parametric_var
from src.risk.risk_contribution import pct_contribution_to_risk

st.set_page_config(page_title="Risk Analytics", page_icon="⚠️", layout="wide")
st.title("⚠️ Risk Analytics")

cfg = render_sidebar()
tickers, returns, portfolio_value = cfg["tickers"], cfg["returns"], cfg["portfolio_value"]

st.subheader("Allocation")
weights = weight_editor(tickers)
port_returns = portfolio_returns(returns, weights)

col_a, col_b = st.columns(2)
confidence_level = col_a.slider("Confidence level", 0.90, 0.99, CONFIG["risk"]["confidence_level"], 0.01)
horizon_days = col_b.slider("Horizon (trading days)", 1, 20, CONFIG["risk"]["var_horizon_days"])

hvar = historical_var(port_returns, confidence_level, horizon_days)
pvar = parametric_var(port_returns, confidence_level, horizon_days)
cvar = historical_cvar(port_returns, confidence_level, horizon_days)

col1, col2, col3 = st.columns(3)
col1.metric(f"Historical VaR ({confidence_level:.0%})", f"{hvar*100:.2f}%", help=f"≈ ${hvar*portfolio_value:,.0f}")
col2.metric(f"Parametric VaR ({confidence_level:.0%})", f"{pvar*100:.2f}%", help=f"≈ ${pvar*portfolio_value:,.0f}")
col3.metric(f"CVaR / Expected Shortfall ({confidence_level:.0%})", f"{cvar*100:.2f}%", help=f"≈ ${cvar*portfolio_value:,.0f}")

if not cfg["used_synthetic"]:
    if st.button("💾 Save these risk metrics to MySQL", key="save_risk_metrics"):
        from datetime import date as _date

        from src.data.repository import Repository
        from src.portfolio.analytics import performance_report

        repo = Repository()
        portfolio_id = repo.get_or_create_portfolio(f"dashboard-{'-'.join(tickers)}")
        full_report = performance_report(port_returns, confidence_level=confidence_level, horizon_days=horizon_days)
        repo.write_risk_metrics(portfolio_id, _date.today().isoformat(), full_report)
        st.success(f"Saved to `risk_metrics` (portfolio_id={portfolio_id}, date={_date.today().isoformat()}).")
else:
    st.caption("💾 Saving to MySQL is only available in MySQL data-source mode (synthetic demo tickers aren't in the `assets` table).")

st.caption(
    "VaR: the loss you would not expect to exceed, at the given confidence "
    "level, over the given horizon. CVaR: the average loss **given** that "
    "VaR is exceeded — always ≥ VaR, and more informative about tail severity."
)

st.divider()
st.subheader("Return distribution")
fig = go.Figure()
fig.add_trace(go.Histogram(x=port_returns.values, nbinsx=60, name="Daily returns"))
fig.add_vline(x=-hvar, line_dash="dash", line_color="orange", annotation_text="Historical VaR")
fig.add_vline(x=-cvar, line_dash="dash", line_color="red", annotation_text="CVaR")
fig.update_layout(height=400, xaxis_title="Daily return", xaxis_tickformat=".1%")
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Risk contribution by asset")
st.caption(
    "How much each asset actually contributes to total portfolio "
    "volatility — not the same as its portfolio weight. A small weight in "
    "a volatile, highly-correlated asset can dominate portfolio risk."
)
cov = returns[tickers].cov().values * 252  # annualized covariance
w = weights.reindex(tickers).values
ccr_pct = pct_contribution_to_risk(w, cov, tickers)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=weights.reindex(tickers).values * 100, y=tickers, name="Weight %", orientation="h"))
fig2.add_trace(go.Bar(x=ccr_pct.values * 100, y=tickers, name="Risk contribution %", orientation="h"))
fig2.update_layout(barmode="group", height=350, xaxis_title="%")
st.plotly_chart(fig2, width='stretch')
