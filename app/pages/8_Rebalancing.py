import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.state import render_sidebar, weight_editor
from src.config import CONFIG
from src.portfolio.rebalancing import rebalance_plan

st.set_page_config(page_title="Rebalancing", page_icon="🔄", layout="wide")
st.title("🔄 Rebalancing")
st.caption(
    "Analytical recommendation only — this project never places real "
    "trades. Set a target allocation on the Portfolio Optimization page "
    "first, or enter one manually below."
)

cfg = render_sidebar()
tickers, portfolio_value = cfg["tickers"], cfg["portfolio_value"]

st.subheader("Current allocation")
current_weights = weight_editor(tickers, key_prefix="current")

st.subheader("Target allocation")
target_from_optimizer = st.session_state.get("target_weights")

use_optimizer_target = (
    target_from_optimizer is not None
    and set(target_from_optimizer.index) == set(tickers)
    and st.checkbox("Use target weights from Portfolio Optimization page", value=True)
)

if use_optimizer_target:
    target_weights = target_from_optimizer.reindex(tickers)
    st.bar_chart(target_weights)
else:
    if target_from_optimizer is not None and set(target_from_optimizer.index) != set(tickers):
        st.info("Optimizer target weights don't match the current ticker list — enter target weights manually.")
    cols = st.columns(len(tickers))
    raw = {}
    for col, t in zip(cols, tickers):
        raw[t] = col.number_input(f"{t} target %", min_value=0.0, max_value=100.0, value=float(100 / len(tickers)), step=1.0, key=f"target_{t}")
    total = sum(raw.values()) or 1.0
    target_weights = pd.Series({t: v / total for t, v in raw.items()})

transaction_cost_bps = st.slider("Transaction cost (bps)", 0, 100, CONFIG["backtest"]["transaction_cost_bps"])

plan = rebalance_plan(current_weights, target_weights, portfolio_value, transaction_cost_bps)

st.divider()
st.subheader("Rebalancing plan")

display_plan = plan.copy()
display_plan["current_weight"] = (display_plan["current_weight"] * 100).round(1).astype(str) + "%"
display_plan["target_weight"] = (display_plan["target_weight"] * 100).round(1).astype(str) + "%"
display_plan["weight_diff"] = (display_plan["weight_diff"] * 100).round(1).astype(str) + "%"
display_plan["trade_value"] = display_plan["trade_value"].map(lambda v: f"${v:,.0f}")
display_plan["estimated_transaction_cost"] = display_plan["estimated_transaction_cost"].map(lambda v: f"${v:,.2f}")
st.dataframe(display_plan, width='stretch')

col1, col2, col3 = st.columns(3)
n_buys = (plan["action"] == "BUY").sum()
n_sells = (plan["action"] == "SELL").sum()
total_cost = plan["estimated_transaction_cost"].sum()
col1.metric("Buys", n_buys)
col2.metric("Sells", n_sells)
col3.metric("Total estimated cost", f"${total_cost:,.2f}")

fig = go.Figure()
fig.add_trace(go.Bar(x=plan.index, y=plan["current_weight"] * 100, name="Current"))
fig.add_trace(go.Bar(x=plan.index, y=plan["target_weight"] * 100, name="Target"))
fig.update_layout(barmode="group", yaxis_title="Weight (%)", height=400, title="Current vs target allocation")
st.plotly_chart(fig, width='stretch')

if not cfg["used_synthetic"]:
    if st.button("💾 Save this rebalancing plan to MySQL", key="save_rebalance_plan"):
        from datetime import date as _date

        from src.data.repository import Repository

        repo = Repository()
        portfolio_id = repo.get_or_create_portfolio(f"dashboard-{'-'.join(tickers)}")
        n_saved = repo.write_rebalance_recommendations(portfolio_id, _date.today().isoformat(), plan)
        st.success(f"Saved {n_saved} recommendation(s) to `rebalance_recommendations` (portfolio_id={portfolio_id}).")
else:
    st.caption("💾 Saving to MySQL is only available in MySQL data-source mode (synthetic demo tickers aren't in the `assets` table).")
