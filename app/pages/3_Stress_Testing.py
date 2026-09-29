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
from src.risk.stress_testing import run_all_scenarios, run_stress_test

st.set_page_config(page_title="Stress Testing", page_icon="🧨", layout="wide")
st.title("🧨 Stress Testing")
st.caption(
    "These are **hypothetical scenarios**, not predictions or historical "
    "replays. They show what would happen to your current allocation if a "
    "shock like this occurred."
)

cfg = render_sidebar()
tickers, portfolio_value = cfg["tickers"], cfg["portfolio_value"]

st.subheader("Allocation")
weights = weight_editor(tickers)

st.divider()
st.subheader("Predefined scenarios")
st.caption(
    "Each scenario applies a shock per asset class; here it is applied "
    "uniformly to every held ticker as a simplification (a real "
    "implementation would map each ticker to its asset class)."
)

scenario_choice = st.selectbox(
    "Assume every holding behaves like:",
    options=["equities", "bonds", "gold"],
    help="Which shock line from each scenario definition applies to your tickers.",
)

results = []
for name, shocks in CONFIG["stress_scenarios"].items():
    shock_value = shocks.get(scenario_choice, 0.0)
    uniform_shocks = {t: shock_value for t in tickers}
    result = run_stress_test(name, weights, uniform_shocks, portfolio_value)
    results.append(result)

summary = pd.DataFrame(
    {
        "Scenario": [r.scenario_name.replace("_", " ").title() for r in results],
        "Shock applied": [f"{CONFIG['stress_scenarios'][r.scenario_name][scenario_choice]*100:+.0f}%" for r in results],
        "Portfolio loss": [f"{r.portfolio_loss_pct*100:.1f}%" for r in results],
        "New portfolio value": [f"${r.portfolio_new_value:,.0f}" for r in results],
    }
)
st.dataframe(summary, width='stretch', hide_index=True)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=[r.scenario_name.replace("_", " ").title() for r in results],
        y=[r.portfolio_loss_pct * 100 for r in results],
        marker_color=["crimson" if r.portfolio_loss_pct < 0 else "seagreen" for r in results],
    )
)
fig.update_layout(title="Hypothetical portfolio impact by scenario", yaxis_title="Loss / gain (%)", height=400)
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Custom scenario")
st.caption("Define your own shock (%) for every holding and see the impact.")
custom_shocks = {}
cols = st.columns(len(tickers))
for col, t in zip(cols, tickers):
    custom_shocks[t] = col.number_input(f"{t} shock %", value=-10.0, step=5.0, key=f"shock_{t}") / 100

custom_result = run_stress_test("custom", weights, custom_shocks, portfolio_value)
st.metric(
    "Custom scenario portfolio impact",
    f"{custom_result.portfolio_loss_pct*100:.1f}%",
    delta=f"${custom_result.portfolio_new_value - portfolio_value:,.0f}",
)
