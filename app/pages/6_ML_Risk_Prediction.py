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
from src.features.feature_pipeline import build_features_for_asset
from src.models.evaluate import classification_report
from src.models.preprocessing import chronological_split
from src.models.targets import build_loss_event_target
from src.models.train import classify_risk, select_risk_thresholds, train_model
from src.portfolio.analytics import portfolio_returns

st.set_page_config(page_title="ML Risk Prediction", page_icon="🤖", layout="wide")
st.title("🤖 ML Risk Prediction")
st.caption(
    "Trains a model to estimate the probability that the portfolio loses "
    "more than a threshold over the next few days, using only leakage-safe, "
    "trailing features. This is a demonstration model trained on a single "
    "portfolio history — treat the output as illustrative, not a production "
    "risk signal."
)

cfg = render_sidebar()
tickers, returns = cfg["tickers"], cfg["returns"]
weights = weight_editor(tickers)
port_returns = portfolio_returns(returns, weights)

col1, col2 = st.columns(2)
horizon_days = col1.slider("Target horizon (days)", 3, 10, CONFIG["ml"]["target_horizon_days"])
loss_threshold = col2.slider("Loss threshold", 0.01, 0.10, CONFIG["ml"]["loss_threshold"], 0.01, format="%.2f")

wealth = (1 + port_returns).cumprod()
prices_df = wealth.reset_index()
prices_df.columns = ["date", "adj_close"]
prices_df["volume"] = 1

feats = build_features_for_asset(prices_df).drop(columns=["beta_60d", "corr_market_60d", "volume_zscore_20d"])
target = build_loss_event_target(port_returns.reset_index(drop=True), horizon_days=horizon_days, loss_threshold=loss_threshold)

df = feats.copy()
df["target"] = target.values
df = df.dropna()

min_rows_needed = 300
if len(df) < min_rows_needed:
    st.warning(f"Only {len(df)} usable rows after building features/targets — pick a longer history for a meaningful model.")
    st.stop()

n = len(df)
train_end = df["date"].iloc[int(n * 0.55)]
validation_end = df["date"].iloc[int(n * 0.80)]

split = chronological_split(
    df, train_end=train_end, validation_end=validation_end,
    purge_days=horizon_days, embargo_days=2,
)

if split.train.empty or split.validation.empty or split.test.empty:
    st.warning("Not enough history to form train/validation/test splits with the required purge/embargo gaps.")
    st.stop()

feature_cols = [c for c in df.columns if c not in ("date", "ticker", "target")]

X_train, y_train = split.train[feature_cols], split.train["target"].astype(int)
X_val, y_val = split.validation[feature_cols], split.validation["target"].astype(int)
X_test, y_test = split.test[feature_cols], split.test["target"].astype(int)

# Remove collinearity before training -- fit on TRAIN only, then apply the
# same reduced feature list unchanged to validation/test. See
# docs/PREPROCESSING_AND_FEATURE_SELECTION.md §3 for the full rationale.
from src.models.feature_selection import drop_low_variance_features, iterative_vif_removal

after_variance = drop_low_variance_features(X_train)
feature_cols = iterative_vif_removal(X_train[after_variance], vif_threshold=10.0)
X_train, X_val, X_test = X_train[feature_cols], X_val[feature_cols], X_test[feature_cols]
st.caption(f"Using {len(feature_cols)} features after removing near-constant and collinear ones: {', '.join(feature_cols)}")

model_name = st.selectbox("Model", ["random_forest", "logistic_regression", "hist_gradient_boosting"])

if y_train.nunique() < 2:
    st.warning("Training data contains only one class (no loss events observed) — try a lower loss threshold or longer history.")
    st.stop()

# TWO SEPARATE MODELS (see docs/SECOND_FORENSIC_AUDIT.md, Issue H6): the
# original version used a single model, fit on TRAIN only, for BOTH the
# historical TEST evaluation AND the "current risk" inference shown to
# the user -- conflating two different questions. "How good was this
# methodology, evaluated honestly" needs a model that never saw
# VALIDATION or TEST. "What's my best current estimate" should use every
# bit of labeled data available before TEST (TRAIN+VALIDATION), which
# TEST evaluation deliberately withholds. Using the TRAIN-only model for
# both meant the "current" number was needlessly less informed than it
# could be, for no benefit to the (separately correct) historical
# evaluation.
test_eval_model = train_model(X_train, y_train, model_name, numeric_features=feature_cols)

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val]).reset_index(drop=True)
production_model = train_model(X_train_val, y_train_val, model_name, numeric_features=feature_cols)

# Thresholds are calibrated using the SAME model that will actually
# produce the "current" probability shown to the user (production_model),
# evaluated on the same TRAIN+VALIDATION data it was fit on -- an
# in-sample calibration, unavoidable without a second holdout, but at
# least self-consistent with the model doing the real inference below.
production_proba_train_val = pd.Series(production_model.pipeline.predict_proba(X_train_val)[:, 1])
thresholds = select_risk_thresholds(production_proba_train_val, y_train_val)

test_proba = test_eval_model.pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

report = classification_report(y_test, pd.Series(test_pred), pd.Series(test_proba))

st.subheader("Held-out test performance")
st.caption(
    "Computed on the TEST split using a model fit on TRAIN only — never "
    "used for training or threshold selection, and never the same fitted "
    "model as the one used for the current risk estimate below (see "
    "docs/SECOND_FORENSIC_AUDIT.md Issue H6)."
)
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", f"{report['roc_auc']:.3f}" if report["roc_auc"] == report["roc_auc"] else "N/A")
col2.metric("Recall (risk events caught)", f"{report['recall']*100:.0f}%")
col3.metric("False negative rate", f"{report['false_negative_rate']*100:.0f}%")
col4.metric("Precision", f"{report['precision']*100:.0f}%")

st.divider()
st.subheader("Current risk estimate")
st.caption(
    "Uses a SEPARATE model fit on TRAIN+VALIDATION (more data than the "
    "TEST-evaluation model above), since this estimate isn't the one "
    "being scored against TEST — see docs/SECOND_FORENSIC_AUDIT.md Issue H6."
)
latest_features = feats[feature_cols].dropna().iloc[[-1]]
if latest_features.empty:
    st.info("Not enough recent data to compute a current prediction.")
else:
    current_proba = float(production_model.pipeline.predict_proba(latest_features)[:, 1][0])
    current_class = classify_risk(current_proba, thresholds)
    color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[current_class]
    st.markdown(
        f"### Estimated probability of a >{loss_threshold*100:.0f}% loss "
        f"in the next {horizon_days} trading days: **{current_proba*100:.1f}%**"
    )
    st.markdown(f"### Risk classification: :{color}[**{current_class}**]")

    if not cfg["used_synthetic"]:
        if st.button("💾 Save this prediction to MySQL", key="save_risk_prediction"):
            from datetime import date as _date

            from src.data.repository import Repository

            repo = Repository()
            portfolio_id = repo.get_or_create_portfolio(f"dashboard-{'-'.join(tickers)}")
            model_run_id = repo.get_or_create_model_run(
                experiment_id=f"dashboard-{model_name}-{_date.today().isoformat()}",
                model_name=model_name,
            )
            repo.write_risk_prediction(
                portfolio_id, model_run_id, _date.today().isoformat(), current_proba, current_class,
            )
            st.success(f"Saved to `risk_predictions` (portfolio_id={portfolio_id}, model_run_id={model_run_id}).")
    else:
        st.caption("💾 Saving to MySQL is only available in MySQL data-source mode (synthetic demo tickers aren't in the `assets` table).")

st.caption(
    "Thresholds are absolute, outcome-calibrated precision levels on "
    "TRAIN+VALIDATION data only (never chosen using the test set) -- see "
    "docs/SECOND_FORENSIC_AUDIT.md Issue H4."
)
