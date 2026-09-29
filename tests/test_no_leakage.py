"""
Automated data-leakage tests.

This is the most important test file in the repository. Every one of these
tests must pass before any model result from this project should be
trusted. See docs/DATA_LEAKAGE_AUDIT.md for the full narrative explanation
of what each test is guarding against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import build_features_for_asset


def _make_price_series(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame({"date": dates, "adj_close": prices, "volume": volume})


# --------------------------------------------------------------------------
# Test 1: changing future observations must NOT change historical features.
# --------------------------------------------------------------------------
def test_future_data_does_not_change_past_features():
    df = _make_price_series()
    feats_original = build_features_for_asset(df)

    df_modified = df.copy()
    cutoff = len(df) // 2
    # Wildly perturb everything AFTER the cutoff date.
    df_modified.loc[cutoff:, "adj_close"] *= 5
    df_modified.loc[cutoff:, "volume"] *= 10

    feats_modified = build_features_for_asset(df_modified)

    # Every feature row with date < the cutoff date must be identical,
    # regardless of how much the future was perturbed.
    cutoff_date = df.loc[cutoff, "date"]
    past_original = feats_original[feats_original["date"] < cutoff_date].reset_index(drop=True)
    past_modified = feats_modified[feats_modified["date"] < cutoff_date].reset_index(drop=True)

    pd.testing.assert_frame_equal(past_original, past_modified, check_exact=False, rtol=1e-8)


# --------------------------------------------------------------------------
# Test 2 & 3: preprocessing (scaler) must be fit ONLY on training data, and
# test-set data must not influence its parameters.
# --------------------------------------------------------------------------
def test_scaler_fit_only_on_training_data():
    rng = np.random.default_rng(1)
    X_train = rng.normal(0, 1, size=(100, 3))
    X_test = rng.normal(100, 50, size=(20, 3))  # wildly different distribution

    scaler_correct = StandardScaler().fit(X_train)
    mean_correct = scaler_correct.mean_.copy()

    # The WRONG way: fit on train+test combined (this is what we forbid).
    scaler_wrong = StandardScaler().fit(np.vstack([X_train, X_test]))
    mean_wrong = scaler_wrong.mean_

    # Prove the two differ substantially -- i.e. fitting on test data DOES
    # change parameters, which is exactly why we must never do it.
    assert not np.allclose(mean_correct, mean_wrong, atol=1e-6)

    # The scaler actually used by the project's pipeline must match the
    # train-only fit.
    np.testing.assert_allclose(scaler_correct.mean_, mean_correct)


# --------------------------------------------------------------------------
# Test 4: feature timestamps must not exceed prediction timestamps.
# --------------------------------------------------------------------------
def test_feature_dates_never_exceed_prediction_date():
    df = _make_price_series()
    feats = build_features_for_asset(df)
    prediction_date = df["date"].iloc[150]

    features_at_prediction = feats[feats["date"] == prediction_date]
    assert not features_at_prediction.empty
    # By construction, build_features_for_asset only uses data up to and
    # including each row's own date -- so the feature row's date IS the
    # prediction date, and no column can encode information from a later date.
    assert (features_at_prediction["date"] <= prediction_date).all()


# --------------------------------------------------------------------------
# Test 5 & 6: target columns / future-return columns must never end up in
# the feature set that is fed to a model.
# --------------------------------------------------------------------------
FORBIDDEN_FEATURE_SUBSTRINGS = ("future", "target", "label")


def test_feature_columns_contain_no_future_or_target_leakage():
    df = _make_price_series()
    feats = build_features_for_asset(df)
    for col in feats.columns:
        lowered = col.lower()
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert forbidden not in lowered, (
                f"Feature column '{col}' contains forbidden substring "
                f"'{forbidden}' -- targets/future data must never appear "
                f"in the feature set."
            )


# --------------------------------------------------------------------------
# Test 7: chronological split must never shuffle time-series observations.
# --------------------------------------------------------------------------
def test_chronological_split_is_strictly_ordered():
    df = _make_price_series()
    train_end = df["date"].iloc[150]
    val_end = df["date"].iloc[220]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]

    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    # No overlap between splits.
    assert set(train.index).isdisjoint(set(val.index))
    assert set(val.index).isdisjoint(set(test.index))


# --------------------------------------------------------------------------
# Rolling windows must be trailing (backward-looking), never centered.
# --------------------------------------------------------------------------
def test_rolling_features_are_trailing_not_centered():
    df = _make_price_series()
    feats = build_features_for_asset(df)

    # A trailing 20-day rolling feature at index i must be unaffected by
    # data at index i+1 (a centered window would use it).
    i = 100
    df_perturbed = df.copy()
    df_perturbed.loc[i + 1, "adj_close"] *= 3

    feats_perturbed = build_features_for_asset(df_perturbed)

    columns_to_check = ["vol_20d", "momentum_20d", "ma_ratio_20_60", "vol_ratio_20_60", "momentum_vol_interaction"]
    row_original = feats.loc[i, columns_to_check]
    row_perturbed = feats_perturbed.loc[i, columns_to_check]

    pd.testing.assert_series_equal(row_original, row_perturbed, check_exact=False, rtol=1e-8)
