"""
Market regime detection.

Two distinct modes, matching docs/DATA_LEAKAGE_AUDIT.md §11:

  * `fit_exploratory_regimes`: fits ONE clustering model on the entire
    historical dataset. Useful for visualizing "what regimes has this
    market been in historically", but NOT valid for backtesting a strategy
    that reacts to regime, because the model has already seen the whole
    future when labeling any given date. Every row produced this way is
    marked `is_exploratory=True` and must never feed into a backtest.

  * `fit_walk_forward_regimes`: refits the clustering model at each step of
    an expanding window, so the regime label assigned to date t only used
    information available up to date t. This is the only mode allowed to
    feed a walk-forward backtest.

Features used for clustering: trailing volatility and trailing return
(both already leakage-safe, trailing-window quantities from
src/features/feature_pipeline.py) -- never a centered or forward-looking
statistic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG

logger = logging.getLogger(__name__)

RANDOM_SEED = CONFIG["general"]["random_seed"]
N_REGIMES = 3  # e.g. low-vol bull / choppy / high-vol bear

REGIME_FEATURE_COLUMNS = ["ret_20d", "vol_20d"]


def _label_regimes_by_risk(centers: np.ndarray) -> dict[int, str]:
    """Assign human-readable labels to cluster indices by ranking cluster
    centers on volatility (2nd feature column), since raw KMeans cluster
    indices (0,1,2,...) are arbitrary and would otherwise flip between
    refits.
    """
    vol_col = 1
    order = np.argsort(centers[:, vol_col])  # ascending volatility
    names = ["low_vol", "medium_vol", "high_vol"][: len(order)]
    return {cluster_idx: names[rank] for rank, cluster_idx in enumerate(order)}


def fit_exploratory_regimes(
    features: pd.DataFrame, n_regimes: int = N_REGIMES, random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Fit KMeans on the FULL history at once. Output rows are marked
    is_exploratory=True; see module docstring for why this must never be
    used in a backtest.
    """
    df = features.dropna(subset=REGIME_FEATURE_COLUMNS).copy()
    X = StandardScaler().fit_transform(df[REGIME_FEATURE_COLUMNS])

    model = KMeans(n_clusters=n_regimes, random_state=random_seed, n_init=10)
    labels = model.fit_predict(X)
    label_map = _label_regimes_by_risk(model.cluster_centers_)

    df["regime_label"] = [label_map[l] for l in labels]
    df["is_exploratory"] = True
    return df[["date", "regime_label", "is_exploratory"]]


def fit_walk_forward_regimes(
    features: pd.DataFrame,
    min_train_days: int = 252,
    refit_every_days: int = 21,
    n_regimes: int = N_REGIMES,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Expanding-window regime detection safe for backtesting.

    At each refit point, fit KMeans using only data up to (and including)
    that point, then label subsequent dates (until the next refit) with
    that frozen model -- never re-labeling past dates once a later refit
    happens.
    """
    df = features.dropna(subset=REGIME_FEATURE_COLUMNS).sort_values("date").reset_index(drop=True)
    if len(df) < min_train_days + 1:
        raise ValueError(
            f"Not enough history ({len(df)} rows) for min_train_days={min_train_days}."
        )

    results = []
    i = min_train_days
    while i < len(df):
        train_slice = df.iloc[:i]
        label_slice = df.iloc[i : i + refit_every_days]
        if label_slice.empty:
            break

        scaler = StandardScaler().fit(train_slice[REGIME_FEATURE_COLUMNS])
        X_train = scaler.transform(train_slice[REGIME_FEATURE_COLUMNS])
        model = KMeans(n_clusters=n_regimes, random_state=random_seed, n_init=10).fit(X_train)
        label_map = _label_regimes_by_risk(model.cluster_centers_)

        X_label = scaler.transform(label_slice[REGIME_FEATURE_COLUMNS])
        preds = model.predict(X_label)

        out = label_slice[["date"]].copy()
        out["regime_label"] = [label_map[p] for p in preds]
        out["is_exploratory"] = False
        results.append(out)

        i += refit_every_days

    result_df = pd.concat(results, ignore_index=True)
    logger.info(
        "Walk-forward regimes: %d labeled rows from %d total (%d used only as warm-up train data)",
        len(result_df), len(df), min_train_days,
    )
    return result_df
