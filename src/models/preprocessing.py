"""
Chronological splitting, purge/embargo, and preprocessing for the ML pipeline.

See docs/DATA_LEAKAGE_AUDIT.md §4, §9 for the full explanation of why purge
and embargo periods are needed whenever a target spans multiple future days.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG

TRAIN_END = pd.Timestamp(CONFIG["ml"]["train_end"])
VALIDATION_END = pd.Timestamp(CONFIG["ml"]["validation_end"])
PURGE_DAYS = CONFIG["ml"]["purge_days"]
EMBARGO_DAYS = CONFIG["ml"]["embargo_days"]


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: pd.Timestamp = TRAIN_END,
    validation_end: pd.Timestamp = VALIDATION_END,
    purge_days: int = PURGE_DAYS,
    embargo_days: int = EMBARGO_DAYS,
) -> ChronologicalSplit:
    """Split a time-ordered DataFrame into TRAIN / VALIDATION / TEST with a
    purge gap removed from the end of each earlier split and an embargo gap
    removed from the start of each later split.

    Why both purge AND embargo: a training row at date t whose target looks
    forward to t+horizon can overlap the validation period's *features* if
    the two are adjacent. Purging removes the last `purge_days` rows of the
    earlier split (so its target window can't reach into the next split),
    and embargo removes the first `embargo_days` rows of the later split
    (so its target window can't reach backward into information that was
    influenced by the earlier split's own targets). Using both is a common,
    conservative choice for horizon-based classification targets (see
    docs/DATA_LEAKAGE_AUDIT.md §9); either alone can leave a thin overlap.

    TRADING DAYS, NOT CALENDAR DAYS (see docs/SECOND_FORENSIC_AUDIT.md,
    Issue H2): `purge_days`/`embargo_days` are documented (config.yaml,
    src/models/targets.py) as trading-day counts -- `purge_days` must be
    `>= target_horizon_days`, itself a trading-day horizon. An earlier
    version of this function subtracted/added `pd.Timedelta(days=N)`,
    which is CALENDAR days: a 5-calendar-day purge typically spans only
    3-4 actual trading-day observations depending on weekend placement,
    silently under-purging relative to the stated intent. This version
    counts purge/embargo in actual ROW POSITIONS within the sorted,
    one-row-per-trading-day sequence, so `purge_days=5` always excludes
    exactly 5 real trading-day observations, regardless of how many
    calendar days (weekends, holidays) those 5 rows happen to span.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col]

    train_end_candidates = df.index[dates <= train_end]
    if len(train_end_candidates) == 0:
        raise ValueError(f"No rows on or before train_end={train_end}")
    train_end_idx = int(train_end_candidates.max())

    validation_end_candidates = df.index[dates <= validation_end]
    if len(validation_end_candidates) == 0:
        raise ValueError(f"No rows on or before validation_end={validation_end}")
    validation_end_idx = int(validation_end_candidates.max())

    # Purge/embargo measured in trading-day ROW POSITIONS, both anchored
    # to the same nominal boundary (train_end_idx / validation_end_idx),
    # matching the original semantics: purge trims backward from the
    # boundary, embargo skips forward from the SAME boundary (not
    # stacked on top of purge) -- i.e. the boundary itself is excluded
    # from both splits, with `purge_days` trading days removed before it
    # and `embargo_days` trading days removed after it.
    train_last_idx = train_end_idx - purge_days
    val_first_idx = train_end_idx + embargo_days + 1
    val_last_idx = validation_end_idx - purge_days
    test_first_idx = validation_end_idx + embargo_days + 1

    train_end_slice = max(train_last_idx + 1, 0)
    val_start_slice = max(val_first_idx, 0)
    val_end_slice = max(val_last_idx + 1, 0)
    test_start_slice = max(test_first_idx, 0)

    train = df.iloc[:train_end_slice]
    validation = df.iloc[val_start_slice:val_end_slice] if val_end_slice > val_start_slice else df.iloc[0:0]
    test = df.iloc[test_start_slice:]

    return ChronologicalSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def build_preprocessing_pipeline(
    numeric_features: list[str], use_power_transform: bool = False
) -> ColumnTransformer:
    """Impute (median, fit on train only) then scale (fit on train only).

    `use_power_transform=True` inserts a Yeo-Johnson power transform
    (`sklearn.preprocessing.PowerTransformer`) BEFORE scaling. Why
    Yeo-Johnson specifically, and why conditionally rather than always-on:

      - Several engineered features are structurally skewed: volatility
        (vol_5d/10d/20d/60d) is bounded at zero and right-skewed by
        construction (a std-dev can't be negative), and
        `momentum_vol_interaction` inherits that skew from `vol_20d`. A
        linear model (Logistic Regression) fits an additive effect in the
        feature's own scale, so heavy skew means a few extreme values can
        dominate the fitted coefficient far more than the bulk of
        typical-day observations should justify.
      - Box-Cox (the classical power transform) requires strictly positive
        input; several features here (returns, momentum, beta) can be
        negative, ruling it out. Yeo-Johnson is the standard extension that
        supports zero and negative values while keeping the same goal
        (making a skewed distribution closer to Normal), which is exactly
        why it's used here instead.
      - It's conditional (`use_power_transform`) rather than unconditional
        because tree-based models (Random Forest/HistGradientBoostingClassifier) are invariant to
        monotonic transformations of a single feature -- a power transform
        changes nothing about the splits a tree would choose, so applying
        it there adds computation for zero benefit. It's intended for the
        Logistic Regression path; see
        docs/PREPROCESSING_AND_FEATURE_SELECTION.md §2 for the full
        before/after skewness comparison.
      - Like the imputer and scaler, `PowerTransformer` is fit on
        X_train only and applied via `.transform()` to validation/test --
        the same leakage rule applies to every step in this pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_power_transform:
        from sklearn.preprocessing import PowerTransformer

        steps.append(("power_transform", PowerTransformer(method="yeo-johnson", standardize=False)))
    steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, numeric_features)],
        remainder="drop",
    )
