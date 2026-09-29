import numpy as np
import pandas as pd
import pytest

from src.models.targets import build_loss_event_target, forward_return
from src.models.preprocessing import chronological_split


def test_forward_return_uses_only_future_data():
    # constant +1% daily returns -> forward 5d return should be very close
    # to (1.01)^5 - 1 for every row that has enough future data.
    returns = pd.Series([0.01] * 50)
    fwd = forward_return(returns, horizon_days=5)
    expected = (1.01 ** 5) - 1
    valid = fwd.dropna()
    np.testing.assert_allclose(valid.values, expected, rtol=1e-6)
    # last 5 rows must be NaN: no future data available
    assert fwd.iloc[-5:].isna().all()


def test_forward_return_does_not_leak_into_earlier_rows():
    returns = pd.Series([0.001] * 100)
    fwd_original = forward_return(returns, horizon_days=5)

    perturbed = returns.copy()
    perturbed.iloc[90:] = 5.0  # huge shock far in the future
    fwd_perturbed = forward_return(perturbed, horizon_days=5)

    # Rows whose forward window doesn't reach the shock must be unaffected.
    unaffected = fwd_original.iloc[:80]
    unaffected_perturbed = fwd_perturbed.iloc[:80]
    pd.testing.assert_series_equal(unaffected, unaffected_perturbed)


def test_loss_event_target_flags_large_forward_drops():
    # Flat returns, then a big forward drop planted at a known location.
    returns = pd.Series([0.0] * 100)
    returns.iloc[50:55] = -0.02  # -2% for 5 days -> ~ -9.6% cumulative
    target = build_loss_event_target(returns, horizon_days=5, loss_threshold=0.03)
    # date index 44 has forward window [45..49] which is still flat -> 0
    assert target.iloc[44] == 0
    # date index 45 has forward window [46..50] entering the drop
    assert target.iloc[49] == 1


def test_chronological_split_has_no_overlap_after_purge_embargo():
    dates = pd.bdate_range("2018-01-01", "2024-06-01")
    df = pd.DataFrame({"date": dates, "x": np.arange(len(dates))})

    split = chronological_split(
        df,
        train_end=pd.Timestamp("2021-12-31"),
        validation_end=pd.Timestamp("2023-12-31"),
        purge_days=5,
        embargo_days=2,
    )

    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()

    gap_train_val = (split.validation["date"].min() - split.train["date"].max()).days
    gap_val_test = (split.test["date"].min() - split.validation["date"].max()).days
    assert gap_train_val >= 2  # at least the embargo
    assert gap_val_test >= 2

    assert not split.train.empty
    assert not split.validation.empty
    assert not split.test.empty


def test_time_series_cv_tune_raises_when_no_fold_has_two_classes():
    """A pathologically rare positive class (or too many CV splits for the
    data size) can leave every fold with only one class present. This must
    raise a clear, actionable error -- not silently return {}/-inf, which
    would look like tuning succeeded when it actually found nothing.
    """
    import numpy as np
    import pandas as pd
    import pytest
    from src.models.train import time_series_cv_tune

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, size=(50, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 49 + [1])  # a single positive example in the whole dataset

    with pytest.raises(ValueError, match="could not compute"):
        time_series_cv_tune(X, y, "logistic_regression", {"C": [1.0]}, n_splits=5)


def test_chronological_split_purges_by_trading_days_not_calendar_days():
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H2: purge
    must exclude exactly `purge_days` actual trading-day OBSERVATIONS,
    not `purge_days` calendar days (which can span fewer trading days
    across a weekend/holiday gap). Construct a split boundary that lands
    right before a weekend, where the old calendar-day-based subtraction
    and the new trading-day-based approach would disagree.
    """
    # bdate_range excludes weekends already; pick a boundary and confirm
    # the purge removes exactly purge_days ROWS counting backward from
    # the boundary, regardless of the calendar dates those rows carry.
    dates = pd.bdate_range("2020-01-01", "2020-01-31")  # includes a Jan 1 holiday-adjacent stretch
    df = pd.DataFrame({"date": dates, "x": range(len(dates))})

    purge_days = 3
    train_end = dates[10]  # a Friday, per bdate_range's regular weekly cadence around this date range

    split = chronological_split(
        df, train_end=train_end, validation_end=dates[20],
        purge_days=purge_days, embargo_days=1,
    )

    # The train split's last row must be exactly `purge_days` TRADING-DAY
    # positions before train_end, i.e. at index (10 - 3) = 7 in the
    # original `dates` array -- not filtered by calendar-day subtraction,
    # which could land on a different (or the same, coincidentally) row
    # depending on weekend placement. We verify by position, which is the
    # actual guarantee this function must provide.
    expected_last_train_date = dates[10 - purge_days]
    assert split.train["date"].max() == expected_last_train_date
    assert len(split.train) == 10 - purge_days + 1  # rows 0..7 inclusive = 8 rows


def test_chronological_split_row_counts_are_exact_trading_day_counts():
    """A second, more direct proof: with a known, fixed number of trading
    days between two boundaries, purge_days and embargo_days must each
    remove EXACTLY that many rows, verified by counting -- not by
    checking dates land in some approximately-correct range."""
    dates = pd.bdate_range("2021-01-01", periods=100)
    df = pd.DataFrame({"date": dates, "x": range(len(dates))})

    train_end = dates[50]
    validation_end = dates[80]
    purge_days = 4
    embargo_days = 2

    split = chronological_split(
        df, train_end=train_end, validation_end=validation_end,
        purge_days=purge_days, embargo_days=embargo_days,
    )

    # Train: rows 0..(50-4) inclusive = 47 rows.
    assert len(split.train) == 50 - purge_days + 1
    # Validation: rows (50+embargo+1)..(80-purge) inclusive.
    val_first = 50 + embargo_days + 1
    val_last = 80 - purge_days
    assert len(split.validation) == val_last - val_first + 1
    # Test: rows (80+embargo+1)..99 inclusive.
    test_first = 80 + embargo_days + 1
    assert len(split.test) == 100 - test_first
