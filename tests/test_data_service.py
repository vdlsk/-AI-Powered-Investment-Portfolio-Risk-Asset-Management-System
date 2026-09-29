import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def test_synthetic_seed_is_stable_across_processes():
    """Regression test for a real bug: the old code used Python's built-in
    `hash(ticker)`, which is randomized per-process by default
    (PYTHONHASHSEED), so the same ticker produced DIFFERENT synthetic data
    every time a new process started -- silently breaking the
    "reproducible demo data" guarantee. This test proves the fix by
    actually spawning a separate Python process (not just calling the
    function twice in-process, which would not have caught the original
    bug at all) and comparing output byte-for-byte.
    """
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "from app.components.data_loader import _synthetic_price_panel; "
        "panel = _synthetic_price_panel(('AAPL',), '2020-01-01', '2020-03-01'); "
        "print(panel['AAPL'].to_csv())"
    )

    result_1 = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    result_2 = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert result_1.stdout == result_2.stdout
    assert len(result_1.stdout) > 0  # sanity: the script actually produced output


def test_stable_seed_differs_between_tickers():
    from app.components.data_loader import _stable_seed

    assert _stable_seed("AAPL") != _stable_seed("MSFT")


def test_stable_seed_is_deterministic_within_process():
    from app.components.data_loader import _stable_seed

    assert _stable_seed("AAPL") == _stable_seed("AAPL")
    assert isinstance(_stable_seed("AAPL"), int)
    assert 0 <= _stable_seed("AAPL") < 2**32


# --------------------------------------------------------------------- #
# Data service: dashboard must read from MySQL, never silently fall back
# --------------------------------------------------------------------- #
def test_data_service_never_calls_provider(monkeypatch):
    """The dashboard's data service must read exclusively through
    Repository/MySQL and must never import or call MarketDataProvider --
    that class belongs only to the explicit ingestion pipeline
    (src/data/ingestion.py), not to normal dashboard analytics.
    """
    from src.data import service

    called = {"provider_used": False}

    class FakeProviderThatShouldNeverRun:
        def __init__(self):
            called["provider_used"] = True

    monkeypatch.setattr("src.data.ingestion.MarketDataProvider", FakeProviderThatShouldNeverRun)

    class FakeRepository:
        def read_clean_prices(self, tickers):
            return pd.DataFrame(columns=["ticker", "date", "adj_close"])

    with pytest.raises(service.NoDataInDatabaseError):
        service.DataService(repository=FakeRepository()).get_price_panel(["AAPL"])

    assert called["provider_used"] is False


def test_no_silent_fallback_on_missing_data():
    """Requesting data through DataService when MySQL has none for the
    requested tickers must raise a clear, typed error -- never silently
    return synthetic or empty data disguised as a real result.
    """
    from src.data.service import DataService, NoDataInDatabaseError

    class EmptyRepository:
        def read_clean_prices(self, tickers):
            return pd.DataFrame(columns=["ticker", "date", "adj_close"])

    service = DataService(repository=EmptyRepository())
    with pytest.raises(NoDataInDatabaseError, match="No market data found in MySQL"):
        service.get_price_panel(["AAPL", "MSFT"])


def test_data_service_returns_wide_panel_when_data_exists():
    from src.data.service import DataService

    class FakeRepositoryWithData:
        def read_clean_prices(self, tickers):
            dates = pd.bdate_range("2024-01-01", periods=5)
            rows = []
            for t in tickers:
                for i, d in enumerate(dates):
                    rows.append({"ticker": t, "date": d, "adj_close": 100 + i})
            return pd.DataFrame(rows)

    service = DataService(repository=FakeRepositoryWithData())
    panel = service.get_price_panel(["AAPL", "MSFT"])
    assert list(panel.columns) == ["AAPL", "MSFT"]
    assert len(panel) == 5


def test_synthetic_price_panel_works_with_no_end_date():
    """Regression test for a real bug caught via AppTest during
    development: `_synthetic_price_panel`'s `end` parameter had no default
    value, so calling it with only (tickers, start) -- exactly how the
    dashboard's synthetic-mode branch calls it -- raised
    `TypeError: missing 1 required positional argument: 'end'`.
    """
    from app.components.data_loader import _synthetic_price_panel

    panel = _synthetic_price_panel(("AAPL", "MSFT"), "2023-01-01")
    assert not panel.empty
    assert list(panel.columns) == ["AAPL", "MSFT"]


def test_data_service_returns_datetime64_index_not_raw_date_objects():
    """Regression test for a real bug caught via AppTest during
    development: pandas.read_sql returns MySQL DATE columns as plain
    `datetime.date` objects, not pandas Timestamps. Pivoting on that
    column without converting first produces an `object`-dtype index that
    silently breaks standard comparisons like `panel.index >= pd.Timestamp(x)`
    (raised `TypeError: Cannot compare Timestamp with datetime.date`).
    """
    import datetime as dt

    from src.data.service import DataService

    class FakeRepositoryWithRawDateObjects:
        def read_clean_prices(self, tickers):
            # Simulate exactly what pandas.read_sql returns for a MySQL
            # DATE column: plain datetime.date objects, not Timestamps.
            raw_dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
            rows = []
            for t in tickers:
                for i, d in enumerate(raw_dates):
                    rows.append({"ticker": t, "date": d, "adj_close": 100 + i})
            return pd.DataFrame(rows)

    service = DataService(repository=FakeRepositoryWithRawDateObjects())
    panel = service.get_price_panel(["AAPL"])

    assert pd.api.types.is_datetime64_any_dtype(panel.index)
    # The comparison that previously crashed must now work cleanly.
    filtered = panel[panel.index >= pd.Timestamp("2024-01-02")]
    assert len(filtered) == 2
