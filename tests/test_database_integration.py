"""
Integration tests against a REAL, running MySQL database.

Unlike every other test file in this project, these tests do not run
against in-memory pandas objects — they open an actual connection, load
the actual `database/schema.sql`, and verify real database behavior:
foreign key enforcement, unique constraints, CASCADE deletes, and the
NaN -> SQL NULL conversion that pymysql requires but pandas doesn't.

WHY THIS FILE EXISTS: a real bug was caught here during development that
no unit test in this project could have found. `Repository.write_features`
passed raw pandas NaN values (present by design during rolling-window
warm-up periods) straight to pymysql's parameter binding, which raises
`ProgrammingError: nan can not be used with MySQL` -- pymysql has no
implicit NaN -> NULL conversion, unlike pandas/numpy which tolerate NaN
everywhere. Every other test in this project mocks or stays in-memory and
so never exercised this code path. See `_records_with_null_safe_nans` in
`src/data/repository.py` for the fix.

ISOLATED TEST DATABASE (see docs/SECOND_FORENSIC_AUDIT.md, Issue C3):
these tests run destructive `DELETE FROM` statements before every test to
get a clean slate. An earlier version of this file connected via the same
`MYSQL` settings the application itself uses -- meaning `pytest tests/ -v`
(this project's own documented first command) would silently wipe a
developer's real ingested data. This file now connects EXCLUSIVELY via
`src.config.get_test_mysql_settings()`, which resolves to a distinct
database (`MYSQL_TEST_DATABASE` env var, or `f"{MYSQL_DATABASE}_test"` by
default -- e.g. `finrisk` -> `finrisk_test`), and there is no code path
here that can silently fall back to the application's real database.

These tests are skipped automatically (not failed) if no MySQL server is
reachable, since not every environment running `pytest tests/` will have
a database available -- this matches standard practice for integration
tests that depend on external infrastructure. To run them:

    sudo service mysql start
    mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS finrisk_test"
    mysql -u root -p finrisk_test < database/schema.sql
    cp .env.example .env  # fill in real credentials (MYSQL_DATABASE=finrisk is fine;
                           # the test database is derived automatically as finrisk_test)
    pytest tests/test_database_integration.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from src.config import MYSQL, get_test_mysql_settings


def _mysql_test_db_available() -> bool:
    """Provision the ISOLATED TEST database from scratch (not the
    application's real database) and confirm connectivity.

    Unconditionally DROP + CREATE + reload schema.sql every time this is
    called, rather than checking "do the expected tables already exist"
    first. That existence-check approach was tried and found insufficient
    during development: it only checked TABLE NAMES, not column
    correctness, so a test database provisioned from an earlier schema
    version (before a migration added new columns) silently passed the
    check and caused a real `Unknown column` error on first use, rather
    than being detected and reprovisioned. Since this is a disposable test
    database (never the application's real one -- see
    `get_test_mysql_settings`), always reprovisioning fresh is cheap and
    removes an entire category of "stale test database" bugs.
    """
    test_settings = get_test_mysql_settings()
    try:
        server_url = test_settings.sqlalchemy_url.rsplit("/", 1)[0]
        server_engine = create_engine(server_url, pool_pre_ping=True)
        with server_engine.begin() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(text(f"DROP DATABASE IF EXISTS `{test_settings.database}`"))
            conn.execute(text(f"CREATE DATABASE `{test_settings.database}`"))

        import os
        import pathlib
        import subprocess

        schema_path = pathlib.Path(__file__).resolve().parents[1] / "database" / "schema.sql"
        schema_sql = schema_path.read_text()
        # schema.sql's own CREATE DATABASE/USE statements (a multi-line
        # CREATE DATABASE ... CHARACTER SET ... COLLATE ...; followed by
        # USE ...;) target the application's database name; strip exactly
        # those two leading statements (split on the statement
        # terminator, not a per-line prefix check -- a per-line check
        # previously left the CHARACTER SET/COLLATE continuation lines
        # dangling as invalid orphaned SQL) and run the remaining DDL
        # against the already-created, already-selected test database.
        _, _, remaining_after_use = schema_sql.partition(";")  # drop "CREATE DATABASE ... utf8mb4_unicode_ci"
        _, _, ddl_only = remaining_after_use.partition(";")    # drop "USE finrisk"

        env = dict(os.environ)
        if test_settings.password:
            env["MYSQL_PWD"] = test_settings.password
        subprocess.run(
            [
                "mysql",
                f"--host={test_settings.host}",
                f"--port={test_settings.port}",
                f"--user={test_settings.user}",
                test_settings.database,
            ],
            input=ddl_only,
            text=True,
            env=env,
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        # Broad on purpose here: this is a test-collection-time
        # availability probe (schema provisioning via subprocess can
        # raise CalledProcessError, connectivity issues raise
        # OperationalError, etc.) -- any failure means "can't run these
        # integration tests in this environment," which should skip the
        # tests, not crash pytest collection.
        return False


requires_mysql = pytest.mark.skipif(
    not _mysql_test_db_available(),
    reason="No reachable/provisionable MySQL TEST database (see this file's docstring to set one up)",
)


def test_test_database_is_distinct_from_application_database():
    """Guard against this exact mistake (Issue C3) ever silently
    reappearing: the integration-test database name must never equal the
    application's configured database name.
    """
    test_settings = get_test_mysql_settings()
    assert test_settings.database != MYSQL.database


@pytest.fixture
def repo():
    from src.data.repository import Repository

    test_settings = get_test_mysql_settings()
    from sqlalchemy import create_engine as _create_engine

    engine = _create_engine(test_settings.sqlalchemy_url, pool_pre_ping=True)
    r = Repository(engine=engine)
    # Clean slate: delete in FK-safe order (children first). This is safe
    # because `r.engine` is guaranteed (by get_test_mysql_settings) to
    # point at the isolated test database, never the application's real one.
    with r.engine.begin() as conn:
        conn.execute(text("DELETE FROM market_features"))
        conn.execute(text("DELETE FROM clean_market_prices"))
        conn.execute(text("DELETE FROM raw_market_prices"))
        conn.execute(text("DELETE FROM assets"))
    yield r


@requires_mysql
def test_schema_loaded_all_expected_tables(repo):
    """Confirm database/schema.sql actually loaded successfully into a
    real server -- this is a live check that the DDL is valid SQL, not
    just plausible-looking text."""
    expected_tables = {
        "assets", "raw_market_prices", "clean_market_prices", "market_features",
        "portfolio", "portfolio_holdings", "portfolio_transactions", "risk_metrics",
        "market_regimes", "stress_scenarios", "model_runs", "risk_predictions",
        "backtest_results", "rebalance_recommendations",
    }
    with repo.engine.connect() as conn:
        tables = {row[0] for row in conn.execute(text("SHOW TABLES"))}
    assert expected_tables.issubset(tables)


@requires_mysql
def test_asset_upsert_is_idempotent(repo):
    repo.upsert_assets(["AAPL", "MSFT"])
    first_map = repo.get_asset_id_map()
    repo.upsert_assets(["AAPL", "MSFT"])  # calling again must not create duplicates or change IDs
    second_map = repo.get_asset_id_map()
    assert first_map == second_map
    assert len(first_map) == 2


@requires_mysql
def test_raw_and_clean_price_roundtrip(repo):
    repo.upsert_assets(["AAPL"])
    dates = pd.bdate_range("2024-01-01", periods=20)
    df = pd.DataFrame({
        "ticker": "AAPL", "date": dates,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0,
        "volume": 1_000_000, "source": "test",
    })
    n_written = repo.write_raw_prices(df)
    assert n_written == 20

    raw_back = repo.read_raw_prices(["AAPL"])
    assert len(raw_back) == 20

    from src.data.cleaning import clean_prices
    cleaned = clean_prices(raw_back.rename(columns={
        "price_date": "date", "open_price": "open", "high_price": "high",
        "low_price": "low", "close_price": "close",
    }))
    n_clean = repo.write_clean_prices(cleaned)
    assert n_clean == 20

    clean_back = repo.read_clean_prices(["AAPL"])
    assert len(clean_back) == 20


@requires_mysql
def test_feature_write_converts_nan_to_sql_null_not_error(repo):
    """Regression test for the exact bug this file's docstring describes:
    writing feature rows with NaN (from rolling-window warm-up) must
    succeed and be stored as SQL NULL, not raise pymysql ProgrammingError.
    """
    repo.upsert_assets(["AAPL"])
    dates = pd.bdate_range("2024-01-01", periods=5)
    # Simulate a features DataFrame with real NaN, as build_features_for_asset
    # produces during warm-up -- this is exactly what crashed before the fix.
    feats = pd.DataFrame({
        "ticker": "AAPL",
        "date": dates,
        "ret_1d": [np.nan, 0.01, -0.02, 0.005, np.nan],
        "vol_20d": [np.nan] * 5,  # entirely NaN, like real 20d-warmup rows
    })

    n_written = repo.write_features(feats)
    assert n_written == 5

    with repo.engine.connect() as conn:
        null_count = conn.execute(
            text("SELECT COUNT(*) FROM market_features WHERE vol_20d IS NULL")
        ).scalar()
        non_null_ret_count = conn.execute(
            text("SELECT COUNT(*) FROM market_features WHERE ret_1d IS NOT NULL")
        ).scalar()
    assert null_count == 5  # every vol_20d was NaN -> every row stored as SQL NULL
    assert non_null_ret_count == 3  # 3 of 5 ret_1d values were real numbers


@requires_mysql
def test_foreign_key_constraint_rejects_orphaned_row(repo):
    """A raw_market_prices row referencing a non-existent asset_id must be
    rejected by the database itself (not just by application-level checks)."""
    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO raw_market_prices (asset_id, price_date, close_price, adj_close, source) "
                    "VALUES (999999, '2024-01-01', 100, 100, 'test')"
                )
            )


@requires_mysql
def test_unique_constraint_prevents_duplicate_asset_date(repo):
    repo.upsert_assets(["AAPL"])
    asset_id = repo.get_asset_id_map()["AAPL"]
    with repo.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO raw_market_prices (asset_id, price_date, close_price, adj_close, source) "
                "VALUES (:aid, '2024-01-01', 100, 100, 'test')"
            ),
            {"aid": asset_id},
        )
    with pytest.raises(IntegrityError):
        with repo.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO raw_market_prices (asset_id, price_date, close_price, adj_close, source) "
                    "VALUES (:aid, '2024-01-01', 105, 105, 'test')"
                ),
                {"aid": asset_id},
            )


@requires_mysql
def test_cascade_delete_removes_dependent_rows(repo):
    """Deleting an asset must cascade-delete its raw prices and features
    (ON DELETE CASCADE in the schema) -- verified against real row counts,
    not just assumed from the DDL text."""
    repo.upsert_assets(["AAPL", "MSFT"])
    dates = pd.bdate_range("2024-01-01", periods=10)
    for ticker in ["AAPL", "MSFT"]:
        df = pd.DataFrame({
            "ticker": ticker, "date": dates,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0,
            "volume": 1_000_000, "source": "test",
        })
        repo.write_raw_prices(df)

    with repo.engine.connect() as conn:
        total_before = conn.execute(text("SELECT COUNT(*) FROM raw_market_prices")).scalar()
    assert total_before == 20

    with repo.engine.begin() as conn:
        conn.execute(text("DELETE FROM assets WHERE ticker = 'AAPL'"))

    with repo.engine.connect() as conn:
        total_after = conn.execute(text("SELECT COUNT(*) FROM raw_market_prices")).scalar()
        remaining_tickers = {
            row[0] for row in conn.execute(text(
                "SELECT DISTINCT a.ticker FROM assets a JOIN raw_market_prices r ON r.asset_id = a.asset_id"
            ))
        }
    assert total_after == 10  # AAPL's 10 rows cascade-deleted, MSFT's 10 remain
    assert remaining_tickers == {"MSFT"}


@requires_mysql
def test_all_feature_spec_columns_actually_round_trip_through_mysql(repo):
    """The strongest possible proof for docs/SECOND_FORENSIC_AUDIT.md
    Issue C4: write a features dataframe containing EVERY column in
    FEATURE_SPEC to a real MySQL database, read it back, and confirm
    every single one survived -- not silently dropped, and not rejected
    by the database (which would mean the schema migration didn't
    actually add the missing columns).
    """
    from src.features.feature_pipeline import FEATURE_SPEC

    repo.upsert_assets(["AAPL"])
    dates = pd.bdate_range("2024-01-01", periods=3)
    feats = pd.DataFrame({
        "ticker": "AAPL",
        "date": dates,
        **{col: [0.11, 0.22, 0.33] for col in FEATURE_SPEC},
    })

    n_written = repo.write_features(feats)
    assert n_written == 3

    with repo.engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT * FROM market_features WHERE feature_date = :d"),
            {"d": dates[0].date()},
        ).mappings().first()

    assert row is not None
    for feature_name in FEATURE_SPEC:
        assert feature_name in row, f"'{feature_name}' column doesn't exist in market_features table"
        assert row[feature_name] == pytest.approx(0.11), (
            f"'{feature_name}' was written but did not round-trip correctly "
            f"(got {row[feature_name]!r})"
        )


@requires_mysql
def test_get_or_create_portfolio_is_idempotent(repo):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H11
    (portfolio FK prerequisite): calling get_or_create_portfolio twice
    with the same name must return the same portfolio_id, not create a
    duplicate row."""
    id_1 = repo.get_or_create_portfolio("Test Portfolio")
    id_2 = repo.get_or_create_portfolio("Test Portfolio")
    assert id_1 == id_2

    with repo.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM portfolio WHERE name = 'Test Portfolio'")
        ).scalar()
    assert count == 1


@requires_mysql
def test_risk_metrics_round_trip_through_mysql(repo):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H11:
    risk_metrics previously had zero write path -- verify a real
    performance_report-shaped dict actually persists and reads back."""
    portfolio_id = repo.get_or_create_portfolio("Test Portfolio")
    metrics = {
        "cumulative_return": 0.15,  # not a risk_metrics column -- must be silently ignored, not error
        "annualized_return": 0.10,
        "annualized_volatility": 0.18,
        "sharpe_ratio": 0.55,
        "max_drawdown": -0.12,
        "historical_var": 0.021,
        "parametric_var": float("nan"),  # must become SQL NULL, not crash
        "cvar": 0.035,
    }
    repo.write_risk_metrics(portfolio_id, "2024-01-15", metrics)

    with repo.engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM risk_metrics WHERE portfolio_id = :pid AND metric_date = '2024-01-15'"),
            {"pid": portfolio_id},
        ).mappings().first()

    assert row is not None
    assert row["sharpe_ratio"] == pytest.approx(0.55)
    assert row["var_historical"] == pytest.approx(0.021)  # renamed from historical_var
    assert row["var_parametric"] is None  # NaN -> SQL NULL


@requires_mysql
def test_risk_prediction_round_trip_through_mysql(repo):
    portfolio_id = repo.get_or_create_portfolio("Test Portfolio")
    model_run_id = repo.get_or_create_model_run("test_experiment_001", "logistic_regression")
    repo.write_risk_prediction(portfolio_id, model_run_id, "2024-01-15", 0.42, "MEDIUM")

    with repo.engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM risk_predictions WHERE portfolio_id = :pid"),
            {"pid": portfolio_id},
        ).mappings().first()

    assert row is not None
    assert row["predicted_probability"] == pytest.approx(0.42)
    assert row["predicted_class"] == "MEDIUM"


@requires_mysql
def test_backtest_result_round_trip_through_mysql(repo):
    portfolio_id = repo.get_or_create_portfolio("Test Portfolio")
    performance = {
        "cumulative_return": 0.22,
        "total_transaction_costs": 15.50,
        "sharpe_ratio": 0.61,
        "max_drawdown": -0.14,
        "num_rebalances": 24,
    }
    repo.write_backtest_result("test_backtest_exp_001", portfolio_id, "2020-01-01", "2023-12-31", performance)

    with repo.engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM backtest_results WHERE experiment_id = 'test_backtest_exp_001'"),
        ).mappings().first()

    assert row is not None
    assert row["sharpe_ratio"] == pytest.approx(0.61)
    assert row["num_rebalances"] == 24


@requires_mysql
def test_rebalance_recommendations_round_trip_through_mysql(repo):
    portfolio_id = repo.get_or_create_portfolio("Test Portfolio")
    repo.upsert_assets(["AAPL", "MSFT"])

    plan_df = pd.DataFrame(
        {
            "current_weight": [0.5, 0.5],
            "target_weight": [0.3, 0.7],
            "weight_diff": [-0.2, 0.2],
            "action": ["SELL", "BUY"],
            "estimated_transaction_cost": [10.0, 10.0],
        },
        index=["AAPL", "MSFT"],
    )
    n_written = repo.write_rebalance_recommendations(portfolio_id, "2024-01-15", plan_df)
    assert n_written == 2

    with repo.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM rebalance_recommendations WHERE portfolio_id = :pid ORDER BY asset_id"),
            {"pid": portfolio_id},
        ).mappings().all()

    assert len(rows) == 2
    assert rows[0]["action"] == "SELL"
    assert rows[1]["action"] == "BUY"
