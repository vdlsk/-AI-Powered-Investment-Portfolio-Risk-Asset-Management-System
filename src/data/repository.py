"""
Data access layer.

All database reads/writes go through this module. Nothing else in the
codebase should import sqlalchemy or hold a raw connection — this keeps
schema/connection changes contained to one place.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import MYSQL

logger = logging.getLogger(__name__)


def _records_with_null_safe_nans(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of row-dicts, replacing NaN/NaT with
    None so pymysql sends a real SQL NULL instead of raising
    `ProgrammingError: nan can not be used with MySQL`.

    WHY THIS EXISTS (a real bug caught during integration testing against
    a live MySQL instance, not found by any purely in-memory/pandas test):
    rolling-window features are NaN during their warm-up period by design
    (e.g. `vol_60d` has no defined value for the first 60 rows of a new
    asset's history -- see src/features/feature_pipeline.py). Pandas and
    every unit test in this project tolerate NaN everywhere, so this bug
    was invisible until the exact moment real feature rows were written to
    a real database. `.to_dict(orient="records")` on a DataFrame containing
    NaN produces Python `float('nan')` values, and pymysql's parameter
    escaping explicitly rejects NaN (it has no meaningful SQL
    representation) rather than silently converting it -- which is the
    right behavior from pymysql, but means every write path in this module
    must proactively convert NaN/NaT to None before calling execute().
    """
    clean_df = df.astype(object).where(pd.notnull(df), None)
    return clean_df.to_dict(orient="records")


class Repository:
    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or create_engine(MYSQL.sqlalchemy_url, pool_pre_ping=True)

    # ------------------------------------------------------------------ #
    # assets
    # ------------------------------------------------------------------ #
    def upsert_assets(self, tickers: list[str]) -> None:
        with self.engine.begin() as conn:
            for ticker in tickers:
                conn.execute(
                    text(
                        """
                        INSERT INTO assets (ticker)
                        VALUES (:ticker)
                        ON DUPLICATE KEY UPDATE ticker = ticker
                        """
                    ),
                    {"ticker": ticker},
                )

    def get_asset_id_map(self) -> dict[str, int]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT asset_id, ticker FROM assets")).fetchall()
        return {ticker: asset_id for asset_id, ticker in rows}

    # ------------------------------------------------------------------ #
    # raw prices
    # ------------------------------------------------------------------ #
    def write_raw_prices(self, df: pd.DataFrame) -> int:
        """Insert raw price rows. Duplicate (asset, date) rows are skipped,
        not overwritten — raw data is append-only/idempotent, never mutated.
        """
        asset_map = self.get_asset_id_map()
        df = df.copy()
        df["asset_id"] = df["ticker"].map(asset_map)
        missing = df[df["asset_id"].isna()]
        if not missing.empty:
            logger.warning(
                "Dropping %d rows with unknown tickers: %s",
                len(missing),
                missing["ticker"].unique().tolist(),
            )
        df = df.dropna(subset=["asset_id"])

        rows = df.rename(
            columns={
                "date": "price_date",
                "open": "open_price",
                "high": "high_price",
                "low": "low_price",
                "close": "close_price",
            }
        )[
            [
                "asset_id",
                "price_date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "adj_close",
                "volume",
                "source",
            ]
        ]
        rows = _records_with_null_safe_nans(rows)

        if not rows:
            return 0

        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO raw_market_prices
                            (asset_id, price_date, open_price, high_price, low_price,
                             close_price, adj_close, volume, source)
                        VALUES
                            (:asset_id, :price_date, :open_price, :high_price, :low_price,
                             :close_price, :adj_close, :volume, :source)
                        ON DUPLICATE KEY UPDATE
                            source = source  -- no-op: keep first-seen raw row untouched
                        """
                    ),
                    row,
                )
        return len(rows)

    def read_raw_prices(self, tickers: list[str] | None = None) -> pd.DataFrame:
        query = """
            SELECT a.ticker, r.price_date, r.open_price, r.high_price, r.low_price,
                   r.close_price, r.adj_close, r.volume
            FROM raw_market_prices r
            JOIN assets a ON a.asset_id = r.asset_id
        """
        params = {}
        if tickers:
            placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
            query += f" WHERE a.ticker IN ({placeholders})"
            params = {f"t{i}": t for i, t in enumerate(tickers)}
        query += " ORDER BY a.ticker, r.price_date"
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    # ------------------------------------------------------------------ #
    # clean prices
    # ------------------------------------------------------------------ #
    def write_clean_prices(self, df: pd.DataFrame) -> int:
        asset_map = self.get_asset_id_map()
        df = df.copy()
        df["asset_id"] = df["ticker"].map(asset_map)
        df = df.dropna(subset=["asset_id"])

        rows = df.rename(
            columns={
                "date": "price_date",
                "open": "open_price",
                "high": "high_price",
                "low": "low_price",
                "close": "close_price",
            }
        )[
            [
                "asset_id",
                "price_date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "adj_close",
                "volume",
                "is_imputed",
            ]
        ]
        rows = _records_with_null_safe_nans(rows)

        if not rows:
            return 0

        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO clean_market_prices
                            (asset_id, price_date, open_price, high_price, low_price,
                             close_price, adj_close, volume, is_imputed)
                        VALUES
                            (:asset_id, :price_date, :open_price, :high_price, :low_price,
                             :close_price, :adj_close, :volume, :is_imputed)
                        ON DUPLICATE KEY UPDATE
                            close_price = VALUES(close_price),
                            adj_close = VALUES(adj_close),
                            is_imputed = VALUES(is_imputed)
                        """
                    ),
                    row,
                )
        return len(rows)

    def read_clean_prices(self, tickers: list[str] | None = None) -> pd.DataFrame:
        query = """
            SELECT a.ticker, c.price_date AS date, c.open_price AS open,
                   c.high_price AS high, c.low_price AS low,
                   c.close_price AS close, c.adj_close, c.volume
            FROM clean_market_prices c
            JOIN assets a ON a.asset_id = c.asset_id
        """
        params = {}
        if tickers:
            placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
            query += f" WHERE a.ticker IN ({placeholders})"
            params = {f"t{i}": t for i, t in enumerate(tickers)}
        query += " ORDER BY a.ticker, c.price_date"
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    # ------------------------------------------------------------------ #
    # features
    # ------------------------------------------------------------------ #
    def write_features(self, df: pd.DataFrame, feature_version: str = "v1") -> int:
        """Persist computed features to market_features.

        IMPORTANT (see docs/SECOND_FORENSIC_AUDIT.md, Issue C4): the
        column list below is derived from
        `src.features.feature_pipeline.FEATURE_SPEC` -- the single
        authoritative registry of what features exist -- rather than a
        separately hand-maintained literal list. An earlier version of
        this method had its own hardcoded column list that silently drifted
        out of sync with FEATURE_SPEC when two new features
        (vol_ratio_20_60, momentum_vol_interaction) were added elsewhere
        in the codebase: those two were computed, used for model training
        throughout the rest of the project, and then silently DROPPED
        right here, before ever reaching the database, with no warning.
        Deriving from FEATURE_SPEC means a newly-added feature is
        automatically included in what gets written (and, separately, the
        database schema itself must still be migrated to have the matching
        column -- see database/migrations/ -- this method can't create
        columns, only insert into ones that already exist).
        """
        from src.features.feature_pipeline import FEATURE_SPEC

        asset_map = self.get_asset_id_map()
        df = df.copy()
        df["asset_id"] = df["ticker"].map(asset_map)
        df = df.dropna(subset=["asset_id"])
        df["feature_version"] = feature_version

        cols = ["asset_id", "date", "feature_version"] + list(FEATURE_SPEC.keys())
        cols = [c for c in cols if c in df.columns]
        rows = df[cols].rename(columns={"date": "feature_date"})
        rows = _records_with_null_safe_nans(rows)
        if not rows:
            return 0

        col_names = list(rows[0].keys())
        insert_cols = ", ".join(col_names)
        insert_vals = ", ".join(f":{c}" for c in col_names)
        update_clause = ", ".join(
            f"{c} = VALUES({c})" for c in col_names if c not in ("asset_id", "feature_date", "feature_version")
        )

        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(
                    text(
                        f"""
                        INSERT INTO market_features ({insert_cols})
                        VALUES ({insert_vals})
                        ON DUPLICATE KEY UPDATE {update_clause}
                        """
                    ),
                    row,
                )
        return len(rows)

    # ------------------------------------------------------------------ #
    # analytical-output persistence (see docs/SECOND_FORENSIC_AUDIT.md,
    # Issue H11): risk_metrics, risk_predictions, backtest_results, and
    # rebalance_recommendations previously existed only as unused schema
    # -- no code path ever wrote to them, so none of this project's
    # actual analytical outputs (a computed VaR number, a backtest's
    # Sharpe ratio, a model's risk prediction, a rebalancing
    # recommendation) could be looked up later or audited after the fact.
    # These methods are the fix. All four tables have a portfolio_id
    # foreign key, so `get_or_create_portfolio` comes first.
    # ------------------------------------------------------------------ #
    def get_or_create_portfolio(self, name: str, base_currency: str = "USD") -> int:
        """Return the portfolio_id for `name`, creating the row if it
        doesn't exist yet. This is the minimal persistence needed to give
        the four analytical-output tables below a valid foreign key --
        it does NOT itself decide whether the dashboard's session-based
        portfolio state should be fully persisted (see
        docs/SECOND_FORENSIC_AUDIT.md, Issue M6, which remains open).
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO portfolio (name, base_currency)
                    VALUES (:name, :base_currency)
                    ON DUPLICATE KEY UPDATE name = name
                    """
                ),
                {"name": name, "base_currency": base_currency},
            )
            portfolio_id = conn.execute(
                text("SELECT portfolio_id FROM portfolio WHERE name = :name"),
                {"name": name},
            ).scalar_one()
        return int(portfolio_id)

    def get_or_create_model_run(
        self,
        experiment_id: str,
        model_name: str,
        feature_version: str = "v1",
        hyperparameters: dict | None = None,
        random_seed: int = 42,
        train_start: str | None = None,
        train_end: str | None = None,
        validation_start: str | None = None,
        validation_end: str | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
        metrics: dict | None = None,
    ) -> int:
        """Return the model_run_id for `experiment_id`, creating the row
        if it doesn't exist yet. Required as a foreign key by
        `write_risk_prediction` below. Date-range fields accept `None`
        for callers (e.g. a quick dashboard inference) that don't have a
        full experiment record to attach -- placeholder dates are used
        rather than making the columns nullable, since `model_runs`'s
        schema declares them NOT NULL for the fuller case of a real
        training-experiment record (see `src.models.registry.ExperimentRecord`,
        which captures the same fields for the on-disk joblib artifact).
        """
        import json as _json

        placeholder = "1900-01-01"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO model_runs
                        (experiment_id, model_name, feature_version, hyperparameters,
                         random_seed, train_start, train_end, validation_start,
                         validation_end, test_start, test_end, metrics)
                    VALUES
                        (:experiment_id, :model_name, :feature_version, :hyperparameters,
                         :random_seed, :train_start, :train_end, :validation_start,
                         :validation_end, :test_start, :test_end, :metrics)
                    ON DUPLICATE KEY UPDATE experiment_id = experiment_id
                    """
                ),
                {
                    "experiment_id": experiment_id,
                    "model_name": model_name,
                    "feature_version": feature_version,
                    "hyperparameters": _json.dumps(hyperparameters) if hyperparameters else None,
                    "random_seed": random_seed,
                    "train_start": train_start or placeholder,
                    "train_end": train_end or placeholder,
                    "validation_start": validation_start or placeholder,
                    "validation_end": validation_end or placeholder,
                    "test_start": test_start or placeholder,
                    "test_end": test_end or placeholder,
                    "metrics": _json.dumps(metrics) if metrics else None,
                },
            )
            model_run_id = conn.execute(
                text("SELECT model_run_id FROM model_runs WHERE experiment_id = :experiment_id"),
                {"experiment_id": experiment_id},
            ).scalar_one()
        return int(model_run_id)

    def write_risk_metrics(self, portfolio_id: int, metric_date, metrics: dict) -> None:
        """Persist a computed performance/risk report (the dict returned
        by `src.portfolio.analytics.performance_report`) for one
        portfolio/date. Column names are matched by key where they exist
        in the `risk_metrics` table; extra keys in `metrics` are ignored
        rather than causing an error, since `performance_report`'s exact
        key set has evolved over this project's development and likely
        will again.
        """
        allowed_cols = {
            "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "beta", "var_historical", "var_parametric",
            "var_monte_carlo", "cvar",
        }
        key_map = {"historical_var": "var_historical", "parametric_var": "var_parametric"}
        row = {"portfolio_id": portfolio_id, "metric_date": metric_date}
        for k, v in metrics.items():
            mapped_key = key_map.get(k, k)
            if mapped_key in allowed_cols:
                row[mapped_key] = None if (isinstance(v, float) and np.isnan(v)) else v

        cols = list(row.keys())
        insert_cols = ", ".join(cols)
        insert_vals = ", ".join(f":{c}" for c in cols)
        update_clause = ", ".join(f"{c} = VALUES({c})" for c in cols if c not in ("portfolio_id", "metric_date"))
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO risk_metrics ({insert_cols})
                    VALUES ({insert_vals})
                    ON DUPLICATE KEY UPDATE {update_clause}
                    """
                ),
                row,
            )

    def write_risk_prediction(
        self, portfolio_id: int, model_run_id: int, prediction_date, predicted_probability: float, predicted_class: str,
    ) -> None:
        """Persist one ML risk-prediction result."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO risk_predictions
                        (portfolio_id, model_run_id, prediction_date, predicted_probability, predicted_class)
                    VALUES
                        (:portfolio_id, :model_run_id, :prediction_date, :predicted_probability, :predicted_class)
                    ON DUPLICATE KEY UPDATE
                        predicted_probability = VALUES(predicted_probability),
                        predicted_class = VALUES(predicted_class)
                    """
                ),
                {
                    "portfolio_id": portfolio_id,
                    "model_run_id": model_run_id,
                    "prediction_date": prediction_date,
                    "predicted_probability": predicted_probability,
                    "predicted_class": predicted_class,
                },
            )

    def write_backtest_result(
        self, experiment_id: str, portfolio_id: int, period_start, period_end, performance: dict,
    ) -> None:
        """Persist one walk-forward backtest run's summary performance
        (the dict returned by `run_walk_forward_backtest(...).performance`).
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO backtest_results
                        (experiment_id, portfolio_id, period_start, period_end, gross_return,
                         net_return, transaction_cost_total, sharpe_ratio, max_drawdown, num_rebalances)
                    VALUES
                        (:experiment_id, :portfolio_id, :period_start, :period_end, :gross_return,
                         :net_return, :transaction_cost_total, :sharpe_ratio, :max_drawdown, :num_rebalances)
                    """
                ),
                {
                    "experiment_id": experiment_id,
                    "portfolio_id": portfolio_id,
                    "period_start": period_start,
                    "period_end": period_end,
                    "gross_return": None,
                    "net_return": performance.get("cumulative_return"),
                    "transaction_cost_total": performance.get("total_transaction_costs"),
                    "sharpe_ratio": performance.get("sharpe_ratio"),
                    "max_drawdown": performance.get("max_drawdown"),
                    "num_rebalances": performance.get("num_rebalances"),
                },
            )

    def write_rebalance_recommendations(self, portfolio_id: int, recommendation_date, plan_df: pd.DataFrame) -> int:
        """Persist a rebalancing plan (the DataFrame returned by
        `src.portfolio.rebalancing.rebalance_plan`, indexed by ticker).
        Each row needs a valid asset_id, so tickers not already present in
        `assets` are skipped (not silently invented).
        """
        asset_map = self.get_asset_id_map()
        rows = []
        for ticker, row in plan_df.iterrows():
            if ticker not in asset_map:
                continue
            rows.append(
                {
                    "portfolio_id": portfolio_id,
                    "asset_id": asset_map[ticker],
                    "recommendation_date": recommendation_date,
                    "current_weight": float(row["current_weight"]),
                    "target_weight": float(row["target_weight"]),
                    "weight_diff": float(row["weight_diff"]),
                    "action": row["action"],
                    "estimated_transaction_cost": float(row["estimated_transaction_cost"]),
                }
            )
        if not rows:
            return 0
        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO rebalance_recommendations
                            (portfolio_id, asset_id, recommendation_date, current_weight,
                             target_weight, weight_diff, action, estimated_transaction_cost)
                        VALUES
                            (:portfolio_id, :asset_id, :recommendation_date, :current_weight,
                             :target_weight, :weight_diff, :action, :estimated_transaction_cost)
                        ON DUPLICATE KEY UPDATE
                            current_weight = VALUES(current_weight),
                            target_weight = VALUES(target_weight),
                            weight_diff = VALUES(weight_diff),
                            action = VALUES(action),
                            estimated_transaction_cost = VALUES(estimated_transaction_cost)
                        """
                    ),
                    row,
                )
        return len(rows)
