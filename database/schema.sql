-- ============================================================================
-- FinRisk database schema
--
-- Design principles:
--   1. Raw data is NEVER overwritten by processed data (separate tables).
--   2. Every fact table carries created_at (and updated_at where mutable).
--   3. Foreign keys enforce referential integrity between assets/portfolios
--      and everything that references them.
--   4. Composite unique constraints prevent duplicate (asset, date) rows,
--      which is one of the most common sources of silent data bugs.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS finrisk
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE finrisk;

-- ----------------------------------------------------------------------------
-- assets: master list of tradable instruments.
-- Why it exists: every price/feature/holding row references an asset_id
-- rather than repeating the ticker string, keeping the schema normalized.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id      INT AUTO_INCREMENT PRIMARY KEY,
    ticker        VARCHAR(20)  NOT NULL,
    name          VARCHAR(255) NULL,
    asset_class   VARCHAR(50)  NULL,   -- e.g. 'equity', 'bond', 'commodity'
    currency      VARCHAR(10)  NOT NULL DEFAULT 'USD',
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                                ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_assets_ticker (ticker)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- raw_market_prices: exactly what the data provider returned, untouched.
-- Why it exists: an audit trail. If a downstream number ever looks wrong,
-- we can always compare back to this table to see if the bug is in our
-- cleaning logic or in the source data itself.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_market_prices (
    raw_price_id   BIGINT AUTO_INCREMENT PRIMARY KEY,
    asset_id       INT NOT NULL,
    price_date     DATE NOT NULL,
    open_price     DECIMAL(18,6) NULL,
    high_price     DECIMAL(18,6) NULL,
    low_price      DECIMAL(18,6) NULL,
    close_price    DECIMAL(18,6) NULL,
    adj_close      DECIMAL(18,6) NULL,
    volume         BIGINT NULL,
    source         VARCHAR(50) NOT NULL DEFAULT 'yfinance',
    ingested_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_raw_prices_asset FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id) ON DELETE CASCADE,
    UNIQUE KEY uq_raw_asset_date (asset_id, price_date),
    INDEX idx_raw_asset_date (asset_id, price_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- clean_market_prices: validated/cleaned prices used by every downstream
-- calculation (returns, risk, features, backtests).
-- Why it exists: separates "what came in" from "what we trust to compute
-- with", so cleaning logic can be re-run/audited without touching raw data.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clean_market_prices (
    clean_price_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    asset_id       INT NOT NULL,
    price_date     DATE NOT NULL,
    open_price     DECIMAL(18,6) NULL,
    high_price     DECIMAL(18,6) NULL,
    low_price      DECIMAL(18,6) NULL,
    close_price    DECIMAL(18,6) NOT NULL,
    adj_close      DECIMAL(18,6) NOT NULL,
    volume         BIGINT NULL,
    is_imputed     BOOLEAN NOT NULL DEFAULT FALSE,  -- true if this row was filled, not observed
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_clean_prices_asset FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id) ON DELETE CASCADE,
    UNIQUE KEY uq_clean_asset_date (asset_id, price_date),
    INDEX idx_clean_asset_date (asset_id, price_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- market_features: leakage-safe engineered features, one row per asset/date.
-- Why it exists: keeps feature generation reproducible and inspectable
-- (e.g. you can check that the row for date t contains no t+1 information)
-- without recomputing pandas rolling windows every time a model needs data.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_features (
    feature_id     BIGINT AUTO_INCREMENT PRIMARY KEY,
    asset_id       INT NOT NULL,
    feature_date   DATE NOT NULL,     -- the "as-of" date t; features use data <= t
    feature_version VARCHAR(20) NOT NULL DEFAULT 'v1',
    ret_1d         DOUBLE NULL,
    ret_5d         DOUBLE NULL,
    ret_10d        DOUBLE NULL,
    ret_20d        DOUBLE NULL,
    ret_60d        DOUBLE NULL,
    vol_5d         DOUBLE NULL,
    vol_10d        DOUBLE NULL,
    vol_20d        DOUBLE NULL,
    vol_60d        DOUBLE NULL,
    vol_ratio_20_60 DOUBLE NULL,
    momentum_20d   DOUBLE NULL,
    ma_ratio_20_60 DOUBLE NULL,
    rsi_14         DOUBLE NULL,
    rolling_drawdown DOUBLE NULL,
    momentum_vol_interaction DOUBLE NULL,
    beta_60d       DOUBLE NULL,
    corr_market_60d DOUBLE NULL,
    volume_zscore_20d DOUBLE NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_features_asset FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id) ON DELETE CASCADE,
    UNIQUE KEY uq_features_asset_date_version (asset_id, feature_date, feature_version),
    INDEX idx_features_asset_date (asset_id, feature_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- portfolio: a named portfolio definition.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio (
    portfolio_id   INT AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(100) NOT NULL,
    base_currency  VARCHAR(10)  NOT NULL DEFAULT 'USD',
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                               ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_portfolio_name (name)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- portfolio_holdings: point-in-time weights per asset in a portfolio.
-- Why it exists: lets us track how a portfolio's composition evolved
-- (needed for realistic, non-look-ahead backtests and for the dashboard's
-- "current allocation" view).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    holding_id     BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id   INT NOT NULL,
    asset_id       INT NOT NULL,
    as_of_date     DATE NOT NULL,
    weight         DECIMAL(9,6) NOT NULL,   -- 0.0 - 1.0
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_holdings_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES portfolio(portfolio_id) ON DELETE CASCADE,
    CONSTRAINT fk_holdings_asset FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id) ON DELETE CASCADE,
    UNIQUE KEY uq_holdings_portfolio_asset_date (portfolio_id, asset_id, as_of_date),
    INDEX idx_holdings_portfolio_date (portfolio_id, as_of_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- portfolio_transactions: analytical log of buy/sell/rebalance actions.
-- These are NEVER executed against a real brokerage — this project only
-- produces recommendations.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_transactions (
    transaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id   INT NOT NULL,
    asset_id       INT NOT NULL,
    transaction_date DATE NOT NULL,
    action         ENUM('BUY','SELL','HOLD') NOT NULL,
    weight_before  DECIMAL(9,6) NOT NULL,
    weight_after   DECIMAL(9,6) NOT NULL,
    transaction_cost_bps DECIMAL(9,4) NOT NULL DEFAULT 0,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_txn_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES portfolio(portfolio_id) ON DELETE CASCADE,
    CONSTRAINT fk_txn_asset FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id) ON DELETE CASCADE,
    INDEX idx_txn_portfolio_date (portfolio_id, transaction_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- risk_metrics: computed risk metrics per portfolio/date (VaR, CVaR, vol, ...).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_metrics (
    risk_metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id   INT NOT NULL,
    metric_date    DATE NOT NULL,
    annualized_return DOUBLE NULL,
    annualized_volatility DOUBLE NULL,
    sharpe_ratio   DOUBLE NULL,
    sortino_ratio  DOUBLE NULL,
    max_drawdown   DOUBLE NULL,
    calmar_ratio   DOUBLE NULL,
    beta           DOUBLE NULL,
    var_historical DOUBLE NULL,
    var_parametric DOUBLE NULL,
    var_monte_carlo DOUBLE NULL,
    cvar           DOUBLE NULL,
    confidence_level DECIMAL(5,4) NOT NULL DEFAULT 0.95,
    horizon_days   INT NOT NULL DEFAULT 1,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_riskmetrics_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES portfolio(portfolio_id) ON DELETE CASCADE,
    UNIQUE KEY uq_riskmetrics_portfolio_date (portfolio_id, metric_date),
    INDEX idx_riskmetrics_portfolio_date (portfolio_id, metric_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- market_regimes: detected regime label per date (unsupervised).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_regimes (
    regime_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
    regime_date    DATE NOT NULL,
    regime_label   VARCHAR(50) NOT NULL,     -- e.g. 'low_vol_bull', 'high_vol_bear'
    model_version  VARCHAR(20) NOT NULL DEFAULT 'v1',
    is_exploratory BOOLEAN NOT NULL DEFAULT FALSE, -- TRUE = fit on full history, not backtest-safe
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_regime_date_version (regime_date, model_version),
    INDEX idx_regime_date (regime_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- stress_scenarios: named hypothetical shock definitions and their results.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stress_scenarios (
    scenario_id    INT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id   INT NOT NULL,
    scenario_name  VARCHAR(100) NOT NULL,
    scenario_date  DATE NOT NULL,
    shock_definition JSON NOT NULL,   -- e.g. {"equities": -0.30, "gold": 0.05}
    portfolio_loss_pct DOUBLE NULL,
    portfolio_loss_value DOUBLE NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_stress_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES portfolio(portfolio_id) ON DELETE CASCADE,
    INDEX idx_stress_portfolio_date (portfolio_id, scenario_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- model_runs: reproducibility record for every trained model.
-- Why it exists: lets us answer "which exact data/hyperparameters produced
-- this number" months later.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id   INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id  VARCHAR(64) NOT NULL,
    model_name     VARCHAR(100) NOT NULL,
    feature_version VARCHAR(20) NOT NULL,
    hyperparameters JSON NULL,
    random_seed    INT NOT NULL,
    train_start    DATE NOT NULL,
    train_end      DATE NOT NULL,
    validation_start DATE NOT NULL,
    validation_end DATE NOT NULL,
    test_start     DATE NOT NULL,
    test_end       DATE NOT NULL,
    metrics        JSON NULL,
    artifact_path  VARCHAR(255) NULL,   -- path to joblib file with full pipeline
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_model_experiment (experiment_id),
    INDEX idx_model_name (model_name)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- risk_predictions: ML model output (probability, class) per portfolio/date.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_predictions (
    prediction_id  BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id   INT NOT NULL,
    model_run_id   INT NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_probability DOUBLE NOT NULL,
    predicted_class VARCHAR(10) NOT NULL,   -- LOW / MEDIUM / HIGH
    actual_outcome TINYINT NULL,            -- filled in later once target is realized
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_pred_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES portfolio(portfolio_id) ON DELETE CASCADE,
    CONSTRAINT fk_pred_model_run FOREIGN KEY (model_run_id)
        REFERENCES model_runs(model_run_id) ON DELETE CASCADE,
    UNIQUE KEY uq_pred_portfolio_date_model (portfolio_id, prediction_date, model_run_id),
    INDEX idx_pred_portfolio_date (portfolio_id, prediction_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- backtest_results: walk-forward backtest performance per experiment.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_results (
    backtest_id    INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id  VARCHAR(64) NOT NULL,
    portfolio_id   INT NOT NULL,
    period_start   DATE NOT NULL,
    period_end     DATE NOT NULL,
    gross_return   DOUBLE NULL,
    net_return     DOUBLE NULL,
    transaction_cost_total DOUBLE NULL,
    sharpe_ratio   DOUBLE NULL,
    max_drawdown   DOUBLE NULL,
    num_rebalances INT NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_backtest_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES portfolio(portfolio_id) ON DELETE CASCADE,
    INDEX idx_backtest_experiment (experiment_id)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- rebalance_recommendations: current vs target weights and suggested trades.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rebalance_recommendations (
    recommendation_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    portfolio_id   INT NOT NULL,
    asset_id       INT NOT NULL,
    recommendation_date DATE NOT NULL,
    current_weight DECIMAL(9,6) NOT NULL,
    target_weight  DECIMAL(9,6) NOT NULL,
    weight_diff    DECIMAL(9,6) NOT NULL,
    action         ENUM('BUY','SELL','HOLD') NOT NULL,
    estimated_transaction_cost DOUBLE NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_rebalance_portfolio FOREIGN KEY (portfolio_id)
        REFERENCES portfolio(portfolio_id) ON DELETE CASCADE,
    CONSTRAINT fk_rebalance_asset FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id) ON DELETE CASCADE,
    UNIQUE KEY uq_rebalance_portfolio_asset_date (portfolio_id, asset_id, recommendation_date),
    INDEX idx_rebalance_portfolio_date (portfolio_id, recommendation_date)
) ENGINE=InnoDB;
