"""
Walk-forward backtesting engine.

Core timing convention (config.yaml: backtest.execution_lag_days, README
§63, docs/DATA_LEAKAGE_AUDIT.md §10):

    At the close of day t:
        - all features/signals use data <= t (already guaranteed by
          src/features/feature_pipeline.py)
        - a rebalancing decision is made using only that information
    The resulting trade is assumed to EXECUTE at day t + execution_lag_days
    (default 1), not at day t's own close. This prevents the backtest from
    implicitly assuming you can trade at a price you only "saw" after the
    fact.

Costs: every rebalance charges `transaction_cost_bps` (config.yaml) on the
turnover (sum of absolute weight changes), consistent with
src/portfolio/rebalancing.py's convention.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.portfolio.rebalancing import rebalance_plan
from src.risk import metrics as risk_metrics

logger = logging.getLogger(__name__)

EXECUTION_LAG_DAYS = CONFIG["backtest"]["execution_lag_days"]
TRANSACTION_COST_BPS = CONFIG["backtest"]["transaction_cost_bps"]

REBALANCE_FREQ_TO_OFFSET = {
    "daily": pd.DateOffset(days=1),
    "weekly": pd.DateOffset(weeks=1),
    "monthly": pd.DateOffset(months=1),
    "quarterly": pd.DateOffset(months=3),
}


@dataclass
class BacktestResult:
    portfolio_value: pd.Series          # date-indexed net-of-cost portfolio value
    gross_returns: pd.Series
    net_returns: pd.Series
    weights_history: pd.DataFrame       # date-indexed realized weights after each rebalance
    transaction_costs: pd.Series        # cost charged on each rebalance date (0 elsewhere)
    num_rebalances: int
    performance: dict[str, float] = field(default_factory=dict)


def _rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    """First trading date on/after each period boundary, matching
    `frequency` (daily/weekly/monthly/quarterly).
    """
    if frequency == "daily":
        return list(dates)

    offset = REBALANCE_FREQ_TO_OFFSET[frequency]
    boundaries = []
    current = dates[0]
    while current <= dates[-1]:
        boundaries.append(current)
        current = current + offset

    result = []
    for b in boundaries:
        candidates = dates[dates >= b]
        if len(candidates):
            result.append(candidates[0])
    return sorted(set(result))


def run_walk_forward_backtest(
    asset_returns: pd.DataFrame,
    target_weight_fn,
    initial_value: float = 1.0,
    rebalance_frequency: str = None,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    execution_lag_days: int = EXECUTION_LAG_DAYS,
) -> BacktestResult:
    """Run a walk-forward backtest.

    `target_weight_fn(as_of_date, historical_returns) -> pd.Series` is
    called with ONLY the asset_returns available up to and including
    `as_of_date` (never later data), and must return target weights
    (summing to 1) for the assets in `asset_returns.columns`. This is the
    hook where optimization.py, a fixed strategy, or an ML-driven allocation
    rule plugs in.

    Trades decided using information through day t are QUEUED and only
    actually take effect at day t + execution_lag_days: both the weight
    change and its transaction cost are applied at the start of that exact
    execution date, before that date's own return is computed -- so the
    position genuinely starts earning the new weights from the execution
    date onward, priced against the portfolio's value AT that date, not
    at signal time.

    (See docs/FORENSIC_AUDIT.md, Issue 5: an earlier version of this
    function updated `current_weights` immediately at signal time, which
    happened to look correct for the default `execution_lag_days=1` but
    was silently wrong for any other lag -- the new weights and cost were
    applied `execution_lag_days - 1` days too early. This version uses an
    explicit pending-trade queue to guarantee the labeled execution date
    is when the change actually takes effect, for any lag value.)
    """
    rebalance_frequency = rebalance_frequency or CONFIG["backtest"]["rebalance_frequency"]
    dates = asset_returns.index
    rebalance_dates = _rebalance_dates(dates, rebalance_frequency)

    current_weights = pd.Series(0.0, index=asset_returns.columns)
    portfolio_value = initial_value
    value_history = {}
    weight_rows = {}
    cost_history = {}
    gross_return_history = {}
    n_rebalances = 0

    # Trades decided on a signal date are queued here and only actually
    # applied when the loop reaches the corresponding execution date.
    pending_trades: dict[pd.Timestamp, pd.Series] = {}

    for i, current_date in enumerate(dates):
        # Apply any trade scheduled to execute TODAY, before computing
        # today's return -- the new weights must be in place for the
        # execution date to actually earn that date's return under them,
        # and the cost is charged against the portfolio's value at this
        # exact point in time (not the value back at signal time, which
        # could be materially different if the lag spans several days).
        if current_date in pending_trades:
            target_weights = pending_trades.pop(current_date)
            plan = rebalance_plan(current_weights, target_weights, portfolio_value, transaction_cost_bps)
            total_cost = float(plan["estimated_transaction_cost"].sum())
            portfolio_value -= total_cost
            cost_history[current_date] = cost_history.get(current_date, 0.0) + total_cost
            current_weights = target_weights
            weight_rows[current_date] = current_weights.copy()

        # Daily P&L from whatever weights are currently held (set by the
        # most recent EXECUTED trade -- never today's signal, unless a
        # trade also executes today, handled just above).
        if i > 0:
            day_return = float((current_weights * asset_returns.loc[current_date]).sum())
        else:
            day_return = 0.0
        portfolio_value *= (1 + day_return)
        gross_return_history[current_date] = day_return
        value_history[current_date] = portfolio_value
        cost_history.setdefault(current_date, 0.0)

        if current_date in rebalance_dates:
            # Signal generated using data through current_date only...
            historical = asset_returns.loc[:current_date]
            target_weights = target_weight_fn(current_date, historical)
            target_weights = target_weights.reindex(asset_returns.columns).fillna(0.0)

            # ...but merely QUEUED for execution execution_lag_days later.
            # Nothing about current_weights or portfolio_value changes yet.
            exec_idx = i + execution_lag_days
            if exec_idx < len(dates):
                exec_date = dates[exec_idx]
                pending_trades[exec_date] = target_weights
                n_rebalances += 1

    weights_history = pd.DataFrame(weight_rows).T.reindex(dates).ffill().fillna(0.0)
    value_series = pd.Series(value_history).reindex(dates)
    gross_returns = pd.Series(gross_return_history).reindex(dates)
    net_returns = value_series.pct_change().fillna(0.0)
    costs = pd.Series(cost_history).reindex(dates).fillna(0.0)

    wealth = value_series / value_series.iloc[0]
    performance = {
        "cumulative_return": float(value_series.iloc[-1] / value_series.iloc[0] - 1),
        "annualized_return": risk_metrics.annualized_return(net_returns),
        "annualized_volatility": risk_metrics.annualized_volatility(net_returns),
        "sharpe_ratio": risk_metrics.sharpe_ratio(net_returns, CONFIG["risk"]["risk_free_rate_annual"]),
        "max_drawdown": risk_metrics.max_drawdown(wealth),
        "total_transaction_costs": float(costs.sum()),
        "num_rebalances": n_rebalances,
    }

    return BacktestResult(
        portfolio_value=value_series,
        gross_returns=gross_returns,
        net_returns=net_returns,
        weights_history=weights_history,
        transaction_costs=costs,
        num_rebalances=n_rebalances,
        performance=performance,
    )


def equal_weight_strategy(as_of_date: pd.Timestamp, historical_returns: pd.DataFrame) -> pd.Series:
    """Simplest possible target_weight_fn: always equal-weight. Useful as a
    baseline/benchmark to compare smarter strategies against.
    """
    n = len(historical_returns.columns)
    return pd.Series(1 / n, index=historical_returns.columns)


def make_periodic_reoptimization_strategy(
    optimizer_fn, lookback_days: int = 252, **optimizer_kwargs
):
    """Factory returning a target_weight_fn that re-runs a
    src.portfolio.optimization function using only the trailing
    `lookback_days` of historical returns as of each rebalance date --
    keeping the optimizer's mean/cov estimation strictly in-sample-to-date,
    per the module docstring in optimization.py.
    """

    def strategy(as_of_date: pd.Timestamp, historical_returns: pd.DataFrame) -> pd.Series:
        window = historical_returns.tail(lookback_days)
        if len(window) < max(20, lookback_days // 4):
            n = len(window.columns)
            return pd.Series(1 / n, index=window.columns)
        mean_returns = window.mean()
        cov_matrix = window.cov()
        result = optimizer_fn(mean_returns, cov_matrix, **optimizer_kwargs)
        if not result.success:
            n = len(window.columns)
            return pd.Series(1 / n, index=window.columns)
        return result.weights

    return strategy
