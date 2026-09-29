"""
Rebalancing engine.

Compares current portfolio weights to a target allocation (e.g. the output
of optimization.py) and produces a BUY/SELL/HOLD recommendation per asset,
including an estimated transaction cost. This module never places real
trades — it is analytical output only, matching the "recommendation, not
execution" scope of the whole project.
"""
from __future__ import annotations

import pandas as pd

from src.config import CONFIG

TRANSACTION_COST_BPS = CONFIG["backtest"]["transaction_cost_bps"]
HOLD_TOLERANCE = 0.0005  # weight differences smaller than this are treated as HOLD, not noise trades


def rebalance_plan(
    current_weights: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Return a DataFrame with one row per asset:
    current_weight, target_weight, weight_diff, action, trade_value,
    estimated_transaction_cost.
    """
    all_assets = current_weights.index.union(target_weights.index)
    current = current_weights.reindex(all_assets).fillna(0.0)
    target = target_weights.reindex(all_assets).fillna(0.0)
    diff = target - current

    def classify(d: float) -> str:
        if d > HOLD_TOLERANCE:
            return "BUY"
        if d < -HOLD_TOLERANCE:
            return "SELL"
        return "HOLD"

    actions = diff.apply(classify)
    trade_value = diff.abs() * portfolio_value
    cost = trade_value * (transaction_cost_bps / 10_000)

    return pd.DataFrame(
        {
            "current_weight": current,
            "target_weight": target,
            "weight_diff": diff,
            "action": actions,
            "trade_value": trade_value,
            "estimated_transaction_cost": cost,
        }
    ).sort_values("weight_diff")


def apply_transaction_costs(gross_return: float, total_cost: float, portfolio_value: float) -> float:
    """Net return after subtracting rebalancing transaction costs, expressed
    as a fraction of portfolio value (matches README §36 gross/cost/net convention).
    """
    return gross_return - (total_cost / portfolio_value)
