"""
Stress testing.

Applies predefined (or custom) hypothetical shocks to a portfolio's current
holdings, e.g. "equities -30%, gold +5%, bonds +2%". These are hypothetical
scenarios, NOT predictions or historical replays — always labeled as such
in any output shown to a user.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StressTestResult:
    scenario_name: str
    asset_losses: pd.Series      # asset_class or ticker -> loss fraction applied
    contribution_to_loss: pd.Series  # weight * shock, per holding
    portfolio_loss_pct: float
    portfolio_new_value: float


def run_stress_test(
    scenario_name: str,
    holdings: pd.Series,          # index: asset_class or ticker, values: weight (sums to 1)
    shocks: dict[str, float],     # index key -> fractional shock, e.g. {"equities": -0.30}
    portfolio_value: float,
) -> StressTestResult:
    """Apply `shocks` to `holdings` and compute the resulting hypothetical
    portfolio loss. Any holding not present in `shocks` is assumed
    unaffected (shock = 0) and this is not silently hidden — callers should
    check `shocks.keys()` against `holdings.index` if full coverage matters.
    """
    aligned_shocks = holdings.index.map(lambda k: shocks.get(k, 0.0))
    aligned_shocks = pd.Series(aligned_shocks, index=holdings.index)

    contribution = holdings * aligned_shocks
    portfolio_loss_pct = float(contribution.sum())
    new_value = portfolio_value * (1 + portfolio_loss_pct)

    return StressTestResult(
        scenario_name=scenario_name,
        asset_losses=aligned_shocks,
        contribution_to_loss=contribution,
        portfolio_loss_pct=portfolio_loss_pct,
        portfolio_new_value=new_value,
    )


def run_all_scenarios(
    holdings: pd.Series, scenarios: dict[str, dict[str, float]], portfolio_value: float
) -> list[StressTestResult]:
    return [
        run_stress_test(name, holdings, shocks, portfolio_value)
        for name, shocks in scenarios.items()
    ]
