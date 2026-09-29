import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.risk import metrics
from src.risk.cvar import historical_cvar
from src.risk.var import historical_var, parametric_var
from src.risk.risk_contribution import component_contribution_to_risk


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, size=500))


def test_portfolio_variance_matches_manual_calc():
    weights = np.array([0.5, 0.5])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    expected = weights @ cov @ weights
    assert metrics.portfolio_variance(weights, cov) == pytest.approx(expected)


def test_max_drawdown_is_non_positive():
    wealth = pd.Series([1.0, 1.1, 0.9, 1.05, 0.8, 1.2])
    mdd = metrics.max_drawdown(wealth)
    assert mdd <= 0
    # worst peak (1.1) to trough (0.8) = -0.2727...
    assert mdd == pytest.approx((0.8 - 1.1) / 1.1, rel=1e-6)


def test_sharpe_ratio_zero_vol_returns_nan():
    flat_returns = pd.Series([0.0] * 100)
    assert np.isnan(metrics.sharpe_ratio(flat_returns, 0.02))


def test_historical_var_is_positive_for_typical_returns(sample_returns):
    var = historical_var(sample_returns, confidence_level=0.95)
    assert var > 0  # VaR expressed as a positive loss number


def test_cvar_is_at_least_as_large_as_var(sample_returns):
    var = historical_var(sample_returns, confidence_level=0.95)
    cvar = historical_cvar(sample_returns, confidence_level=0.95)
    assert cvar >= var  # expected shortfall in the tail is never smaller than the threshold


def test_parametric_var_scales_mu_and_sigma_separately_not_combined_sqrt_h(sample_returns):
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue C2: mu must
    scale by h (not sqrt(h)) while sigma scales by sqrt(h) -- these are
    NOT the same as scaling the combined (mu + z*sigma) expression by
    sqrt(h), which was the original bug. This is verified against a
    manually-computed worked example, not merely a self-consistency check
    against another horizon (the old test compared h=4 against h=1 scaled
    by sqrt(4)==2, which is exactly the WRONG formula and could never have
    caught this bug).
    """
    mu, sigma = 0.001, 0.02
    confidence_level = 0.95
    horizon_days = 4
    returns = pd.Series([mu] * 500)  # mean exactly mu; std computed below separately
    # Construct a series with the exact target mean and std for a clean hand-check.
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, 10000)
    raw = (raw - raw.mean()) / raw.std(ddof=1)  # standardize exactly
    returns = pd.Series(mu + sigma * raw)

    z = norm.ppf(1 - confidence_level)
    expected_mu_h = horizon_days * mu
    expected_sigma_h = np.sqrt(horizon_days) * sigma
    expected_var_h = -(expected_mu_h + z * expected_sigma_h)

    actual = parametric_var(returns, confidence_level=confidence_level, horizon_days=horizon_days)
    assert actual == pytest.approx(expected_var_h, rel=1e-6)

    # And confirm this does NOT match the old (buggy) formula, which would
    # have scaled the whole (mu + z*sigma) expression by sqrt(h) instead.
    old_buggy_formula = -(mu + z * sigma) * np.sqrt(horizon_days)
    assert actual != pytest.approx(old_buggy_formula, rel=1e-6)


def test_parametric_var_h1_is_unaffected_by_the_fix(sample_returns):
    """horizon_days=1 must give identical results before and after the
    fix (h=1 is the only case where the two formulas coincide) -- this is
    exactly why the original bug was invisible under the default config
    (var_horizon_days: 1)."""
    var_h1 = parametric_var(sample_returns, confidence_level=0.95, horizon_days=1)
    mu, sigma = sample_returns.mean(), sample_returns.std(ddof=1)
    z = norm.ppf(0.05)
    expected = -(mu + z * sigma)
    assert var_h1 == pytest.approx(expected, rel=1e-6)


def test_component_contribution_sums_to_portfolio_volatility():
    weights = np.array([0.4, 0.6])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    ccr = component_contribution_to_risk(weights, cov, asset_names=["A", "B"])
    port_vol = metrics.portfolio_volatility(weights, cov)
    assert ccr.sum() == pytest.approx(port_vol, rel=1e-6)
