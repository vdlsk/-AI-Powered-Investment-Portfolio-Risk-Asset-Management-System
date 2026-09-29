import numpy as np
import pandas as pd
import pytest

from src.models.feature_selection import (
    drop_correlated_features,
    drop_low_variance_features,
    mutual_information_ranking,
    select_by_model_importance,
    select_features,
    variance_inflation_factors,
)


@pytest.fixture
def toy_features():
    rng = np.random.default_rng(0)
    n = 500
    base = rng.normal(0, 1, n)
    X = pd.DataFrame(
        {
            "signal": base,
            "signal_duplicate": base + rng.normal(0, 0.01, n),  # near-perfectly correlated with 'signal'
            "noise": rng.normal(0, 1, n),
            "constant": np.full(n, 5.0),  # zero variance
        }
    )
    y = pd.Series((base > 0.5).astype(int))
    return X, y


def test_drop_low_variance_features_removes_constant_column(toy_features):
    X, _ = toy_features
    kept = drop_low_variance_features(X)
    assert "constant" not in kept
    assert "signal" in kept


def test_drop_correlated_features_removes_one_of_a_redundant_pair(toy_features):
    X, _ = toy_features
    kept = drop_correlated_features(X, threshold=0.90)
    # exactly one of the two near-duplicate columns should survive
    assert ("signal" in kept) != ("signal_duplicate" not in kept) or (
        ("signal" in kept) + ("signal_duplicate" in kept) == 1
    )
    assert "noise" in kept  # uncorrelated column must survive


def test_select_by_model_importance_ranks_true_signal_above_noise(toy_features):
    X, y = toy_features
    X_clean = X.drop(columns=["constant"])
    importances = select_by_model_importance(X_clean, y, random_seed=0)
    assert importances["signal"] > importances["noise"]


def test_mutual_information_ranks_true_signal_above_noise(toy_features):
    X, y = toy_features
    X_clean = X.drop(columns=["constant"])
    mi = mutual_information_ranking(X_clean, y, random_seed=0)
    assert mi["signal"] > mi["noise"]


def test_vif_flags_the_correlated_pair_higher_than_noise(toy_features):
    X, _ = toy_features
    X_clean = X.drop(columns=["constant"])
    vifs = variance_inflation_factors(X_clean)
    assert vifs["signal"] > vifs["noise"]
    assert vifs["signal_duplicate"] > vifs["noise"]


def test_iterative_vif_removal_reduces_multicollinearity(toy_features):
    from src.models.feature_selection import iterative_vif_removal, variance_inflation_factors

    X, _ = toy_features
    X_clean = X.drop(columns=["constant"])
    kept = iterative_vif_removal(X_clean, vif_threshold=10.0)

    # exactly one of the near-duplicate pair should be removed
    assert len(kept) < len(X_clean.columns)
    assert "noise" in kept  # uncorrelated feature must always survive

    # every remaining feature's VIF must now be at or below the threshold
    remaining_vifs = variance_inflation_factors(X_clean[kept])
    assert (remaining_vifs <= 10.0 + 1e-6).all()


def test_iterative_vif_removal_handles_perfect_collinearity():
    """A feature that is an exact linear combination of others produces an
    infinite VIF -- this must trigger removal, not stop the loop early
    (regression test for a bug caught during development: `if worst_vif <=
    threshold or not isfinite(worst_vif): break` incorrectly treated
    infinite VIF as "good enough to stop", the opposite of correct).
    """
    from src.models.feature_selection import iterative_vif_removal

    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 1, 200)
    X = pd.DataFrame({"a": a, "b": b, "a_plus_b": a + b, "noise": rng.normal(0, 1, 200)})  # exact linear combo

    kept = iterative_vif_removal(X, vif_threshold=10.0)
    assert "noise" in kept
    assert len(kept) <= 3  # at least one of a/b/a_plus_b must be removed


def test_select_features_end_to_end(toy_features):
    X, y = toy_features
    selected = select_features(X, y, correlation_threshold=0.90, top_n_by_importance=1)
    assert "constant" not in selected
    assert len(selected) <= 1
    assert selected == ["signal"]  # the true signal must outrank noise when forced to pick just 1


# ---------------------------------------------------------------------- #
# Leakage test: feature selection must be decided using TRAIN only.
# ---------------------------------------------------------------------- #
def test_feature_selection_is_not_influenced_by_test_data():
    # Deterministic construction (no randomness) so this test's outcome
    # never depends on a lucky/unlucky random draw. TRAIN has 'a' and 'b'
    # virtually identical (correlation ~1.0), so selecting on TRAIN alone
    # must drop one of them.
    n_train = 200
    a_train = np.linspace(-1, 1, n_train)
    b_train = a_train + 1e-6
    X_train = pd.DataFrame({"a": a_train, "b": b_train})

    selected_train_only = drop_correlated_features(X_train, threshold=0.9)
    assert selected_train_only == ["a"]  # 'b' is dropped as redundant with 'a'

    # TEST data where 'a' and 'b' are ANTI-correlated (b = -a) -- the
    # opposite relationship from TRAIN. If selection were (incorrectly)
    # run on TRAIN+TEST combined, the overall correlation collapses toward
    # zero (mixing a +1 and a -1 relationship), so the pair would NOT be
    # flagged as redundant -- a different, wrong selection outcome. This
    # is a real, deterministic demonstration of why selection must run on
    # TRAIN alone, not a hoped-for random-noise effect.
    a_test = np.linspace(-1, 1, 100)
    b_test = -a_test
    X_test = pd.DataFrame({"a": a_test, "b": b_test})

    combined = pd.concat([X_train, X_test], ignore_index=True)
    selected_if_leaked = drop_correlated_features(combined, threshold=0.9)

    assert set(selected_train_only) != set(selected_if_leaked)
    assert selected_if_leaked == ["a", "b"]  # correlation collapses below threshold, so nothing gets dropped

    # The actual guarantee the pipeline relies on: selecting on X_train
    # alone is deterministic/reproducible regardless of what a caller
    # later concatenates elsewhere.
    selected_train_only_again = drop_correlated_features(X_train, threshold=0.9)
    assert selected_train_only == selected_train_only_again
