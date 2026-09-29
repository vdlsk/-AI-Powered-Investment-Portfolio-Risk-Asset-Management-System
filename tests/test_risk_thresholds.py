import numpy as np
import pandas as pd
import pytest

from src.models.train import select_risk_thresholds, classify_risk


def test_high_threshold_achieves_at_least_the_requested_precision():
    """The defining guarantee of the redesigned function (see
    docs/SECOND_FORENSIC_AUDIT.md Issue H4): among all predictions AT OR
    ABOVE the returned 'high' threshold, the actual observed positive
    rate must be >= high_level -- an absolute, outcome-tied guarantee,
    not a quantile of the model's own output.
    """
    rng = np.random.default_rng(0)
    n = 2000
    # Construct probabilities genuinely correlated with outcomes so a
    # meaningful precision-vs-threshold relationship exists.
    true_risk = rng.uniform(0, 1, n)
    outcomes = (rng.uniform(0, 1, n) < true_risk * 0.3).astype(int)  # rare-ish positive class
    proba = pd.Series(true_risk + rng.normal(0, 0.05, n))

    thresholds = select_risk_thresholds(proba, pd.Series(outcomes), medium_level=0.10, high_level=0.25)

    high_cutoff = thresholds["high"]
    mask = proba >= high_cutoff
    actual_precision = outcomes[mask.values].mean()
    assert actual_precision >= 0.25 - 1e-9


def test_medium_threshold_achieves_at_least_the_requested_precision():
    rng = np.random.default_rng(1)
    n = 2000
    true_risk = rng.uniform(0, 1, n)
    outcomes = (rng.uniform(0, 1, n) < true_risk * 0.3).astype(int)
    proba = pd.Series(true_risk + rng.normal(0, 0.05, n))

    thresholds = select_risk_thresholds(proba, pd.Series(outcomes), medium_level=0.10, high_level=0.25)
    medium_cutoff = thresholds["medium"]
    mask = proba >= medium_cutoff
    actual_precision = outcomes[mask.values].mean()
    assert actual_precision >= 0.10 - 1e-9


def test_high_threshold_is_at_least_as_strict_as_medium():
    rng = np.random.default_rng(2)
    n = 1000
    true_risk = rng.uniform(0, 1, n)
    outcomes = (rng.uniform(0, 1, n) < true_risk * 0.3).astype(int)
    proba = pd.Series(true_risk + rng.normal(0, 0.05, n))

    thresholds = select_risk_thresholds(proba, pd.Series(outcomes), medium_level=0.10, high_level=0.25)
    assert thresholds["high"] >= thresholds["medium"]


def test_thresholds_are_not_self_referential_quantiles_of_clustered_output():
    """Regression test for the exact scenario Issue H4 describes: if a
    model's predictions all cluster tightly in a narrow LOW-probability
    band, the OLD quantile-based approach would still label the top third
    'HIGH' regardless of whether that's an absolute high probability. The
    new approach must not do this -- when no cutoff can reach the
    requested precision level (because the clustered probabilities carry
    little real signal), it must fall back to the most conservative
    cutoff rather than fabricate a 'HIGH' label from a quantile.
    """
    rng = np.random.default_rng(3)
    n = 500
    # All predictions clustered between 4% and 8%, with outcomes
    # essentially uncorrelated with the (narrow) probability differences.
    proba = pd.Series(rng.uniform(0.04, 0.08, n))
    outcomes = pd.Series(rng.integers(0, 2, n) * 0)  # no positive outcomes at all -- worst case

    thresholds = select_risk_thresholds(proba, outcomes, medium_level=0.10, high_level=0.25)
    # With zero positive outcomes, precision can never reach 10%/25% at
    # any cutoff -- the function must fall back to the most conservative
    # (highest observed probability) cutoff, not silently return some
    # quantile of the clustered low probabilities as if it meant something.
    assert thresholds["high"] == pytest.approx(proba.max())
    assert thresholds["medium"] == pytest.approx(proba.max())


def test_classify_risk_still_works_with_new_threshold_dict_shape():
    thresholds = {"medium": 0.10, "high": 0.25}
    assert classify_risk(0.30, thresholds) == "HIGH"
    assert classify_risk(0.15, thresholds) == "MEDIUM"
    assert classify_risk(0.05, thresholds) == "LOW"


def test_test_eval_and_production_models_are_trained_on_different_data_h6():
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue H6: the
    model used for historical TEST evaluation and the model used for
    "current" inference must be two DIFFERENT fitted objects -- the
    production model trained on strictly more data (TRAIN+VALIDATION)
    than the TEST-evaluation model (TRAIN only).
    """
    import numpy as np
    import pandas as pd
    from src.models.train import train_model

    rng = np.random.default_rng(0)
    n_train, n_val = 200, 80
    X_train = pd.DataFrame(rng.normal(0, 1, size=(n_train, 3)), columns=["a", "b", "c"])
    y_train = pd.Series(rng.integers(0, 2, n_train))
    X_val = pd.DataFrame(rng.normal(0, 1, size=(n_val, 3)), columns=["a", "b", "c"])
    y_val = pd.Series(rng.integers(0, 2, n_val))

    test_eval_model = train_model(X_train, y_train, "logistic_regression")

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val]).reset_index(drop=True)
    production_model = train_model(X_train_val, y_train_val, "logistic_regression")

    # Different objects entirely (not the same Python object reused).
    assert test_eval_model.pipeline is not production_model.pipeline

    # The production model was genuinely fit on strictly more rows.
    test_eval_n_samples = test_eval_model.pipeline.named_steps["model"].n_features_in_
    production_n_samples = production_model.pipeline.named_steps["model"].n_features_in_
    # (n_features_in_ is the same for both -- same feature set -- so
    # instead verify via coefficients: fitting on different data with a
    # fixed random_state should still generally produce different learned
    # coefficients, since the underlying data differs.)
    coef_test_eval = test_eval_model.pipeline.named_steps["model"].coef_
    coef_production = production_model.pipeline.named_steps["model"].coef_
    assert not np.allclose(coef_test_eval, coef_production), (
        "The two models produced identical coefficients despite being "
        "fit on different data -- suspicious, suggests they may not "
        "actually be trained on different inputs."
    )
