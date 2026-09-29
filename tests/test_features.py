import numpy as np
import pandas as pd

from src.features.returns import log_returns, multi_day_return, simple_returns
from src.features.feature_pipeline import build_features_for_asset


def test_simple_vs_log_return_relationship():
    prices = pd.Series([100, 110, 99, 108.9])
    simple = simple_returns(prices)
    log = log_returns(prices)
    # log(1+R) == r
    np.testing.assert_allclose(np.log1p(simple.dropna()), log.dropna(), rtol=1e-10)


def test_multi_day_return_matches_manual_calc():
    prices = pd.Series(range(100, 130))  # 100..129
    ret_5d = multi_day_return(prices, 5)
    assert ret_5d.iloc[10] == prices.iloc[10] / prices.iloc[5] - 1


def test_feature_pipeline_produces_expected_columns():
    dates = pd.bdate_range("2022-01-01", periods=300)
    rng = np.random.default_rng(2)
    prices = pd.DataFrame(
        {
            "date": dates,
            "adj_close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300)),
            "volume": rng.integers(1e6, 5e6, 300),
        }
    )
    feats = build_features_for_asset(prices)
    expected_cols = {
        "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "vol_5d", "vol_10d", "vol_20d", "vol_60d", "vol_ratio_20_60",
        "momentum_20d", "ma_ratio_20_60", "rsi_14",
        "rolling_drawdown", "momentum_vol_interaction",
        "beta_60d", "corr_market_60d", "volume_zscore_20d",
    }
    assert expected_cols.issubset(set(feats.columns))
    # first row must have no lookback data yet
    assert pd.isna(feats.loc[0, "ret_1d"])


def test_power_transform_reduces_skew_on_skewed_feature():
    from src.models.preprocessing import build_preprocessing_pipeline

    rng = np.random.default_rng(0)
    skewed = pd.DataFrame({"vol_like": rng.chisquare(df=2, size=1000)})  # right-skewed, like a volatility feature

    pipeline_no_transform = build_preprocessing_pipeline(["vol_like"], use_power_transform=False)
    pipeline_with_transform = build_preprocessing_pipeline(["vol_like"], use_power_transform=True)

    out_no_transform = pipeline_no_transform.fit_transform(skewed)
    out_with_transform = pipeline_with_transform.fit_transform(skewed)

    skew_before = pd.Series(out_no_transform.ravel()).skew()
    skew_after = pd.Series(out_with_transform.ravel()).skew()

    assert abs(skew_after) < abs(skew_before)


def test_train_model_uses_power_transform_only_for_logistic_regression():
    from src.models.train import train_model

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.chisquare(df=2, size=200), "b": rng.normal(0, 1, 200)})
    y = pd.Series(rng.integers(0, 2, 200))

    trained_lr = train_model(X, y, "logistic_regression")
    trained_rf = train_model(X, y, "random_forest")

    lr_steps = dict(trained_lr.pipeline.named_steps["preprocessing"].transformers_[0][1].named_steps)
    rf_steps = dict(trained_rf.pipeline.named_steps["preprocessing"].transformers_[0][1].named_steps)

    assert "power_transform" in lr_steps
    assert "power_transform" not in rf_steps


def test_model_registry_has_no_external_native_library_dependency():
    """This project previously used XGBoost as its gradient-boosting model,
    which pulls in a separately-compiled C++ library with its own binary
    wheel per platform -- a real failure mode this project hit directly:
    `XGBoostError: Library (libxgboost.dylib) could not be loaded` on
    macOS without the OpenMP runtime installed, which crashed importing
    this entire module (including logistic_regression and random_forest,
    which have nothing to do with gradient boosting at all).
    `HistGradientBoostingClassifier` replaces it specifically because it
    ships inside scikit-learn itself with no separate native library to
    fail to load. This test asserts all three registered models can be
    imported and instantiated using ONLY the sklearn import already
    required by the rest of this module -- i.e. building the registry can
    never fail due to a missing/broken external ML library.
    """
    from src.models.train import _build_model_registry

    registry = _build_model_registry()
    assert set(registry.keys()) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    for name, factory in registry.items():
        model = factory()
        assert model is not None, f"{name} factory returned None"
        assert hasattr(model, "fit"), f"{name} is not a valid sklearn estimator"


def test_hist_gradient_boosting_handles_class_imbalance():
    """HistGradientBoostingClassifier must be configured with
    class_weight="balanced", the same as the other two models -- risk
    events are rare by construction (see src/models/targets.py), and an
    unweighted classifier would learn to always predict "no risk".
    """
    from src.models.train import _build_model_registry

    model = _build_model_registry()["hist_gradient_boosting"]()
    assert model.get_params()["class_weight"] == "balanced"


def test_write_features_column_list_matches_feature_spec_not_a_stale_hardcoded_list():
    """Regression test for docs/SECOND_FORENSIC_AUDIT.md Issue C4:
    Repository.write_features must derive its column list from
    FEATURE_SPEC (the single source of truth), not a separately
    maintained literal that can silently drift out of sync -- which
    previously caused vol_ratio_20_60 and momentum_vol_interaction to be
    silently dropped before ever reaching the database.
    """
    import inspect

    from src.data import repository as repository_module
    from src.features.feature_pipeline import FEATURE_SPEC

    source = inspect.getsource(repository_module.Repository.write_features)
    # The fixed implementation must reference FEATURE_SPEC directly,
    # rather than spelling out every feature name as a literal.
    assert "FEATURE_SPEC" in source

    # And every currently-registered feature must be a plausible target
    # column this method could write -- i.e. present in df.columns after
    # the standard rename, using a fabricated dataframe with all of them.
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({
        "ticker": ["AAPL"] * 3,
        "date": pd.bdate_range("2024-01-01", periods=3),
        **{col: [0.1, 0.2, 0.3] for col in FEATURE_SPEC},
    })

    class FakeRepo:
        def get_asset_id_map(self):
            return {"AAPL": 1}

    # Reuse the real method, bound to a minimal fake self, to inspect the
    # column list it would actually attempt to insert (without needing a
    # live database for this particular check).
    captured = {}

    class CapturingEngine:
        def begin(self):
            class Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

                def execute(self_inner, stmt, row=None):
                    captured["insert_sql"] = str(stmt)
                    return None

            return Ctx()

    fake_repo = FakeRepo()
    fake_repo.engine = CapturingEngine()
    repository_module.Repository.write_features(fake_repo, df)

    assert "insert_sql" in captured
    for feature_name in FEATURE_SPEC:
        assert feature_name in captured["insert_sql"], (
            f"'{feature_name}' from FEATURE_SPEC was not included in the "
            f"INSERT statement -- the write path is still dropping a "
            f"registered feature."
        )
