# Second Forensic Audit — FinRisk

**Method:** every finding below comes from reading the actual current
source file/notebook/schema and, where feasible, executing it — not from
trusting prior audit documents, comments, or docstrings in this
repository. Several findings directly contradict claims made in
`docs/FORENSIC_AUDIT.md` (the first audit pass) — those are called out
explicitly. This document does not modify any source file; per the
instructions this responds to, this is Phase 0 (audit only).

**What was actually run for this pass:** `pytest tests/ -q` (real result
recorded below), and targeted read/grep inspection of every file named in
this document. Notebooks were **not** re-executed in this pass — any
notebook-related finding below is based on reading notebook source cells,
not on confirmed execution output, and is labeled as such.

**Actual current test suite result** (not fabricated, run just before
writing this document):
```
64 passed, 7 skipped
```
The 7 skips are `tests/test_database_integration.py`, which auto-skip
without a reachable MySQL instance (no `.env` was present at run time).
This confirms Issue C3 below — worth noting immediately, since it means
the "71 tests, all passing" claim in `README.md` is only true under a
specific, undocumented precondition (a live MySQL instance with the exact
credentials in `.env`), not unconditionally.

---

## CRITICAL

### Issue C1 — Risk-managed strategy predicts using a feature row from ~5 trading days ago, not the current date

**Severity:** CRITICAL
**Status:** **FIXED** — separated the target-labeled training frame (`train_df`, correctly `dropna()`'d) from the true latest-date feature row (`latest_row`, taken from the unfiltered `feats` frame). Verified with `tests/test_risk_managed_strategy.py::test_prediction_uses_current_date_features_not_stale_ones`, which monkeypatches `predict_proba` to capture the actual row passed in and compares it against independently-computed current-date vs. stale-date features. **Proven to catch the original bug**: reconstructed the pre-fix code and confirmed this exact test fails against it (captured row matched the stale date, not the current one) before confirming it passes against the fix.
**File:** `src/backtesting/risk_managed_strategy.py`
**Function:** `make_risk_managed_strategy.strategy` (inner closure)

**Problem:**
```python
df = feats.copy()
df["target"] = target.values
df = df.dropna()          # <-- drops the LAST horizon_days rows, since their target is undefined
...
latest_row = df[selected].iloc[[-1]]   # <-- "latest" AFTER dropna, not latest overall
```
`build_loss_event_target` (see `src/models/targets.py`) is a
forward-looking target: the last `horizon_days` rows of any series cannot
have a defined target, because their forward window extends past the end
of available data. `df.dropna()` removes those rows. `.iloc[[-1]]` is then
taken from the row that remains after that removal — which is
`as_of_date - horizon_days` (5 trading days, by default config), **not**
`as_of_date`.

**Evidence:** traced directly from `src/models/targets.py::forward_return`
(`fwd = growth.shift(-horizon_days) - 1`, "the LAST `horizon_days` rows of
the series will be NaN") into this file's `df.dropna()` call. No
execution needed to confirm this — it follows deterministically from the
target construction contract, which is itself correctly documented
elsewhere in the codebase (the bug is that this file doesn't account for
it).

**Why it matters:** the entire point of this module is to answer "what is
today's estimated risk." Predicting from 5-day-stale features means the
"current risk prediction" this strategy acts on in the backtest is
systematically lagged by the target horizon, every single time it's
computed.

**Financial/ML consequence:** the risk-managed strategy in notebook 12
that reported "the ML overlay did not produce the shallowest drawdown"
was evaluated using a mechanism that was never actually looking at
current data. That specific empirical conclusion (documented as an
"honest, run-specific result" in the notebook) may be an artifact of this
bug rather than a genuine finding about the strategy's merit — the
strategy as *intended* (react to today's risk) was never actually tested.

**Correct approach:** compute the feature row for `as_of_date` directly
from the feature dataframe BEFORE merging with (and dropping rows for) the
target. The target-labeled subset is for TRAINING only; the prediction
input for "today" must come from the unfiltered feature series' actual
last row.

**Recommended fix:** separate `df_train` (features + target, `dropna()`'d,
used only for `X, y` in `train_model`) from `latest_features` (the full
`feats` dataframe's true last row, independent of target availability).

**Required test:** a regression test asserting
`prediction_row_date == as_of_date` — i.e. the row passed to
`predict_proba` for "today's" estimate has a date/index position matching
`as_of_date`, not `as_of_date` minus the horizon.

---

### Issue C2 — Parametric VaR incorrectly scales the drift term by `sqrt(h)` instead of `h`

**Severity:** CRITICAL (this is a mathematical error, not a documented approximation)
**Status:** **FIXED** — `mu` now scales by `horizon_days` and `sigma` by `sqrt(horizon_days)` separately, matching the textbook formula stated in the function's own docstring. Verified with a manual worked example in `tests/test_risk.py::test_parametric_var_scales_mu_and_sigma_separately_not_combined_sqrt_h` (constructs a series with an exact known mean/std, computes the expected value by hand, and confirms the function matches it — and explicitly confirms it does NOT match the old buggy formula). The old test that encoded the bug (`test_parametric_var_scales_with_sqrt_horizon`, which compared h=4 against h=1×√4 — exactly the wrong formula, incapable of catching this) was replaced.
**File:** `src/risk/var.py`
**Function:** `parametric_var`

**Problem:**
```python
daily_var = -(mu + z * sigma)
return float(daily_var * np.sqrt(horizon_days))
```
This computes `-(mu + z*sigma) * sqrt(h) = -sqrt(h)*mu - sqrt(h)*z*sigma`.

The textbook h-day parametric VaR under i.i.d. Normal daily returns is:
```
mu_h    = h * mu
sigma_h = sqrt(h) * sigma
VaR_h   = -(mu_h + z * sigma_h) = -(h*mu + sqrt(h)*z*sigma)
```
The volatility term (`sqrt(h)*sigma`) happens to come out the same either
way. The drift term does not: the code effectively uses `sqrt(h)*mu`
where it should use `h*mu`.

**Evidence:** direct algebraic derivation, shown above. For `h=1` both
formulas coincide (this is presumably why it was never caught — the
default `var_horizon_days: 1` in `config.yaml` masks the bug in the
default configuration). For `h=4`: current code gives `-2mu - 2*z*sigma`;
correct formula gives `-4mu - 2*z*sigma`. The two diverge specifically in
the return-drift contribution, and diverge more as `h` grows.

**Why it matters:** for a positive-drift asset/portfolio (the normal
case), the current formula understates how much the multi-day expected
gain should offset the multi-day risk, making multi-day parametric VaR
numbers subtly wrong in a directionally-inconsistent way — not merely
"an approximation with documented limitations" (which is how `var.py`'s
own docstring currently frames its limitations; drift-scaling is not
mentioned there because it isn't understood as an approximation, it's an
implementation error against the formula the docstring itself states two
lines above: "VaR = -(mu + z * sigma), scaled to horizon").

**Financial/ML consequence:** any dashboard page or notebook cell that
uses `parametric_var` with `horizon_days > 1` (the Risk Analytics page's
horizon slider defaults to 1 but allows up to 20) reports an incorrect
number for the mean-return contribution to VaR.

**Correct approach:** scale `mu` by `h` and `sigma` by `sqrt(h)`
separately, not the combined `(mu + z*sigma)` expression by `sqrt(h)`.

**Recommended fix:**
```python
def parametric_var(returns, confidence_level=0.95, horizon_days=1):
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = norm.ppf(1 - confidence_level)
    mu_h = horizon_days * mu
    sigma_h = np.sqrt(horizon_days) * sigma
    return float(-(mu_h + z * sigma_h))
```

**Required test:** a manually-calculable example (e.g. `mu=0.001,
sigma=0.02, h=4, confidence=0.95`) with the expected value computed by
hand from the correct formula, asserting the function output matches —
not merely comparing against `h=1` scaled by `sqrt(h)`, which would not
have caught this bug.

---

### Issue C3 — The full test suite runs destructive `DELETE` statements against the developer's normal MySQL database, not an isolated test database

**Severity:** CRITICAL
**Status:** **FIXED** — added `src.config.get_test_mysql_settings()`, which resolves to a distinct database (`MYSQL_TEST_DATABASE` env var, or `f"{MYSQL_DATABASE}_test"` by default). `tests/test_database_integration.py` now connects exclusively through it, and unconditionally provisions that database fresh (DROP + CREATE + reload `schema.sql`) at test-collection time rather than reusing a possibly-stale existing one. Verified two ways: (1) `test_test_database_is_distinct_from_application_database` asserts the two database names can never match, and (2) empirically — seeded 1 row into the real `finrisk` database, ran the full 76-test suite, and confirmed that row still existed afterward.
**File:** `tests/test_database_integration.py`, `src/config/__init__.py`
**Function:** `repo` fixture, `get_test_mysql_settings`

**Problem:**
```python
@pytest.fixture
def repo():
    r = Repository()
    with r.engine.begin() as conn:
        conn.execute(text("DELETE FROM market_features"))
        conn.execute(text("DELETE FROM clean_market_prices"))
        conn.execute(text("DELETE FROM raw_market_prices"))
        conn.execute(text("DELETE FROM assets"))
    yield r
```
`Repository()` connects using whatever `MYSQL_DATABASE` is set in `.env`
(default template value: `finrisk` — see `.env.example`). There is no
separate `finrisk_test` database, and no check preventing this fixture
from running against a database that also holds real ingested data a
developer is using for dashboard/notebook work.

**Evidence:** confirmed by reading `.env.example` (single database name,
no test-specific variable) and this fixture's literal `DELETE FROM`
statements with no scoping/guard.

**Why it matters:** running `pytest tests/ -v` — the exact command this
project's own `README.md` tells a new user to run first — silently wipes
`assets`, `raw_market_prices`, `clean_market_prices`, and
`market_features` in whatever database `.env` points at, every single
test run. A developer who ingests real data, then runs the test suite to
"make sure everything works," loses that data with no warning.

**Financial/ML consequence:** none directly (this is a developer-workflow
correctness issue, not a financial-math issue), but it directly
undermines the project's own claim (`docs/FORENSIC_AUDIT.md`, Issue 1)
that "MySQL is the central persistent data source" — the test suite
itself treats that persistent store as disposable.

**Correct approach:** integration tests must run against a dedicated test
database (e.g. `finrisk_test`), never the database a developer/dashboard
is using for real data.

**Recommended fix:** add a distinct `MYSQL_TEST_DATABASE` environment
variable (or derive `f"{MYSQL_DATABASE}_test"`), have
`tests/test_database_integration.py` connect to that database
specifically, and fail loudly (not silently proceed) if it isn't
configured — rather than defaulting to the same database the app uses.

**Required test:** a test asserting the integration-test `Repository`'s
configured database name is different from a freshly-loaded
production-config `Repository`'s database name (or equivalent guard),
so this exact mistake can't silently reappear.

---

### Issue C4 — Two production features are computed in code but do not exist in the database schema, and are silently dropped by `Repository.write_features`

**Severity:** CRITICAL
**Status:** **FIXED** — added `vol_ratio_20_60` and `momentum_vol_interaction` to `database/schema.sql`'s `market_features` table (plus `database/migrations/001_add_vol_ratio_and_momentum_interaction.sql` for already-deployed databases), and rewrote `Repository.write_features` to derive its column list from `FEATURE_SPEC` directly rather than a separately-maintained literal — so any future drift between the two will now fail loudly (a real SQL `Unknown column` error) instead of silently dropping data. Verified with a live round-trip test, `tests/test_database_integration.py::test_all_feature_spec_columns_actually_round_trip_through_mysql`, which writes every single `FEATURE_SPEC` feature to a real MySQL table and confirms every one reads back correctly.
**File:** `database/schema.sql`, `src/data/repository.py`
**Function:** `write_features`

**Problem:** `src/features/feature_pipeline.py::FEATURE_SPEC` and
`build_features_for_asset` both compute `vol_ratio_20_60` and
`momentum_vol_interaction` (added in a later development pass than the
original schema). Neither column exists in `database/schema.sql`'s
`market_features` table, and `Repository.write_features`'s hardcoded
`cols` list also omits both:
```python
cols = [
    "asset_id", "date", "feature_version", "ret_1d", "ret_5d", "ret_10d",
    "ret_20d", "ret_60d", "vol_5d", "vol_10d", "vol_20d", "vol_60d",
    "momentum_20d", "ma_ratio_20_60", "rsi_14", "rolling_drawdown",
    "beta_60d", "corr_market_60d", "volume_zscore_20d",
]
cols = [c for c in cols if c in df.columns]   # silently filters to this fixed list
```

**Evidence:** direct comparison of `FEATURE_SPEC`'s 17 feature keys against
`market_features`'s DDL columns (verified via `grep` against both files)
— `vol_ratio_20_60` and `momentum_vol_interaction` are present in the
former, absent in the latter and in `write_features`'s `cols` list.

**Why it matters:** this is a real, silent production data-loss bug in
the path the second-pass audit's own target architecture (item 36 in the
prompt this responds to: "FEATURE ENGINE → FEATURE MYSQL") requires. If
anyone actually calls `Repository.write_features(feats)` where `feats`
includes these two engineered columns (which every model-training call
site does compute and use in-memory), the two columns are silently
dropped before the SQL `INSERT`, with no warning printed or raised.

**Financial/ML consequence:** a model trained by reading features back
OUT of MySQL (the architecture the audit requires, not the current
notebook/dashboard behavior which computes features in-memory and never
persists them) would silently lose 2 of 15 engineered features, changing
which features are actually available for training without any error.

**Correct approach:** the feature list embedded in `write_features` must
be derived from the same single source of truth as
`FEATURE_SPEC`/`build_features_for_asset`'s output columns, not a second,
manually-maintained list that can drift — and the database schema must be
kept in lock-step with it (via a migration when new features are added).

**Recommended fix:** either (a) make `write_features` iterate over
`df.columns` intersected with the actual DB table's columns (introspected
via `information_schema`, not a hardcoded Python list), or (b) add a
migration adding `vol_ratio_20_60 DOUBLE` and
`momentum_vol_interaction DOUBLE` to `market_features`, and update
`write_features`'s list to match `FEATURE_SPEC.keys()` directly (single
source of truth) rather than a separately maintained literal.

**Required test:** a test asserting
`set(FEATURE_SPEC.keys()) == set(columns actually insertable by write_features)`,
so this can never silently drift again — this is exactly the "single
authoritative feature registry" the audit prompt asks for (its item 18).

---

## HIGH

### Issue H1 — [STILL OPEN — not addressed in this pass] Notebooks and the Streamlit dashboard use two different, inconsistent data architectures

**Severity:** HIGH
**File:** `notebooks/nb_utils.py`, all of `notebooks/*.ipynb` except conceptually notebook 01
**Function:** `fetch_price_panel`

**Problem:** `docs/FORENSIC_AUDIT.md` (first pass) correctly fixed the
Streamlit dashboard to read exclusively through
`src/data/service.py::DataService` → `Repository` → MySQL, with an
explicit, mandatory user choice between MySQL and synthetic data (no
silent fallback). **The notebooks were never updated to match.**
`notebooks/nb_utils.py::fetch_price_panel` still calls
`src.data.ingestion.MarketDataProvider` (Yahoo Finance) directly, and
silently returns synthetic data on any exception:
```python
try:
    ... MarketDataProvider().fetch(...) ...
except Exception as e:
    print(f"[info] Live data unavailable ({e.__class__.__name__}); using synthetic demo data instead.")
    return synthetic_price_panel(tickers, start, end), True
```
This is printed to notebook output (better than nothing), but it is still
an automatic, silent-to-the-architecture switch with no user gate — the
exact pattern `docs/FORENSIC_AUDIT.md` Issue 2 fixed for the dashboard and
never applied to notebooks.

**Evidence:** read directly from `notebooks/nb_utils.py`, current version.
`docs/FORENSIC_AUDIT.md` itself acknowledges this gap ("flagged as an
open item") but frames it as out-of-scope rather than as an
architecture-consistency defect — this audit disagrees with that framing,
given the target architecture (item 36 in the second-pass prompt)
requires notebooks and Streamlit to share the same MySQL-first path.

**Why it matters:** a reader of these notebooks (an MSc examiner, an
interviewer) cannot tell, without running them and reading printed output
carefully, whether a given notebook's numbers came from real Yahoo
Finance data, MySQL, or a random-walk generator — and the architecture
diagrams in `README.md` describe MySQL as central, which the notebooks
don't actually honor.

**Financial/ML consequence:** none of the numbers/conclusions in
notebooks 03–14 can be trusted to reflect real market behavior without
independently checking `used_synthetic` in each notebook's own output —
several notebooks' "Honest discussion" sections already hedge their
conclusions for this reason, but the underlying architectural
inconsistency is the root cause, not just the small-sample caveat those
sections state.

**Correct approach:** per the second-pass prompt's explicit instruction —
notebook 01 is the only notebook that should demonstrate external
ingestion (Provider → Validation → Cleaning → MySQL); every other
notebook should read through the same `Repository`/`DataService` path the
dashboard uses, with synthetic data available only as an explicit,
separately-invoked demo mode, never an automatic fallback.

**Recommended fix:** rewrite `notebooks/nb_utils.py` to expose a
MySQL-reading function built on `src.data.service.DataService`, used by
notebooks 02–14; keep `MarketDataProvider`/synthetic generation isolated
to notebook 01 and an explicitly-named "demo mode" helper, never invoked
implicitly by the other notebooks.

**Required test:** a static/text-scan test asserting no notebook file
other than `01_data_collection_and_storage.ipynb` contains the string
`MarketDataProvider` in its source cells.

---

### Issue H2 — [FIXED] `chronological_split` purges/embargoes by calendar days, not trading days

**Severity:** HIGH
**File:** `src/models/preprocessing.py`
**Function:** `chronological_split`

**Problem:** unchanged since the first audit pass (which correctly
identified this and left it explicitly open):
```python
train_mask = df[date_col] <= (train_end - pd.Timedelta(days=purge_days))
val_start = train_end + pd.Timedelta(days=embargo_days)
```
`purge_days`/`embargo_days` (`config.yaml`: 5 and 2 respectively) are
documented as trading-day counts (`ml.purge_days: "≥ target_horizon_days"`,
and `target_horizon_days` is explicitly a trading-day horizon in
`targets.py`), but `pd.Timedelta(days=N)` subtracts N **calendar** days.
A 5-calendar-day purge typically spans only 3–4 trading days depending on
weekend placement, under-purging relative to the stated intent.

**Evidence:** direct reading of the function; cross-referenced against
`config.yaml`'s own comment ("purge_days: 5 # >= target_horizon_days,
prevents train/val overlap leakage" — `target_horizon_days` is a trading-
day count per `src/models/targets.py`'s docstring).

**Why it matters:** this is the exact mechanism meant to prevent a
forward-looking target's window from overlapping the next split — an
under-sized purge gap partially defeats the purpose the parameter exists
for, meaning some genuine train/validation overlap in trading-day terms
can still occur near a split boundary.

**Financial/ML consequence:** any model comparison/tuning number reported
in notebooks 11/13 or the ML dashboard page carries a small residual risk
of train/validation contamination near split boundaries — a partial, not
total, defeat of the leakage-prevention guarantee the project's central
claim rests on.

**Correct approach:** purge/embargo must operate on the actual ordered
sequence of trading-day observations present in the data, not on calendar
arithmetic.

**Recommended fix:** rewrite `chronological_split` to index into the
sorted, deduplicated trading-day sequence directly (e.g. find the row
index of `train_end`, then exclude the next `purge_days` rows by
position, not by date subtraction).

**Required test:** construct a dataset spanning a real weekend/holiday
gap; assert the number of rows actually excluded by the purge equals
`purge_days` counted in trading-day observations, not calendar days.

---

### Issue H3 — [STILL OPEN — not addressed in this pass] No inner train/validation loop inside the walk-forward risk-managed strategy: no tuning, no threshold calibration, single untuned model per call

**Severity:** HIGH
**File:** `src/backtesting/risk_managed_strategy.py`

**Problem:** each walk-forward call does feature selection + a single
`train_model(X, y, model_name)` call on the ENTIRE available history up to
`as_of_date`, with no inner split, no hyperparameter search, and a fixed,
globally-configured `high_risk_threshold` (from `config.yaml`, not
recalibrated per walk-forward step).

**Evidence:** read directly from the function body (already reproduced
under Issue C1 above) — there is exactly one `train_model` call, no
`time_series_cv_tune` call, and `high_risk_threshold` is resolved once,
outside the returned closure, from `CONFIG["risk"]["risk_class_thresholds"]["high"]`.

**Why it matters:** the second-pass prompt's item 6 explicitly requires
"inner training set / inner validation set / feature selection /
preprocessing / model selection / hyperparameter tuning / threshold
selection / final model / prediction at t" inside every walk-forward
decision. The current implementation is a deliberately simple version
(its own docstring says "deliberately simple... rather than a second
optimization layered on top") — that design choice is reasonable as a
first version, but does not meet the stated requirement, and the module's
docstring doesn't flag this gap against that specific standard.

**Financial/ML consequence:** the risk-managed strategy's threshold for
"HIGH risk" never adapts to how well-calibrated the model actually is at
each point in the backtest — a systematically over- or under-confident
model at some historical period would trigger the defensive branch too
often or too rarely, with no mechanism to catch this.

**Correct approach:** within each walk-forward call, split the available
history into an inner train/validation portion, use the validation
portion for threshold calibration (at minimum), and document the
trade-off if full hyperparameter tuning per call is judged too
computationally expensive for a multi-year backtest.

**Recommended fix:** at minimum, add an inner chronological split and
recompute `select_risk_thresholds` per walk-forward call rather than using
a single global config value; document clearly if hyperparameter tuning
itself is intentionally left out per-call for runtime reasons.

**Required test:** assert the effective HIGH-risk probability threshold
used differs across at least two different `as_of_date` calls with
genuinely different underlying data — proving it's actually calibrated
per-call, not a disguised constant.

---

### Issue H4 — [FIXED]

**Fix applied:** redesigned `select_risk_thresholds` to require both
predicted probabilities AND actual outcomes, and to select the lowest
probability cutoff at which the ACTUAL observed positive rate (precision)
reaches an absolute, configured level (`config.yaml`'s
`risk_class_thresholds.medium/high`, now 0.10/0.25 — re-documented as
precision levels, not quantiles) rather than a quantile of the model's
own output. Verified with `tests/test_risk_thresholds.py`, including a
direct reconstruction of the scenario this issue describes (predictions
clustered in a narrow low-probability band with zero real signal),
confirming the function now falls back to the most conservative cutoff
instead of fabricating a "HIGH" label from a meaningless quantile. Risk classification thresholds are self-referential quantiles of the model's own output, not tied to any absolute financial criterion

**Severity:** HIGH
**File:** `src/models/train.py`
**Function:** `select_risk_thresholds`

**Problem:**
```python
return {
    "medium": float(predicted_probabilities.quantile(medium_quantile)),
    "high": float(predicted_probabilities.quantile(high_quantile)),
}
```
"HIGH" is defined as "top ~34% of whatever probabilities this particular
model happened to output on TRAIN+VALIDATION," not "predicted probability
of loss exceeds some absolute, financially-meaningful level." If a model's
predictions cluster tightly between 4% and 8% (plausible for a rare-event
classifier), the top third of that narrow range still gets labeled "HIGH"
even though every value is objectively a low probability.

**Evidence:** read directly from the function; confirmed the only two
call sites (`app/pages/6_ML_Risk_Prediction.py`, notebook 11 per
`build_notebooks.py`) pass `train_val_proba` with no absolute-threshold
alternative offered anywhere in the codebase.

**Why it matters:** the dashboard displays "Risk classification: HIGH" in
red to a user — a label that reads as an absolute statement about danger
level, when it is actually a purely relative statement about this
specific model run's own output distribution.

**Financial/ML consequence:** the same real-world 6% loss-probability
estimate could be labeled "LOW", "MEDIUM", or "HIGH" purely depending on
what other probabilities the model happened to produce elsewhere in the
TRAIN+VALIDATION window — not a defensible basis for a label a user is
meant to act on.

**Correct approach:** per the second-pass prompt's item 7, redesign using
one defensible, documented, absolute approach: a calibrated probability
threshold (e.g. "HIGH = calibrated P(loss) > 20%"), an expected-loss
minimization threshold, or a cost-sensitive precision/recall-optimized
threshold — chosen on VALIDATION only, never TEST.

**Recommended fix:** replace the quantile-based `select_risk_thresholds`
with a threshold selection function built on the calibration table
already computed in `evaluate.py::calibration_table`, picking cut-points
where the *calibrated* predicted probability crosses fixed, documented
absolute levels (e.g. 10%/25%) rather than percentiles of raw model
output.

**Required test:** assert the HIGH threshold, once selected, corresponds
to a calibrated probability at or above a documented fixed floor (e.g.
≥15%) — not merely "the 66th percentile of whatever this run produced."

---

### Issue H5 — [STILL OPEN — not addressed in this pass] Streamlit ML page retrains a model on every page interaction; `src/models/registry.py` is never used by the dashboard

**Severity:** HIGH
**File:** `app/pages/6_ML_Risk_Prediction.py`

**Problem:** `trained = train_model(X_train, y_train, model_name, ...)` is
called directly inside the page script body, which Streamlit re-executes
on every widget interaction (moving a slider, changing the model
dropdown). There is no call to `src.models.registry.save_model` or
`load_model` anywhere in this file.

**Evidence:** read directly from the file; confirmed via `grep -n
"registry" app/pages/6_ML_Risk_Prediction.py` returning no matches.

**Why it matters:** this directly contradicts the second-pass prompt's
item 9 ("The Streamlit app should NOT retrain models every time the ML
page is opened") and makes the "current risk estimate" non-reproducible
run to run within the same session if any upstream input changes even
slightly, since there is no persisted, versioned artifact behind the
displayed number.

**Financial/ML consequence:** two users opening the same page with the
same inputs at different times could see different "current risk"
percentages purely due to retraining variance (though `random_state` is
fixed, so this is more about staleness/reproducibility than randomness —
still, there is no way to answer "what model produced this number" after
the fact, since nothing is saved).

**Correct approach:** separate OFFLINE training (a script/notebook that
trains, validates, and saves a versioned artifact via
`registry.save_model`) from ONLINE inference (the dashboard loads the
saved artifact via `registry.load_model` and only computes fresh
features, never retrains).

**Recommended fix:** add a `scripts/train_production_model.py` (or a
notebook) that performs the full train → validate → tune → select →
save-to-registry pipeline; change the dashboard page to call
`registry.load_model(latest_experiment_id)` and show a clear "no trained
model found — run training first" message (not a silent retrain) if none
exists.

**Required test:** a test asserting the dashboard page's data-loading
function never calls `train_model` — mirroring
`tests/test_data_service.py::test_data_service_never_calls_provider`'s
pattern of proving an architectural boundary via a mocked call-count
assertion, not just an docstring claim.

---

### Issue H6 — [FIXED]

**Fix applied:** the ML dashboard page now trains two separate models:
`test_eval_model` (fit on TRAIN only, used exclusively for the historical
TEST-split metrics shown) and `production_model` (fit on TRAIN+VALIDATION,
used exclusively for the "current risk estimate" and its threshold
calibration). Verified with a test confirming the two are different
Python objects fit on different data, with measurably different learned
coefficients despite a fixed random seed — proving they are not,
accidentally, the same model reused. The same model instance is used for historical TEST evaluation and "current" real-time inference

**Severity:** HIGH
**File:** `app/pages/6_ML_Risk_Prediction.py`

**Problem:** the single `trained` object (fit on `X_train` only) is used
both to compute `report = classification_report(y_test, ...)` (a
statement about historical methodology quality) and
`current_proba = trained.pipeline.predict_proba(latest_features)` (a
statement about "what should I believe about risk right now").

**Evidence:** read directly from the file (reproduced in Issue H5's
context above).

**Why it matters:** conflates two different questions with two different
correct answers: "how good was this modeling approach, historically,
evaluated honestly" (needs a model that never saw VALIDATION or TEST) vs.
"what's my best current estimate" (should use every bit of available
labeled data, i.e. TRAIN+VALIDATION at minimum, ideally
TRAIN+VALIDATION+TEST once TEST has already served its one-time
evaluation purpose). Using the TRAIN-only model for both means the
"current" number is needlessly less informed than it could be.

**Financial/ML consequence:** the displayed "current risk" probability is
systematically based on less data than is actually available, understating
the model's true potential accuracy for the decision a user actually
cares about (today's risk), for no benefit to the (also compromised, per
Issue H5) historical evaluation.

**Correct approach:** per the second-pass prompt's item 8 — train a
separate "historical evaluation" model (TRAIN → VALIDATION → TEST, exactly
once) and a separate "final production" model (TRAIN+VALIDATION, no TEST
exposure) used only for current inference. TEST remains untouched by the
production model's training.

**Recommended fix:** in the offline training script (see Issue H5), fit
and save two artifacts, or fit the final artifact on TRAIN+VALIDATION
after TEST evaluation is complete and frozen; the dashboard only ever
loads the TRAIN+VALIDATION-fit artifact for current inference.

**Required test:** assert the model used for `current_proba` was fit on a
strictly larger date range than the model used for TEST evaluation (i.e.
they are provably different fitted objects, not the same one reused).

---

### Issue H7 — [FIXED] `random_feasible_portfolios` does not enforce `max_weight`/`min_weight`, despite its name

**Severity:** HIGH
**File:** `src/portfolio/optimization.py`
**Function:** `random_feasible_portfolios`

**Problem:** weights are drawn from `rng.dirichlet(np.ones(n_assets), ...)`,
which only guarantees `sum(weights)=1` and `weights >= 0` — it does not
respect `max_weight`/`min_weight`, the same constraints
`minimum_variance_portfolio`/`maximum_sharpe_portfolio` enforce via SLSQP
bounds. With few assets, Dirichlet(1,...,1) frequently produces samples
where a single asset's weight exceeds any reasonable `max_weight` (e.g.
0.30).

**Evidence:** read directly from the function; no `max_weight`/
`min_weight` parameters exist in its signature at all.

**Why it matters:** the function's own name and its use on the Portfolio
Optimization dashboard page (scattering these points "behind" the
efficient frontier) implies these are portfolios a user could actually
hold under the same constraints as the optimized portfolios shown
alongside them — they are not.

**Financial/ML consequence:** a viewer of the dashboard's efficient-
frontier scatter plot can be misled into thinking the "cloud" of random
points represents genuinely achievable alternative allocations; some
fraction of them violate the same concentration limits the optimizer
itself is constrained by.

**Correct approach:** per the second-pass prompt's item 12 — either (A)
implement genuinely constrained random sampling (e.g. rejection sampling
against the bounds, or a constrained Dirichlet variant), or (B) rename to
`random_long_only_portfolios` and explicitly document "not necessarily
optimizer-feasible."

**Recommended fix:** Option A preferred per the audit prompt — rejection-
sample from the Dirichlet distribution, discarding draws that violate
`max_weight`/`min_weight`, until the requested count is reached (with a
sane iteration cap and a clear error if the constraints are too tight to
sample from efficiently).

**Required test:** assert every returned portfolio's weights satisfy
`min_weight <= w <= max_weight` for all assets, given non-trivial bounds.

---

### Issue H8 — [FIXED] `efficient_frontier`'s target-return range ignores portfolio constraints

**Severity:** HIGH
**File:** `src/portfolio/optimization.py`
**Function:** `efficient_frontier`

**Problem:** `lo, hi = mean_returns.min(), mean_returns.max()` — the
target-return sweep range is the single highest/lowest INDIVIDUAL asset's
own mean return, not the actual achievable range under
`max_weight`/`min_weight`/`long_only` constraints. If `max_weight < 1.0`,
no feasible portfolio can actually achieve `hi` (that would require 100%
allocation to the highest-returning asset). The function currently
handles this by silently dropping infeasible target points
(`if res.success: rows.append(...)`), rather than computing the true
feasible range up front.

**Evidence:** read directly from the function (reproduced above).

**Why it matters:** currently harmless in that no crash occurs and no
literally wrong point is plotted (infeasible targets are filtered out) —
but the frontier's target grid is not evenly/meaningfully spaced across
the ACTUAL feasible region, since a chunk of the requested `n_points`
range near `hi` is systematically wasted on infeasible targets that get
silently discarded, leaving fewer effective points to trace the truly
achievable frontier than `n_points` suggests.

**Financial/ML consequence:** none quantitatively wrong, but the visual
frontier density near the top-return end is misleadingly sparse relative
to what `n_points=40` implies to a viewer.

**Correct approach:** compute the actual feasible target-return range
(e.g. by evaluating `minimum_variance_portfolio`'s achieved return as the
floor, and solving a constrained "maximize expected return subject to
weight bounds" LP as the ceiling) before building the `linspace` sweep.

**Recommended fix:** replace `mean_returns.min()/.max()` with the result
of two auxiliary optimizations — a max-return-subject-to-constraints
solve for `hi`, and (optionally) the min-variance portfolio's own return
for a more representative `lo`.

**Required test:** assert every point in the returned frontier
dataframe's weights satisfy the constraints (already implicitly true via
`res.success`, but should be asserted directly), and assert the fraction
of requested `n_points` that end up feasible is above some reasonable
threshold (e.g. >80%) — a regression guard against the current silent
waste.

---

### Issue H9 — [FIXED] Defensive "safe asset" selection, weight-constraint validation, and true exposure reduction are all unaddressed in the risk-managed strategy

**Severity:** HIGH (three related sub-issues in the same code path)
**File:** `src/backtesting/risk_managed_strategy.py`

**H9a — `historical_returns.std().idxmin()` as a universal "safe asset":**
lowest historical volatility does not necessarily mean defensive in a
crisis (a historically-low-vol asset can still crash hard in a genuine
tail event; correlations often rise precisely when it matters). This is
hard-coded with no configuration override.

**H9b — no weight-constraint validation after the defensive reallocation:**
```python
defensive_weights[safe_asset] = defensive_weights.get(safe_asset, 0.0) + freed_weight
```
adds the entire freed weight to one asset with no check against
`max_weight` — a defensive move could concentrate the portfolio beyond
limits the "normal" optimizer strategies are held to elsewhere in the
same project.

**H9c — this is a reallocation, not an exposure reduction:** weights still
sum to 1.0 after the "defensive" adjustment (still 100% invested,
just concentrated differently) — there is no actual cash/low-risk-asset
de-risking option, despite the module's own docstring describing this as
"cut exposure by `defensive_scale`," which is not literally what happens
(total market exposure is unchanged; only its distribution changes).

**Evidence:** all three read directly from the function body (already
reproduced under Issue C1).

**Why it matters:** H9a and H9c together mean the strategy's actual
behavior during a genuine systemic risk-off event (the exact scenario it
exists to help with) could be to concentrate more heavily into a single
asset that itself might be crashing — the opposite of the intended
protective effect, and worse than doing nothing in the worst case.

**Financial/ML consequence:** any backtest result attributed to this
"risk management" strategy should not be interpreted as evidence for or
against genuine risk-reduction efficacy, only for or against this specific
reallocation rule.

**Correct approach:** make the defensive-asset choice configurable
(explicit ticker(s), asset class, or a proper minimum-volatility
sub-portfolio); validate the post-adjustment weights against the same
`max_weight`/`min_weight` bounds used elsewhere; and support an actual
cash/low-risk allocation option so "reduce exposure" can mean what it
says, with the rotation-vs-reduction distinction documented explicitly.

**Recommended fix:** add a `defensive_assets: list[str] | None` and
`allow_cash_allocation: bool` parameter to `make_risk_managed_strategy`;
if `allow_cash_allocation=True`, defensive weight reduction actually
lowers total invested weight (summing to `1 - reduction`, with the
"cash" portion implicitly earning zero return) rather than always
reallocating to 100% invested.

**Required test:** assert post-defensive-adjustment weights respect
`max_weight`; assert that with `allow_cash_allocation=True`, total
invested weight is genuinely below 1.0 during a HIGH-risk period.

---

### Issue H10 — [FIXED] Broad `except Exception` in the risk-managed strategy converts unexpected programming errors into silent fallback

**Severity:** HIGH
**File:** `src/backtesting/risk_managed_strategy.py`

**Problem:**
```python
try:
    after_variance = drop_low_variance_features(X)
    selected = iterative_vif_removal(X[after_variance], vif_threshold=10.0)
    X = X[selected]
    trained = train_model(X, y, model_name, numeric_features=selected)
except Exception as e:
    logger.warning("Risk-managed strategy: training failed at %s (%s), using base weights", as_of_date, e)
    return base_weights
```
This catches EVERYTHING — a genuine bug (e.g. a typo introduced in a
future edit, a broken import, a real `TypeError` from a code change
elsewhere) is indistinguishable from an expected, benign data-quality
issue (e.g. too few varying rows at an early backtest date) and both are
silently converted into "use base weights" with only a `logger.warning`
call, which is easy to miss in a long backtest's output.

**Evidence:** read directly from the function.

**Why it matters:** a real bug introduced anywhere in the training
pipeline this function calls into (`train_model`,
`iterative_vif_removal`, `drop_low_variance_features`) would not crash the
backtest — it would silently degrade every single risk-managed strategy
call for the rest of the run into the equal-weight base strategy, with no
visible failure, potentially for months of backtest history, while the
notebook/dashboard output looks entirely normal.

**Financial/ML consequence:** any historical conclusion drawn from this
strategy's backtest results could actually be describing the BASE
strategy's behavior (silently substituted in) rather than the risk-
managed strategy's, with no way to tell from the output alone.

**Correct approach:** catch specific, expected exception types only (e.g.
a custom `InsufficientDataError`, or `ValueError` from the known-safe
feature-selection edge cases), and let genuinely unexpected exception
types propagate (or be recorded in a way that's impossible to miss, e.g.
accumulated into the `BacktestResult` and surfaced prominently).

**Recommended fix:** narrow the `except` clause to the specific,
enumerable failure modes already anticipated elsewhere in this codebase
(e.g. `ValueError` from `iterative_vif_removal`'s infeasibility check),
and add an explicit counter/log of how many calls fell back due to
genuine vs. expected reasons, surfaced in `BacktestResult`.

**Required test:** inject a deliberately-broken `train_model` (e.g. via
monkeypatching to raise `AttributeError`) and assert the backtest either
propagates the error or records it in a way a test can detect — not that
it silently produces a "successful"-looking `BacktestResult`.

---

### Issue H11 — [FIXED] Four MySQL tables are defined in the schema but never written to by any code path

**Severity:** HIGH
**File:** `database/schema.sql`, `src/data/repository.py`

**Problem:** `risk_metrics`, `risk_predictions`, `backtest_results`, and
`rebalance_recommendations` all exist as fully-specified tables in
`database/schema.sql`, but `grep`-ing the entire `src/` and `app/` trees
for actual `INSERT INTO`/repository write methods targeting these four
table names returns zero matches. The only hits for these names in
`src/backtesting/engine.py` and `src/portfolio/analytics.py` are a Python
module alias (`from src.risk import metrics as risk_metrics`) — unrelated
to the database table of the same name.

**Evidence:** targeted `grep` across the codebase, confirmed zero
`Repository` methods exist for any of these four tables.

**Why it matters:** directly contradicts the second-pass prompt's item 30
("If MySQL is intended to be the central analytical database, integrate
[these tables] into the actual pipeline. Do not create tables that are
never used.") and the broader claim that MySQL is the project's
persistent analytical store — backtest results, computed risk metrics,
ML predictions, and rebalancing recommendations currently exist only
transiently in memory/Streamlit session state, never persisted anywhere.

**Financial/ML consequence:** none of this project's actual analytical
outputs (a computed VaR number, a backtest's Sharpe ratio, a model's risk
prediction, a rebalancing recommendation) can currently be looked up
later, compared across runs, or audited after the fact — undermining the
reproducibility goal stated elsewhere in this project's own documentation.

**Correct approach:** either implement the write paths for these four
tables (backtest results at the end of `run_walk_forward_backtest`, risk
metrics after `performance_report`, predictions after each ML dashboard
inference, rebalance recommendations after `rebalance_plan`), or remove
the unused tables from the schema and be explicit in documentation that
this project's persistence layer currently covers only prices/features,
not analytical outputs.

**Recommended fix:** add `Repository.write_backtest_result`,
`write_risk_metrics`, `write_risk_prediction`, `write_rebalance_recommendation`
methods; call them from the natural completion points of
`run_walk_forward_backtest`, `performance_report`, the ML dashboard page,
and `rebalance_plan`'s dashboard caller respectively.

**Required test:** an integration test (added to
`tests/test_database_integration.py`, using the isolated test database
from Issue C3's fix) that runs a small backtest, calls the new write
method, and reads the row back from MySQL to confirm it round-trips.

---

### Issue H12 — [FIXED] Notebook 14 hard-codes a stale test-pass count

**Severity:** HIGH
**File:** `notebooks/14_final_summary_and_conclusions.ipynb`

**Problem:**
```python
results_summary = {
    "tests_passing": "29 / 29 (see tests/)",
    ...
}
```
The actual current count (verified by running `pytest tests/ -q` for this
audit) is 64 passed + 7 skipped = 71 total, itself a number that has
already changed multiple times across this project's development and
will change again. "29/29" reflects an early snapshot, not the current
repository state.

**Evidence:** read directly from the notebook's source cell; cross-checked
against this audit's own freshly-run `pytest` output above.

**Why it matters:** directly matches the second-pass prompt's item 25
("NOTEBOOK 14 MUST NOT HARD-CODE TEST RESULTS... Never fabricate test
status"). A hard-coded number in a presentation notebook is exactly the
kind of stale claim an MSc examiner or interviewer could catch and
reasonably read as evidence the project isn't actually maintained /
re-verified.

**Financial/ML consequence:** none directly (documentation-accuracy
issue), but it's a credibility risk for the exact audience this project
is being prepared for.

**Correct approach:** either dynamically run `pytest` from within the
notebook and report the live result, or remove the hard-coded number
entirely and instruct the reader to run `pytest tests/ -v` themselves.

**Recommended fix:** replace the hard-coded string with a `subprocess.run(["pytest", "tests/", "-q"], ...)`
call whose output is captured and displayed, OR simply state "Run `pytest
tests/ -v` to see the current, live result" with no embedded number at
all (safer, since `subprocess`-invoking pytest from inside a notebook
that's itself sometimes run inside a test/CI context can have surprising
interactions).

**Required test:** a documentation-scan test asserting no notebook
contains a hard-coded `"N / N"`-shaped test-count string.

---

## MEDIUM

### Issue M1 — Historical VaR/CVaR use single-day-quantile-times-sqrt(h), not a true rolling h-day empirical distribution

**Severity:** MEDIUM
**File:** `src/risk/var.py::historical_var`, `src/risk/cvar.py::historical_cvar`

**Problem:** both compute a 1-day empirical quantile/tail-mean, then
multiply by `sqrt(horizon_days)`. The second-pass prompt's item 10
correctly points out the more rigorous approach is to build the actual
empirical distribution of h-day CUMULATIVE returns (via a rolling h-day
window over history) and take the quantile/tail-mean of THAT
distribution directly — which captures real multi-day return
autocorrelation/clustering, rather than assuming i.i.d. days as the
sqrt(h) shortcut does.

**Evidence:** read directly from both functions; this is already
partially disclosed as a limitation in both docstrings (following the
first audit pass's Issue 6 fix for CVaR) — but disclosure is not the same
as implementing the more correct method the second-pass prompt asks for.

**Why it matters:** distinguishes this from Issue C2 (parametric VaR):
this is not a mathematical error, it's a documented-but-still-present
methodological simplification the second-pass prompt asks to be replaced
with the more rigorous rolling-window approach.

**Financial/ML consequence:** historical VaR/CVaR at horizons > 1 day
somewhat understate true multi-day tail risk on data with volatility
clustering (which `notebooks/03_exploratory_data_analysis.ipynb`'s own
EDA demonstrates is present) — the same category of concern that
motivated adding the block-bootstrap Monte Carlo option in an earlier
pass, not yet extended to historical VaR/CVaR.

**Correct approach:** compute rolling h-day cumulative returns from the
historical daily series, then take the empirical quantile/tail-mean of
that distribution directly, per the formulas given in the second-pass
prompt's item 10.

**Recommended fix:** add `historical_var_multiday`/`historical_cvar_multiday`
(or an `h_day_method: "sqrt_scaling" | "rolling_window"` parameter) that
builds `returns.rolling(h).apply(lambda r: (1+r).prod()-1)` and computes
the quantile/tail-mean directly on that series.

**Required test:** a manually-verifiable small example (e.g. 20 known
daily returns, h=5) comparing the rolling-window method's output against
a by-hand calculation.

---

### Issue M2 — Stress testing asks the user to manually pretend every ticker belongs to one asset class, instead of using `assets.asset_class` from the database

**Severity:** MEDIUM
**File:** `app/pages/3_Stress_Testing.py`

**Problem:**
```python
scenario_choice = st.selectbox(
    "Assume every holding behaves like:",
    options=["equities", "bonds", "gold"], ...
)
...
uniform_shocks = {t: shock_value for t in tickers}
```
Every ticker in the portfolio is shocked identically, using whichever
single asset class the user manually selects for the whole portfolio at
once — even though `database/schema.sql`'s `assets` table has an
`asset_class` column specifically meant to support per-ticker
classification, and `Repository`/`DataService` already have access to it.

**Evidence:** read directly from the file; confirmed `assets.asset_class`
exists in schema but is never queried by this page (`grep -n
"asset_class" app/pages/3_Stress_Testing.py` returns nothing).

**Why it matters:** a genuinely mixed portfolio (the dashboard's own
default tickers are `AAPL, MSFT, GLD, BND, SPY` — equities, gold, and
bonds together) cannot be stress-tested realistically at all with this
page as built; the user is forced into a single-asset-class fiction for
the whole portfolio.

**Financial/ML consequence:** any stress-test number shown on this page
for the default (or any mixed) portfolio is not a meaningful estimate of
that portfolio's actual behavior under the named scenario.

**Correct approach:** query `assets.asset_class` per ticker via
`DataService`/`Repository`, apply each scenario's per-asset-class shock to
the correct subset of holdings, and warn (not silently zero-shock) for
any ticker with no recorded asset class.

**Recommended fix:** add `Repository.get_asset_classes(tickers) ->
dict[str, str]`; rewrite the stress-test page to map each ticker to its
real class and apply the corresponding shock line from
`config.yaml`'s `stress_scenarios`, with an explicit UI warning listing
any tickers with unmapped/missing asset class.

**Required test:** a test asserting a mixed-asset-class portfolio's
stress-test result differs correctly depending on which subset of
holdings falls in the shocked asset class, using a fake repository
returning known asset classes.

---

### Issue M3 — Portfolio-derived features reuse stock-technical-indicator names (RSI, moving-average ratio) on portfolio wealth without justifying the interpretation

**Severity:** MEDIUM
**File:** `src/backtesting/risk_managed_strategy.py`, `app/pages/6_ML_Risk_Prediction.py`, `app/pages/5_Market_Regime.py`

**Problem:** all three call `build_features_for_asset` on a synthetic
"price" series built from cumulative PORTFOLIO wealth
(`wealth = (1 + port_returns).cumprod()`), producing columns named
`rsi_14`, `ma_ratio_20_60`, `momentum_20d` — names borrowed unchanged
from single-stock technical analysis, applied to a portfolio-level
aggregate wealth curve.

**Evidence:** read directly from all three files' construction of
`prices_df`/`wealth` fed into `build_features_for_asset`.

**Why it matters:** RSI's textbook interpretation (overbought/oversold
momentum oscillator) was designed for a single security's price series;
whether it retains the same interpretation applied to a multi-asset
portfolio's aggregate wealth curve (which behaves differently — smoother,
different autocorrelation structure — than any single asset) is not
obviously true and is not discussed anywhere in this project's
documentation.

**Financial/ML consequence:** none of the reported feature importances
(e.g. notebook 11's Random-Forest importance chart) are wrong in a
mechanical sense, but a reader could reasonably misinterpret
"`rsi_14` was an important feature" as carrying the same textbook meaning
it has for single-stock RSI, when the underlying series it was computed
on is structurally different.

**Correct approach:** per the second-pass prompt's item 19 — prefer
portfolio-native names (`portfolio_20d_return`, `portfolio_20d_volatility`,
`portfolio_drawdown`, `portfolio_momentum`) that don't imply an
interpretation borrowed from single-stock technical analysis without
justification.

**Recommended fix:** either rename the portfolio-level feature columns
(a `feature_pipeline.py` parameter like `is_portfolio_level: bool`
controlling naming), or add an explicit note wherever these features are
computed on portfolio wealth, stating the reused stock-indicator names are
kept for code-reuse convenience only and their textbook interpretation is
not being claimed to transfer.

**Required test:** none required for a naming/documentation change; if
renamed, existing tests referencing the old column names would need
updating (a mechanical, not a leakage/correctness, concern).

---

### Issue M4 — `ON DELETE CASCADE` deletes financial history; no soft-delete (`is_active`) path for assets

**Severity:** MEDIUM
**File:** `database/schema.sql`

**Problem:** `raw_market_prices`, `clean_market_prices`, `market_features`,
etc. all reference `assets(asset_id) ON DELETE CASCADE`. Deleting a row
from `assets` (e.g. to remove a mistakenly-added ticker) permanently
destroys all historical price/feature data for that asset, with no
recovery path. `assets.is_active` exists as a column (confirmed in
schema) but nothing in the codebase currently uses it as a soft-delete
mechanism instead of a hard `DELETE`.

**Evidence:** read directly from `database/schema.sql`; confirmed no
`UPDATE assets SET is_active = FALSE` call exists anywhere in
`src/`/`app/` (`grep -rn "is_active" src/ app/` returns no matches outside
the schema file itself).

**Why it matters:** for a project whose stated goal includes
reproducibility, an accidental or mistaken `DELETE FROM assets WHERE
ticker = 'X'` silently and irreversibly destroys that ticker's entire
historical record, including any features/backtest inputs derived from
it — a much higher-consequence operation than the cascade's convenience
justifies for what is, in effect, financial historical data.

**Financial/ML consequence:** loss of ability to reproduce any earlier
analysis that depended on the deleted asset's data.

**Correct approach:** per the second-pass prompt's item 28 — prefer
`is_active = FALSE` (already a schema column) over a hard delete for
assets with real historical data attached; reserve hard `DELETE` for
genuine data-entry mistakes with no dependent rows yet.

**Recommended fix:** add `Repository.deactivate_asset(ticker)` using
`UPDATE assets SET is_active = FALSE`; update `get_asset_id_map` (or add a
variant) to optionally exclude inactive assets from normal queries while
preserving their historical rows.

**Required test:** assert `raw_market_prices` rows for a deactivated asset
still exist and are readable after `deactivate_asset` is called, unlike
after a hard `DELETE`.

---

### Issue M5 — No dataset/feature/model version metadata sufficient to reproduce a result, despite reproducibility being a repeatedly-stated project goal

**Severity:** MEDIUM
**File:** `database/schema.sql`, `src/models/registry.py`

**Problem:** `market_features.feature_version` exists as a column
(default `'v1'`, never incremented anywhere in the codebase), and
`model_runs`/`ExperimentRecord` (`src/models/registry.py`) capture
hyperparameters/seed/date-ranges — but there is no `dataset_version` or
"ingestion run ID" concept anywhere connecting a specific set of
`raw_market_prices` rows (which could later be revised if re-ingested
from a provider that itself revises historical data, e.g. corporate
action adjustments) to a specific trained model.

**Evidence:** read directly from `database/schema.sql` (single
`feature_version` column, statically defaulted); `grep -rn
"feature_version" src/` shows it's always passed as the literal default
`'v1'`, never derived from anything that would change if the underlying
raw data changed.

**Why it matters:** per the second-pass prompt's item 27 — "A model must
be reproducible from: dataset version + feature version + configuration +
model version. Do not falsely claim exact reproducibility if the external
provider can revise historical data." This project does not currently
make that claim explicitly, but also doesn't have the metadata that would
be needed to honestly answer "exactly which data produced this model" if
asked.

**Financial/ML consequence:** if a data provider (Yahoo Finance) later
revises historical adjusted-close values (a real, documented phenomenon
for dividend/split adjustments), there is currently no way to detect that
a previously-trained model's inputs have silently changed underneath it.

**Correct approach:** add an `ingestion_run_id` (or timestamp-based
dataset version) to `raw_market_prices`, propagate a real, incrementing
`feature_version` when the feature computation logic changes, and record
both in `model_runs`.

**Recommended fix:** add a lightweight `ingestion_runs` table (run id,
timestamp, tickers, date range, source); tag `raw_market_prices` rows with
the run id that inserted them; extend `ExperimentRecord` with
`dataset_version`/actual `feature_version` fields sourced from these.

**Required test:** none required beyond the schema/repository changes
themselves being exercised by existing/new integration tests.

---

### Issue M6 — Portfolio persistence intent is undecided and undocumented

**Severity:** MEDIUM
**File:** `database/schema.sql` (`portfolio`, `portfolio_holdings`,
`portfolio_transactions` tables), `app/components/state.py`

**Problem:** the schema defines full portfolio/holdings/transaction
tables, but the dashboard's actual portfolio state lives entirely in
`st.session_state` (`app/components/state.py`) and is never written to
these tables. Nothing in `README.md` explicitly states "the dashboard
portfolio is session-only, not persisted" — a reader could reasonably
assume, given the schema's existence, that portfolios ARE saved.

**Evidence:** `grep -rn "portfolio_holdings\|portfolio_transactions"
src/ app/` (excluding schema.sql itself) returns no write-path matches.

**Why it matters:** matches the second-pass prompt's item 29 directly —
an undecided, undocumented gap between what the schema implies and what
the application actually does.

**Financial/ML consequence:** none directly; this is a documentation/
scope-clarity issue, not a correctness issue — a user's portfolio
configuration is lost when the Streamlit session ends, which may or may
not be the intended design, but isn't stated either way.

**Correct approach:** make an explicit decision (persist portfolios in
MySQL, or don't) and document it clearly in `README.md`'s architecture
section.

**Recommended fix (if persistence is desired):** add
`Repository.save_portfolio`/`load_portfolio` methods and wire them into
the dashboard sidebar (e.g. a "save this portfolio" button); (if not
desired) add one clear sentence to `README.md` stating the dashboard
portfolio is session-based only, and that `portfolio`/`portfolio_holdings`/
`portfolio_transactions` are schema provided for future/optional use, not
currently wired to the dashboard.

**Required test:** if persistence is implemented, a round-trip
save/load integration test; if not, no test needed beyond the
documentation fix.

---

## LOW

### Issue L1 — Two remaining weak-but-not-fabricated test assertions

**Severity:** LOW
**File:** `tests/test_data_service.py`, `tests/test_features.py`

**Problem:** `assert len(result_1.stdout) > 0` (a sanity check
supplementary to the real assertion `result_1.stdout == result_2.stdout`
in the same test) and `assert model is not None, f"{name} factory
returned None"` (supplementary to `assert hasattr(model, "fit")` in the
same test) are present. Unlike the `assert ... or True` pattern fixed in
the first audit pass, these are not unconditionally-true — they can fail
— but they are weak in isolation (not the load-bearing assertion of their
respective tests).

**Evidence:** direct `grep` match, reviewed in context — both are
secondary assertions alongside a stronger, load-bearing assertion in the
same test function, not standalone weak tests.

**Why it matters:** minor; included for completeness per the second-pass
prompt's item 31 instruction to search for this pattern, but these two
instances do not rise to the same severity as a genuinely fabricated
assertion.

**Financial/ML consequence:** none.

**Correct approach:** no urgent action needed; could be tightened
(e.g. `assert len(result_1.stdout) > 100` with a comment on why 100, or
removed as redundant with the stronger assertion) for stylistic rigor.

**Recommended fix:** low priority — optionally strengthen or remove in a
future pass.

**Required test:** none (this issue is about existing test quality, not
missing coverage).

---

### Issue L2 — Future-perturbation leakage tests exist for the equal-weight and risk-managed strategies, but not explicitly for the optimization-based (`make_periodic_reoptimization_strategy`) strategy

**Severity:** LOW
**File:** `tests/test_backtesting.py`, `tests/test_risk_managed_strategy.py`

**Problem:** `tests/test_backtesting.py::test_backtest_signal_uses_only_past_data`
uses an equal-weight-wrapping strategy; `tests/test_risk_managed_strategy.py::test_risk_managed_strategy_retraining_does_not_use_future_data`
covers the risk-managed strategy. No test in the current suite runs the
same future-perturbation pattern specifically against
`make_periodic_reoptimization_strategy` (the actual portfolio-optimization
based strategy used in notebook 12's "reoptimized_min_var" comparison
arm).

**Evidence:** `grep -n "future\|perturb"` across both test files, cross-
referenced against which strategy factory each test actually exercises.

**Why it matters:** matches the second-pass prompt's item 21, which
explicitly asks for coverage across "Equal-weight strategy / Optimization
strategy / ML strategy / Risk-managed strategy" as four distinct cases —
currently 2 of 4 are covered by a dedicated future-perturbation test.

**Financial/ML consequence:** low direct risk, since
`make_periodic_reoptimization_strategy` is a thin wrapper whose only
data dependency is `historical_returns.tail(lookback_days)`, which is
itself already truncated to `as_of_date` by the backtest engine before
being passed in (verified in the first audit pass) — but an explicit test
would remove the need to trust that reasoning rather than verify it
directly for this specific strategy.

**Correct approach:** add the missing test for symmetry/completeness with
the other two.

**Recommended fix:** mirror
`test_backtest_signal_uses_only_past_data`'s pattern, substituting
`make_periodic_reoptimization_strategy(minimum_variance_portfolio, ...)`
as the strategy under test.

**Required test:** as described above.

---

### Issue L3 — Minor imprecise use of "leakage-free" phrasing in one doc

**Severity:** LOW
**File:** `docs/MODEL_SELECTION_AND_EVALUATION.md`

**Problem:** "A leakage-free pipeline that finds no signal in random data
is a *sign the pipeline is trustworthy*" — used descriptively/
conditionally about a hypothetical pipeline, not as a direct claim about
this project, but the second-pass prompt's item 35 asks specifically to
avoid the phrase "100% leakage-free" (and, by clear extension, unqualified
"leakage-free") anywhere in the repository, in favor of "leakage-aware and
protected against the tested forms of temporal and data leakage."
`docs/AUDIT_SUMMARY.md`'s own usage is already correctly hedged ("does not
claim to be 'leakage-free'").

**Evidence:** `grep -rn "leakage-free"` across the repository, both
instances reviewed in context.

**Why it matters:** minor wording precision; worth a one-line edit for
strict consistency with the second-pass prompt's exact phrasing
requirement.

**Financial/ML consequence:** none.

**Correct approach:** reword to "a pipeline that is genuinely leakage-
aware and finds no signal in random data..." to avoid the flagged phrase
even in a conditional/hypothetical context.

**Recommended fix:** one-line text edit.

**Required test:** a documentation-scan test asserting the literal string
"leakage-free" does not appear anywhere in `docs/`, `README.md`, or
notebook source cells (case-insensitive), enforcing this permanently.

---

## ENHANCEMENT

### Issue E1 — No `docs/TEMPORAL_DATA_CONTRACT.md`

**Severity:** ENHANCEMENT
**File:** N/A (missing file)

Per the second-pass prompt's item 34: a single document defining, for
every pipeline stage (feature engineering, target creation, ML training,
ML prediction, optimization, backtesting, risk management, rebalancing),
exactly what information is available, what's forbidden, the prediction
timestamp, and the execution timestamp does not currently exist. The
information is scattered across module docstrings
(`feature_pipeline.py`, `targets.py`, `engine.py`,
`risk_managed_strategy.py`) but not consolidated into one authoritative
contract document. Recommended as a Phase 10 deliverable, consolidating
already-correct scattered documentation rather than establishing new
rules.

### Issue E2 — No `docs/FINAL_LEAKAGE_AUDIT.md` with the specific category breakdown requested

**Severity:** ENHANCEMENT
**File:** N/A (missing file)

Per the second-pass prompt's item 35: a document with explicit
per-category coverage (target/rolling/preprocessing/feature-selection/
hyperparameter/threshold/optimization/backtest/regime/database/notebook
leakage) does not currently exist as a single file — the closest existing
document, `docs/DATA_LEAKAGE_AUDIT.md` (from a much earlier development
pass), predates several of these categories (threshold leakage, regime
leakage, database leakage, notebook leakage as separate concerns weren't
explicitly enumerated there) and should be revisited/superseded once the
fixes in this document's HIGH/CRITICAL section are actually implemented —
producing it before the fixes land would risk documenting intent rather
than verified behavior.

### Issue E3 — `requirements.txt` completeness was spot-checked, not exhaustively verified via a clean-environment install

**Severity:** ENHANCEMENT
**File:** `requirements.txt`

A grep-based cross-check of imports used in `tests/`/`src/`/`app/`
against `requirements.txt` found no obvious gap, but this was not
verified by actually creating a fresh virtual environment, running `pip
install -r requirements.txt` with nothing else installed, and confirming
`pytest tests/ -v` succeeds from that clean state — the second-pass
prompt's item 32 asks specifically for this. Recommended as a concrete,
cheap verification step for Phase 9 (Tests) rather than a code change.
