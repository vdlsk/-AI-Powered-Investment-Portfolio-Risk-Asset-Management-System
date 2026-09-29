# Audit Summary — FinRisk

## Scope honesty

The master prompt this audit responds to specifies a 12-phase,
170-section correction process. This pass covers **Phase 0 (audit) plus
the highest-severity items from the 30-item priority list**, fixed with
real tests, not the full 26-category exhaustive re-audit of every file in
the project. Claiming otherwise would itself violate the master prompt's
own instruction not to fabricate results. See `docs/FORENSIC_AUDIT.md` for
full evidence on each item.

## What was found and fixed

1. **Streamlit bypassed MySQL entirely** — the dashboard called Yahoo
   Finance directly through `MarketDataProvider`, never through
   `Repository`/MySQL, contradicting the documented architecture. Fixed
   with a new `src/data/service.py` that is the dashboard's only read path.
2. **Silent synthetic-data fallback** — the app could switch from real to
   synthetic data automatically, with only an after-the-fact banner. Fixed
   with an explicit, mandatory data-source choice before any data loads.
3. **`hash(ticker)` is not stable across processes** — Python randomizes
   string hashing per process by default, so the "reproducible synthetic
   data" claim was false across separate runs. Fixed with a `sha256`-based
   deterministic seed.
4. **A test contained `assert ... or True`** — an unconditionally-true
   assertion that could never fail, exactly the anti-pattern the master
   prompt names explicitly. Removed and replaced with a real assertion of
   the property the test was meant to check.
5. **Backtest timing bug for `execution_lag_days != 1`** — weights and
   transaction costs were applied immediately at signal time instead of at
   the labeled execution date. Invisible with the default lag of 1 day,
   real for any other value. Fixed with a pending-trade queue; proven with
   a new test that fails against the old code.
6. **CVaR's `sqrt(time)` horizon scaling was undisclosed as an
   approximation**, unlike the matching VaR code. Fixed with a docstring
   correction — no behavior change needed here, just an honest limitation
   statement.

## What was found but NOT yet fixed

7. **Purge/embargo gaps use calendar days, not trading days**
   (`chronological_split`). This under-purges slightly relative to the
   documented intent. Flagged HIGH priority, deliberately left for a
   dedicated follow-up pass rather than bundled in under time pressure.

## Two more bugs found while verifying the fixes above

Not in the original priority list — found by actually running the fixed
code through `streamlit.testing.v1.AppTest` end-to-end rather than
assuming it worked once it imported cleanly:

8. `_synthetic_price_panel`'s `end` parameter had no default value,
   crashing the moment a user picked "Synthetic demo data" in the new
   sidebar. Fixed.
9. `DataService` returned MySQL's `date` column as raw `datetime.date`
   objects (what `pandas.read_sql` actually returns for a MySQL DATE
   column), not pandas Timestamps — breaking a downstream comparison with
   `TypeError: Cannot compare Timestamp with datetime.date`. Fixed by
   normalizing to `datetime64` once, inside `DataService`, rather than
   patching each call site that happened to hit it.

Both are covered by dedicated regression tests. This is offered as direct
evidence for why "run it end-to-end and check" matters more than "it
imports without error" — the second bug in particular would not have
been caught by any unit test that mocked the repository with
already-correct pandas Timestamps instead of what MySQL actually returns.

## What was not re-audited in this pass

Everything not listed in `docs/FORENSIC_AUDIT.md` — database migrations,
every notebook cell individually, every Streamlit page's UI copy, security
hardening beyond what was already covered in earlier work on this project,
and full static-analysis/compileall sweeps. These may still be correct;
they simply were not independently re-verified from scratch in this pass.

## Standing claim about leakage prevention

Per the master prompt's own required standard: this project does not
claim to be "leakage-free." It implements explicit controls and automated
tests designed to prevent common forms of temporal and data leakage. No
software project can mathematically guarantee the absence of every
possible implementation error — this audit itself found three real bugs
that earlier "leakage-safe" comments had missed, which is direct evidence
for why that more modest claim is the honest one.
