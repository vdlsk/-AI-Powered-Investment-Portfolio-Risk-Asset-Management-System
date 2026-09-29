# Second Audit Summary — FinRisk

**Phase 1 status: all 4 CRITICAL issues, and 9 of 12 HIGH issues, are
fixed and verified.** The remaining 3 HIGH, all 6 MEDIUM, all 3 LOW, and
all 3 ENHANCEMENT issues from Phase 0 are documented in
`docs/SECOND_FORENSIC_AUDIT.md` and not yet addressed — this document
states plainly what was and wasn't done.

## Actual current test result (run live, not assumed)

```
98 passed, 0 skipped
```
(with a reachable MySQL instance and `.env` configured). All 14 notebooks
re-executed end-to-end with no errors after every fix below. All 9
Streamlit dashboard pages re-verified via `streamlit.testing.v1.AppTest`,
including three new "Save to MySQL" buttons clicked and confirmed to
write real rows.

## Critical issues — all 4 fixed and verified (see previous summary section below for detail)

1. `risk_managed_strategy.py` stale prediction bug — fixed.
2. Parametric VaR drift-scaling math error — fixed.
3. Test suite deleting real developer data — fixed with an isolated database.
4. Silently-dropped engineered features — fixed with schema + registry-derived write path.

## HIGH issues — 9 of 12 fixed and verified this pass

- **H2** (calendar-day vs. trading-day purge/embargo): fixed by rewriting
  `chronological_split` to count purge/embargo in actual trading-day row
  positions. Proof: on a Monday-boundary example, the old code purged only
  2 trading days when 4 were configured (a 50% under-purge); the fix
  purges exactly 4.
- **H7** (`random_feasible_portfolios` ignoring weight bounds): fixed with
  rejection sampling against `max_weight`/`min_weight`.
- **H8** (`efficient_frontier`'s target range ignoring constraints): fixed
  by solving the true feasible return range via linear programming instead
  of using raw individual-asset min/max.
- **H9** (three risk-managed-strategy design gaps — hardcoded safe asset,
  no weight-cap validation after reallocation, reallocation mislabeled as
  exposure reduction): fixed with a configurable `defensive_assets`
  parameter, `max_weight`-respecting redistribution logic, and a genuine
  `allow_cash_allocation` option. Caught and fixed a real bug in this fix
  itself during testing (a mutating-vs-fixed variable mixup that let the
  cap-respecting logic silently fail) — see the module's inline comments.
- **H10** (overly broad `except Exception`): narrowed to the specific,
  enumerable expected failure types; a genuinely unexpected exception now
  propagates instead of being silently swallowed for an entire backtest run.
- **H11** (4 MySQL tables never written to): fixed with real repository
  write methods for all four tables, wired into three dashboard pages
  with explicit "Save to MySQL" buttons (not automatic writes), each
  verified with a real button click producing a real row in MySQL.
- **H12** (notebook 14's hard-coded stale test count): fixed by running
  `pytest` live from within the notebook itself.
- **H4** (self-referential quantile-based risk thresholds): redesigned
  `select_risk_thresholds` to find the lowest probability cutoff at which
  the ACTUAL observed positive rate (precision) reaches an absolute,
  configured level (10%/25% by default), rather than a quantile of the
  model's own output distribution. Proof: constructed the exact scenario
  the issue describes (predictions clustered in a narrow low-probability
  band with zero real signal) and confirmed the function now falls back
  to the most conservative cutoff instead of fabricating a "HIGH" label
  from a meaningless quantile; also confirmed on realistic correlated
  data that the returned "high" cutoff genuinely achieves >= the
  requested precision, measured independently.

Also fixed, found only while testing H9: `param or CONFIG[...]` silently
discards an explicitly-passed `0.0` (falsy in Python) — the same class of
bug documented as already-fixed elsewhere in this project's history, now
also fixed here.

- **H6** (same fitted model used for TEST evaluation and current
  inference): fixed by training two separate models in the ML dashboard
  page — one on TRAIN only (TEST evaluation), one on TRAIN+VALIDATION
  (current inference and its threshold calibration). Verified with a test
  confirming the two are genuinely different fitted objects with
  different learned coefficients.

## What remains open (documented, not yet fixed)

**3 HIGH-severity items**: H1 (notebooks still call Yahoo Finance directly
with a silent fallback, architecturally inconsistent with the dashboard's
explicit-choice fix), H3 (no inner train/validation loop or per-call
threshold calibration inside the walk-forward risk-managed strategy), H5
(the dashboard still retrains a model on every page interaction;
`src/models/registry.py` remains unused by the dashboard).

**All 6 MEDIUM, all 3 LOW, and all 3 ENHANCEMENT items** remain exactly as
documented in Phase 0.

## Honest scope statement

This document and the fixes it describes cover the 4 CRITICAL issues plus
9 of the 12 HIGH issues from `docs/SECOND_FORENSIC_AUDIT.md`, each with
the same before/after verification standard: a real reproduction of the
bug where feasible, a test that fails against the old behavior and passes
against the fix, and — for the database-touching fixes — an actual live
MySQL round-trip, not a mock. It does not claim the remaining 3 HIGH, 6
MEDIUM, 3 LOW, or 3 ENHANCEMENT items have been addressed, and does not
claim the codebase is free of undiscovered issues.
