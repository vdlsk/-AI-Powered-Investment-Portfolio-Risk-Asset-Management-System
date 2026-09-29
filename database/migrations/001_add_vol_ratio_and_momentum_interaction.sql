-- Migration: add vol_ratio_20_60 and momentum_vol_interaction to market_features
--
-- WHY (see docs/SECOND_FORENSIC_AUDIT.md, Issue C4): these two features
-- were added to src/features/feature_pipeline.py::FEATURE_SPEC in a later
-- development pass than the original schema, but the schema and
-- Repository.write_features's column list were never updated to match --
-- meaning both columns were silently dropped before any SQL INSERT if a
-- caller ever persisted computed features to MySQL. This migration brings
-- an already-deployed database in line with the fixed schema.sql; for a
-- fresh database, database/schema.sql already includes both columns.
--
-- Apply with:
--   mysql -u root -p finrisk < database/migrations/001_add_vol_ratio_and_momentum_interaction.sql

USE finrisk;

ALTER TABLE market_features
    ADD COLUMN vol_ratio_20_60 DOUBLE NULL AFTER vol_60d,
    ADD COLUMN momentum_vol_interaction DOUBLE NULL AFTER rolling_drawdown;
