# Tiered Storage Architecture

> **Routing note:** `HOT_YEAR_CUTOFF` in `scripts/config.py` is **deprecated** and
> retained only for backward compatibility. Tier routing uses
> `get_tier_for_date()` (rolling window from `as_of`). See `scripts/config.py`.

## Why

GDELT 2.0 is ~180 GB raw for 2015–2024. Available local disk is 53 GB.
Tiered storage filters older data at ingest, keeping recent data full-fidelity
for streaming + analytics, and older data compressed + filtered for batch ML training.

## Tiers

| Tier | Years | Schema | Filter | Est. Size | HDFS Path |
|------|-------|--------|--------|-----------|-----------|
| Hot | 2023–2024 | Full GDELT v1 (58 cols) | None | ~36 GB | `/supply-chain/raw/hot/year=YYYY/month=MM/` |
| Warm | 2015–2022 | 12 cols | CAMEO 14/17/18/19/20 + chokepoint geo | ~12 GB | `/supply-chain/raw/warm/year=YYYY/month=MM/` |
| Cold | 2015–2024 | Daily aggregates per chokepoint | Pre-aggregated | <1 GB | `/supply-chain/features/` |

## Schema contract

Parquet layouts for **HOT**, **WARM**, **COLD** (future), and **FEATURES** are
defined as PySpark `StructType`s in [`scripts/schemas.py`](../scripts/schemas.py),
with version constants `HOT_SCHEMA_VERSION`, `WARM_SCHEMA_VERSION`,
`COLD_SCHEMA_VERSION`, and `FEATURES_SCHEMA_VERSION` (all currently `1`).
The pipeline calls `validate_*()` at read/write boundaries so drift fails fast.

| Table | Version | Description |
|-------|---------|-------------|
| HOT | 1 | Full GDELT row + `event_date` as written by `ingest_gdelt_tiered` (hot path). |
| WARM | 1 | `WARM_COLUMNS` projection for warm/cold ingest path until cold aggregates exist. |
| COLD | 1 | Speculative monthly aggregates per chokepoint (PR L will populate). |
| FEATURES | 1 | Daily training table from `build_features()` (GDELT rollups, oil, FRED, label). |

## Pipeline

1. `ingest_gdelt_tiered(path, year, month)` routes to hot or warm (or warm with cold warning) via `get_tier_for_date()`.
2. `clean_and_filter_tiered()` reads BOTH tiers, normalizes hot→warm schema, unions, dedupes, tags chokepoints.
3. `build_features()` consumes the unified output and writes the **features** Parquet (not the cold tier; cold tier is separate, PR L).

## Why Hot vs Warm Cutoff Is 2023

Two reasons:

- Live demo (Kafka streaming) replays recent events — needs full schema fidelity.
- Red Sea case study (Nov 2023+) needs full data for narrative impact.
- Older data is for ML signal only — filtered subset is sufficient.

## Adding/Demoting Months

- New months: backfill orchestrator (PR #2) calls `ingest_gdelt_tiered` per month.
- Demote hot → warm: `lifecycle_demote.py` (PR #5) re-projects + writes warm, then deletes hot.

## Verification

Use `scripts/ingest_one_month.py` with `spark-submit` for a single month. After hot and warm samples exist, copy Parquet to `data/parquet/hot_sample/` and `data/parquet/warm_sample/` and run `python3 tests/test_tier_ingest.py`.

End-to-end union check:

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --driver-memory 2g --executor-memory 4g \
  /opt/scripts/smoke_clean_tiered.py
```

Note: `spark-submit` does not accept a Python `-c` string; use `smoke_clean_tiered.py` (or an equivalent small driver script).
