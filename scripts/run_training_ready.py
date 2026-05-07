#!/usr/bin/env python3
"""Create ML training-ready dataset from baseline features and write to HDFS.

Reads:
  - {HDFS_BASE}/features/baseline/ (Parquet)

Writes:
  - {HDFS_BASE}/features/training_ready/ (Parquet, overwrite)

Transforms (in Spark):
  - Time-based 70/15/15 split by event_date ordering (percentile_approx)
  - Add a 'split' column: 'train' | 'val' | 'test'
  - Drop rows with nulls in any required feature + label column
  - Validate input/output schemas
"""

from __future__ import annotations

from pyspark.sql import functions as F

from config import HDFS_BASE
from pipeline_features import spark
from schemas import validate_features, validate_training_ready


FEATURE_COLS = [
    "event_count_7d",
    "avg_goldstein_7d",
    "avg_tone_7d",
    "total_mentions_7d",
    "conflict_ratio_7d",
    "event_count_30d",
    "avg_goldstein_30d",
    "avg_tone_30d",
    "total_mentions_30d",
    "conflict_ratio_30d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "treasury_10y",
    "usd_index",
]


def main() -> int:
    in_path = f"{HDFS_BASE}/features/baseline"
    baseline = spark.read.parquet(in_path)
    validate_features(baseline)

    # NOTE: baseline event_date is a DateType; cast to timestamp/long for quantiles.
    quantiles = (
        baseline.selectExpr(
            "percentile_approx(cast(cast(event_date as timestamp) as long), 0.70) as q70",
            "percentile_approx(cast(cast(event_date as timestamp) as long), 0.85) as q85",
        )
        .first()
    )
    train_end = quantiles["q70"]
    val_end = quantiles["q85"]

    if train_end is None or val_end is None:
        raise RuntimeError("Could not compute split quantiles from event_date (q70/q85 are null).")

    training_ready = baseline.withColumn(
        "split",
        F.when(
            F.col("event_date").cast("timestamp").cast("long") < F.lit(train_end), F.lit("train")
        )
        .when(
            F.col("event_date").cast("timestamp").cast("long") < F.lit(val_end), F.lit("val")
        )
        .otherwise(F.lit("test")),
    )

    training_ready = training_ready.na.drop(subset=FEATURE_COLS + ["label"])
    validate_training_ready(training_ready)

    out_path = f"{HDFS_BASE}/features/training_ready"
    training_ready.write.mode("overwrite").parquet(out_path)
    print(f"[run_training_ready] Wrote training_ready to {out_path}")
    print("[run_training_ready] Splits:")
    training_ready.groupBy("split").count().orderBy("split").show(truncate=False)

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

