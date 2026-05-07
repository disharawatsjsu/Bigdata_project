#!/usr/bin/env python3
"""
Supply Chain Disruption Intelligence — Spark Pipeline (V1)

Runs end-to-end: GDELT ingestion → cleaning → feature engineering → ML training.
Designed to run inside the spark-master container with:
    spark-submit --master spark://spark-master:7077 spark_pipeline.py

Pipeline stages live in pipeline_ingest, pipeline_features, and pipeline_train.

V1 scope: single-commodity (crude_oil), single-region (Strait of Hormuz + Red Sea)
to prove the pipeline works before scaling.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Canonical entrypoint: create Spark first; pipeline modules use getOrCreate() and share this session.
spark = (
    SparkSession.builder
    .appName("SupplyChainIntel_V1")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")  # small for local dev
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

from config import (
    COMMODITY_RAW,
    FRED_RAW,
    GDELT_RAW,
    HDFS_FEATURES,
    HDFS_GDELT_PARQUET,
    HDFS_MODEL,
    LOCAL_MODE,
)
from schemas import validate_features
from pipeline_ingest import ingest_gdelt
from pipeline_features import clean_and_filter, build_features
from pipeline_train import train_model

if LOCAL_MODE:
    print("⚠ Running in LOCAL mode (no HDFS)")


if __name__ == "__main__":
    raw_df = ingest_gdelt(GDELT_RAW, HDFS_GDELT_PARQUET)

    clean_df = clean_and_filter(HDFS_GDELT_PARQUET)

    features_df = build_features(
        clean_df,
        commodity_path=COMMODITY_RAW,
        fred_path=FRED_RAW,
    )

    features_df.cache()
    validate_features(features_df)
    features_df.write.mode("overwrite").parquet(HDFS_FEATURES)
    print(f"\nFeatures saved to {HDFS_FEATURES}")

    features_df = features_df.filter(F.col("event_date") >= "2024-01-01")
    model = train_model(features_df, HDFS_MODEL)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — V1 vertical slice done.")
    print("=" * 60)

    spark.stop()
