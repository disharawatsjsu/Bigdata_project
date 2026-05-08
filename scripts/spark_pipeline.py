#!/usr/bin/env python3
"""Legacy Spark entrypoint for ingest + baseline feature generation."""

import json

from pyspark.sql import SparkSession

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
    LOCAL_MODE,
)
from schemas import validate_features
from pipeline_ingest import ingest_gdelt
from pipeline_features import clean_and_filter, build_features, summarize_features


def _write_text(path: str, content: str) -> None:
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    jvm = spark.sparkContext._jvm
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI.create(path), hadoop_conf)
    hdfs_path = jvm.org.apache.hadoop.fs.Path(path)
    if fs.exists(hdfs_path):
        fs.delete(hdfs_path, True)
    stream = fs.create(hdfs_path, True)
    try:
        stream.write(bytearray(content.encode("utf-8")))
    finally:
        stream.close()

if LOCAL_MODE:
    print("⚠ Running in LOCAL mode (no HDFS)")


if __name__ == "__main__":
    raw_df = ingest_gdelt(GDELT_RAW, HDFS_GDELT_PARQUET)

    clean_df = clean_and_filter(HDFS_GDELT_PARQUET)

    commodity_features = build_features(
        clean_df,
        commodity_path=COMMODITY_RAW,
        fred_path=FRED_RAW,
    )

    for commodity, (features_df, threshold, train_end_date) in commodity_features.items():
        features_df.cache()
        validate_features(features_df)
        out_path = f"{HDFS_FEATURES}/baseline/{commodity}"
        features_df.write.mode("overwrite").parquet(out_path)
        summary = summarize_features(commodity, features_df, threshold, train_end_date)
        _write_text(f"{out_path}/_summary.json", json.dumps(summary, indent=2, sort_keys=True))
        print(f"\nFeatures saved to {out_path}: rows={features_df.count():,}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — baseline features written.")
    print("=" * 60)

    spark.stop()
