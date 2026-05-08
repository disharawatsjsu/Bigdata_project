#!/usr/bin/env python3
"""Build wide commodity training features from tiered GDELT and write to HDFS.

Reads:
  - HDFS hot + warm tiers (tiered Parquet)
  - Local (mounted) commodity + FRED CSVs under /opt/data

Writes:
  - {HDFS_BASE}/features/baseline/{commodity}/ (Parquet, overwrite)
  - {HDFS_BASE}/features/baseline/{commodity}/_summary.json
"""

from __future__ import annotations

import json

from pipeline_features import build_features, clean_and_filter_tiered, spark, summarize_features
from config import HDFS_BASE


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


def main() -> int:
    events = clean_and_filter_tiered()

    # Spark uses fs.defaultFS=hdfs://... so absolute paths resolve as HDFS unless file:// is used.
    commodity_path = "file:///opt/data/commodities/commodity_prices.csv"
    fred_path = "file:///opt/data/fred/fred_macro.csv"

    commodity_features = build_features(
        events,
        commodity_path=commodity_path,
        fred_path=fred_path,
    )

    out_path = f"{HDFS_BASE}/features/baseline"
    for commodity, (features, threshold, train_end_date) in commodity_features.items():
        commodity_out = f"{out_path}/{commodity}"
        features.write.mode("overwrite").parquet(commodity_out)

        summary = summarize_features(commodity, features, threshold, train_end_date)
        _write_text(f"{commodity_out}/_summary.json", json.dumps(summary, indent=2, sort_keys=True))
        rows = summary["row_count"]
        print(
            f"[run_feature_refresh] {commodity}: "
            f"source/output date range {summary['date_min']} -> {summary['date_max']}, "
            f"rows={rows:,}, output={commodity_out}"
        )
        print(f"[run_feature_refresh] Wrote {commodity} features and summary to {commodity_out}")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

