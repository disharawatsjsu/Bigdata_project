#!/usr/bin/env python3
"""Build baseline training features from tiered GDELT and write to HDFS.

Reads:
  - HDFS hot + warm tiers (tiered Parquet)
  - Local (mounted) commodity + FRED CSVs under /opt/data

Writes:
  - {HDFS_BASE}/features/baseline/ (Parquet, overwrite)
"""

from __future__ import annotations

from pipeline_features import build_features, clean_and_filter_tiered, spark
from config import HDFS_BASE


def main() -> int:
    events = clean_and_filter_tiered()

    # Spark uses fs.defaultFS=hdfs://... so absolute paths resolve as HDFS unless file:// is used.
    commodity_path = "file:///opt/data/commodities/commodity_prices.csv"
    fred_path = "file:///opt/data/fred/fred_macro.csv"

    features = build_features(
        events,
        commodity_path=commodity_path,
        fred_path=fred_path,
    )

    out_path = f"{HDFS_BASE}/features/baseline"
    features.write.mode("overwrite").parquet(out_path)
    print(f"[run_feature_refresh] Wrote features to {out_path}")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

