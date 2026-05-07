#!/usr/bin/env python3
"""Ingest a single month of GDELT into the tiered HDFS layout.
Called by the backfill orchestrator (one month at a time, resumable).

Usage:
    spark-submit /opt/scripts/ingest_one_month.py --year 2024 --month 1 \\
        --input hdfs://namenode:9000/tmp/gdelt_chunk
"""
import argparse
import sys

# Spark: pipeline_ingest creates the shared SparkSession via getOrCreate() (same process as spark-submit).
from pipeline_ingest import ingest_gdelt_tiered, spark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument(
        "--input",
        required=True,
        help="HDFS path to directory of month's CSV files (or glob)",
    )
    args = ap.parse_args()

    count, tier = ingest_gdelt_tiered(args.input, args.year, args.month)
    print(
        f"[ingest_one_month] DONE: {args.year}-{args.month:02d} "
        f"-> {tier} tier, {count:,} rows"
    )
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
