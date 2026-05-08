#!/usr/bin/env python3
"""CLI wrapper around ingest_gdelt_tiered for spark-submit (Airflow DAGs).

Expects a single day's GDELT export CSV under GDELT_RAW, e.g.
  /opt/data/gdelt/20260504.export.CSV

Year/month are derived from --date for partition routing inside ingest_gdelt_tiered.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from config import GDELT_RAW
from pipeline_ingest import ingest_gdelt_tiered, spark


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiered GDELT ingest for one calendar day")
    parser.add_argument(
        "--date",
        required=True,
        help="Target calendar day (YYYY-MM-DD); must match GDELT file  YYYYMMDD.export.CSV",
    )
    args = parser.parse_args()

    d = date.fromisoformat(args.date)
    local_csv = f"{GDELT_RAW}/{d.strftime('%Y%m%d')}.export.CSV"
    # Spark is configured with fs.defaultFS=hdfs://... so absolute paths get treated as HDFS.
    # Force a local filesystem read for the host-mounted /opt/data/gdelt files.
    input_csv = f"file://{local_csv}" if local_csv.startswith("/") else local_csv

    cnt, tier = ingest_gdelt_tiered(input_csv, d.year, d.month, day=d.day)
    print(f"[run_tiered_ingest] Ingested {cnt:,} rows -> {tier} tier (day {args.date})")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
