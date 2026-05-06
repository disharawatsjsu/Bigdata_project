#!/usr/bin/env python3
"""One-shot spark-submit helper: run clean_and_filter_tiered() and show sample rows."""
from spark_pipeline import clean_and_filter_tiered, spark

if __name__ == "__main__":
    clean_and_filter_tiered().show(5)
    spark.stop()
