#!/usr/bin/env python3
"""One-shot spark-submit helper: run clean_and_filter_tiered() and show sample rows."""
# Spark: pipeline_features shares the session via getOrCreate() with other pipeline modules.
from pipeline_features import clean_and_filter_tiered, spark

if __name__ == "__main__":
    clean_and_filter_tiered().show(5)
    spark.stop()
