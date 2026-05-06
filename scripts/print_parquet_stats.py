#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    args = ap.parse_args()

    spark = SparkSession.builder.appName('print_parquet_stats').getOrCreate()
    df = spark.read.parquet(args.path)
    print(f"PATH: {args.path}")
    print(f"COLS: {len(df.columns)}")
    print(f"ROWS: {df.count()}")
    spark.stop()

if __name__ == '__main__':
    main()
