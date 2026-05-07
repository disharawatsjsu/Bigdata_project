#!/usr/bin/env python3
"""
GDELT ingestion: raw CSV → Parquet (full or tiered hot/warm).

# SparkSession is reused across modules; getOrCreate() returns the same
# session within a single Python process.
"""
from __future__ import annotations

from datetime import date
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
)

from config import CHOKEPOINTS, HDFS_HOT, HDFS_WARM, WARM_COLUMNS, get_tier_for_date
from schemas import validate_hot, validate_warm

spark = (
    SparkSession.builder.appName("SupplyChainIntel_V1")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Supply-chain CAMEO event root codes (integers; EventRootCode is string in GDELT)
SUPPLY_CHAIN_CAMEO = [14, 17, 18, 19, 20]

# GDELT Events 2.0 is tab-delimited with no header row
GDELT_SCHEMA = StructType([
    StructField("GLOBALEVENTID", IntegerType()),
    StructField("SQLDATE", IntegerType()),       # YYYYMMDD
    StructField("MonthYear", IntegerType()),
    StructField("Year", IntegerType()),
    StructField("FractionDate", FloatType()),
    StructField("Actor1Code", StringType()),
    StructField("Actor1Name", StringType()),
    StructField("Actor1CountryCode", StringType()),
    StructField("Actor1KnownGroupCode", StringType()),
    StructField("Actor1EthnicCode", StringType()),
    StructField("Actor1Religion1Code", StringType()),
    StructField("Actor1Religion2Code", StringType()),
    StructField("Actor1Type1Code", StringType()),
    StructField("Actor1Type2Code", StringType()),
    StructField("Actor1Type3Code", StringType()),
    StructField("Actor2Code", StringType()),
    StructField("Actor2Name", StringType()),
    StructField("Actor2CountryCode", StringType()),
    StructField("Actor2KnownGroupCode", StringType()),
    StructField("Actor2EthnicCode", StringType()),
    StructField("Actor2Religion1Code", StringType()),
    StructField("Actor2Religion2Code", StringType()),
    StructField("Actor2Type1Code", StringType()),
    StructField("Actor2Type2Code", StringType()),
    StructField("Actor2Type3Code", StringType()),
    StructField("IsRootEvent", IntegerType()),
    StructField("EventCode", StringType()),
    StructField("EventBaseCode", StringType()),
    StructField("EventRootCode", StringType()),     # key — CAMEO root
    StructField("QuadClass", IntegerType()),
    StructField("GoldsteinScale", FloatType()),      # key — conflict intensity
    StructField("NumMentions", IntegerType()),        # key — signal strength
    StructField("NumSources", IntegerType()),
    StructField("NumArticles", IntegerType()),
    StructField("AvgTone", FloatType()),              # key — sentiment
    StructField("Actor1Geo_Type", IntegerType()),
    StructField("Actor1Geo_FullName", StringType()),
    StructField("Actor1Geo_CountryCode", StringType()),
    StructField("Actor1Geo_ADM1Code", StringType()),
    StructField("Actor1Geo_Lat", FloatType()),
    StructField("Actor1Geo_Long", FloatType()),
    StructField("Actor1Geo_FeatureID", StringType()),
    StructField("Actor2Geo_Type", IntegerType()),
    StructField("Actor2Geo_FullName", StringType()),
    StructField("Actor2Geo_CountryCode", StringType()),
    StructField("Actor2Geo_ADM1Code", StringType()),
    StructField("Actor2Geo_Lat", FloatType()),
    StructField("Actor2Geo_Long", FloatType()),
    StructField("Actor2Geo_FeatureID", StringType()),
    StructField("ActionGeo_Type", IntegerType()),
    StructField("ActionGeo_FullName", StringType()),
    StructField("ActionGeo_CountryCode", StringType()),
    StructField("ActionGeo_ADM1Code", StringType()),
    StructField("ActionGeo_Lat", FloatType()),        # key — event location
    StructField("ActionGeo_Long", FloatType()),       # key — event location
    StructField("ActionGeo_FeatureID", StringType()),
    StructField("DATEADDED", StringType()),
    StructField("SOURCEURL", StringType()),
])


def _in_any_chokepoint_bbox(lat_col, lon_col):
    """True if (lat, lon) falls in ANY chokepoint bbox. Uses CHOKEPOINTS tuple layout."""
    conditions = []
    for _name, (lat_min, lat_max, lon_min, lon_max) in CHOKEPOINTS.items():
        conditions.append(
            (lat_col.between(lat_min, lat_max)) &
            (lon_col.between(lon_min, lon_max))
        )
    return reduce(lambda a, b: a | b, conditions)


def _tag_with_chokepoint(df):
    """Explode to one row per (event, chokepoint); expects ActionGeo_Lat/Long."""
    for name, (lat_min, lat_max, lon_min, lon_max) in CHOKEPOINTS.items():
        df = df.withColumn(
            f"near_{name}",
            (F.col("ActionGeo_Lat").between(lat_min, lat_max)) &
            (F.col("ActionGeo_Long").between(lon_min, lon_max))
        )

    chokepoint_cols = [f"near_{n}" for n in CHOKEPOINTS]
    df_tagged = df.select(
        "*",
        F.explode(
            F.array([
                F.when(F.col(f"near_{name}"), F.lit(name))
                for name in CHOKEPOINTS
            ])
        ).alias("chokepoint")
    ).filter(F.col("chokepoint").isNotNull())

    return df_tagged.drop(*chokepoint_cols)


def ingest_gdelt(raw_path: str, parquet_path: str):
    """Read tab-delimited GDELT CSVs, write as partitioned Parquet."""
    print(f"\n{'='*60}")
    print("STEP 1: Ingesting GDELT raw CSVs → Parquet")
    print(f"{'='*60}")

    raw_df = (
        spark.read
        .option("delimiter", "\t")
        .option("header", "false")
        .schema(GDELT_SCHEMA)
        .csv(f"{raw_path}/*.CSV")
    )

    raw_df = raw_df.withColumn(
        "event_date",
        F.to_date(F.col("SQLDATE").cast("string"), "yyyyMMdd")
    ).withColumn(
        "year", F.year("event_date")
    ).withColumn(
        "month", F.month("event_date")
    )

    row_count = raw_df.count()
    print(f"  Raw rows loaded: {row_count:,}")

    (
        raw_df.write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(parquet_path)
    )
    print(f"  → Written to {parquet_path}")
    return raw_df


def ingest_gdelt_tiered(input_path: str, year: int, month: int, as_of=None):
    """Ingest one month of GDELT CSVs, routing to hot or warm tier by rolling window.

    Hot: representative month start falls in hot window relative to as_of (full schema).
    Warm / cold: same write path and projection as warm (cold logged until PR L).

    Returns (row_count, tier_name) for logging.
    """
    print(f"[ingest_tiered] reading {input_path} for {year}-{month:02d}")

    rep_date = date(year, month, 1)
    tier = get_tier_for_date(rep_date, as_of=as_of)

    base = input_path.rstrip("/")
    csv_path = base if base.lower().endswith(".csv") else f"{base}/*.CSV"
    df = (
        spark.read
        .option("delimiter", "\t")
        .option("header", "false")
        .option("mode", "PERMISSIVE")
        .schema(GDELT_SCHEMA)
        .csv(csv_path)
    )

    df = df.withColumn(
        "event_date",
        F.to_date(F.col("SQLDATE").cast("string"), "yyyyMMdd")
    )

    if tier == "hot":
        out_path = f"{HDFS_HOT}/year={year}/month={month:02d}"
        cnt = df.count()
        validate_hot(df)
        df.write.mode("overwrite").parquet(out_path)
        print(f"[ingest_tiered] HOT: wrote {cnt:,} rows to {out_path}")
        return cnt, "hot"

    if tier == "cold":
        print(
            f"[ingest_gdelt_tiered] WARNING: tier=cold for {year}-{month:02d}; "
            f"using warm-tier path until PR L adds cold aggregates"
        )
        # TODO(PR-L): validate_cold(...) once cold-tier Parquet writes land here.

    cameo_filter = F.col("EventRootCode").cast("int").isin(SUPPLY_CHAIN_CAMEO)
    warm_df = (
        df.filter(cameo_filter)
        .filter(_in_any_chokepoint_bbox(F.col("ActionGeo_Lat"), F.col("ActionGeo_Long")))
        .select(*WARM_COLUMNS)
    )
    out_path = f"{HDFS_WARM}/year={year}/month={month:02d}"
    cnt = warm_df.count()
    validate_warm(warm_df)
    warm_df.write.mode("overwrite").parquet(out_path)
    print(
        f"[ingest_tiered] WARM: wrote {cnt:,} rows to {out_path} "
        f"(filtered from full month)"
    )
    return cnt, "warm"
