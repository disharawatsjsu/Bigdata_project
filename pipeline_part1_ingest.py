#!/usr/bin/env python3
"""
PART 2 — Ingest, Clean + Feature Engineering
=============================================
What is NEW in this file (on top of Part 1):
  - Daily aggregation of GDELT events per chokepoint
    (event count, avg Goldstein score, avg tone, conflict ratio)
  - 7-day and 30-day rolling window averages of all signals
  - Load commodity prices (crude oil) from CSV
  - Calculate price features: 5-day return, 20-day return, volatility
  - Load FRED macro indicators (treasury yield, USD index)
  - Forward-fill sparse FRED data to daily frequency
  - Create prediction LABEL using tercile-based classification
    (0 = negative shock, 1 = normal, 2 = positive shock)
  - Join everything: GDELT features + price features + macro features

Output: final feature table saved as Parquet — ready for ML training
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType
)
from pyspark.sql.window import Window
import os

# =============================================================================
# SPARK SESSION
# =============================================================================
spark = (
    SparkSession.builder
    .appName("SupplyChain_Part2_Features")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# --- Paths ---
LOCAL_MODE = not os.environ.get("HADOOP_CONF_DIR")
BASE = "/opt/data" if LOCAL_MODE else "hdfs://namenode:9000/supply-chain"

GDELT_RAW        = "/opt/data/gdelt"
HDFS_GDELT_PARQUET = f"{BASE}/{'parquet/' if LOCAL_MODE else ''}gdelt_events"
GDELT_CLEAN      = f"{BASE}/{'parquet/' if LOCAL_MODE else ''}gdelt_clean"
HDFS_FEATURES    = f"{BASE}/{'parquet/' if LOCAL_MODE else ''}features"
COMMODITY_PATH   = f"{BASE}/commodities/commodity_prices.csv"
FRED_PATH        = f"{BASE}/fred/fred_macro.csv"

if LOCAL_MODE:
    print("⚠ Running in LOCAL mode (no HDFS)")


# =============================================================================
# GDELT SCHEMA (same as Part 1 — needed to re-ingest if clean data not saved)
# =============================================================================
from pyspark.sql.types import DoubleType

GDELT_SCHEMA = StructType([
    StructField("GLOBALEVENTID", IntegerType()),
    StructField("SQLDATE", IntegerType()),
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
    StructField("EventRootCode", StringType()),
    StructField("QuadClass", IntegerType()),
    StructField("GoldsteinScale", FloatType()),
    StructField("NumMentions", IntegerType()),
    StructField("NumSources", IntegerType()),
    StructField("NumArticles", IntegerType()),
    StructField("AvgTone", FloatType()),
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
    StructField("ActionGeo_Lat", FloatType()),
    StructField("ActionGeo_Long", FloatType()),
    StructField("ActionGeo_FeatureID", StringType()),
    StructField("DATEADDED", StringType()),
    StructField("SOURCEURL", StringType()),
])

SC_CAMEO_ROOTS = ["14", "17", "18", "19", "20"]

CHOKEPOINTS = {
    "hormuz":    (25.06, 28.06, 54.75, 57.75),
    "suez":      (29.46, 31.46, 31.34, 33.34),
    "red_sea":   (11.00, 17.00, 40.00, 46.00),
    "black_sea": (41.60, 47.60, 30.50, 36.50),
    "malacca":   (0.50,  4.50,  99.50, 103.50),
    "panama":    (8.58,  9.58, -80.18, -79.18),
    "taiwan":    (23.00, 26.00, 118.00, 121.00),
    "chile":     (-26.50, -20.50, -72.50, -66.50),
}


# =============================================================================
# STEP 1 — INGEST (same as Part 1)
# =============================================================================
def ingest_gdelt(raw_path, parquet_path):
    print(f"\n{'='*60}\nSTEP 1: Ingesting GDELT raw CSVs → Parquet\n{'='*60}")
    raw_df = (
        spark.read
        .option("delimiter", "\t")
        .option("header", "false")
        .schema(GDELT_SCHEMA)
        .csv(f"{raw_path}/*.CSV")
    )
    raw_df = (
        raw_df
        .withColumn("event_date", F.to_date(F.col("SQLDATE").cast("string"), "yyyyMMdd"))
        .withColumn("year", F.year("event_date"))
        .withColumn("month", F.month("event_date"))
    )
    print(f"  Raw rows loaded: {raw_df.count():,}")
    raw_df.write.mode("overwrite").partitionBy("year", "month").parquet(parquet_path)
    print(f"  → Written to {parquet_path}")
    return raw_df


# =============================================================================
# STEP 2 — CLEAN & FILTER (same as Part 1)
# =============================================================================
def clean_and_filter(parquet_path):
    print(f"\n{'='*60}\nSTEP 2: Cleaning & Filtering\n{'='*60}")
    df = spark.read.parquet(parquet_path)
    df = df.dropDuplicates(["SOURCEURL", "SQLDATE", "Actor1Code", "Actor2Code"])
    df_sc = df.filter(F.col("EventRootCode").isin(SC_CAMEO_ROOTS))

    geo_filter = None
    for name, (lat_min, lat_max, lon_min, lon_max) in CHOKEPOINTS.items():
        cond = (
            F.col("ActionGeo_Lat").between(lat_min, lat_max) &
            F.col("ActionGeo_Long").between(lon_min, lon_max)
        )
        geo_filter = cond if geo_filter is None else (geo_filter | cond)

    df_geo = df_sc.filter(geo_filter)

    for name, (lat_min, lat_max, lon_min, lon_max) in CHOKEPOINTS.items():
        df_geo = df_geo.withColumn(
            f"near_{name}",
            F.col("ActionGeo_Lat").between(lat_min, lat_max) &
            F.col("ActionGeo_Long").between(lon_min, lon_max)
        )

    df_tagged = df_geo.select(
        "*",
        F.explode(F.array([
            F.when(F.col(f"near_{name}"), F.lit(name)) for name in CHOKEPOINTS
        ])).alias("chokepoint")
    ).filter(F.col("chokepoint").isNotNull())

    df_tagged = df_tagged.drop(*[f"near_{n}" for n in CHOKEPOINTS])
    print(f"  Tagged event-chokepoint rows: {df_tagged.count():,}")
    return df_tagged


# =============================================================================
# STEP 3 — FEATURE ENGINEERING (NEW in Part 2)
# =============================================================================
def build_features(events_df, commodity_path, fred_path):
    print(f"\n{'='*60}\nSTEP 3: Feature Engineering\n{'='*60}")

    # --- 3a. Daily aggregation per chokepoint ---
    # For each day × chokepoint: count events, avg Goldstein, avg tone, conflict ratio
    daily = (
        events_df
        .groupBy("event_date", "chokepoint")
        .agg(
            F.count("*").alias("event_count"),
            F.avg("GoldsteinScale").alias("avg_goldstein"),
            F.avg("AvgTone").alias("avg_tone"),
            F.sum("NumMentions").alias("total_mentions"),
            # Conflict ratio = fraction of events that are actual violence (18/19/20)
            F.avg(
                F.when(F.col("EventRootCode").isin(["18", "19", "20"]), 1.0)
                .otherwise(0.0)
            ).alias("conflict_ratio"),
        )
    )
    print(f"  Daily rows (date × chokepoint): {daily.count():,}")

    # --- 3b. Rolling windows: 7-day and 30-day averages ---
    # This captures short-term spikes vs long-term trends
    w7  = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-6, 0)
    w30 = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-29, 0)

    for col_name in ["event_count", "avg_goldstein", "avg_tone", "total_mentions", "conflict_ratio"]:
        daily = daily.withColumn(f"{col_name}_7d",  F.avg(col_name).over(w7))
        daily = daily.withColumn(f"{col_name}_30d", F.avg(col_name).over(w30))

    # --- 3c. Load crude oil prices ---
    prices = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(commodity_path)
    )
    prices = prices.withColumn("date", F.to_date("date"))

    # Focus on crude oil only (V1 scope)
    oil = (
        prices
        .filter(F.col("commodity") == "crude_oil")
        .select(F.col("date").alias("price_date"), F.col("close1").cast("float").alias("oil_close"))
    )

    # --- 3d. Price features: returns and volatility ---
    oil = oil.withColumn("commodity", F.lit("crude_oil"))
    w_price = Window.partitionBy("commodity").orderBy("price_date")

    oil = (
        oil
        .withColumn("close_lag5",  F.lag("oil_close", 5).over(w_price))
        .withColumn("close_lag20", F.lag("oil_close", 20).over(w_price))
        .withColumn("return_5d",   (F.col("oil_close") - F.col("close_lag5"))  / F.col("close_lag5"))
        .withColumn("return_20d",  (F.col("oil_close") - F.col("close_lag20")) / F.col("close_lag20"))
        .withColumn("daily_return",
            (F.col("oil_close") - F.lag("oil_close", 1).over(w_price)) /
             F.lag("oil_close", 1).over(w_price)
        )
    )

    w_vol = Window.partitionBy("commodity").orderBy("price_date").rowsBetween(-19, 0)
    oil = oil.withColumn("volatility_20d", F.stddev("daily_return").over(w_vol))

    # --- 3e. Forward 5-day return = what we are predicting ---
    oil = oil.withColumn(
        "close_fwd5", F.lead("oil_close", 5).over(w_price)
    ).withColumn(
        "fwd_return_5d", (F.col("close_fwd5") - F.col("oil_close")) / F.col("oil_close")
    )

    # --- 3f. Create labels using terciles (guarantees 3 balanced classes) ---
    # Label 0 = negative shock (bottom 33%), 1 = normal, 2 = positive shock (top 33%)
    oil_labelled = oil.filter(F.col("fwd_return_5d").isNotNull())
    terciles = oil_labelled.approxQuantile("fwd_return_5d", [0.33, 0.66], 0.01)
    t_low, t_high = terciles[0], terciles[1]
    print(f"  Label thresholds: negative < {t_low:.4f} | positive > {t_high:.4f}")

    oil = oil.withColumn(
        "label",
        F.when(F.col("fwd_return_5d").isNull(), None)
        .when(F.col("fwd_return_5d") > t_high, 2)   # positive shock
        .when(F.col("fwd_return_5d") < t_low,  0)   # negative shock
        .otherwise(1)                                 # normal
    )

    # --- 3g. Load FRED macro data and forward-fill to daily ---
    # FRED reports weekly/monthly — we spread each value forward to cover every day
    fred = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(fred_path)
    )
    fred = fred.withColumn("date", F.to_date("date"))

    # Build daily date spine from oil price range
    date_bounds = oil.select(F.min("price_date").alias("mn"), F.max("price_date").alias("mx")).first()
    date_spine = spark.sql(
        f"SELECT explode(sequence(to_date('{date_bounds['mn']}'), "
        f"to_date('{date_bounds['mx']}'), interval 1 day)) AS fred_date"
    )

    # Left-join FRED onto spine, then forward-fill nulls
    fred_daily = date_spine.join(
        fred.select(F.col("date").alias("fred_date"), "treasury_10y", "usd_index"),
        on="fred_date", how="left"
    )
    w_ffill = Window.orderBy("fred_date").rowsBetween(Window.unboundedPreceding, 0)
    fred_daily = (
        fred_daily
        .withColumn("treasury_10y", F.last("treasury_10y", ignorenulls=True).over(w_ffill))
        .withColumn("usd_index",    F.last("usd_index",    ignorenulls=True).over(w_ffill))
    )
    print(f"  FRED daily rows (after forward-fill): {fred_daily.count():,}")

    # --- 3h. Join everything together ---
    # V1 scope: only oil-relevant chokepoints
    oil_chokepoints = ["hormuz", "suez", "red_sea"]
    features_gdelt = daily.filter(F.col("chokepoint").isin(oil_chokepoints))

    # GDELT features + oil price label
    features = features_gdelt.join(
        oil.select("price_date", "return_5d", "return_20d", "volatility_20d", "label"),
        features_gdelt["event_date"] == oil["price_date"],
        "inner"
    ).drop("price_date")

    # + FRED macro
    features = features.join(
        fred_daily,
        features["event_date"] == fred_daily["fred_date"],
        "left"
    ).drop("fred_date")

    # Drop rows with any null in model input columns
    model_cols = [
        "event_count_7d", "avg_goldstein_7d", "avg_tone_7d",
        "total_mentions_7d", "conflict_ratio_7d",
        "event_count_30d", "avg_goldstein_30d", "avg_tone_30d",
        "total_mentions_30d", "conflict_ratio_30d",
        "return_5d", "return_20d", "volatility_20d",
        "treasury_10y", "usd_index", "label",
    ]
    features = features.na.drop(subset=model_cols)
    print(f"  Final feature table rows: {features.count():,}")
    print("  Label distribution:")
    features.groupBy("label").count().orderBy("label").show()

    return features


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Step 1: Ingest
    ingest_gdelt(GDELT_RAW, HDFS_GDELT_PARQUET)

    # Step 2: Clean & filter
    clean_df = clean_and_filter(HDFS_GDELT_PARQUET)

    # Step 3: Feature engineering (NEW)
    features_df = build_features(clean_df, COMMODITY_PATH, FRED_PATH)

    # Save features for Part 3 (ML training)
    features_df.cache()
    features_df.write.mode("overwrite").parquet(HDFS_FEATURES)
    print(f"\nFeatures saved to {HDFS_FEATURES}")
    print("\nPART 2 COMPLETE — Ready for ML training (Part 3)")

    spark.stop()
