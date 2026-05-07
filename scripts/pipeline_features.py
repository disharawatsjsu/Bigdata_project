#!/usr/bin/env python3
"""
Feature engineering: clean / filter GDELT, build training features.

# SparkSession is reused across modules; getOrCreate() returns the same
# session within a single Python process.
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import HDFS_HOT, HDFS_WARM, WARM_COLUMNS
from pipeline_ingest import SUPPLY_CHAIN_CAMEO, _in_any_chokepoint_bbox, _tag_with_chokepoint
from schemas import validate_features

spark = (
    SparkSession.builder.appName("SupplyChainIntel_V1")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Supply-chain-relevant CAMEO root codes
SC_CAMEO_ROOTS = ["14", "17", "18", "19", "20"]  # protest, coerce, assault, fight, mass violence


def clean_and_filter(parquet_path: str):
    """Dedup, filter to supply-chain events near chokepoints."""
    print(f"\n{'='*60}")
    print("STEP 2: Cleaning & Filtering")
    print(f"{'='*60}")

    df = spark.read.parquet(parquet_path)
    print(f"  Parquet rows: {df.count():,}")

    before = df.count()
    df = df.dropDuplicates(["SOURCEURL", "SQLDATE", "Actor1Code", "Actor2Code"])
    after = df.count()
    print(f"  After dedup: {after:,} (removed {before - after:,}, {(before-after)/before*100:.0f}%)")

    df_sc = df.filter(F.col("EventRootCode").isin(SC_CAMEO_ROOTS))
    print(f"  Supply-chain events (CAMEO 14/17/18/19/20): {df_sc.count():,}")

    df_geo = df_sc.filter(_in_any_chokepoint_bbox(F.col("ActionGeo_Lat"), F.col("ActionGeo_Long")))
    print(f"  After geo-filter (near chokepoints): {df_geo.count():,}")

    df_tagged = _tag_with_chokepoint(df_geo)
    print(f"  Tagged event-chokepoint rows: {df_tagged.count():,}")

    print("  Per-chokepoint counts:")
    df_tagged.groupBy("chokepoint").count().orderBy(F.desc("count")).show(truncate=False)

    return df_tagged


def clean_and_filter_tiered():
    """Read hot + warm tiers, normalize hot to WARM_COLUMNS, union, dedup, tag chokepoints."""
    print(f"[clean_tiered] reading hot tier from {HDFS_HOT}")
    try:
        hot_df = spark.read.parquet(HDFS_HOT)
        hot_count = hot_df.count()
        print(f"[clean_tiered] hot tier: {hot_count:,} raw rows")
    except Exception as e:
        print(f"[clean_tiered] hot tier empty or unreadable: {e}")
        hot_df = None
        hot_count = 0

    print(f"[clean_tiered] reading warm tier from {HDFS_WARM}")
    try:
        warm_df = spark.read.parquet(HDFS_WARM).select(*WARM_COLUMNS)
        warm_count = warm_df.count()
        print(f"[clean_tiered] warm tier: {warm_count:,} pre-filtered rows")
    except Exception as e:
        print(f"[clean_tiered] warm tier empty or unreadable: {e}")
        warm_df = None
        warm_count = 0

    if hot_df is None and warm_df is None:
        raise RuntimeError("Both tiers empty. Run ingest_gdelt_tiered first.")

    cameo_filter = F.col("EventRootCode").cast("int").isin(SUPPLY_CHAIN_CAMEO)
    if hot_df is not None:
        hot_filtered = (
            hot_df.filter(cameo_filter)
            .filter(_in_any_chokepoint_bbox(F.col("ActionGeo_Lat"), F.col("ActionGeo_Long")))
            .select(*WARM_COLUMNS)
        )
    else:
        hot_filtered = None

    if hot_filtered is not None and warm_df is not None:
        unified = hot_filtered.unionByName(warm_df)
    elif hot_filtered is not None:
        unified = hot_filtered
    else:
        unified = warm_df

    unified_rows = unified.count()
    print(f"[clean_tiered] unified: {unified_rows:,} rows entering dedup")

    deduped = unified.dropDuplicates(
        ["SOURCEURL", "event_date", "Actor1Code", "Actor2Code"]
    )
    print(f"[clean_tiered] after dedup: {deduped.count():,} rows")

    tagged = _tag_with_chokepoint(deduped)
    print(f"[clean_tiered] tagged rows: {tagged.count():,}")
    return tagged


def build_features(events_df, commodity_path: str, fred_path: str):
    """Build training features: GDELT signals + price lags + macro indicators."""
    print(f"\n{'='*60}")
    print("STEP 3: Feature Engineering")
    print(f"{'='*60}")

    daily = (
        events_df
        .groupBy("event_date", "chokepoint")
        .agg(
            F.count("*").alias("event_count"),
            F.avg("GoldsteinScale").alias("avg_goldstein"),
            F.avg("AvgTone").alias("avg_tone"),
            F.sum("NumMentions").alias("total_mentions"),
            F.avg(
                F.when(F.col("EventRootCode").isin(["18", "19", "20"]), 1.0)
                .otherwise(0.0)
            ).alias("conflict_ratio"),
        )
    )

    w7 = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-6, 0)
    w30 = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-29, 0)

    for col_name in ["event_count", "avg_goldstein", "avg_tone", "total_mentions", "conflict_ratio"]:
        daily = daily.withColumn(f"{col_name}_7d", F.avg(col_name).over(w7))
        daily = daily.withColumn(f"{col_name}_30d", F.avg(col_name).over(w30))

    print(f"  Daily feature rows (date × chokepoint): {daily.count():,}")

    prices = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(commodity_path)
    )
    prices = prices.withColumn("date", F.to_date("date"))

    oil = prices.filter(F.col("commodity") == "crude_oil").select(
        F.col("date").alias("price_date"),
        F.col("close1").cast("float").alias("oil_close"),
    )

    oil = oil.withColumn("commodity", F.lit("crude_oil"))
    w_price = Window.partitionBy("commodity").orderBy("price_date")
    oil = (
        oil
        .withColumn("close_lag5", F.lag("oil_close", 5).over(w_price))
        .withColumn("close_lag20", F.lag("oil_close", 20).over(w_price))
        .withColumn("return_5d", (F.col("oil_close") - F.col("close_lag5")) / F.col("close_lag5"))
        .withColumn("return_20d", (F.col("oil_close") - F.col("close_lag20")) / F.col("close_lag20"))
    )

    oil = oil.withColumn(
        "daily_return",
        (F.col("oil_close") - F.lag("oil_close", 1).over(w_price)) / F.lag("oil_close", 1).over(w_price)
    )
    w_vol = Window.partitionBy("commodity").orderBy("price_date").rowsBetween(-19, 0)
    oil = oil.withColumn("volatility_20d", F.stddev("daily_return").over(w_vol))

    oil = oil.withColumn(
        "close_fwd5", F.lead("oil_close", 5).over(w_price)
    ).withColumn(
        "fwd_return_5d", (F.col("close_fwd5") - F.col("oil_close")) / F.col("oil_close")
    )

    oil_labelled = oil.filter(F.col("fwd_return_5d").isNotNull())
    terciles = oil_labelled.approxQuantile("fwd_return_5d", [0.33, 0.66], 0.01)
    t_low, t_high = terciles[0], terciles[1]
    print(f"  Label tercile thresholds: negative < {t_low:.4f} | normal | positive > {t_high:.4f}")

    oil = oil.withColumn(
        "label",
        F.when(F.col("fwd_return_5d").isNull(), None)
        .when(F.col("fwd_return_5d") > t_high, 2)
        .when(F.col("fwd_return_5d") < t_low, 0)
        .otherwise(1)
    )

    fred = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(fred_path)
    )
    fred = fred.withColumn("date", F.to_date("date"))

    date_bounds = oil.select(
        F.min("price_date").alias("mn"), F.max("price_date").alias("mx")
    ).first()
    date_spine = spark.sql(
        f"SELECT explode(sequence(to_date('{date_bounds['mn']}'), "
        f"to_date('{date_bounds['mx']}'), interval 1 day)) AS fred_date"
    )

    fred_daily = date_spine.join(
        fred.select(F.col("date").alias("fred_date"), "treasury_10y", "usd_index"),
        on="fred_date",
        how="left",
    )
    w_ffill = Window.orderBy("fred_date").rowsBetween(Window.unboundedPreceding, 0)
    fred_daily = fred_daily.withColumn(
        "treasury_10y", F.last("treasury_10y", ignorenulls=True).over(w_ffill)
    ).withColumn(
        "usd_index", F.last("usd_index", ignorenulls=True).over(w_ffill)
    )
    print(f"  FRED daily rows (after forward-fill): {fred_daily.count():,}")

    oil_chokepoints = ["hormuz", "suez", "red_sea"]
    features_gdelt = daily.filter(F.col("chokepoint").isin(oil_chokepoints))

    features = features_gdelt.join(
        oil.select("price_date", "return_5d", "return_20d", "volatility_20d", "label"),
        features_gdelt["event_date"] == oil["price_date"],
        "inner"
    ).drop("price_date")

    features = features.join(
        fred_daily,
        features["event_date"] == fred_daily["fred_date"],
        "left"
    ).drop("fred_date")

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
    print(f"  Label distribution:")
    features.groupBy("label").count().orderBy("label").show()

    validate_features(features)
    return features
