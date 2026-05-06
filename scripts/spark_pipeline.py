#!/usr/bin/env python3
"""
Supply Chain Disruption Intelligence — Spark Pipeline (V1)

Runs end-to-end: GDELT ingestion → cleaning → feature engineering → ML training.
Designed to run inside the spark-master container with:
    spark-submit --master spark://spark-master:7077 spark_pipeline.py

Or cell-by-cell in a Jupyter notebook for learning/debugging.

V1 scope: single-commodity (crude_oil), single-region (Strait of Hormuz + Red Sea)
to prove the pipeline works before scaling.
"""

# =============================================================================
# 0. SPARK SESSION
# =============================================================================
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, DoubleType
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from functools import reduce
import os
from datetime import date

spark = (
    SparkSession.builder
    .appName("SupplyChainIntel_V1")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")  # small for local dev
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

from config import (
    CHOKEPOINTS,
    COMMODITY_RAW,
    FRED_RAW,
    GDELT_RAW,
    HDFS_FEATURES,
    HDFS_GDELT_PARQUET,
    HDFS_HOT,
    HDFS_MODEL,
    HDFS_WARM,
    LOCAL_MODE,
    WARM_COLUMNS,
    get_tier_for_date,
)

if LOCAL_MODE:
    print("⚠ Running in LOCAL mode (no HDFS)")

# Supply-chain CAMEO event root codes (integers; EventRootCode is string in GDELT)
SUPPLY_CHAIN_CAMEO = [14, 17, 18, 19, 20]

# =============================================================================
# 1. GDELT SCHEMA — all 61 columns, but we only use ~15
# =============================================================================
# GDELT Events 2.0 is tab-delimited with no header row
GDELT_SCHEMA = StructType([
    StructField("GLOBALEVENTID", IntegerType()),
    StructField("SQLDATE", IntegerType()),       # YYYYMMDD
    StructField("MonthYear", IntegerType()),
    StructField("Year", IntegerType()),
    StructField("FractionDate", FloatType()),
    # Actor1 fields (11)
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
    # Actor2 fields (10)
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
    # Event classification
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
    # Actor1 geo
    StructField("Actor1Geo_Type", IntegerType()),
    StructField("Actor1Geo_FullName", StringType()),
    StructField("Actor1Geo_CountryCode", StringType()),
    StructField("Actor1Geo_ADM1Code", StringType()),
    StructField("Actor1Geo_Lat", FloatType()),
    StructField("Actor1Geo_Long", FloatType()),
    StructField("Actor1Geo_FeatureID", StringType()),
    # Actor2 geo
    StructField("Actor2Geo_Type", IntegerType()),
    StructField("Actor2Geo_FullName", StringType()),
    StructField("Actor2Geo_CountryCode", StringType()),
    StructField("Actor2Geo_ADM1Code", StringType()),
    StructField("Actor2Geo_Lat", FloatType()),
    StructField("Actor2Geo_Long", FloatType()),
    StructField("Actor2Geo_FeatureID", StringType()),
    # Action geo — WHERE the event happened
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


# =============================================================================
# 2. INGEST: Read raw GDELT CSVs → Parquet on HDFS
# =============================================================================
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

    # Parse date for partitioning
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

    # Write partitioned by year/month for efficient time-range queries
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
        df.write.mode("overwrite").parquet(out_path)
        print(f"[ingest_tiered] HOT: wrote {cnt:,} rows to {out_path}")
        return cnt, "hot"

    if tier == "cold":
        print(
            f"[ingest_gdelt_tiered] WARNING: tier=cold for {year}-{month:02d}; "
            f"using warm-tier path until PR L adds cold aggregates"
        )

    cameo_filter = F.col("EventRootCode").cast("int").isin(SUPPLY_CHAIN_CAMEO)
    warm_df = (
        df.filter(cameo_filter)
        .filter(_in_any_chokepoint_bbox(F.col("ActionGeo_Lat"), F.col("ActionGeo_Long")))
        .select(*WARM_COLUMNS)
    )
    out_path = f"{HDFS_WARM}/year={year}/month={month:02d}"
    cnt = warm_df.count()
    warm_df.write.mode("overwrite").parquet(out_path)
    print(
        f"[ingest_tiered] WARM: wrote {cnt:,} rows to {out_path} "
        f"(filtered from full month)"
    )
    return cnt, "warm"


# =============================================================================
# 3. CLEAN & FILTER: Dedup, supply-chain events, geo-filter to chokepoints
# =============================================================================

# Supply-chain-relevant CAMEO root codes
SC_CAMEO_ROOTS = ["14", "17", "18", "19", "20"]  # protest, coerce, assault, fight, mass violence

# Chokepoint bounding boxes: (lat_min, lat_max, lon_min, lon_max)
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


def clean_and_filter(parquet_path: str):
    """Dedup, filter to supply-chain events near chokepoints."""
    print(f"\n{'='*60}")
    print("STEP 2: Cleaning & Filtering")
    print(f"{'='*60}")

    df = spark.read.parquet(parquet_path)
    print(f"  Parquet rows: {df.count():,}")

    # --- Dedup: same URL + date + actor pair = same event ---
    before = df.count()
    df = df.dropDuplicates(["SOURCEURL", "SQLDATE", "Actor1Code", "Actor2Code"])
    after = df.count()
    print(f"  After dedup: {after:,} (removed {before - after:,}, {(before-after)/before*100:.0f}%)")

    # --- Filter to supply-chain CAMEO codes ---
    df_sc = df.filter(F.col("EventRootCode").isin(SC_CAMEO_ROOTS))
    print(f"  Supply-chain events (CAMEO 14/17/18/19/20): {df_sc.count():,}")

    df_geo = df_sc.filter(_in_any_chokepoint_bbox(F.col("ActionGeo_Lat"), F.col("ActionGeo_Long")))
    print(f"  After geo-filter (near chokepoints): {df_geo.count():,}")

    df_tagged = _tag_with_chokepoint(df_geo)
    print(f"  Tagged event-chokepoint rows: {df_tagged.count():,}")

    # Diagnostic: per-chokepoint breakdown so we can see dead zones
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
        # Reading from year=YYYY/month=MM paths adds partition cols (year, month).
        # Normalize to the warm schema before union.
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


# =============================================================================
# 4. FEATURE ENGINEERING: Aggregate GDELT → daily features per chokepoint
# =============================================================================
def build_features(events_df, commodity_path: str, fred_path: str):
    """Build training features: GDELT signals + price lags + macro indicators."""
    print(f"\n{'='*60}")
    print("STEP 3: Feature Engineering")
    print(f"{'='*60}")

    # --- 4a. Daily aggregation per chokepoint ---
    daily = (
        events_df
        .groupBy("event_date", "chokepoint")
        .agg(
            F.count("*").alias("event_count"),
            F.avg("GoldsteinScale").alias("avg_goldstein"),
            F.avg("AvgTone").alias("avg_tone"),
            F.sum("NumMentions").alias("total_mentions"),
            # Conflict ratio: fraction of events that are CAMEO 18/19/20
            F.avg(
                F.when(F.col("EventRootCode").isin(["18", "19", "20"]), 1.0)
                .otherwise(0.0)
            ).alias("conflict_ratio"),
        )
    )

    # --- 4b. Rolling windows (7-day, 30-day) ---
    w7 = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-6, 0)
    w30 = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-29, 0)

    for col_name in ["event_count", "avg_goldstein", "avg_tone", "total_mentions", "conflict_ratio"]:
        daily = daily.withColumn(f"{col_name}_7d", F.avg(col_name).over(w7))
        daily = daily.withColumn(f"{col_name}_30d", F.avg(col_name).over(w30))

    print(f"  Daily feature rows (date × chokepoint): {daily.count():,}")

    # --- 4c. Load commodity prices ---
    prices = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(commodity_path)
    )
    prices = prices.withColumn("date", F.to_date("date"))

    # For V1: focus on crude oil
    oil = prices.filter(F.col("commodity") == "crude_oil").select(
        F.col("date").alias("price_date"),
        F.col("close1").cast("float").alias("oil_close"),
    )

    # Price features: lagged returns + volatility
    # Tag commodity so partitionBy works now and scales to multi-commodity later
    oil = oil.withColumn("commodity", F.lit("crude_oil"))
    w_price = Window.partitionBy("commodity").orderBy("price_date")
    oil = (
        oil
        .withColumn("close_lag5", F.lag("oil_close", 5).over(w_price))
        .withColumn("close_lag20", F.lag("oil_close", 20).over(w_price))
        .withColumn("return_5d", (F.col("oil_close") - F.col("close_lag5")) / F.col("close_lag5"))
        .withColumn("return_20d", (F.col("oil_close") - F.col("close_lag20")) / F.col("close_lag20"))
    )

    # 20-day rolling volatility (std of daily returns)
    oil = oil.withColumn(
        "daily_return",
        (F.col("oil_close") - F.lag("oil_close", 1).over(w_price)) / F.lag("oil_close", 1).over(w_price)
    )
    w_vol = Window.partitionBy("commodity").orderBy("price_date").rowsBetween(-19, 0)
    oil = oil.withColumn("volatility_20d", F.stddev("daily_return").over(w_vol))

    # Forward 5-day return — this is what we're predicting
    oil = oil.withColumn(
        "close_fwd5", F.lead("oil_close", 5).over(w_price)
    ).withColumn(
        "fwd_return_5d", (F.col("close_fwd5") - F.col("oil_close")) / F.col("oil_close")
    )

    # --- 4d. Classification label: tercile-based for V1 (guarantees class balance) ---
    # With small data, 2σ threshold produces single-class; terciles always give 3 classes.
    # Upgrade to 2σ once multi-year data is ingested.
    oil_labelled = oil.filter(F.col("fwd_return_5d").isNotNull())
    terciles = oil_labelled.approxQuantile("fwd_return_5d", [0.33, 0.66], 0.01)
    t_low, t_high = terciles[0], terciles[1]
    print(f"  Label tercile thresholds: negative < {t_low:.4f} | normal | positive > {t_high:.4f}")

    oil = oil.withColumn(
        "label",
        F.when(F.col("fwd_return_5d").isNull(), None)           # keep null for rows without fwd data
        .when(F.col("fwd_return_5d") > t_high, 2)               # positive shock
        .when(F.col("fwd_return_5d") < t_low, 0)                # negative shock
        .otherwise(1)                                             # normal
    )

    # --- 4e. Load FRED macro data & forward-fill to daily ---
    fred = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(fred_path)
    )
    fred = fred.withColumn("date", F.to_date("date"))

    # Build a daily date spine covering the full GDELT+price range, then
    # forward-fill FRED values so every calendar day has macro data.
    date_bounds = oil.select(
        F.min("price_date").alias("mn"), F.max("price_date").alias("mx")
    ).first()
    date_spine = spark.sql(
        f"SELECT explode(sequence(to_date('{date_bounds['mn']}'), "
        f"to_date('{date_bounds['mx']}'), interval 1 day)) AS fred_date"
    )

    # Left-join sparse FRED onto the spine, then fill forward (unbounded preceding)
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

    # --- 4f. Join everything ---
    # V1: focus on chokepoints relevant to oil (hormuz, suez, red_sea)
    oil_chokepoints = ["hormuz", "suez", "red_sea"]
    features_gdelt = daily.filter(F.col("chokepoint").isin(oil_chokepoints))

    # Join: events ↔ prices on date
    features = features_gdelt.join(
        oil.select("price_date", "return_5d", "return_20d", "volatility_20d", "label"),
        features_gdelt["event_date"] == oil["price_date"],
        "inner"
    ).drop("price_date")

    # Join: features ↔ FRED (pre-filled daily, so equi-join is clean)
    features = features.join(
        fred_daily,
        features["event_date"] == fred_daily["fred_date"],
        "left"
    ).drop("fred_date")

    # Only drop rows where actual model inputs are null — not every column
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

    return features


# =============================================================================
# 5. ML TRAINING: Random Forest in Spark MLlib
# =============================================================================
def train_model(features_df, model_path: str):
    """Train RF classifier with cross-validation, evaluate on time-split test set."""
    print(f"\n{'='*60}")
    print("STEP 4: ML Training (Random Forest)")
    print(f"{'='*60}")

    # --- Feature columns ---
    feature_cols = [
        # GDELT 7-day signals
        "event_count_7d", "avg_goldstein_7d", "avg_tone_7d",
        "total_mentions_7d", "conflict_ratio_7d",
        # GDELT 30-day signals
        "event_count_30d", "avg_goldstein_30d", "avg_tone_30d",
        "total_mentions_30d", "conflict_ratio_30d",
        # Price features
        "return_5d", "return_20d", "volatility_20d",
        # Macro
        "treasury_10y", "usd_index",
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=8,
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, rf])

    # --- Time-based split (no data leakage) ---
    # Dynamic: derive boundaries from actual data so it works at any scale
    date_range = features_df.select(
        F.min("event_date").alias("mn"), F.max("event_date").alias("mx")
    ).first()
    from datetime import timedelta
    total_days = (date_range["mx"] - date_range["mn"]).days
    train_end = date_range["mn"] + timedelta(days=int(total_days * 0.6))
    val_end = date_range["mn"] + timedelta(days=int(total_days * 0.8))

    train_df = features_df.filter(F.col("event_date") < F.lit(train_end))
    val_df = features_df.filter(
        (F.col("event_date") >= F.lit(train_end)) & (F.col("event_date") < F.lit(val_end))
    )
    test_df = features_df.filter(F.col("event_date") >= F.lit(val_end))

    print(f"  Date range: {date_range['mn']} → {date_range['mx']} ({total_days} days)")
    print(f"  Train: {train_df.count():,} rows (< {train_end})")
    print(f"  Val:   {val_df.count():,} rows ({train_end} – {val_end})")
    print(f"  Test:  {test_df.count():,} rows (≥ {val_end})")

    # --- Cross-validation on train set ---
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100])
        .addGrid(rf.maxDepth, [5, 8])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=42,
    )

    print("  Training with 3-fold cross-validation...")
    train_count = train_df.count()
    if train_count < 10:
        print(f"  ⚠ Only {train_count} train rows — skipping CV, fitting directly")
        best_model = pipeline.fit(train_df)
    else:
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel

    # --- Evaluate ---
    for name, df in [("Validation", val_df), ("Test", test_df)]:
        if df.count() == 0:
            print(f"\n  {name} — SKIPPED (0 rows)")
            continue
        preds = best_model.transform(df)
        f1 = evaluator.evaluate(preds)
        acc_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        acc = acc_eval.evaluate(preds)
        print(f"\n  {name} — F1: {f1:.4f}, Accuracy: {acc:.4f}")

        # Confusion matrix
        print(f"  {name} confusion matrix:")
        preds.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

    # --- Feature importance ---
    rf_model = best_model.stages[-1]  # RandomForestClassificationModel
    importances = rf_model.featureImportances.toArray()
    print("\n  Feature importances:")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {col:30s} {imp:.4f} {bar}")

    # --- Save model ---
    best_model.write().overwrite().save(model_path)
    print(f"\n  Model saved to {model_path}")

    return best_model


# =============================================================================
# 6. MAIN — Run the full pipeline
# =============================================================================
if __name__ == "__main__":
    # Step 1: Ingest
    raw_df = ingest_gdelt(GDELT_RAW, HDFS_GDELT_PARQUET)

    # Step 2: Clean & filter
    clean_df = clean_and_filter(HDFS_GDELT_PARQUET)

    # Step 3: Features
    features_df = build_features(
        clean_df,
        commodity_path=COMMODITY_RAW,
        fred_path=FRED_RAW,
    )

    # Save features for reuse
    features_df.cache()
    features_df.write.mode("overwrite").parquet(HDFS_FEATURES)
    print(f"\nFeatures saved to {HDFS_FEATURES}")

    # Step 4: Train
    features_df = features_df.filter(F.col("event_date") >= "2024-01-01")
    model = train_model(features_df, HDFS_MODEL)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — V1 vertical slice done.")
    print("=" * 60)

    spark.stop()
