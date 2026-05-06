#!/usr/bin/env python3
"""
PART 3 — Full Pipeline: Ingest + Clean + Feature Engineering + ML Training
===========================================================================
What is NEW in this file (on top of Part 2):
  - Loads saved feature table from HDFS (output of Part 2)
  - Time-based train/val/test split (60% / 20% / 20%) — no data leakage
  - Builds Spark ML Pipeline: VectorAssembler → RandomForestClassifier
  - Cross-validation with ParamGrid (numTrees, maxDepth) on training set
  - Evaluates best model on validation and test sets (F1 + accuracy)
  - Prints confusion matrix for both sets
  - Prints feature importances
  - Saves trained model to HDFS for use by spark_streaming.py

This is the complete end-to-end pipeline.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from datetime import timedelta
import os

# =============================================================================
# SPARK SESSION
# =============================================================================
spark = (
    SparkSession.builder
    .appName("SupplyChain_Part3_FullPipeline")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# --- Paths ---
LOCAL_MODE = not os.environ.get("HADOOP_CONF_DIR")
BASE = "/opt/data" if LOCAL_MODE else "hdfs://namenode:9000/supply-chain"

GDELT_RAW          = "/opt/data/gdelt"
HDFS_GDELT_PARQUET = f"{BASE}/{'parquet/' if LOCAL_MODE else ''}gdelt_events"
HDFS_FEATURES      = f"{BASE}/{'parquet/' if LOCAL_MODE else ''}features"
HDFS_MODEL         = f"{BASE}/model_rf_v1"
COMMODITY_PATH     = f"{BASE}/commodities/commodity_prices.csv"
FRED_PATH          = f"{BASE}/fred/fred_macro.csv"

if LOCAL_MODE:
    print("⚠ Running in LOCAL mode (no HDFS)")


# =============================================================================
# GDELT SCHEMA
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
# STEP 1 — INGEST (same as Parts 1 & 2)
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
# STEP 2 — CLEAN & FILTER (same as Parts 1 & 2)
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
# STEP 3 — FEATURE ENGINEERING (same as Part 2)
# =============================================================================
def build_features(events_df, commodity_path, fred_path):
    print(f"\n{'='*60}\nSTEP 3: Feature Engineering\n{'='*60}")

    daily = (
        events_df.groupBy("event_date", "chokepoint").agg(
            F.count("*").alias("event_count"),
            F.avg("GoldsteinScale").alias("avg_goldstein"),
            F.avg("AvgTone").alias("avg_tone"),
            F.sum("NumMentions").alias("total_mentions"),
            F.avg(
                F.when(F.col("EventRootCode").isin(["18", "19", "20"]), 1.0).otherwise(0.0)
            ).alias("conflict_ratio"),
        )
    )

    w7  = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-6, 0)
    w30 = Window.partitionBy("chokepoint").orderBy("event_date").rowsBetween(-29, 0)
    for col_name in ["event_count", "avg_goldstein", "avg_tone", "total_mentions", "conflict_ratio"]:
        daily = daily.withColumn(f"{col_name}_7d",  F.avg(col_name).over(w7))
        daily = daily.withColumn(f"{col_name}_30d", F.avg(col_name).over(w30))

    prices = spark.read.option("header", "true").option("inferSchema", "true").csv(commodity_path)
    prices = prices.withColumn("date", F.to_date("date"))
    oil = (
        prices.filter(F.col("commodity") == "crude_oil")
        .select(F.col("date").alias("price_date"), F.col("close1").cast("float").alias("oil_close"))
        .withColumn("commodity", F.lit("crude_oil"))
    )

    w_price = Window.partitionBy("commodity").orderBy("price_date")
    oil = (
        oil
        .withColumn("close_lag5",  F.lag("oil_close", 5).over(w_price))
        .withColumn("close_lag20", F.lag("oil_close", 20).over(w_price))
        .withColumn("return_5d",   (F.col("oil_close") - F.col("close_lag5"))  / F.col("close_lag5"))
        .withColumn("return_20d",  (F.col("oil_close") - F.col("close_lag20")) / F.col("close_lag20"))
        .withColumn("daily_return",
            (F.col("oil_close") - F.lag("oil_close", 1).over(w_price)) /
             F.lag("oil_close", 1).over(w_price))
    )
    w_vol = Window.partitionBy("commodity").orderBy("price_date").rowsBetween(-19, 0)
    oil = (
        oil
        .withColumn("volatility_20d", F.stddev("daily_return").over(w_vol))
        .withColumn("close_fwd5", F.lead("oil_close", 5).over(w_price))
        .withColumn("fwd_return_5d", (F.col("close_fwd5") - F.col("oil_close")) / F.col("oil_close"))
    )

    oil_labelled = oil.filter(F.col("fwd_return_5d").isNotNull())
    terciles = oil_labelled.approxQuantile("fwd_return_5d", [0.33, 0.66], 0.01)
    t_low, t_high = terciles[0], terciles[1]
    oil = oil.withColumn(
        "label",
        F.when(F.col("fwd_return_5d").isNull(), None)
        .when(F.col("fwd_return_5d") > t_high, 2)
        .when(F.col("fwd_return_5d") < t_low,  0)
        .otherwise(1)
    )

    fred = spark.read.option("header", "true").option("inferSchema", "true").csv(fred_path)
    fred = fred.withColumn("date", F.to_date("date"))
    date_bounds = oil.select(F.min("price_date").alias("mn"), F.max("price_date").alias("mx")).first()
    date_spine = spark.sql(
        f"SELECT explode(sequence(to_date('{date_bounds['mn']}'), "
        f"to_date('{date_bounds['mx']}'), interval 1 day)) AS fred_date"
    )
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

    oil_chokepoints = ["hormuz", "suez", "red_sea"]
    features = (
        daily.filter(F.col("chokepoint").isin(oil_chokepoints))
        .join(oil.select("price_date", "return_5d", "return_20d", "volatility_20d", "label"),
              daily["event_date"] == oil["price_date"], "inner")
        .drop("price_date")
        .join(fred_daily, daily["event_date"] == fred_daily["fred_date"], "left")
        .drop("fred_date")
    )

    model_cols = [
        "event_count_7d", "avg_goldstein_7d", "avg_tone_7d",
        "total_mentions_7d", "conflict_ratio_7d",
        "event_count_30d", "avg_goldstein_30d", "avg_tone_30d",
        "total_mentions_30d", "conflict_ratio_30d",
        "return_5d", "return_20d", "volatility_20d",
        "treasury_10y", "usd_index", "label",
    ]
    features = features.na.drop(subset=model_cols)
    print(f"  Final feature rows: {features.count():,}")
    features.groupBy("label").count().orderBy("label").show()
    return features


# =============================================================================
# STEP 4 — ML TRAINING (NEW in Part 3)
# =============================================================================
def train_model(features_df, model_path):
    print(f"\n{'='*60}\nSTEP 4: ML Training (Random Forest)\n{'='*60}")

    # --- Feature columns fed into the model ---
    feature_cols = [
        "event_count_7d", "avg_goldstein_7d", "avg_tone_7d",       # GDELT 7-day
        "total_mentions_7d", "conflict_ratio_7d",
        "event_count_30d", "avg_goldstein_30d", "avg_tone_30d",     # GDELT 30-day
        "total_mentions_30d", "conflict_ratio_30d",
        "return_5d", "return_20d", "volatility_20d",                # Price features
        "treasury_10y", "usd_index",                                 # Macro
    ]

    # VectorAssembler packs all feature columns into a single vector for Spark ML
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=8,
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, rf])

    # --- Time-based split — IMPORTANT: must not shuffle, preserves temporal order ---
    # This prevents future data leaking into training (would inflate accuracy)
    date_range = features_df.select(
        F.min("event_date").alias("mn"), F.max("event_date").alias("mx")
    ).first()

    total_days = (date_range["mx"] - date_range["mn"]).days
    train_end = date_range["mn"] + timedelta(days=int(total_days * 0.6))  # 60% train
    val_end   = date_range["mn"] + timedelta(days=int(total_days * 0.8))  # 20% val, 20% test

    train_df = features_df.filter(F.col("event_date") <  F.lit(train_end))
    val_df   = features_df.filter(
        (F.col("event_date") >= F.lit(train_end)) & (F.col("event_date") < F.lit(val_end))
    )
    test_df  = features_df.filter(F.col("event_date") >= F.lit(val_end))

    print(f"  Date range: {date_range['mn']} → {date_range['mx']} ({total_days} days)")
    print(f"  Train: {train_df.count():,} rows  (< {train_end})")
    print(f"  Val:   {val_df.count():,}   rows  ({train_end} – {val_end})")
    print(f"  Test:  {test_df.count():,}   rows  (≥ {val_end})")

    # --- Cross-validation: try different numTrees and maxDepth combinations ---
    # Picks the best hyperparameters automatically using 3-fold CV on training set
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

    print("  Training with 3-fold cross-validation (4 param combos × 3 folds = 12 fits)...")
    train_count = train_df.count()
    if train_count < 10:
        print(f"  ⚠ Only {train_count} train rows — skipping CV, fitting directly")
        best_model = pipeline.fit(train_df)
    else:
        cv_model   = cv.fit(train_df)
        best_model = cv_model.bestModel

    # --- Evaluate on validation and test sets ---
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    for name, df in [("Validation", val_df), ("Test", test_df)]:
        if df.count() == 0:
            print(f"\n  {name} — SKIPPED (0 rows)")
            continue

        preds = best_model.transform(df)
        f1  = evaluator.evaluate(preds)
        acc = acc_evaluator.evaluate(preds)
        print(f"\n  {name} — F1: {f1:.4f} | Accuracy: {acc:.4f}")

        print(f"  {name} confusion matrix (rows=actual, cols=predicted):")
        preds.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

    # --- Feature importances — tells us which signals matter most ---
    rf_model    = best_model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    print("\n  Feature importances:")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {col:30s} {imp:.4f} {bar}")

    # --- Save model to HDFS — spark_streaming.py will load this for live inference ---
    best_model.write().overwrite().save(model_path)
    print(f"\n  ✓ Model saved to {model_path}")

    return best_model


# =============================================================================
# MAIN — Full end-to-end pipeline
# =============================================================================
if __name__ == "__main__":
    # Step 1: Ingest raw CSVs → Parquet
    ingest_gdelt(GDELT_RAW, HDFS_GDELT_PARQUET)

    # Step 2: Clean, dedup, geo-filter, tag chokepoints
    clean_df = clean_and_filter(HDFS_GDELT_PARQUET)

    # Step 3: Feature engineering — GDELT signals + price features + macro
    features_df = build_features(clean_df, COMMODITY_PATH, FRED_PATH)

    # Save features to HDFS (also used by Hive and graph_analysis.py)
    features_df.cache()
    features_df.write.mode("overwrite").parquet(HDFS_FEATURES)
    print(f"\nFeatures saved to {HDFS_FEATURES}")

    # Step 4: Train Random Forest, evaluate, save model (NEW)
    features_df = features_df.filter(F.col("event_date") >= "2024-01-01")
    model = train_model(features_df, HDFS_MODEL)

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("  → Features:  " + HDFS_FEATURES)
    print("  → Model:     " + HDFS_MODEL)
    print("  Next steps:  run hive_setup.py | graph_analysis.py | dashboard.py")
    print("=" * 60)

    spark.stop()
