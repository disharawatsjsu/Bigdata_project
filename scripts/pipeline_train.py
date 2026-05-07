#!/usr/bin/env python3
"""
ML training: Random Forest on engineered features.

# SparkSession is reused across modules; getOrCreate() returns the same
# session within a single Python process.
"""
from __future__ import annotations

from datetime import timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

from config import HDFS_FEATURES, HDFS_MODEL
from schemas import validate_features

spark = (
    SparkSession.builder.appName("SupplyChainIntel_V1")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


def train_model(features_df, model_path: str):
    """Train RF classifier with cross-validation, evaluate on time-split test set."""
    print(f"\n{'='*60}")
    print("STEP 4: ML Training (Random Forest)")
    print(f"{'='*60}")

    validate_features(features_df)

    feature_cols = [
        "event_count_7d", "avg_goldstein_7d", "avg_tone_7d",
        "total_mentions_7d", "conflict_ratio_7d",
        "event_count_30d", "avg_goldstein_30d", "avg_tone_30d",
        "total_mentions_30d", "conflict_ratio_30d",
        "return_5d", "return_20d", "volatility_20d",
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

    date_range = features_df.select(
        F.min("event_date").alias("mn"), F.max("event_date").alias("mx")
    ).first()
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

        print(f"  {name} confusion matrix:")
        preds.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

    rf_model = best_model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    print("\n  Feature importances:")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {col:30s} {imp:.4f} {bar}")

    best_model.write().overwrite().save(model_path)
    print(f"\n  Model saved to {model_path}")

    return best_model
