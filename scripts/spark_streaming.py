#!/usr/bin/env python3
"""
Spark Structured Streaming Consumer — reads GDELT events from Kafka,
computes windowed features, and runs inference against the trained model.

This is the "Mode 3" from the architecture doc — the live demo pipeline.

Usage (inside spark-master container):
    spark-submit --master spark://spark-master:7077 \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
        spark_streaming.py

Prerequisites:
    - Kafka running with events in 'gdelt-events-raw' topic
    - Trained model saved at HDFS_MODEL path (from spark_pipeline.py)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType
)
from pyspark.ml import PipelineModel
import os

# --- Config ---


def _kafka_bootstrap() -> str:
    if os.environ.get("KAFKA_BROKER"):
        return os.environ["KAFKA_BROKER"]
    host, port = os.environ.get("KAFKA_BROKER_HOST"), os.environ.get("KAFKA_HOST_PORT")
    if host and port:
        return f"{host}:{port}"
    return "kafka:9092"


KAFKA_BROKER = _kafka_bootstrap()
TOPIC = "gdelt-events-raw"

from config import HDFS_MODEL, LOCAL_MODE
OUTPUT_PATH = (
    "/opt/data/streaming_output" if LOCAL_MODE
    else "hdfs://namenode:9000/supply-chain/streaming_output"
)
CHECKPOINT_PATH = (
    "/opt/data/checkpoints/streaming" if LOCAL_MODE
    else "hdfs://namenode:9000/supply-chain/checkpoints/streaming"
)

# Chokepoint bounding boxes (same as spark_pipeline.py)
CHOKEPOINTS = {
    "hormuz":   (25.06, 28.06, 54.75, 57.75),
    "suez":     (29.46, 31.46, 31.34, 33.34),
    "red_sea":  (11.00, 17.00, 40.00, 46.00),
}

SC_CAMEO_ROOTS = ["14", "17", "18", "19", "20"]


# --- Spark Session ---
spark = (
    SparkSession.builder
    .appName("SupplyChainIntel_Streaming")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# --- Schema for JSON events from Kafka ---
event_schema = StructType([
    StructField("global_event_id", StringType()),
    StructField("sql_date", StringType()),
    StructField("event_root_code", StringType()),
    StructField("goldstein_scale", FloatType()),
    StructField("num_mentions", IntegerType()),
    StructField("avg_tone", FloatType()),
    StructField("action_geo_lat", FloatType()),
    StructField("action_geo_long", FloatType()),
    StructField("source_url", StringType()),
    StructField("ingested_at", StringType()),
])


def assign_chokepoint(lat, lon):
    """Check if event falls within any oil-relevant chokepoint bbox."""
    for name, (lat_min, lat_max, lon_min, lon_max) in CHOKEPOINTS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return None


# Register as UDF
from pyspark.sql.types import StringType as ST
assign_chokepoint_udf = F.udf(assign_chokepoint, ST())


def main():
    print("=" * 60)
    print("Starting Spark Structured Streaming Consumer")
    print(f"Kafka: {KAFKA_BROKER} | Topic: {TOPIC}")
    print(f"Model: {HDFS_MODEL}")
    print("=" * 60)

    # --- 1. Read from Kafka ---
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON value
    events = (
        raw_stream
        .select(F.from_json(F.col("value").cast("string"), event_schema).alias("data"))
        .select("data.*")
        # Parse timestamp for windowing
        .withColumn("event_ts", F.to_timestamp("ingested_at"))
        .withColumn("event_date", F.to_date("sql_date", "yyyyMMdd"))
    )

    # --- 2. Filter: supply-chain events near oil chokepoints ---
    sc_events = (
        events
        .filter(F.col("event_root_code").isin(SC_CAMEO_ROOTS))
        .filter(F.col("action_geo_lat").isNotNull())
        .withColumn(
            "chokepoint",
            assign_chokepoint_udf(F.col("action_geo_lat"), F.col("action_geo_long"))
        )
        .filter(F.col("chokepoint").isNotNull())
    )

    # --- 3. Windowed aggregation (5-minute tumbling windows for demo speed) ---
    windowed = (
        sc_events
        .withWatermark("event_ts", "10 minutes")
        .groupBy(
            F.window("event_ts", "5 minutes"),
            "chokepoint"
        )
        .agg(
            F.count("*").alias("event_count"),
            F.avg("goldstein_scale").alias("avg_goldstein"),
            F.avg("avg_tone").alias("avg_tone"),
            F.sum("num_mentions").alias("total_mentions"),
            F.avg(
                F.when(F.col("event_root_code").isin(["18", "19", "20"]), 1.0)
                .otherwise(0.0)
            ).alias("conflict_ratio"),
        )
    )

    # --- 4. Load trained model and run inference ---
    try:
        model = PipelineModel.load(HDFS_MODEL)
        print(f"Loaded model from {HDFS_MODEL}")
        has_model = True
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        print("  Running without inference — will just show aggregated features")
        has_model = False

    # --- 5. Output ---
    if has_model:
        # For streaming inference, we need to add placeholder columns
        # that the model expects but we don't have in real-time
        # (rolling windows, price lags, macro). Use defaults for demo.
        enriched = (
            windowed
            .withColumn("event_count_7d", F.col("event_count"))
            .withColumn("avg_goldstein_7d", F.col("avg_goldstein"))
            .withColumn("avg_tone_7d", F.col("avg_tone"))
            .withColumn("total_mentions_7d", F.col("total_mentions"))
            .withColumn("conflict_ratio_7d", F.col("conflict_ratio"))
            .withColumn("event_count_30d", F.col("event_count"))  # approximation
            .withColumn("avg_goldstein_30d", F.col("avg_goldstein"))
            .withColumn("avg_tone_30d", F.col("avg_tone"))
            .withColumn("total_mentions_30d", F.col("total_mentions"))
            .withColumn("conflict_ratio_30d", F.col("conflict_ratio"))
            .withColumn("return_5d", F.lit(0.0))        # placeholder
            .withColumn("return_20d", F.lit(0.0))
            .withColumn("volatility_20d", F.lit(0.015))
            .withColumn("treasury_10y", F.lit(4.2))
            .withColumn("usd_index", F.lit(104.0))
        )

        # Run model — this will create 'prediction' and 'probability' columns
        predictions = model.transform(enriched)

        query = (
            predictions
            .select(
                "window", "chokepoint",
                "event_count", "avg_goldstein", "conflict_ratio",
                "prediction",
            )
            .writeStream
            .outputMode("update")
            .format("console")
            .option("truncate", "false")
            .option("checkpointLocation", CHECKPOINT_PATH)
            .trigger(processingTime="30 seconds")
            .start()
        )
    else:
        # Just show raw windowed aggregates
        query = (
            windowed
            .writeStream
            .outputMode("update")
            .format("console")
            .option("truncate", "false")
            .option("checkpointLocation", CHECKPOINT_PATH)
            .trigger(processingTime="30 seconds")
            .start()
        )

    print("\nStreaming started — waiting for events...")
    print("Press Ctrl+C to stop\n")
    query.awaitTermination()


if __name__ == "__main__":
    main()
