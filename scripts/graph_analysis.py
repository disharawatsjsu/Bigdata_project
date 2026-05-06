#!/usr/bin/env python3
"""
GraphFrames Analysis — Region-Commodity Dependency Network

Builds a bipartite graph:
  - Nodes: chokepoint regions + commodities
  - Edges: trade dependency links (weighted by disruption frequency)

Runs:
  1. PageRank — which regions are most critical?
  2. Connected Components — which commodities cluster together?
  3. Centrality scores fed back as ML features

Then does an ablation study: RF with vs without graph features.

Usage (inside spark-master):
    spark-submit --master spark://spark-master:7077 \
        --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
        graph_analysis.py

NOTE: GraphFrames requires the package flag above. If that fails,
install manually:
    pyspark --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import os

# --- Spark Session ---
spark = (
    SparkSession.builder
    .appName("SupplyChainIntel_GraphAnalysis")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

from config import HDFS_FEATURES
from schemas import validate_features

FEATURES_PATH = HDFS_FEATURES


# =============================================================================
# 1. BUILD THE GRAPH
# =============================================================================
def build_graph():
    """Create the region-commodity dependency graph from reference data."""
    print("=" * 60)
    print("GRAPH ANALYSIS: Building region-commodity network")
    print("=" * 60)

    # --- Nodes ---
    # Regions (chokepoints)
    regions = [
        ("hormuz",     "region", "Strait of Hormuz"),
        ("suez",       "region", "Suez Canal"),
        ("red_sea",    "region", "Red Sea / Yemen"),
        ("black_sea",  "region", "Black Sea Ports"),
        ("malacca",    "region", "Strait of Malacca"),
        ("panama",     "region", "Panama Canal"),
        ("taiwan",     "region", "Taiwan Strait"),
        ("chile",      "region", "Chile Copper Belt"),
    ]

    # Commodities
    commodities = [
        ("crude_oil",    "commodity", "Crude Oil"),
        ("natural_gas",  "commodity", "Natural Gas"),
        ("wheat",        "commodity", "Wheat"),
        ("copper",       "commodity", "Copper"),
        ("coffee",       "commodity", "Coffee"),
        ("gold",         "commodity", "Gold"),
    ]

    vertices = spark.createDataFrame(
        regions + commodities,
        ["id", "node_type", "display_name"]
    )
    print(f"  Vertices: {vertices.count()} ({len(regions)} regions + {len(commodities)} commodities)")

    # --- Edges ---
    # (src=region, dst=commodity, weight based on trade importance)
    # Weights: primary=1.0, secondary=0.5 — represents relative disruption impact
    edge_data = [
        # Oil supply chain
        ("hormuz",    "crude_oil",    1.0),
        ("suez",      "crude_oil",    0.8),
        ("red_sea",   "crude_oil",    0.9),   # Houthi disruption zone
        ("malacca",   "crude_oil",    0.6),
        # Natural gas
        ("hormuz",    "natural_gas",  0.9),
        ("red_sea",   "natural_gas",  0.5),
        ("black_sea", "natural_gas",  0.4),
        # Wheat
        ("black_sea", "wheat",        1.0),   # Ukraine/Russia grain corridor
        ("suez",      "wheat",        0.6),
        # Copper
        ("chile",     "copper",       1.0),   # Atacama mines
        ("panama",    "copper",       0.7),   # shipping route
        # Coffee
        ("panama",    "coffee",       0.8),
        ("suez",      "coffee",       0.5),
        ("malacca",   "coffee",       0.4),
        # Gold (geopolitical hedge — indirect links)
        ("hormuz",    "gold",         0.4),
        ("red_sea",   "gold",         0.3),
        ("taiwan",    "gold",         0.5),   # geopolitical tension
        # Cross-connections (regions that bridge multiple commodities)
        ("suez",      "natural_gas",  0.3),
        ("malacca",   "natural_gas",  0.3),
    ]

    edges = spark.createDataFrame(
        edge_data,
        ["src", "dst", "weight"]
    )
    # Make edges bidirectional (commodity depends on region AND region serves commodity)
    edges_reverse = edges.select(
        F.col("dst").alias("src"),
        F.col("src").alias("dst"),
        F.col("weight"),
    )
    edges_all = edges.union(edges_reverse)
    print(f"  Edges: {edges_all.count()} (bidirectional)")

    return vertices, edges_all


# =============================================================================
# 2. RUN GRAPH ALGORITHMS
# =============================================================================
def run_graph_algorithms(vertices, edges):
    """PageRank, connected components, and degree centrality."""
    from graphframes import GraphFrame

    g = GraphFrame(vertices, edges)

    # --- PageRank ---
    print("\n  Running PageRank...")
    pr = g.pageRank(resetProbability=0.15, maxIter=20)
    pr_results = (
        pr.vertices
        .select("id", "node_type", "display_name", F.col("pagerank").alias("pagerank_score"))
        .orderBy(F.desc("pagerank_score"))
    )
    print("  PageRank scores:")
    pr_results.show(20, truncate=False)

    # --- Degree centrality ---
    print("  Computing degree centrality...")
    in_degree = g.inDegrees
    out_degree = g.outDegrees
    degree = (
        in_degree.join(out_degree, "id", "outer")
        .na.fill(0)
        .withColumn("total_degree", F.col("inDegree") + F.col("outDegree"))
    )

    # --- Connected Components ---
    print("  Running connected components...")
    spark.sparkContext.setCheckpointDir("/tmp/graphframes_checkpoint")
    cc = g.connectedComponents()
    component_sizes = cc.groupBy("component").count().orderBy(F.desc("count"))
    print("  Component sizes:")
    component_sizes.show()

    # --- Combine all graph metrics per node ---
    graph_features = (
        pr_results
        .join(degree.select("id", "total_degree"), "id", "left")
        .join(cc.select("id", "component"), "id", "left")
    )

    # Extract just region scores (for joining with ML features)
    region_scores = (
        graph_features
        .filter(F.col("node_type") == "region")
        .select(
            F.col("id").alias("chokepoint"),
            "pagerank_score",
            "total_degree",
        )
    )
    print("\n  Region centrality scores:")
    region_scores.show()

    return graph_features, region_scores


# =============================================================================
# 3. ABLATION STUDY: RF with vs without graph features
# =============================================================================
def ablation_study(region_scores):
    """Train two RF models — one with graph features, one without — and compare."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY: Impact of Graph Centrality Features")
    print("=" * 60)

    # Load features from pipeline
    features = spark.read.parquet(FEATURES_PATH)
    validate_features(features)
    features = features.na.drop(subset=["label"])

    # Base feature columns (same as spark_pipeline.py)
    base_cols = [
        "event_count_7d", "avg_goldstein_7d", "avg_tone_7d",
        "total_mentions_7d", "conflict_ratio_7d",
        "event_count_30d", "avg_goldstein_30d", "avg_tone_30d",
        "total_mentions_30d", "conflict_ratio_30d",
        "return_5d", "return_20d", "volatility_20d",
        "treasury_10y", "usd_index",
    ]

    # Time split
    train = features.filter(F.col("event_date") < "2024-01-01")
    test = features.filter(F.col("event_date") >= "2024-07-01")

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    # --- Model A: WITHOUT graph features ---
    print("\n  Training Model A (no graph features)...")
    assembler_a = VectorAssembler(inputCols=base_cols, outputCol="features", handleInvalid="skip")
    rf_a = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=8, seed=42)
    pipeline_a = Pipeline(stages=[assembler_a, rf_a])
    model_a = pipeline_a.fit(train)
    preds_a = model_a.transform(test)
    f1_a = evaluator.evaluate(preds_a)
    acc_a = acc_evaluator.evaluate(preds_a)

    # --- Model B: WITH graph features ---
    print("  Training Model B (with graph features)...")
    # Join graph scores into features
    train_g = train.join(region_scores, "chokepoint", "left").na.fill(0)
    test_g = test.join(region_scores, "chokepoint", "left").na.fill(0)

    graph_cols = base_cols + ["pagerank_score", "total_degree"]
    assembler_b = VectorAssembler(inputCols=graph_cols, outputCol="features", handleInvalid="skip")
    rf_b = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=8, seed=42)
    pipeline_b = Pipeline(stages=[assembler_b, rf_b])
    model_b = pipeline_b.fit(train_g)
    preds_b = model_b.transform(test_g)
    f1_b = evaluator.evaluate(preds_b)
    acc_b = acc_evaluator.evaluate(preds_b)

    # --- Report ---
    print("\n  " + "-" * 50)
    print(f"  Model A (base features only):")
    print(f"    F1: {f1_a:.4f}  |  Accuracy: {acc_a:.4f}")
    print(f"  Model B (base + graph centrality):")
    print(f"    F1: {f1_b:.4f}  |  Accuracy: {acc_b:.4f}")
    print(f"  Δ F1:  {f1_b - f1_a:+.4f}")
    print(f"  Δ Acc: {acc_b - acc_a:+.4f}")
    print("  " + "-" * 50)

    if f1_b > f1_a:
        print("  → Graph features IMPROVED the model")
    else:
        print("  → Graph features did not improve (may need more data or tuning)")

    # Feature importance for Model B
    rf_model_b = model_b.stages[-1]
    importances = rf_model_b.featureImportances.toArray()
    print("\n  Feature importances (Model B):")
    for col, imp in sorted(zip(graph_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {col:30s} {imp:.4f} {bar}")

    return {
        "f1_base": f1_a, "acc_base": acc_a,
        "f1_graph": f1_b, "acc_graph": acc_b,
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    vertices, edges = build_graph()

    try:
        graph_features, region_scores = run_graph_algorithms(vertices, edges)
        results = ablation_study(region_scores)
    except ImportError:
        print("\n⚠ GraphFrames not available. Install with:")
        print("  spark-submit --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 graph_analysis.py")
        print("\nSkipping graph algorithms — running ablation with static scores instead.")

        # Fallback: use hardcoded centrality scores from domain knowledge
        static_scores = spark.createDataFrame([
            ("hormuz",  0.18, 8),
            ("suez",    0.16, 10),
            ("red_sea", 0.14, 6),
        ], ["chokepoint", "pagerank_score", "total_degree"])

        results = ablation_study(static_scores)

    print("\n" + "=" * 60)
    print("GRAPH ANALYSIS COMPLETE")
    print("=" * 60)

    spark.stop()
