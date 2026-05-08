#!/usr/bin/env python3
"""Feature engineering: clean/filter GDELT and build commodity feature tables.

Rolling GDELT features are computed from raw tagged events, not from prior
1-day feature output. Events on non-trading days are first folded into the next
available commodity trading day, then daily chokepoint aggregates are expanded
onto a dense calendar spine. Spark range windows over epoch-day order compute
trailing inclusive windows: [event_date - window + 1, event_date].

The output keeps one row per commodity trading day after joining prices. Forward
return targets use trading-day leads: return_5d_fwd and return_20d_fwd are
price[t+horizon] / price[t] - 1. Rows missing either forward target are dropped,
which trims the last ~20 trading rows uniformly. The legacy binary label remains
for backward compatibility and uses a single train-only 1-sigma threshold.
Set TRAIN_END_DATE=YYYY-MM-DD to pin that cutoff; otherwise it defaults to
max(event_date) minus 90 days.
"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import timedelta
from typing import Iterable

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import (
    BASE_METRICS,
    CHOKEPOINTS,
    DEFAULT_TRAIN_END_LOOKBACK_DAYS,
    HDFS_HOT,
    HDFS_WARM,
    TRAIN_END_DATE_ENV,
    WARM_COLUMNS,
    WINDOWS,
)
from schemas import validate_features

JAVA_MIN_VERSION = 17


def _assert_java_version() -> None:
    try:
        proc = subprocess.run(["java", "-version"], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Java is required for Spark. Install Java 17 or newer.") from exc

    version_output = proc.stderr or proc.stdout
    match = re.search(r'version "([^"]+)"', version_output)
    if not match:
        raise RuntimeError(f"Could not determine Java version from: {version_output.strip()}")

    version = match.group(1)
    major = int(version.split(".")[0]) if not version.startswith("1.") else int(version.split(".")[1])
    if major < JAVA_MIN_VERSION:
        raise RuntimeError(
            f"Java {JAVA_MIN_VERSION}+ is required for Spark; found Java {version}. "
            "Install/use Java 17 or newer before running the feature pipeline."
        )


_assert_java_version()

from pipeline_ingest import SUPPLY_CHAIN_CAMEO, _in_any_chokepoint_bbox, _tag_with_chokepoint

spark = (
    SparkSession.builder.appName("SupplyChainIntel_V1")
    .config("spark.sql.parquet.compression.codec", "snappy")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Supply-chain-relevant CAMEO root codes
SC_CAMEO_ROOTS = ["14", "17", "18", "19", "20"]  # protest, coerce, assault, fight, mass violence
COMMODITIES = ("brent", "wti", "copper", "gold", "wheat", "soybeans")
GDELT_WINDOWS = tuple(WINDOWS)
GDELT_BASE_FIELDS = tuple(BASE_METRICS)
MACRO_COLUMNS = ("treasury_10y", "usd_index", "vix")
TARGET_COLUMNS = ("return_5d_fwd", "return_20d_fwd", "abs_return_5d_fwd", "abs_return_20d_fwd")


def gdelt_feature_columns() -> list[str]:
    """Return the expected 8 chokepoints x 4 windows x 3 metrics column names."""
    return [
        f"{chokepoint}_{field}_{window}d"
        for chokepoint in CHOKEPOINTS
        for window in GDELT_WINDOWS
        for field in GDELT_BASE_FIELDS
    ]


def feature_columns() -> list[str]:
    """Return model feature columns, excluding event_date, targets, and label."""
    return [*gdelt_feature_columns(), *MACRO_COLUMNS, "volatility_20d"]


def _read_table(path: str) -> DataFrame:
    """Read parquet by default, with CSV compatibility for older local market files."""
    if path.lower().endswith(".csv"):
        return spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    return spark.read.parquet(path)


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"None of these columns are present: {', '.join(candidates)}")


def _select_warm_columns(df: DataFrame) -> DataFrame:
    """Project warm-tier columns, adding newly required nullable fields for old parquet."""
    for col_name in WARM_COLUMNS:
        if col_name not in df.columns:
            default_value = F.lit(0) if col_name == "NumSources" else F.lit(None)
            df = df.withColumn(col_name, default_value)
    return df.select(*WARM_COLUMNS)


def _resolve_train_end_date(max_event_date) -> str:
    configured = os.environ.get(TRAIN_END_DATE_ENV)
    if configured:
        return configured
    if max_event_date is None:
        raise RuntimeError("Cannot derive default TRAIN_END_DATE because max(event_date) is null.")
    return (max_event_date - timedelta(days=DEFAULT_TRAIN_END_LOOKBACK_DAYS)).isoformat()


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
        warm_df = _select_warm_columns(spark.read.parquet(HDFS_WARM))
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
        )
        hot_filtered = _select_warm_columns(hot_filtered)
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


def _prepare_prices(commodity_path: str) -> DataFrame:
    prices = _read_table(commodity_path).withColumn("date", F.to_date("date"))
    close_col = _first_existing_column(prices.columns, ("close", "Close", "close1", "adj_close"))

    prices = prices.select(
        F.col("date").alias("event_date"),
        F.lower(F.col("commodity")).alias("commodity"),
        F.col(close_col).cast("double").alias("price"),
    ).filter(F.col("event_date").isNotNull() & F.col("price").isNotNull())

    available = [row["commodity"] for row in prices.select("commodity").distinct().orderBy("commodity").collect()]
    missing = sorted(set(COMMODITIES) - set(available))
    print(f"  Commodity names in price input: {available}")
    print(f"  Required output commodities: {list(COMMODITIES)}")
    if missing:
        raise RuntimeError(
            "Missing required commodities in price input: "
            f"{missing}. "
            "the new feature contract requires brent, wti, copper, gold, wheat, soybeans."
        )
    return prices.filter(F.col("commodity").isin(*COMMODITIES))


def _prepare_macro(fred_path: str, min_date, max_date) -> DataFrame:
    fred = _read_table(fred_path).withColumn("date", F.to_date("date"))
    fred = fred.select(
        F.col("date").alias("event_date"),
        F.col("treasury_10y").cast("double").alias("treasury_10y"),
        F.col("usd_index").cast("double").alias("usd_index"),
        F.col("vix").cast("double").alias("vix"),
    )

    date_spine = spark.sql(
        f"SELECT explode(sequence(to_date('{min_date}'), "
        f"to_date('{max_date}'), interval 1 day)) AS event_date"
    )
    macro = date_spine.join(fred, on="event_date", how="left").orderBy("event_date")

    w_ffill = Window.orderBy("event_date").rowsBetween(Window.unboundedPreceding, 0)
    w_bfill = Window.orderBy("event_date").rowsBetween(0, Window.unboundedFollowing)
    for col_name in MACRO_COLUMNS:
        macro = macro.withColumn(col_name, F.last(col_name, ignorenulls=True).over(w_ffill))
        macro = macro.withColumn(
            col_name,
            F.coalesce(F.col(col_name), F.first(col_name, ignorenulls=True).over(w_bfill)),
        )

    print(f"  FRED daily rows (left-joined + filled): {macro.count():,}")
    return macro


def _build_gdelt_wide(events_df: DataFrame, trading_dates: DataFrame, min_date, max_date) -> DataFrame:
    aligned_events = (
        events_df.alias("events")
        .join(
            trading_dates.alias("trading_dates"),
            F.col("trading_dates.event_date") >= F.col("events.event_date"),
            "inner",
        )
        .groupBy("events.GLOBALEVENTID")
        .agg(F.min(F.col("trading_dates.event_date")).alias("aligned_event_date"))
    )
    events_for_features = (
        events_df.join(aligned_events, on="GLOBALEVENTID", how="inner")
        .drop("event_date")
        .withColumnRenamed("aligned_event_date", "event_date")
    )

    daily = (
        events_for_features.groupBy("event_date", "chokepoint")
        .agg(
            F.count("*").cast("double").alias("event_sum"),
            F.sum(F.when(F.col("GoldsteinScale").isNotNull(), F.col("GoldsteinScale")).otherwise(0.0)).alias(
                "_goldstein_sum"
            ),
            F.sum(F.when(F.col("GoldsteinScale").isNotNull(), 1.0).otherwise(0.0)).alias(
                "_goldstein_count"
            ),
            F.sum(F.when(F.col("AvgTone").isNotNull(), F.col("AvgTone")).otherwise(0.0)).alias("_tone_sum"),
            F.sum(F.when(F.col("AvgTone").isNotNull(), 1.0).otherwise(0.0)).alias("_tone_count"),
        )
    )

    date_spine = spark.sql(
        f"SELECT explode(sequence(to_date('{min_date}'), "
        f"to_date('{max_date}'), interval 1 day)) AS event_date"
    )
    chokepoints = spark.createDataFrame([(name,) for name in CHOKEPOINTS], ["chokepoint"])
    dense = date_spine.crossJoin(chokepoints).join(daily, ["event_date", "chokepoint"], "left")

    zero_fill = {
        "event_sum": 0.0,
        "_goldstein_sum": 0.0,
        "_goldstein_count": 0.0,
        "_tone_sum": 0.0,
        "_tone_count": 0.0,
    }
    dense = dense.fillna(zero_fill).withColumn("_event_day", F.unix_date(F.col("event_date")))

    for days in GDELT_WINDOWS:
        suffix = f"{days}d"
        window = (
            Window.partitionBy("chokepoint")
            .orderBy("_event_day")
            .rangeBetween(-(days - 1), 0)
        )
        dense = dense.withColumn(f"event_sum_{suffix}", F.sum("event_sum").over(window))
        dense = dense.withColumn(f"_goldstein_sum_{suffix}", F.sum("_goldstein_sum").over(window))
        dense = dense.withColumn(f"_goldstein_count_{suffix}", F.sum("_goldstein_count").over(window))
        dense = dense.withColumn(f"_tone_sum_{suffix}", F.sum("_tone_sum").over(window))
        dense = dense.withColumn(f"_tone_count_{suffix}", F.sum("_tone_count").over(window))
        dense = dense.withColumn(
            f"goldstein_mean_{suffix}",
            F.when(F.col(f"_goldstein_count_{suffix}") > 0, F.col(f"_goldstein_sum_{suffix}") / F.col(f"_goldstein_count_{suffix}")),
        )
        dense = dense.withColumn(
            f"tone_mean_{suffix}",
            F.when(F.col(f"_tone_count_{suffix}") > 0, F.col(f"_tone_sum_{suffix}") / F.col(f"_tone_count_{suffix}")),
        )

    window_cols = [
        f"{field}_{days}d"
        for days in GDELT_WINDOWS
        for field in GDELT_BASE_FIELDS
    ]
    wide = (
        dense.groupBy("event_date")
        .pivot("chokepoint", list(CHOKEPOINTS))
        .agg(*[F.first(col_name, ignorenulls=True).alias(col_name) for col_name in window_cols])
    )

    count_cols = [
        col_name
        for col_name in wide.columns
        if any(col_name.endswith(f"_event_sum_{days}d") for days in GDELT_WINDOWS)
    ]
    wide = wide.fillna(0.0, subset=count_cols)

    actual_gdelt_cols = [col_name for col_name in wide.columns if col_name != "event_date"]
    print(f"  GDELT wide rows: {wide.count():,}")
    print(f"  GDELT feature columns: {len(actual_gdelt_cols):,} (expected 96)")
    return wide


def _build_commodity_features(
    commodity: str,
    prices: DataFrame,
    gdelt_wide: DataFrame,
    macro: DataFrame,
    source_min_date,
    source_max_date,
    train_end_date: str,
) -> tuple[DataFrame, float, str]:
    price_df = prices.filter(F.col("commodity") == commodity).select("event_date", "price")
    w_price = Window.orderBy("event_date")

    price_df = (
        price_df.withColumn("price_lag1", F.lag("price", 1).over(w_price))
        .withColumn("daily_return", (F.col("price") - F.col("price_lag1")) / F.col("price_lag1"))
        .withColumn("volatility_20d", F.stddev("daily_return").over(w_price.rowsBetween(-19, 0)))
        .withColumn("price_fwd5", F.lead("price", 5).over(w_price))
        .withColumn("price_fwd20", F.lead("price", 20).over(w_price))
        .withColumn("return_5d_fwd", (F.col("price_fwd5") / F.col("price")) - 1.0)
        .withColumn("return_20d_fwd", (F.col("price_fwd20") / F.col("price")) - 1.0)
        .withColumn("abs_return_5d_fwd", F.abs(F.col("return_5d_fwd")))
        .withColumn("abs_return_20d_fwd", F.abs(F.col("return_20d_fwd")))
    )

    threshold_row = (
        price_df.filter(
            (F.col("event_date") < F.to_date(F.lit(train_end_date)))
            & F.col("return_5d_fwd").isNotNull()
        )
        .agg(F.stddev("return_5d_fwd").alias("threshold"))
        .first()
    )
    threshold = threshold_row["threshold"]
    if threshold is None or threshold <= 0:
        raise RuntimeError(f"Could not compute a positive 1-sigma label threshold for {commodity}")

    price_df = price_df.withColumn(
        "label",
        F.when(F.col("return_5d_fwd").isNull(), None)
        .when(F.abs(F.col("return_5d_fwd")) > F.lit(float(threshold)), F.lit(1))
        .otherwise(F.lit(0)),
    )

    features = (
        price_df.select("event_date", "volatility_20d", *TARGET_COLUMNS, "label")
        .filter(
            (F.col("event_date") >= F.lit(source_min_date))
            & (F.col("event_date") <= F.lit(source_max_date))
        )
        .join(gdelt_wide, on="event_date", how="left")
        .join(macro, on="event_date", how="left")
        .orderBy("event_date")
    )

    features = features.filter(
        F.col("return_5d_fwd").isNotNull() & F.col("return_20d_fwd").isNotNull()
    ).select(
        "event_date",
        *gdelt_feature_columns(),
        *MACRO_COLUMNS,
        "volatility_20d",
        *TARGET_COLUMNS,
        "label",
    )
    validate_features(features)
    return features, float(threshold), train_end_date


def _null_counts(df: DataFrame) -> dict[str, int]:
    null_counts = df.select(
        [F.count(F.when(F.col(col_name).isNull(), col_name)).alias(col_name) for col_name in df.columns]
    ).first().asDict()
    return {col_name: int(null_counts[col_name]) for col_name in df.columns}


def _target_stats(df: DataFrame) -> dict[str, dict[str, float | None]]:
    aggregations = []
    for col_name in ("abs_return_5d_fwd", "abs_return_20d_fwd"):
        aggregations.extend(
            [
                F.mean(col_name).alias(f"{col_name}__mean"),
                F.stddev(col_name).alias(f"{col_name}__std"),
                F.expr(f"percentile_approx({col_name}, 0.5, 10000)").alias(f"{col_name}__p50"),
                F.expr(f"percentile_approx({col_name}, 0.9, 10000)").alias(f"{col_name}__p90"),
            ]
        )
    row = df.agg(*aggregations).first().asDict()
    return {
        col_name: {
            stat: (float(row[f"{col_name}__{stat}"]) if row[f"{col_name}__{stat}"] is not None else None)
            for stat in ("mean", "std", "p50", "p90")
        }
        for col_name in ("abs_return_5d_fwd", "abs_return_20d_fwd")
    }


def summarize_features(commodity: str, df: DataFrame, threshold: float, train_end_date: str) -> dict:
    """Build the per-commodity sanity summary written beside the parquet output."""
    row_count = df.count()
    date_range = df.select(F.min("event_date").alias("mn"), F.max("event_date").alias("mx")).first()
    null_counts = _null_counts(df)
    top_null_counts = dict(sorted(null_counts.items(), key=lambda item: item[1], reverse=True)[:10])
    label_counts = {
        str(row["label"]): int(row["count"])
        for row in df.groupBy("label").count().orderBy("label").collect()
    }
    positive_count = label_counts.get("1", 0)

    present_feature_columns = [col_name for col_name in feature_columns() if col_name in df.columns]
    return {
        "commodity": commodity,
        "row_count": int(row_count),
        "date_min": date_range["mn"].isoformat() if date_range["mn"] else None,
        "date_max": date_range["mx"].isoformat() if date_range["mx"] else None,
        "feature_column_count": len(present_feature_columns),
        "gdelt_feature_column_count": len([col_name for col_name in gdelt_feature_columns() if col_name in df.columns]),
        "null_count_per_column": top_null_counts,
        "target_stats": _target_stats(df),
        "label_class_balance": {
            "counts": label_counts,
            "positive_rate": positive_count / row_count if row_count else 0.0,
        },
        "legacy_label_threshold": float(threshold),
        "train_end_date": train_end_date,
    }


def _format_optional_float(value: float | None) -> str:
    return f"{value:.8f}" if value is not None else "null"


def _print_sanity_checks(commodity: str, df: DataFrame, threshold: float, train_end_date: str) -> None:
    summary = summarize_features(commodity, df, threshold, train_end_date)
    label_balance = summary["label_class_balance"]

    print(f"\n  [{commodity}] final parquet sanity checks")
    print(f"    rows: {summary['row_count']:,}")
    print(f"    date range: {summary['date_min']} -> {summary['date_max']}")
    print(f"    feature columns: {summary['feature_column_count']} (expected 100)")
    print(f"    label threshold (train-only 1-sigma): {threshold:.8f}")
    print(
        "    class balance: "
        f"{label_balance['counts'].get('1', 0):,}/{summary['row_count']:,} positive "
        f"({label_balance['positive_rate']:.2%})"
    )
    print("    expected columns: 96 GDELT + 3 macro + 1 volatility + 4 targets + label + event_date = 106")
    print(
        f"    actual columns: {len(df.columns)} total; "
        f"{summary['gdelt_feature_column_count']} GDELT feature columns"
    )
    print("    target stats:")
    for target, stats in summary["target_stats"].items():
        print(
            f"      {target}: mean={_format_optional_float(stats['mean'])}, "
            f"std={_format_optional_float(stats['std'])}, "
            f"p50={_format_optional_float(stats['p50'])}, "
            f"p90={_format_optional_float(stats['p90'])}"
        )
    print("    top 10 null counts:")
    for col_name, null_count in summary["null_count_per_column"].items():
        print(f"      {col_name}: {null_count}")


def build_features(events_df, commodity_path: str, fred_path: str):
    """Build one wide daily feature table per commodity."""
    print(f"\n{'='*60}")
    print("STEP 3: Feature Engineering")
    print(f"{'='*60}")

    prices = _prepare_prices(commodity_path)
    source_bounds = events_df.select(F.min("event_date").alias("mn"), F.max("event_date").alias("mx")).first()
    train_end_date = _resolve_train_end_date(source_bounds["mx"])
    print(f"  Source date range: {source_bounds['mn']} -> {source_bounds['mx']}")
    print(f"  Train threshold cutoff: event_date < {train_end_date}")
    trading_dates = prices.select("event_date").distinct()
    gdelt_wide = _build_gdelt_wide(events_df, trading_dates, source_bounds["mn"], source_bounds["mx"]).cache()
    macro = _prepare_macro(fred_path, source_bounds["mn"], source_bounds["mx"]).cache()

    result: dict[str, tuple[DataFrame, float, str]] = {}
    for commodity in COMMODITIES:
        features, threshold, train_end_date = _build_commodity_features(
            commodity,
            prices,
            gdelt_wide,
            macro,
            source_bounds["mn"],
            source_bounds["mx"],
            train_end_date,
        )
        _print_sanity_checks(commodity, features, threshold, train_end_date)
        result[commodity] = (features, threshold, train_end_date)

    return result
