"""
Schema contract for the Supply Chain Intelligence pipeline.

Defines PySpark StructType schemas for all pipeline-managed tables and
provides validate_schema() for runtime enforcement at read/write boundaries.

Schemas are versioned. Bumping a version is a breaking change — coordinate
across the team before doing it.

Source of truth for HOT/WARM field types: scripts/pipeline_ingest.py (GDELT_SCHEMA),
config (WARM_COLUMNS), pipeline_features (build_features output). Keep in sync when those change.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ---- Schema versions ----
HOT_SCHEMA_VERSION = 1
WARM_SCHEMA_VERSION = 1
COLD_SCHEMA_VERSION = 1
FEATURES_SCHEMA_VERSION = 1
CHOKEPOINTS = ("hormuz", "suez", "red_sea", "black_sea", "malacca", "panama", "taiwan", "chile")
GDELT_BASE_FIELDS = ("event_sum", "goldstein_mean", "tone_mean")
GDELT_WINDOWS = (7, 14, 30, 90)
MACRO_FIELDS = ("treasury_10y", "usd_index", "vix")
TARGET_FIELDS = ("return_5d_fwd", "return_20d_fwd", "abs_return_5d_fwd", "abs_return_20d_fwd")

# ---- HOT tier schema ----
# Full GDELT v1 schema preserved verbatim during ingest, plus event_date added
# in ingest_gdelt_tiered before write. Mirrors GDELT_SCHEMA in pipeline_ingest.py.
# All GDELT fields nullable=True; event_date nullable=False (parsed from SQLDATE).
HOT_SCHEMA = StructType(
    [
        StructField("GLOBALEVENTID", IntegerType(), True),
        StructField("SQLDATE", IntegerType(), True),
        StructField("MonthYear", IntegerType(), True),
        StructField("Year", IntegerType(), True),
        StructField("FractionDate", FloatType(), True),
        StructField("Actor1Code", StringType(), True),
        StructField("Actor1Name", StringType(), True),
        StructField("Actor1CountryCode", StringType(), True),
        StructField("Actor1KnownGroupCode", StringType(), True),
        StructField("Actor1EthnicCode", StringType(), True),
        StructField("Actor1Religion1Code", StringType(), True),
        StructField("Actor1Religion2Code", StringType(), True),
        StructField("Actor1Type1Code", StringType(), True),
        StructField("Actor1Type2Code", StringType(), True),
        StructField("Actor1Type3Code", StringType(), True),
        StructField("Actor2Code", StringType(), True),
        StructField("Actor2Name", StringType(), True),
        StructField("Actor2CountryCode", StringType(), True),
        StructField("Actor2KnownGroupCode", StringType(), True),
        StructField("Actor2EthnicCode", StringType(), True),
        StructField("Actor2Religion1Code", StringType(), True),
        StructField("Actor2Religion2Code", StringType(), True),
        StructField("Actor2Type1Code", StringType(), True),
        StructField("Actor2Type2Code", StringType(), True),
        StructField("Actor2Type3Code", StringType(), True),
        StructField("IsRootEvent", IntegerType(), True),
        StructField("EventCode", StringType(), True),
        StructField("EventBaseCode", StringType(), True),
        StructField("EventRootCode", StringType(), True),
        StructField("QuadClass", IntegerType(), True),
        StructField("GoldsteinScale", FloatType(), True),
        StructField("NumMentions", IntegerType(), True),
        StructField("NumSources", IntegerType(), True),
        StructField("NumArticles", IntegerType(), True),
        StructField("AvgTone", FloatType(), True),
        StructField("Actor1Geo_Type", IntegerType(), True),
        StructField("Actor1Geo_FullName", StringType(), True),
        StructField("Actor1Geo_CountryCode", StringType(), True),
        StructField("Actor1Geo_ADM1Code", StringType(), True),
        StructField("Actor1Geo_Lat", FloatType(), True),
        StructField("Actor1Geo_Long", FloatType(), True),
        StructField("Actor1Geo_FeatureID", StringType(), True),
        StructField("Actor2Geo_Type", IntegerType(), True),
        StructField("Actor2Geo_FullName", StringType(), True),
        StructField("Actor2Geo_CountryCode", StringType(), True),
        StructField("Actor2Geo_ADM1Code", StringType(), True),
        StructField("Actor2Geo_Lat", FloatType(), True),
        StructField("Actor2Geo_Long", FloatType(), True),
        StructField("Actor2Geo_FeatureID", StringType(), True),
        StructField("ActionGeo_Type", IntegerType(), True),
        StructField("ActionGeo_FullName", StringType(), True),
        StructField("ActionGeo_CountryCode", StringType(), True),
        StructField("ActionGeo_ADM1Code", StringType(), True),
        StructField("ActionGeo_Lat", FloatType(), True),
        StructField("ActionGeo_Long", FloatType(), True),
        StructField("ActionGeo_FeatureID", StringType(), True),
        StructField("DATEADDED", StringType(), True),
        StructField("SOURCEURL", StringType(), True),
        StructField("event_date", DateType(), False),
    ]
)

# ---- WARM tier schema ----
# Projected subset: WARM_COLUMNS in config.py; types match HOT_SCHEMA fields.
WARM_SCHEMA = StructType(
    [
        StructField("GLOBALEVENTID", IntegerType(), True),
        StructField("event_date", DateType(), False),
        StructField("Actor1Code", StringType(), True),
        StructField("Actor2Code", StringType(), True),
        StructField("EventCode", StringType(), True),
        StructField("EventRootCode", StringType(), True),
        StructField("GoldsteinScale", FloatType(), True),
        StructField("NumMentions", IntegerType(), True),
        StructField("NumSources", IntegerType(), True),
        StructField("AvgTone", FloatType(), True),
        StructField("ActionGeo_Lat", FloatType(), True),
        StructField("ActionGeo_Long", FloatType(), True),
        StructField("SOURCEURL", StringType(), True),
    ]
)

# ---- COLD tier schema ----
# Aggregated baselines for very old data (PR L will populate).
# Speculative contract for downstream callers.
COLD_SCHEMA = StructType(
    [
        StructField("chokepoint", StringType(), False),
        StructField("year_month", StringType(), False),
        StructField("event_count", LongType(), False),
        StructField("avg_goldstein", DoubleType(), True),
        StructField("avg_tone", DoubleType(), True),
        StructField("conflict_ratio", DoubleType(), True),
        StructField("event_count_p50", DoubleType(), True),
        StructField("event_count_p95", DoubleType(), True),
        StructField("rolling_mean_5y", DoubleType(), True),
    ]
)

# ---- FEATURES schema ----
# Output of build_features(): one row per event_date, with chokepoint-prefixed
# rolling GDELT columns, macro features, commodity volatility, regression
# targets, and a legacy binary label. Total columns: 106.
FEATURES_SCHEMA = StructType(
    [
        StructField("event_date", DateType(), True),
        *[
            StructField(f"{chokepoint}_{field}_{window}d", DoubleType(), True)
            for chokepoint in CHOKEPOINTS
            for window in GDELT_WINDOWS
            for field in GDELT_BASE_FIELDS
        ],
        *[StructField(field, DoubleType(), True) for field in MACRO_FIELDS],
        StructField("volatility_20d", DoubleType(), True),
        *[StructField(field, DoubleType(), True) for field in TARGET_FIELDS],
        StructField("label", IntegerType(), True),
    ]
)


class SchemaValidationError(Exception):
    """Raised when a DataFrame doesn't match its expected schema."""

    pass


def validate_schema(
    df: DataFrame,
    expected: StructType,
    name: str = "<dataframe>",
    strict: bool = True,
) -> None:
    """
    Validate that df conforms to the expected schema.

    Args:
        df: DataFrame to validate
        expected: Expected StructType
        name: Human-readable name for error messages (e.g., "hot_tier")
        strict: If True, df must have exactly the expected columns (in any
                order, with matching types). If False, df may have a
                superset of columns; extras are ignored but missing or
                type-mismatched columns still fail.

    Raises:
        SchemaValidationError: with a clear message about what's wrong.
    """
    actual = df.schema
    actual_fields = {f.name: f for f in actual.fields}
    expected_fields = {f.name: f for f in expected.fields}

    missing = set(expected_fields) - set(actual_fields)
    extra = set(actual_fields) - set(expected_fields)
    type_mismatches = []

    for col_name, expected_field in expected_fields.items():
        if col_name not in actual_fields:
            continue
        actual_field = actual_fields[col_name]
        if expected_field.dataType != actual_field.dataType:
            type_mismatches.append(
                (col_name, expected_field.dataType, actual_field.dataType)
            )

    errors = []
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
    if strict and extra:
        errors.append(f"unexpected extra columns: {sorted(extra)}")
    if type_mismatches:
        errors.append(
            "type mismatches: "
            + ", ".join(f"{n} expected {e} got {a}" for n, e, a in type_mismatches)
        )

    if errors:
        raise SchemaValidationError(
            f"Schema validation failed for {name}:\n  " + "\n  ".join(errors)
        )


def validate_hot(df: DataFrame) -> None:
    validate_schema(df, HOT_SCHEMA, name="hot_tier", strict=True)


def validate_warm(df: DataFrame) -> None:
    validate_schema(df, WARM_SCHEMA, name="warm_tier", strict=True)


def validate_cold(df: DataFrame) -> None:
    validate_schema(df, COLD_SCHEMA, name="cold_tier", strict=True)


def validate_features(df: DataFrame) -> None:
    expected_gdelt = {
        f"{chokepoint}_{field}_{window}d"
        for chokepoint in CHOKEPOINTS
        for window in GDELT_WINDOWS
        for field in GDELT_BASE_FIELDS
    }
    required = {
        "event_date",
        "volatility_20d",
        *MACRO_FIELDS,
        *TARGET_FIELDS,
        "label",
        *expected_gdelt,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SchemaValidationError(f"Schema validation failed for features:\n  missing columns: {missing}")

    extra = sorted(set(df.columns) - required)
    if extra:
        raise SchemaValidationError(f"Schema validation failed for features:\n  unexpected extra columns: {extra}")

    if len([col_name for col_name in df.columns if col_name in expected_gdelt]) != 96:
        raise SchemaValidationError("Schema validation failed for features:\n  expected exactly 96 GDELT columns")

    if len(df.columns) != 106:
        raise SchemaValidationError("Schema validation failed for features:\n  expected exactly 106 total columns")

