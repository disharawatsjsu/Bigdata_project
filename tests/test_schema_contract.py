"""Schema contract tests — validator logic uses lightweight fakes (no JVM)."""

import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from scripts.schemas import (
    COLD_SCHEMA,
    FEATURES_SCHEMA,
    HOT_SCHEMA,
    WARM_SCHEMA,
    SchemaValidationError,
    validate_schema,
)


class _FakeDF:
    """Minimal stand-in for DataFrame; validate_schema only reads .schema."""

    __slots__ = ("schema",)

    def __init__(self, schema: StructType) -> None:
        self.schema = schema


class TestSchemaContract:
    def test_hot_schema_is_nonempty(self):
        assert len(HOT_SCHEMA.fields) >= 50, (
            "HOT schema should have ~61 GDELT columns + event_date"
        )

    def test_warm_schema_subset_of_hot(self):
        hot_fields = {f.name for f in HOT_SCHEMA.fields}
        warm_fields = {f.name for f in WARM_SCHEMA.fields}
        assert warm_fields.issubset(hot_fields), (
            f"WARM should be subset of HOT; extras: {warm_fields - hot_fields}"
        )

    def test_cold_schema_has_chokepoint_and_year_month(self):
        names = {f.name for f in COLD_SCHEMA.fields}
        assert "chokepoint" in names
        assert "year_month" in names

    def test_features_schema_has_label(self):
        names = {f.name for f in FEATURES_SCHEMA.fields}
        assert "label" in names, "features schema must have a label column"

    def test_features_schema_has_new_baseline_contract(self):
        names = {f.name for f in FEATURES_SCHEMA.fields}
        assert len(FEATURES_SCHEMA.fields) == 106
        assert "hormuz_event_sum_7d" in names
        assert "hormuz_goldstein_mean_90d" in names
        assert "vix" in names
        assert "return_5d_fwd" in names
        assert "abs_return_20d_fwd" in names
        assert "hormuz_event_count_1d" not in names
        assert "return_5d" not in names

    def test_validate_passes_correct_df(self):
        schema = StructType(
            [StructField("a", IntegerType()), StructField("b", StringType())]
        )
        df = _FakeDF(schema)
        validate_schema(df, schema, name="test")  # should not raise

    def test_validate_fails_missing_column(self):
        schema = StructType(
            [StructField("a", IntegerType()), StructField("b", StringType())]
        )
        actual = StructType([StructField("a", IntegerType())])
        df = _FakeDF(actual)
        with pytest.raises(SchemaValidationError, match="missing columns"):
            validate_schema(df, schema, name="test")

    def test_validate_fails_type_mismatch(self):
        expected = StructType([StructField("a", IntegerType())])
        actual = StructType([StructField("a", StringType())])
        df = _FakeDF(actual)
        with pytest.raises(SchemaValidationError, match="type mismatches"):
            validate_schema(df, expected, name="test")

    def test_validate_strict_fails_extra_column(self):
        expected = StructType([StructField("a", IntegerType())])
        actual = StructType(
            [StructField("a", IntegerType()), StructField("b", StringType())]
        )
        df = _FakeDF(actual)
        with pytest.raises(SchemaValidationError, match="unexpected extra columns"):
            validate_schema(df, expected, name="test", strict=True)

    def test_validate_nonstrict_allows_extra_column(self):
        expected = StructType([StructField("a", IntegerType())])
        actual = StructType(
            [StructField("a", IntegerType()), StructField("b", StringType())]
        )
        df = _FakeDF(actual)
        validate_schema(df, expected, name="test", strict=False)  # should not raise
