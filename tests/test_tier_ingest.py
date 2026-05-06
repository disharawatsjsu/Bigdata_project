"""Smoke test: after ingest_gdelt_tiered runs, verify tier separation.

Run after manual verification (hot + warm samples copied to local paths below).
"""
import pyarrow.parquet as pq
from pathlib import Path

HOT_SAMPLE = Path("data/parquet/hot_sample/")
WARM_SAMPLE = Path("data/parquet/warm_sample/")


def test_hot_tier_has_full_schema():
    assert HOT_SAMPLE.exists(), "Pull a hot-tier month to local first"
    table = pq.read_table(HOT_SAMPLE)
    cols = set(table.schema.names)
    assert len(cols) >= 50, f"Hot tier should have full schema, got {len(cols)} cols"
    assert "GoldsteinScale" in cols
    assert "ActionGeo_Lat" in cols
    print(f"PASS: hot tier has {len(cols)} columns")


def test_warm_tier_has_filtered_schema():
    assert WARM_SAMPLE.exists(), "Pull a warm-tier month to local first"
    table = pq.read_table(WARM_SAMPLE)
    cols = set(table.schema.names)
    assert len(cols) <= 15, f"Warm tier should be projected, got {len(cols)} cols"
    assert "GoldsteinScale" in cols
    assert "GLOBALEVENTID" in cols
    df = table.to_pandas()
    cameo = df["EventRootCode"].dropna()
    cameo_int = cameo.astype(str).str.strip().astype(int)
    valid_cameo = {14, 17, 18, 19, 20}
    actual_cameo = set(cameo_int.unique())
    assert actual_cameo.issubset(valid_cameo), (
        f"Warm should only have supply-chain CAMEO, found: {actual_cameo}"
    )
    print(f"PASS: warm tier has {len(cols)} columns, CAMEO filtered")


if __name__ == "__main__":
    test_hot_tier_has_full_schema()
    test_warm_tier_has_filtered_schema()
