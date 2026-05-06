"""
Central configuration for the Supply Chain Intelligence pipeline.

All HDFS paths, tier boundaries, chokepoint definitions, and shared
constants live here. Import from this module rather than redefining.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Literal, Union

# ---- Execution mode ----
#
# Preserve current behavior across the codebase:
# - If SC_LOCAL_MODE is explicitly set, respect it.
# - Otherwise, default to "local mode" when HADOOP_CONF_DIR is not set.
_SC_LOCAL_MODE = os.environ.get("SC_LOCAL_MODE")
LOCAL_MODE = (_SC_LOCAL_MODE == "1") if _SC_LOCAL_MODE is not None else (not os.environ.get("HADOOP_CONF_DIR"))

# ---- HDFS cluster base ----
HDFS_BASE = "hdfs://namenode:9000/supply-chain"

# ---- Paths: explicit local vs HDFS (matches pre-refactor spark_pipeline.py) ----
# Raw GDELT is always /opt/data/gdelt (mounted). Parquet outputs live under /opt/data/parquet/
# in local mode; model is /opt/data/model_rf_v1 (no parquet/ prefix). In HDFS mode, commodity/FRED
# use hdfs://.../opt/data/... paths.
if LOCAL_MODE:
    GDELT_RAW = "/opt/data/gdelt"
    COMMODITY_RAW = "/opt/data/commodities/commodity_prices.csv"
    FRED_RAW = "/opt/data/fred/fred_macro.csv"
    HDFS_GDELT_PARQUET = "/opt/data/parquet/gdelt_events"
    HDFS_FEATURES = "/opt/data/parquet/features"
    HDFS_HOT = "/opt/data/parquet/raw/hot"
    HDFS_WARM = "/opt/data/parquet/raw/warm"
    HDFS_COLD = "/opt/data/parquet/raw/cold"
    HDFS_MODEL = "/opt/data/model_rf_v1"
else:
    GDELT_RAW = "/opt/data/gdelt"
    COMMODITY_RAW = "hdfs://namenode:9000/opt/data/commodities/commodity_prices.csv"
    FRED_RAW = "hdfs://namenode:9000/opt/data/fred/fred_macro.csv"
    HDFS_GDELT_PARQUET = f"{HDFS_BASE}/gdelt_events"
    HDFS_FEATURES = f"{HDFS_BASE}/features"
    HDFS_HOT = f"{HDFS_BASE}/raw/hot"
    HDFS_WARM = f"{HDFS_BASE}/raw/warm"
    HDFS_COLD = f"{HDFS_BASE}/raw/cold"
    HDFS_MODEL = f"{HDFS_BASE}/model_rf_v1"

# DEPRECATED: replaced by get_tier_for_date(). Kept temporarily for backward compat.
# Remove once all callers have migrated.
HOT_YEAR_CUTOFF = 2023

# Rolling-window tier boundaries (in months, relative to as_of)
HOT_WINDOW_MONTHS = 18
WARM_WINDOW_MONTHS = 78

Tier = Literal["hot", "warm", "cold"]


def get_tier_for_date(event_date: Union[date, str], as_of: Union[date, str, None] = None) -> Tier:
    """
    Return the storage tier for an event based on its date relative to as_of.

    Tiers (rolling window from as_of):
      - hot:  event_date within the most recent HOT_WINDOW_MONTHS (default 18)
      - warm: older than hot, within WARM_WINDOW_MONTHS total (default 78)
      - cold: older than warm

    Args:
        event_date: The date of the event (date, or "YYYY-MM-DD" string).
        as_of: Reference date for the rolling window. Defaults to today.

    Returns:
        "hot" | "warm" | "cold"

    Raises:
        ValueError: if event_date is in the future relative to as_of.
    """
    if isinstance(event_date, str):
        event_date = date.fromisoformat(event_date)
    if as_of is None:
        as_of = date.today()
    elif isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)

    if event_date > as_of:
        raise ValueError(f"event_date {event_date} is after as_of {as_of}")

    # Approximate months as 30.44 days for boundary math.
    # We're tier-routing, not doing accounting — exact month arithmetic is overkill.
    age_days = (as_of - event_date).days
    hot_cutoff_days = int(HOT_WINDOW_MONTHS * 30.44)
    warm_cutoff_days = int(WARM_WINDOW_MONTHS * 30.44)

    if age_days <= hot_cutoff_days:
        return "hot"
    if age_days <= warm_cutoff_days:
        return "warm"
    return "cold"

# ---- Chokepoint bounding boxes ----
# (lat_min, lat_max, lon_min, lon_max) for each chokepoint.
# Copy values verbatim from current spark_pipeline.py.
CHOKEPOINTS = {
    "hormuz": (25.06, 28.06, 54.75, 57.75),
    "suez": (29.46, 31.46, 31.34, 33.34),
    "red_sea": (11.00, 17.00, 40.00, 46.00),
    "black_sea": (41.60, 47.60, 30.50, 36.50),
    "malacca": (0.50, 4.50, 99.50, 103.50),
    "panama": (8.58, 9.58, -80.18, -79.18),
    "taiwan": (23.00, 26.00, 118.00, 121.00),
    "chile": (-26.50, -20.50, -72.50, -66.50),
}

# ---- Warm tier projection ----
# Columns kept when demoting to warm tier. Verbatim from spark_pipeline.py.
WARM_COLUMNS = [
    "GLOBALEVENTID",
    "event_date",
    "Actor1Code",
    "Actor2Code",
    "EventCode",
    "EventRootCode",
    "GoldsteinScale",
    "NumMentions",
    "AvgTone",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "SOURCEURL",
]

