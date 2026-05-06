#!/usr/bin/env python3
"""
Hive Table Definitions — creates external tables over HDFS Parquet data
so analysts can query with SQL.

This is the "additional Big Data tool not covered in class" requirement.

Usage (inside hive-server container):
    beeline -u jdbc:hive2://localhost:10000 -f hive_tables.sql

Or run this Python script which uses PyHive:
    python hive_setup.py

The script also includes example analytical queries that demonstrate
Hive's value in the pipeline (for the presentation/report).
"""

# SQL statements — can be run directly in Beeline or via PyHive
HIVE_DDL = """
-- ============================================================
-- Database
-- ============================================================
CREATE DATABASE IF NOT EXISTS supply_chain;
USE supply_chain;

-- ============================================================
-- External table over GDELT events Parquet on HDFS
-- Partitioned by year/month for efficient time-range queries
-- ============================================================
CREATE EXTERNAL TABLE IF NOT EXISTS gdelt_events (
    GLOBALEVENTID       INT,
    SQLDATE             INT,
    Actor1Code          STRING,
    Actor1Name          STRING,
    Actor1CountryCode   STRING,
    Actor2Code          STRING,
    Actor2Name          STRING,
    Actor2CountryCode   STRING,
    EventCode           STRING,
    EventBaseCode       STRING,
    EventRootCode       STRING,
    QuadClass           INT,
    GoldsteinScale      FLOAT,
    NumMentions         INT,
    NumSources          INT,
    NumArticles         INT,
    AvgTone             FLOAT,
    ActionGeo_FullName  STRING,
    ActionGeo_CountryCode STRING,
    ActionGeo_Lat       FLOAT,
    ActionGeo_Long      FLOAT,
    event_date          DATE
)
PARTITIONED BY (year INT, month INT)
STORED AS PARQUET
-- Hive LOCATION must be HDFS URI; not config-driven
LOCATION 'hdfs://namenode:9000/supply-chain/gdelt_events';

-- Auto-discover partitions from HDFS directory structure
MSCK REPAIR TABLE gdelt_events;

-- ============================================================
-- External table over feature engineering output
-- ============================================================
CREATE EXTERNAL TABLE IF NOT EXISTS features (
    event_date          DATE,
    chokepoint          STRING,
    event_count         BIGINT,
    avg_goldstein       FLOAT,
    avg_tone            FLOAT,
    total_mentions      BIGINT,
    conflict_ratio      FLOAT,
    event_count_7d      FLOAT,
    avg_goldstein_7d    FLOAT,
    avg_tone_7d         FLOAT,
    total_mentions_7d   FLOAT,
    conflict_ratio_7d   FLOAT,
    event_count_30d     FLOAT,
    avg_goldstein_30d   FLOAT,
    avg_tone_30d        FLOAT,
    total_mentions_30d  FLOAT,
    conflict_ratio_30d  FLOAT,
    return_5d           FLOAT,
    return_20d          FLOAT,
    volatility_20d      FLOAT,
    treasury_10y        FLOAT,
    usd_index           FLOAT,
    label               INT
)
STORED AS PARQUET
-- Hive LOCATION must be HDFS URI; not config-driven
LOCATION 'hdfs://namenode:9000/supply-chain/features';
"""

# Analytical queries for presentation / report
HIVE_QUERIES = {
    "monthly_event_summary": """
    -- Monthly event counts by chokepoint (shows seasonal/geopolitical patterns)
    SELECT year, month, ActionGeo_CountryCode,
           COUNT(*) as event_count,
           AVG(GoldsteinScale) as avg_conflict,
           SUM(NumMentions) as total_mentions
    FROM supply_chain.gdelt_events
    WHERE EventRootCode IN ('14','17','18','19','20')
    GROUP BY year, month, ActionGeo_CountryCode
    ORDER BY year, month, event_count DESC
    LIMIT 50;
    """,

    "top_conflict_regions": """
    -- Which regions had the most intense conflict events?
    SELECT ActionGeo_FullName,
           COUNT(*) as events,
           AVG(GoldsteinScale) as avg_goldstein,
           MIN(GoldsteinScale) as worst_event,
           SUM(NumMentions) as media_attention
    FROM supply_chain.gdelt_events
    WHERE EventRootCode IN ('18','19','20')  -- assault, fight, mass violence
      AND GoldsteinScale < -5
    GROUP BY ActionGeo_FullName
    HAVING COUNT(*) >= 10
    ORDER BY avg_goldstein ASC
    LIMIT 20;
    """,

    "red_sea_crisis_timeline": """
    -- Red Sea / Houthi crisis escalation (the case study)
    SELECT SQLDATE,
           COUNT(*) as daily_events,
           AVG(GoldsteinScale) as avg_goldstein,
           SUM(NumMentions) as mentions
    FROM supply_chain.gdelt_events
    WHERE ActionGeo_Lat BETWEEN 11 AND 17
      AND ActionGeo_Long BETWEEN 40 AND 46
      AND EventRootCode IN ('17','18','19','20')
      AND year = 2024
    GROUP BY SQLDATE
    ORDER BY SQLDATE;
    """,

    "shock_prediction_accuracy": """
    -- Model performance by chokepoint (from features table)
    SELECT chokepoint,
           label,
           COUNT(*) as count,
           AVG(conflict_ratio_7d) as avg_conflict,
           AVG(volatility_20d) as avg_vol
    FROM supply_chain.features
    WHERE event_date >= '2024-07-01'  -- test set
    GROUP BY chokepoint, label
    ORDER BY chokepoint, label;
    """,

    "feature_correlations": """
    -- Which features correlate with price shocks?
    SELECT label,
           AVG(event_count_7d) as avg_events,
           AVG(avg_goldstein_7d) as avg_goldstein,
           AVG(conflict_ratio_7d) as avg_conflict,
           AVG(total_mentions_7d) as avg_mentions,
           AVG(volatility_20d) as avg_vol,
           COUNT(*) as n
    FROM supply_chain.features
    GROUP BY label
    ORDER BY label;
    """,
}


def print_sql_file():
    """Print all DDL + queries as a single .sql file for Beeline."""
    print("-- " + "=" * 60)
    print("-- Supply Chain Disruption Intelligence — Hive Setup")
    print("-- Run: beeline -u jdbc:hive2://localhost:10000 -f hive_tables.sql")
    print("-- " + "=" * 60)
    print()
    print(HIVE_DDL)
    print()
    print("-- " + "=" * 60)
    print("-- ANALYTICAL QUERIES")
    print("-- " + "=" * 60)
    for name, query in HIVE_QUERIES.items():
        print(f"\n-- === {name} ===")
        print(query)


def run_with_pyhive():
    """Run DDL via PyHive (alternative to Beeline)."""
    try:
        from pyhive import hive
    except ImportError:
        print("PyHive not installed. Run: pip install pyhive")
        print("Alternatively, use the SQL output with Beeline.")
        print("\nPrinting SQL instead:\n")
        print_sql_file()
        return

    conn = hive.connect(host="hive-server", port=10000)
    cursor = conn.cursor()

    for statement in HIVE_DDL.split(";"):
        stmt = statement.strip()
        if stmt and not stmt.startswith("--"):
            print(f"  Executing: {stmt[:80]}...")
            cursor.execute(stmt)

    print("\nHive tables created. Running sample query...")
    cursor.execute(HIVE_QUERIES["feature_correlations"])
    for row in cursor.fetchall():
        print(f"  {row}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    import sys
    if "--sql" in sys.argv:
        print_sql_file()
    else:
        run_with_pyhive()
