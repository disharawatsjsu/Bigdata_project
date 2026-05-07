"""
Feature engineering refresh (baseline v1).

After historical_backfill_dag ingests GDELT to HDFS hot/warm tiers, this DAG runs a
single Spark job to build training-ready features and writes:
  {HDFS_BASE}/features/baseline/

Manual trigger only for tonight's pipeline; schedule can be added later.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/opt/airflow/scripts")

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from config import HDFS_BASE

# Airflow containers do not set HADOOP_CONF_DIR, so config.LOCAL_MODE would resolve to True
# if we imported HDFS_FEATURES directly. For validation we always want the HDFS path.
FEATURES_HDFS = f"{HDFS_BASE}/features/baseline"


def _docker_run(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def validate_features_output() -> None:
    proc = _docker_run(["exec", "namenode", "hdfs", "dfs", "-du", "-s", FEATURES_HDFS])
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"Features output missing or unreadable: {FEATURES_HDFS}\n{proc.stderr}"
        )
    size = int(proc.stdout.strip().split("\n")[0].split()[0])
    if size <= 0:
        raise RuntimeError(f"Features output has size 0: {FEATURES_HDFS}")
    print(f"[feature_refresh] OK: {FEATURES_HDFS} size={size} bytes")


with DAG(
    dag_id="feature_refresh_dag",
    description="Build baseline training features from tiered GDELT data",
    schedule=None,
    start_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
    catchup=False,
    is_paused_upon_creation=True,
    default_args={
        "owner": "sc-intel",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=10),
    },
    tags=["features"],
) as dag:
    refresh_features = BashOperator(
        task_id="refresh_features",
        bash_command="""
set -e
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_feature_refresh.py
""",
    )

    validate_output = PythonOperator(
        task_id="validate_output",
        python_callable=validate_features_output,
    )

    refresh_features >> validate_output

