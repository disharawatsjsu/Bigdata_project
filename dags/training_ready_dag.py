"""
Create ML training-ready dataset from baseline features.

This DAG runs a single Spark job that reads:
  {HDFS_BASE}/features/baseline/

and writes:
  {HDFS_BASE}/features/training_ready/

Manual trigger only for tonight's pipeline; scheduling/chaining can be added later.
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

TRAINING_READY_HDFS = f"{HDFS_BASE}/features/training_ready"


def _docker_run(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def validate_training_ready_output() -> None:
    proc = _docker_run(["exec", "namenode", "hdfs", "dfs", "-du", "-s", TRAINING_READY_HDFS])
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"training_ready output missing or unreadable: {TRAINING_READY_HDFS}\n{proc.stderr}"
        )
    size = int(proc.stdout.strip().split("\n")[0].split()[0])
    if size <= 0:
        raise RuntimeError(f"training_ready output has size 0: {TRAINING_READY_HDFS}")
    print(f"[training_ready] OK: {TRAINING_READY_HDFS} size={size} bytes")


with DAG(
    dag_id="training_ready_dag",
    description="Transform baseline features into training_ready with time splits",
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
    tags=["features", "ml"],
) as dag:
    build_training_ready = BashOperator(
        task_id="build_training_ready",
        bash_command="""
set -e
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_training_ready.py
""",
    )

    validate_output = PythonOperator(
        task_id="validate_output",
        python_callable=validate_training_ready_output,
    )

    build_training_ready >> validate_output

