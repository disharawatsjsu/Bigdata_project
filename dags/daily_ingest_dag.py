"""
Daily GDELT ingest to HDFS.

Tasks:
  1. check_not_already_ingested — skip if HDFS partition already populated
  2. download_gdelt — fetch target day's GDELT export from data.gdeltproject.org
  3. ingest_to_hdfs — spark-submit run_tiered_ingest.py for the day
  4. validate_partition — confirm HDFS write succeeded with non-zero size
  5. cleanup_local — delete local CSV after successful HDFS landing

GDELT publishes with ~24h lag; we process (logical_date - 2 days).

Idempotency:
- download_day() skips already-downloaded files
- check_not_already_ingested skips already-populated HDFS partitions
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path

sys.path.insert(0, "/opt/airflow/scripts")

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from config import HDFS_BASE

# Airflow containers do not set HADOOP_CONF_DIR, so config.LOCAL_MODE would resolve to True
# if we imported HDFS_HOT directly. For partition existence checks we always want the HDFS path.
HDFS_HOT_HDFS = f"{HDFS_BASE}/raw/hot"


def _docker_run(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def check_not_already_ingested() -> str:
    """Skip the rest of the DAG if the hot-tier month partition already has data."""
    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    logical = ctx["logical_date"]
    if hasattr(logical, "in_timezone"):
        logical = logical.in_timezone("UTC")
    target = (logical - timedelta(days=2)).date()
    year, month = target.year, target.month
    partition = f"{HDFS_HOT_HDFS}/year={year}/month={month:02d}"

    proc = _docker_run(
        ["exec", "namenode", "hdfs", "dfs", "-du", "-s", partition],
    )
    if proc.returncode == 0 and proc.stdout.strip():
        first = proc.stdout.strip().split("\n")[0]
        try:
            size = int(first.split()[0])
        except (IndexError, ValueError):
            size = 0
        if size > 0:
            raise AirflowSkipException(
                f"Partition already populated ({size} bytes): {partition}"
            )

    target_str = target.isoformat()
    ctx["ti"].xcom_push(key="target_date", value=target_str)
    ctx["ti"].xcom_push(key="target_year", value=year)
    ctx["ti"].xcom_push(key="target_month", value=month)
    return target_str


def download_gdelt_for_date() -> str:
    """Download GDELT export for target_date directly via download_day()."""
    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    target_str = ctx["ti"].xcom_pull(
        task_ids="check_not_already_ingested", key="target_date"
    )
    if not target_str:
        raise ValueError("missing XCom target_date from check_not_already_ingested")

    target = date.fromisoformat(target_str)

    # Shared, host-mounted path: both airflow-* and spark-master see ./data as /opt/data.
    output_dir = "/opt/data/gdelt"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from download_gdelt import download_day

    csv_path = download_day(target, output_dir)
    print(f"Downloaded GDELT for {target_str}: {csv_path}")
    ctx["ti"].xcom_push(key="gdelt_csv_path", value=str(csv_path))
    return str(csv_path)


def validate_partition() -> None:
    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    ti = ctx["ti"]
    year = ti.xcom_pull(task_ids="check_not_already_ingested", key="target_year")
    month = ti.xcom_pull(task_ids="check_not_already_ingested", key="target_month")
    partition = f"{HDFS_HOT_HDFS}/year={year}/month={int(month):02d}"

    proc = _docker_run(
        ["exec", "namenode", "hdfs", "dfs", "-du", "-s", partition],
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"HDFS partition missing or unreadable after ingest: {partition}\n"
            f"{proc.stderr}"
        )
    size = int(proc.stdout.strip().split("\n")[0].split()[0])
    if size <= 0:
        raise RuntimeError(f"HDFS partition has size 0: {partition}")
    print(f"validate_partition OK: {partition} size={size} bytes")


def cleanup_local_csv() -> None:
    """Delete the day's GDELT CSV after successful HDFS write."""
    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    target_str = ctx["ti"].xcom_pull(
        task_ids="check_not_already_ingested", key="target_date"
    )
    if not target_str:
        raise ValueError("missing XCom target_date from check_not_already_ingested")

    ymd = target_str.replace("-", "")
    csv_path = f"/opt/data/gdelt/{ymd}.export.CSV"

    if os.path.exists(csv_path):
        size = os.path.getsize(csv_path)
        os.remove(csv_path)
        print(f"Removed local CSV: {csv_path} ({size:,} bytes freed)")
    else:
        print(f"Local CSV already gone: {csv_path} (skipping)")


with DAG(
    dag_id="daily_ingest_dag",
    description="Daily ingest of GDELT data to HDFS hot/warm tiers",
    schedule="0 6 * * *",
    start_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
    catchup=False,
    is_paused_upon_creation=True,
    default_args={
        "owner": "sc-intel",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=30),
    },
    tags=["ingest", "daily"],
) as dag:
    check_not_already_ingested_task = PythonOperator(
        task_id="check_not_already_ingested",
        python_callable=check_not_already_ingested,
    )

    download_gdelt = PythonOperator(
        task_id="download_gdelt",
        python_callable=download_gdelt_for_date,
    )

    ingest_to_hdfs = BashOperator(
        task_id="ingest_to_hdfs",
        bash_command="""
set -e
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_tiered_ingest.py \
  --date "{{ ti.xcom_pull(task_ids='check_not_already_ingested', key='target_date') }}"
""",
    )

    validate_partition_task = PythonOperator(
        task_id="validate_partition",
        python_callable=validate_partition,
    )

    cleanup_local = PythonOperator(
        task_id="cleanup_local",
        python_callable=cleanup_local_csv,
        trigger_rule="all_success",
    )

    (
        check_not_already_ingested_task
        >> download_gdelt
        >> ingest_to_hdfs
        >> validate_partition_task
        >> cleanup_local
    )
