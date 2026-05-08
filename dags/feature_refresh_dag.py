"""Daily end-to-end feature refresh.

Flow:
  GDELT download -> tiered HDFS ingest -> source date-range discovery ->
  market download -> FRED download -> market/FRED HDFS upload ->
  Spark feature pipeline -> baseline validation.

The ML-consumable Spark output is:
  {HDFS_BASE}/features/baseline/{commodity}/
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/opt/airflow/scripts")

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

from config import HDFS_BASE

# Airflow containers do not set HADOOP_CONF_DIR, so config.LOCAL_MODE would resolve to True
# if we imported HDFS_FEATURES directly. For validation we always want the HDFS path.
FEATURES_HDFS = f"{HDFS_BASE}/features/baseline"
HDFS_HOT = f"{HDFS_BASE}/raw/hot"
HDFS_WARM = f"{HDFS_BASE}/raw/warm"
JAVA17_HOME = "/usr/lib/jvm/java-17-openjdk-arm64"
SPARK_JAVA17_ENV = {
    "JAVA_HOME": JAVA17_HOME,
    "PATH": f"{JAVA17_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}
COMMODITIES = ["brent", "wti", "copper", "gold", "wheat", "soybeans"]


def _docker_run(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _target_date() -> str:
    from airflow.operators.python import get_current_context

    logical = get_current_context()["logical_date"]
    if hasattr(logical, "in_timezone"):
        logical = logical.in_timezone("UTC")
    return (logical - timedelta(days=2)).date().isoformat()


def resolve_target_date() -> str:
    from airflow.operators.python import get_current_context

    target = _target_date()
    get_current_context()["ti"].xcom_push(key="target_date", value=target)
    print(f"[feature_refresh] target GDELT date: {target}")
    return target


def check_hdfs_partition_changed() -> str:
    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    target = datetime.fromisoformat(ctx["ti"].xcom_pull(task_ids="resolve_target_date", key="target_date")).date()
    partition = f"{HDFS_HOT}/year={target.year}/month={target.month:02d}/day={target.day:02d}"
    proc = _docker_run(["exec", "namenode", "hdfs", "dfs", "-du", "-s", partition])
    if proc.returncode == 0 and proc.stdout.strip():
        size = int(proc.stdout.strip().split()[0])
        if size > 0:
            raise AirflowSkipException(f"GDELT HDFS partition already populated: {partition} ({size} bytes)")
    print(f"[feature_refresh] HDFS partition needs ingest: {partition}")
    return partition


def determine_source_date_range() -> dict[str, str]:
    from airflow.operators.python import get_current_context

    script = f"""
import json
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.appName("source-date-range").getOrCreate()
dfs = []
for path in {json.dumps([HDFS_HOT, HDFS_WARM])}:
    try:
        dfs.append(spark.read.parquet(path).select("event_date"))
    except Exception:
        pass
if not dfs:
    raise RuntimeError("No GDELT hot/warm parquet paths are readable.")
df = dfs[0]
for other in dfs[1:]:
    df = df.unionByName(other)
row = df.agg(F.min("event_date").alias("min_date"), F.max("event_date").alias("max_date"), F.count("*").alias("rows")).first()
print(json.dumps({{"start": row["min_date"].isoformat(), "end": row["max_date"].isoformat(), "rows": row["rows"]}}))
spark.stop()
"""
    proc = _docker_run(
        [
            "exec",
            "spark-master",
            "bash",
            "-lc",
            f"export JAVA_HOME={JAVA17_HOME}; export PATH=$JAVA_HOME/bin:$PATH; python - <<'PY'\n{script}\nPY",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Could not determine GDELT source date range:\n{proc.stderr}\n{proc.stdout}")
    json_lines = [line for line in proc.stdout.strip().splitlines() if line.startswith("{") and line.endswith("}")]
    if not json_lines:
        raise RuntimeError(f"Could not parse source date range JSON from:\n{proc.stdout}")
    payload = json.loads(json_lines[-1])
    ti = get_current_context()["ti"]
    ti.xcom_push(key="source_start", value=payload["start"])
    ti.xcom_push(key="source_end", value=payload["end"])
    print(
        "[feature_refresh] source date range: "
        f"{payload['start']} -> {payload['end']} ({payload['rows']:,} rows)"
    )
    return payload


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
    for commodity in COMMODITIES:
        summary = _docker_run(
            ["exec", "namenode", "hdfs", "dfs", "-text", f"{FEATURES_HDFS}/{commodity}/_summary.json"]
        )
        if summary.returncode == 0 and summary.stdout.strip():
            payload = json.loads(summary.stdout)
            print(
                f"[feature_refresh] {commodity}: "
                f"rows={payload['row_count']:,}, "
                f"date_range={payload['date_min']}->{payload['date_max']}, "
                f"output={FEATURES_HDFS}/{commodity}"
            )


with DAG(
    dag_id="feature_refresh_dag",
    description="Download data, ingest GDELT, and build baseline feature parquet per commodity",
    schedule="0 6 * * *",
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
    resolve_target_date_task = PythonOperator(
        task_id="resolve_target_date",
        python_callable=resolve_target_date,
    )

    check_hdfs_partition_changed_task = PythonOperator(
        task_id="check_hdfs_partition_changed",
        python_callable=check_hdfs_partition_changed,
    )

    download_gdelt = BashOperator(
        task_id="download_gdelt",
        bash_command="""
set -e
python /opt/airflow/scripts/download_gdelt.py \
  --start "{{ ti.xcom_pull(task_ids='resolve_target_date', key='target_date') }}" \
  --end "{{ ti.xcom_pull(task_ids='resolve_target_date', key='target_date') }}" \
  --output /opt/data/gdelt
""",
    )

    wait_for_gdelt_source = FileSensor(
        task_id="wait_for_gdelt_source",
        filepath="/opt/data/gdelt/{{ ti.xcom_pull(task_ids='resolve_target_date', key='target_date').replace('-', '') }}.export.CSV",
        poke_interval=60,
        timeout=60 * 30,
        mode="reschedule",
    )

    ingest_gdelt_to_hdfs = BashOperator(
        task_id="ingest_gdelt_to_hdfs",
        env=SPARK_JAVA17_ENV,
        bash_command="""
set -e
docker exec spark-master bash -lc 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64; export PATH=$JAVA_HOME/bin:$PATH; /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_tiered_ingest.py \
  --date "{{ ti.xcom_pull(task_ids='resolve_target_date', key='target_date') }}"'
""",
    )

    determine_source_date_range_task = PythonOperator(
        task_id="determine_source_date_range",
        python_callable=determine_source_date_range,
    )

    download_markets = BashOperator(
        task_id="download_markets",
        bash_command="""
set -e
python /opt/airflow/scripts/download_markets.py \
  --start "{{ ti.xcom_pull(task_ids='determine_source_date_range', key='source_start') }}" \
  --end "{{ ti.xcom_pull(task_ids='determine_source_date_range', key='source_end') }}" \
  --output /opt/data \
  --only commodities
""",
    )

    download_fred = BashOperator(
        task_id="download_fred",
        bash_command="""
set -e
python /opt/airflow/scripts/download_markets.py \
  --start "{{ ti.xcom_pull(task_ids='determine_source_date_range', key='source_start') }}" \
  --end "{{ ti.xcom_pull(task_ids='determine_source_date_range', key='source_end') }}" \
  --output /opt/data \
  --only fred
""",
    )

    upload_market_data_to_hdfs = BashOperator(
        task_id="upload_market_data_to_hdfs",
        bash_command="""
set -e
docker cp /opt/data/commodities namenode:/tmp/commodities
docker cp /opt/data/fred namenode:/tmp/fred
docker exec namenode bash -lc 'hdfs dfs -mkdir -p /opt/data/commodities /opt/data/fred && \
  hdfs dfs -put -f /tmp/commodities/* /opt/data/commodities/ && \
  hdfs dfs -put -f /tmp/fred/* /opt/data/fred/'
echo "[feature_refresh] uploaded market/FRED data for {{ ti.xcom_pull(task_ids='determine_source_date_range', key='source_start') }} -> {{ ti.xcom_pull(task_ids='determine_source_date_range', key='source_end') }}"
""",
    )

    refresh_features = BashOperator(
        task_id="refresh_features",
        env=SPARK_JAVA17_ENV,
        bash_command="""
set -e
docker exec spark-master bash -lc 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64; export PATH=$JAVA_HOME/bin:$PATH; /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_feature_refresh.py'
""",
    )

    validate_output = PythonOperator(
        task_id="validate_output",
        python_callable=validate_features_output,
    )

    (
        resolve_target_date_task
        >> check_hdfs_partition_changed_task
        >> download_gdelt
        >> wait_for_gdelt_source
        >> ingest_gdelt_to_hdfs
        >> determine_source_date_range_task
    )
    determine_source_date_range_task >> [download_markets, download_fred] >> upload_market_data_to_hdfs
    upload_market_data_to_hdfs >> refresh_features >> validate_output

