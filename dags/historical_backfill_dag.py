"""
Historical GDELT backfill to HDFS (manual trigger only).

This DAG expands a user-provided date range into one mapped task per day, where each
day performs:
  - optional idempotency check (skip if HDFS day partition already populated)
  - download_day() to /opt/data/gdelt (host-mounted into spark containers)
  - spark-submit run_tiered_ingest.py --date YYYY-MM-DD via docker exec spark-master
  - validate HDFS day partition has non-zero size
  - optional local cleanup of the downloaded CSV

Operational notes:
  - Some historical dates legitimately 404 on GDELT. These are treated as a non-fatal
    "skipped" outcome and counted in the summary.
  - Disk usage is bounded by concurrency: up to N days of CSVs may exist at once, where
    N is the mapped task's max_active_tis_per_dag. With delete_local_after=True, files
    are removed once HDFS validation succeeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/opt/airflow/scripts")

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param

from config import HDFS_BASE, get_tier_for_date

# Airflow containers do not set HADOOP_CONF_DIR, so config.LOCAL_MODE would resolve to True
# if we imported HDFS_HOT directly. For existence checks we always want the HDFS path.
HDFS_HOT_HDFS = f"{HDFS_BASE}/raw/hot"
HDFS_WARM_HDFS = f"{HDFS_BASE}/raw/warm"


def _docker_run(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _tier_for_day(d: date) -> str:
    # Tier routing matches ingest_gdelt_tiered(): representative date is month start.
    rep_date = date(d.year, d.month, 1)
    tier = get_tier_for_date(rep_date)
    # ingest_gdelt_tiered writes cold data to warm path until cold aggregates land.
    return "warm" if tier == "cold" else tier


def _hdfs_partition_for_day(d: date) -> str:
    tier = _tier_for_day(d)
    base = HDFS_HOT_HDFS if tier == "hot" else HDFS_WARM_HDFS
    return f"{base}/year={d.year}/month={d.month:02d}/day={d.day:02d}"


def _hdfs_du_bytes(path: str) -> Optional[int]:
    proc = _docker_run(["exec", "namenode", "hdfs", "dfs", "-du", "-s", path])
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    first = proc.stdout.strip().split("\n")[0]
    try:
        return int(first.split()[0])
    except (IndexError, ValueError):
        return None


def _backfill_one_day_impl(target_date: str, *, delete_local_after: bool) -> dict:
    """Download + ingest + validate one day. Raises AirflowSkipException when already ingested."""
    d = date.fromisoformat(target_date)
    partition = _hdfs_partition_for_day(d)

    size = _hdfs_du_bytes(partition)
    if size is not None and size > 0:
        raise AirflowSkipException(f"Already ingested (size={size} bytes): {partition}")

    # Shared, host-mounted path: both airflow-* and spark-master see ./data as /opt/data.
    output_dir = "/opt/data/gdelt"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from download_gdelt import download_day

    try:
        csv_path = download_day(d, output_dir)
    except FileNotFoundError as e:
        # Historical gaps are expected; don't fail the task.
        print(f"[backfill_one_day] WARN: {e}")
        return {"date": target_date, "status": "no_file_404"}

    # Ingest via the same spark-submit entrypoint as the daily DAG.
    cmd = [
        "exec",
        "spark-master",
        "/opt/spark/bin/spark-submit",
        "--master",
        "spark://spark-master:7077",
        "/opt/scripts/run_tiered_ingest.py",
        "--date",
        target_date,
    ]
    proc = _docker_run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            "spark-submit failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    # Validate HDFS write.
    size2 = _hdfs_du_bytes(partition)
    if size2 is None:
        raise RuntimeError(f"HDFS partition missing or unreadable after ingest: {partition}")
    if size2 <= 0:
        raise RuntimeError(f"HDFS partition has size 0: {partition}")
    print(f"[backfill_one_day] validate_partition OK: {partition} size={size2} bytes")

    if delete_local_after:
        # download_day() always writes YYYYMMDD.export.CSV
        ymd = target_date.replace("-", "")
        local_csv = f"/opt/data/gdelt/{ymd}.export.CSV"
        if os.path.exists(local_csv):
            freed = os.path.getsize(local_csv)
            os.remove(local_csv)
            print(f"[backfill_one_day] Removed local CSV: {local_csv} ({freed:,} bytes freed)")

    mb = csv_path.stat().st_size / 1e6 if csv_path.exists() else None
    return {"date": target_date, "status": "ok", "local_mb": mb, "hdfs_bytes": size2}


@task
def expand_date_range_chunks() -> list[dict]:
    """Return a list of date-range chunks to stay under Airflow map limits.

    Airflow dynamic task mapping has a default max_map_length=1024. A daily backfill
    for multi-year ranges (e.g. 2018–2020 = 1096 days) will exceed that limit.

    We therefore map over *chunks* and do the per-day work inside each mapped task.
    """
    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    conf = (ctx["dag_run"].conf or {}) if ctx.get("dag_run") else {}
    start = conf.get("start_date", ctx["params"]["start_date"])
    end = conf.get("end_date", ctx["params"]["end_date"])

    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    if e < s:
        raise ValueError(f"end_date must be >= start_date (got {start}..{end})")

    chunk_days = int(conf.get("chunk_days", ctx["params"]["chunk_days"]))
    if chunk_days <= 0:
        raise ValueError(f"chunk_days must be > 0 (got {chunk_days})")

    out: list[dict] = []
    cur = s
    while cur <= e:
        chunk_end = min(e, cur + timedelta(days=chunk_days - 1))
        out.append({"start_date": cur.isoformat(), "end_date": chunk_end.isoformat()})
        cur = chunk_end + timedelta(days=1)
    return out


@task(
    retries=2,
    retry_delay=timedelta(minutes=2),
    retry_exponential_backoff=True,
    max_retry_delay=timedelta(minutes=5),
)
def backfill_one_day(target_date: str, delete_local_after: bool = True) -> dict:
    """Download + ingest + validate one day. Returns a small summary dict.

    Idempotency: skips if the HDFS day partition already has data.
    """
    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    conf = (ctx["dag_run"].conf or {}) if ctx.get("dag_run") else {}
    delete_local_after = bool(conf.get("delete_local_after", ctx["params"]["delete_local_after"]))
    return _backfill_one_day_impl(target_date, delete_local_after=delete_local_after)


@task(
    retries=1,
    retry_delay=timedelta(minutes=2),
)
def backfill_chunk(date_range: dict) -> dict:
    """Backfill a chunk of days inside a single task.

    This avoids Airflow dynamic mapping limits for multi-year ranges.
    Returns a small summary dict (safe for XCom).
    """
    start_s = date_range["start_date"]
    end_s = date_range["end_date"]
    s = date.fromisoformat(start_s)
    e = date.fromisoformat(end_s)
    if e < s:
        raise ValueError(f"chunk end_date must be >= start_date (got {start_s}..{end_s})")

    from airflow.operators.python import get_current_context

    ctx = get_current_context()
    conf = (ctx["dag_run"].conf or {}) if ctx.get("dag_run") else {}
    delete_local_after = bool(conf.get("delete_local_after", ctx["params"]["delete_local_after"]))

    counts = {"ok": 0, "skipped": 0, "no_file_404": 0, "failed": 0}
    cur = s
    while cur <= e:
        ds = cur.isoformat()
        try:
            out = _backfill_one_day_impl(ds, delete_local_after=delete_local_after)
            counts[out.get("status", "ok")] = counts.get(out.get("status", "ok"), 0) + 1
        except AirflowSkipException:
            counts["skipped"] += 1
        except Exception as ex:
            counts["failed"] += 1
            print(f"[backfill_chunk] ERROR day={ds}: {ex}")
        cur += timedelta(days=1)

    print(f"[backfill_chunk] done {start_s}..{end_s}: {counts}")
    return {"start_date": start_s, "end_date": end_s, **counts}


@task(trigger_rule="all_done")
def summarize_backfill() -> None:
    """Log totals and fail the DAG if failures exceed max_failures.

    We count mapped task states from the metadatabase so we still summarize even if
    some mapped tasks fail (and therefore never return XCom results).
    """
    from airflow.operators.python import get_current_context
    from airflow.utils.state import State
    from airflow.models.taskinstance import TaskInstance
    from airflow.utils.session import create_session

    ctx = get_current_context()
    dag_run = ctx["dag_run"]
    conf = (dag_run.conf or {}) if dag_run else {}
    max_failures = int(conf.get("max_failures", ctx["params"]["max_failures"]))

    with create_session() as session:
        tis = (
            session.query(TaskInstance)
            .filter(TaskInstance.dag_id == dag_run.dag_id)
            .filter(TaskInstance.run_id == dag_run.run_id)
            .filter(TaskInstance.task_id == "backfill_chunk")
            .all()
        )

    counts: dict[str, int] = {}
    for ti in tis:
        counts[ti.state or "none"] = counts.get(ti.state or "none", 0) + 1

    n_success = counts.get(State.SUCCESS, 0)
    n_skipped = counts.get(State.SKIPPED, 0)
    n_failed = counts.get(State.FAILED, 0) + counts.get(State.UPSTREAM_FAILED, 0)

    total = len(tis)
    print(
        "[summarize_backfill] done: "
        f"total_days={total} success={n_success} skipped={n_skipped} failed={n_failed} "
        f"(max_failures={max_failures})"
    )
    print(f"[summarize_backfill] raw_state_counts={counts}")

    if n_failed > int(max_failures):
        raise RuntimeError(
            f"Too many failed days: {n_failed} > max_failures={max_failures}. "
            "See mapped task logs for details."
        )


with DAG(
    dag_id="historical_backfill_dag",
    description="Manual backfill of a historical date range of GDELT into HDFS",
    schedule=None,
    start_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
    catchup=False,
    is_paused_upon_creation=True,
    default_args={
        "owner": "sc-intel",
    },
    params={
        "start_date": Param("2024-01-01", type="string", format="date"),
        "end_date": Param("2024-01-31", type="string", format="date"),
        "chunk_days": Param(
            30,
            type="integer",
            description="Days per mapped task (keeps mapping under Airflow limits)",
        ),
        "max_failures": Param(
            5,
            type="integer",
            description="Fail the DAG if more than this many days fail",
        ),
        "delete_local_after": Param(
            True,
            type="boolean",
            description="Delete local CSV after a successful HDFS write",
        ),
    },
    tags=["ingest", "backfill"],
) as dag:
    chunks = expand_date_range_chunks()

    backfill_limited = backfill_chunk.override(max_active_tis_per_dag=4)
    results = backfill_limited.expand(date_range=chunks)
    summary = summarize_backfill()
    results >> summary

