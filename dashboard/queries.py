"""Hive access and in-memory data preparation for the Dash dashboard."""

from __future__ import annotations

import os
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import requests

TABLES = {
    "trigger_log": "trigger_log",
    "predictions": "predictions",
    "shap_attribution": "shap_attribution",
    "event_study": "event_study",
    "market_prices": "market_prices",
    "predictions_v2": "predictions_v2",
    "shap_attribution_v2": "shap_attribution_v2",
    "regression_metrics": "regression_metrics",
}

SHAP_LOCAL_TARGETS = ("abs_return_5d_fwd", "abs_return_20d_fwd")
COMMODITIES = ("brent", "wti", "copper", "gold", "wheat", "soybeans")


@dataclass(frozen=True)
class HiveConfig:
    host: str
    port: int
    user: str
    database: str
    auth: str

    @classmethod
    def from_env(cls) -> "HiveConfig":
        raw_host = os.environ.get("HIVE_HOST", "localhost").strip()
        raw_port = os.environ.get("HIVE_PORT", "10000").strip()
        user = os.environ.get("HIVE_USER", "root").strip()
        database = os.environ.get("HIVE_DATABASE", "supply_chain").strip()
        auth = os.environ.get("HIVE_AUTH", "NOSASL").strip()

        parsed = urlparse(raw_host if "://" in raw_host else f"//{raw_host}")
        host = parsed.hostname or raw_host
        port = parsed.port or int(raw_port)
        return cls(host=host, port=port, user=user, database=database, auth=auth)


def _fetch_table(cursor, database: str, table_name: str) -> pd.DataFrame:
    cursor.execute(f"SELECT * FROM {database}.{table_name}")
    rows = cursor.fetchall()
    columns = [col[0].split(".")[-1] for col in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def load_all_tables() -> tuple[dict[str, pd.DataFrame], str, str | None]:
    """Load all analytics tables into memory.

    Returns `(tables, timestamp, error)`. On failure, `tables` is empty and
    `error` contains the connection/query error for display in the dashboard.
    """
    mode = os.environ.get("DASH_DATA_MODE", "auto").strip().lower()
    if mode == "webhdfs":
        return _load_all_tables_webhdfs()

    hive_result = _load_all_tables_hive()
    if hive_result[2] is None or mode == "hive":
        return hive_result
    if os.environ.get("HDFS_WEBHDFS_URL"):
        tables, timestamp, error = _load_all_tables_webhdfs()
        if error is None:
            return tables, timestamp, None
        return {}, timestamp, f"Hive failed ({hive_result[2]}); WebHDFS fallback failed ({error})"
    return hive_result


def _load_all_tables_hive() -> tuple[dict[str, pd.DataFrame], str, str | None]:
    try:
        from pyhive import hive

        config = HiveConfig.from_env()
        conn = hive.connect(
            host=config.host,
            port=config.port,
            username=config.user,
            database=config.database,
            auth=config.auth,
        )
        cursor = conn.cursor()
        tables = {
            name: _fetch_table(cursor, config.database, table_name)
            for name, table_name in TABLES.items()
        }
        cursor.close()
        conn.close()
        _normalize_tables(tables)
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return tables, timestamp, None
    except Exception as exc:
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return {}, timestamp, f"{type(exc).__name__}: {exc}"


def _webhdfs_url(hdfs_path: str, op: str, **params) -> str:
    base = os.environ.get("HDFS_WEBHDFS_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("HDFS_WEBHDFS_URL is required for WebHDFS data mode.")
    query = {"op": op, "user.name": os.environ.get("HDFS_WEBHDFS_USER", "root"), **params}
    query_string = "&".join(f"{quote(str(k))}={quote(str(v))}" for k, v in query.items())
    return f"{base}/webhdfs/v1{quote(hdfs_path, safe='/')}?{query_string}"


def _webhdfs_json(hdfs_path: str, op: str) -> dict:
    headers = {"ngrok-skip-browser-warning": "true"}
    resp = requests.get(_webhdfs_url(hdfs_path, op), headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _webhdfs_bytes(hdfs_path: str) -> bytes:
    headers = {"ngrok-skip-browser-warning": "true"}
    resp = requests.get(_webhdfs_url(hdfs_path, "OPEN"), headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content


def _read_webhdfs_parquet_dir(hdfs_dir: str) -> pd.DataFrame:
    listing = _webhdfs_json(hdfs_dir, "LISTSTATUS")
    statuses = listing.get("FileStatuses", {}).get("FileStatus", [])
    parquet_paths = sorted(
        f"{hdfs_dir}/{s['pathSuffix']}"
        for s in statuses
        if s.get("type") == "FILE"
        and s.get("pathSuffix", "").endswith(".parquet")
        and int(s.get("length", 0)) > 0
    )
    for status in statuses:
        if status.get("type") != "DIRECTORY":
            continue
        nested_dir = f"{hdfs_dir}/{status['pathSuffix']}"
        nested_listing = _webhdfs_json(nested_dir, "LISTSTATUS")
        nested_statuses = nested_listing.get("FileStatuses", {}).get("FileStatus", [])
        parquet_paths.extend(
            f"{nested_dir}/{s['pathSuffix']}"
            for s in nested_statuses
            if s.get("type") == "FILE"
            and s.get("pathSuffix", "").endswith(".parquet")
            and int(s.get("length", 0)) > 0
        )
    if not parquet_paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(BytesIO(_webhdfs_bytes(path))) for path in sorted(parquet_paths)]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _read_webhdfs_market_prices() -> pd.DataFrame:
    hdfs_path = os.environ.get(
        "HDFS_MARKET_PRICES_PATH",
        "/opt/data/commodities_hive/commodity_prices.csv",
    )
    frame = pd.read_csv(BytesIO(_webhdfs_bytes(hdfs_path)))
    return frame.rename(columns={"date": "price_date", "close": "close_price"})


def _load_all_tables_webhdfs() -> tuple[dict[str, pd.DataFrame], str, str | None]:
    try:
        base = os.environ.get("HDFS_ANALYTICS_BASE", "/supply-chain/analytics").rstrip("/")
        tables = {
            name: _read_webhdfs_parquet_dir(f"{base}/{table_name}")
            for name, table_name in TABLES.items()
            if name != "market_prices"
        }
        tables["market_prices"] = _read_webhdfs_market_prices()
        _normalize_tables(tables)
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return tables, timestamp, None
    except Exception as exc:
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return {}, timestamp, f"{type(exc).__name__}: {exc}"


def _normalize_tables(tables: dict[str, pd.DataFrame]) -> None:
    for table_name in ("trigger_log", "predictions", "predictions_v2"):
        if table_name in tables and "event_date" in tables[table_name].columns:
            tables[table_name]["event_date"] = pd.to_datetime(
                tables[table_name]["event_date"],
                errors="coerce",
            )
    if "market_prices" in tables and "price_date" in tables["market_prices"].columns:
        tables["market_prices"]["price_date"] = pd.to_datetime(
            tables["market_prices"]["price_date"],
            errors="coerce",
        )

    numeric_columns = {
        "trigger_log": ["event_count_1d", "avg_goldstein_1d"],
        "shap_attribution": ["mean_abs_shap", "normalized_share"],
        "shap_attribution_v2": ["mean_abs_shap", "normalized_share"],
        "predictions": ["predicted_proba", "predicted_label", "actual_label"],
        "predictions_v2": ["predicted_value", "actual_value", "naive_value"],
        "market_prices": ["close_price"],
        "regression_metrics": [
            "test_r2",
            "test_mae",
            "naive_mae",
            "improvement_pct",
            "spearman",
        ],
        "event_study": [
            "lift_count_trigger",
            "pval_count_trigger",
            "n_count_trigger_days",
            "lift_goldstein_trigger",
            "pval_goldstein_trigger",
            "n_goldstein_trigger_days",
        ],
    }
    for table_name, columns in numeric_columns.items():
        if table_name not in tables:
            continue
        for col in columns:
            if col in tables[table_name].columns:
                tables[table_name][col] = pd.to_numeric(tables[table_name][col], errors="coerce")


def read_local_shap(commodity: str) -> tuple[pd.DataFrame | None, str | None]:
    """Read optional local SHAP parquet for a commodity.

    `SHAP_LOCAL_PATH` may point to a directory containing `{commodity}.parquet`
    files or `{commodity}` parquet directories.
    """
    base = Path(os.environ.get("SHAP_LOCAL_PATH", "/supply-chain/analytics/shap_local"))
    candidates = [base / f"{commodity}.parquet", base / commodity]
    for candidate in candidates:
        if candidate.exists():
            try:
                df = pd.read_parquet(candidate)
                if "event_date" in df.columns:
                    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
                return df, None
            except Exception as exc:
                return None, f"Could not read SHAP parquet at {candidate}: {exc}"
    if os.environ.get("HDFS_WEBHDFS_URL"):
        hdfs_base = os.environ.get("HDFS_SHAP_LOCAL_BASE", "/supply-chain/analytics/shap_local").rstrip("/")
        hdfs_path = f"{hdfs_base}/{commodity}.parquet"
        try:
            df = pd.read_parquet(BytesIO(_webhdfs_bytes(hdfs_path)))
            if "event_date" in df.columns:
                df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            return df, None
        except Exception as exc:
            return None, f"Could not read SHAP parquet from HDFS {hdfs_path}: {exc}"
    return None, f"Local SHAP not yet computed. Expected {candidates[0]} or {candidates[1]}."


def load_shap_local_v2() -> tuple[dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], str]]:
    """Read optional per-row regression SHAP parquet directly from HDFS/WebHDFS or local paths."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    errors: dict[tuple[str, str], str] = {}
    local_base = Path(os.environ.get("SHAP_LOCAL_V2_PATH", "/supply-chain/analytics/shap_local_v2"))
    hdfs_base = os.environ.get("HDFS_SHAP_LOCAL_V2_BASE", "/supply-chain/analytics/shap_local_v2").rstrip("/")

    for commodity in COMMODITIES:
        for target in SHAP_LOCAL_TARGETS:
            key = (commodity, target)
            name = f"{commodity}_{target}"
            local_candidates = [
                local_base / name / f"{name}.parquet",
                local_base / f"{name}.parquet",
                local_base / name,
            ]
            loaded = None
            read_errors = []
            for candidate in local_candidates:
                if candidate.exists():
                    try:
                        loaded = pd.read_parquet(candidate)
                        break
                    except Exception as exc:
                        read_errors.append(f"{candidate}: {exc}")
            if loaded is None and os.environ.get("HDFS_WEBHDFS_URL"):
                hdfs_candidates = [
                    f"{hdfs_base}/{name}/{name}.parquet",
                    f"{hdfs_base}/{name}.parquet",
                ]
                for hdfs_path in hdfs_candidates:
                    try:
                        loaded = pd.read_parquet(BytesIO(_webhdfs_bytes(hdfs_path)))
                        break
                    except Exception as exc:
                        read_errors.append(f"{hdfs_path}: {exc}")
            if loaded is not None:
                if "event_date" in loaded.columns:
                    loaded["event_date"] = pd.to_datetime(loaded["event_date"], errors="coerce")
                if "shap_value" in loaded.columns:
                    loaded["shap_value"] = pd.to_numeric(loaded["shap_value"], errors="coerce")
                if "feature_value" in loaded.columns:
                    loaded["feature_value"] = pd.to_numeric(loaded["feature_value"], errors="coerce")
                frames[key] = loaded
            else:
                if read_errors:
                    errors[key] = "Could not read shap_local_v2 parquet candidates: " + " | ".join(read_errors)
                else:
                    errors[key] = f"No shap_local_v2 parquet found for {name}"
    return frames, errors


def roc_auc_score(labels: pd.Series, scores: pd.Series) -> float | None:
    df = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if df["label"].nunique() < 2:
        return None
    positives = df["label"].sum()
    negatives = len(df) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = df["score"].rank(method="average")
    pos_rank_sum = ranks[df["label"] == 1].sum()
    auc = (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def average_precision_score(labels: pd.Series, scores: pd.Series) -> float | None:
    df = pd.DataFrame({"label": labels, "score": scores}).dropna()
    positives = int(df["label"].sum())
    if positives == 0:
        return None
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["tp"] = df["label"].cumsum()
    df["precision"] = df["tp"] / (df.index + 1)
    return float(df.loc[df["label"] == 1, "precision"].sum() / positives)


def commodity_metrics(predictions: pd.DataFrame, commodity: str) -> dict[str, float | None]:
    subset = predictions[predictions["commodity"] == commodity].copy()
    if subset.empty:
        return {"roc": None, "pr": None, "naive_pr": None, "lift": None}
    labels = subset["actual_label"].astype(float)
    scores = subset["predicted_proba"].astype(float)
    roc = roc_auc_score(labels, scores)
    pr = average_precision_score(labels, scores)
    naive = float(labels.mean()) if len(labels) else None
    lift = (pr - naive) if pr is not None and naive is not None else None
    return {"roc": roc, "pr": pr, "naive_pr": naive, "lift": lift}


def run_cached_sql(tables: dict[str, pd.DataFrame], sql: str, limit: int = 200) -> tuple[pd.DataFrame, str | None]:
    """Run SQL against the in-memory analytics tables loaded from HDFS.

    This keeps the single-ngrok setup viable: HttpFS serves parquet on port
    14000, and DuckDB provides a local SQL surface over those real tables.
    """
    try:
        import duckdb

        conn = duckdb.connect(database=":memory:")
        for table_name, frame in tables.items():
            if frame.empty and len(frame.columns) == 0:
                continue
            conn.register(table_name, frame)
        query = sql.strip().rstrip(";")
        if not query:
            return pd.DataFrame(), "Enter a SQL query."
        result = conn.execute(f"SELECT * FROM ({query}) AS dashboard_query LIMIT {int(limit)}").fetchdf()
        conn.close()
        return result, None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def run_hive_sql(sql: str, limit: int = 200) -> tuple[pd.DataFrame, str | None]:
    """Run SQL directly against HiveServer2 using PyHive."""
    try:
        from pyhive import hive

        config = HiveConfig.from_env()
        query = sql.strip().rstrip(";")
        if not query:
            return pd.DataFrame(), "Enter a SQL query."
        conn = hive.connect(
            host=config.host,
            port=config.port,
            username=config.user,
            database=config.database,
            auth=config.auth,
        )
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM ({query}) dashboard_query LIMIT {int(limit)}")
        rows = cursor.fetchall()
        columns = [col[0].split(".")[-1] for col in cursor.description]
        cursor.close()
        conn.close()
        return pd.DataFrame(rows, columns=columns), None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"
