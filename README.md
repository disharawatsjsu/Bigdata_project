# Supply Chain Disruption Intelligence

Predicting commodity risk and price shocks from geopolitical events near global shipping chokepoints using **GDELT + Spark + HDFS**, with **Airflow orchestration** and optional **Hive/Kafka** extensions.

**Course**: DATA 228 — Big Data Technologies

---

## What this repository contains

This repo is a self-contained “local big-data stack” (via Docker Compose) plus Python/Spark pipelines that:

- Ingest GDELT daily exports (58 columns) and route them into a **tiered** Parquet layout (**hot/warm**, optional cold later).
- Clean/filter events to chokepoints and conflict-relevant CAMEO roots, dedupe, and build a unified analysis dataset.
- Join market series (Yahoo Finance) and macro data (FRED) and produce **ML-ready baseline features** per commodity.
- Optionally materialize analytic tables in **Hive** and serve them in a **Plotly Dash** intelligence dashboard.
- Provide a lightweight **Streamlit** demo dashboard for feature files (works even before the pipeline runs).
- Orchestrate daily ingest / backfills / feature refresh via **Airflow DAGs**.

High-level flow:

**Raw CSV → tiered Parquet on HDFS → Spark feature build → (Hive analytics tables) → dashboards / notebooks**

---

## Quick start (fastest successful end-to-end run)

### 0) Prerequisites

- Docker Desktop (recommended: **13–16 GB RAM** allocated for the full stack)
- Python **3.10+** (for running host-side helper scripts / dashboards)
- Disk: budget **15–30 GB** depending on date range and whether you backfill

Install local Python deps (Dash + Hive client + common utilities):

```bash
python -m pip install -r requirements.txt
```

If you plan to run host-side download/EDA scripts, also install:

```bash
python -m pip install requests yfinance matplotlib seaborn numpy
```

### 1) Configure environment variables

Docker Compose and the Dash app read variables from a local `.env` file.

```bash
cp .env.example .env
```

You can keep defaults. Ports/credentials are documented inside `.env.example`.

### 2) Start the full stack

```bash
docker compose up -d
```

Wait ~1–2 minutes on first boot (some services install Python packages on startup).

### 3) Validate the stack is healthy

```bash
docker compose ps
```

If something is down, check `docs/STACK_HEALTH.md` for known issues and fixes.

### 4) Run “feature refresh” (Spark output on HDFS)

This produces ML-ready baseline features at:
`hdfs://namenode:9000/supply-chain/features/baseline/{commodity}/`

```bash
docker exec spark-master bash -lc 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64; export PATH=$JAVA_HOME/bin:$PATH; /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_feature_refresh.py'
```

That command assumes the hot/warm tiers already contain some data and `/opt/data/commodities` + `/opt/data/fred` exist. If you’re starting from empty, use one of the paths below:

- **Airflow path (recommended for “from scratch”)**: run `feature_refresh_dag` in the Airflow UI.
- **Manual path (documented below)**: download → ingest → download markets/FRED → upload → feature refresh.

---

## Architecture & services (Docker Compose)

The stack is defined in `docker-compose.yml` and includes:

- **HDFS**: `namenode`, `datanode-1`, `datanode-2`
- **HttpFS**: `httpfs` (single endpoint WebHDFS gateway; useful with ngrok/Colab)
- **Spark**: `spark-master`, `spark-worker-1`, `spark-worker-2`
- **Kafka**: `zookeeper`, `kafka` (used by future/optional streaming demos)
- **Hive**: `hive-server`, `hive-metastore-db` (optional analytics / dashboard backend)
- **Dashboards**:
  - `streamlit` (demo dashboard from `dashboard/dashboard.py`)
  - local-run Plotly Dash app from `dashboard/app.py` (not containerized in compose)
- **Airflow**: `airflow-db`, `airflow-init`, `airflow-webserver`, `airflow-scheduler`

### Web UIs (defaults; configurable via `.env`)

| Service | URL | Notes |
|---------|-----|------|
| Hadoop Namenode | `http://localhost:9870` | HDFS browser, datanodes, file utilities |
| Spark Master | `http://localhost:8080` | Cluster + applications |
| Spark Job UI | `http://localhost:4040` | Only while a job is running |
| Hive Web UI | `http://localhost:10002` | Basic service page |
| Airflow | `http://localhost:8090` | Login from `.env` (`AIRFLOW_ADMIN_USER/PASSWORD`) |
| Streamlit demo | `http://localhost:8501` | Minimal risk monitor demo UI |
| Plotly Dash (local) | `http://localhost:8050` | Run `python dashboard/app.py` |

---

## Running the pipeline

You can operate the project in two ways:

- **Airflow orchestration**: the most reliable, idempotent, “ops-like” path.
- **Manual commands**: good for debugging and one-off demos.

### Option A: Airflow (recommended)

1) Open the Airflow UI at `http://localhost:8090`

2) DAGs live in `dags/`:

- `daily_ingest_dag`: downloads (logical_date - 2 days) and ingests into tiered HDFS hot/warm, validates, then cleans up local CSV.
- `feature_refresh_dag`: end-to-end refresh:
  - download GDELT for target day
  - ingest tiered Parquet
  - discover available GDELT date range from HDFS
  - download market + FRED for that range and upload to HDFS
  - run Spark feature build (`scripts/run_feature_refresh.py`)
  - validate output
- `bulk_download_dag` / `historical_backfill_dag`: batch utilities for larger ranges

### Option B: Manual (from scratch)

#### 1) Download GDELT exports (host)

```bash
python ./scripts/download_gdelt.py --start 2024-01-01 --end 2024-01-10 --output ./data/gdelt
```

#### 2) Ingest one day into tiered HDFS (Spark)

```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_tiered_ingest.py \
  --date 2024-01-01
```

Tip: for month-level ingest, see `scripts/ingest_one_month.py` and `scripts/run_tiered_ingest.py`.

#### 3) Download commodity + macro series (host)

```bash
python ./scripts/download_markets.py --start 2024-01-01 --end 2024-01-10 --output ./data --only commodities
python ./scripts/download_markets.py --start 2024-01-01 --end 2024-01-10 --output ./data --only fred
```

#### 4) Upload market/FRED data to HDFS

```bash
docker cp ./data/commodities namenode:/tmp/commodities
docker cp ./data/fred namenode:/tmp/fred
docker exec namenode bash -lc 'hdfs dfs -mkdir -p /opt/data/commodities /opt/data/fred && \
  hdfs dfs -put -f /tmp/commodities/* /opt/data/commodities/ && \
  hdfs dfs -put -f /tmp/fred/* /opt/data/fred/'
```

#### 5) Build baseline features (Spark)

```bash
docker exec spark-master bash -lc 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64; export PATH=$JAVA_HOME/bin:$PATH; /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/scripts/run_feature_refresh.py'
```

Each commodity output includes a `_summary.json` (row counts, nulls, date bounds) written alongside Parquet.

---

## Tiered storage (hot/warm) and HDFS layout

Tiering exists to fit multi-year GDELT into limited local disk.

- Tier design and schema contracts: `docs/TIERING.md`
- Schema validation and versions: `scripts/schemas.py`

Key HDFS paths (see `scripts/config.py` for authoritative values):

- Raw landing (HDFS): `/opt/data/{commodities,fred,...}`
- Tiered Parquet:
  - Hot: `hdfs://namenode:9000/supply-chain/raw/hot/year=YYYY/month=MM/day=DD/`
  - Warm: `hdfs://namenode:9000/supply-chain/raw/warm/year=YYYY/month=MM/day=DD/`
- Features:
  - `hdfs://namenode:9000/supply-chain/features/baseline/{commodity}/`

---

## Dashboards

### 1) Streamlit demo dashboard (container)

The `streamlit` service runs `dashboard/dashboard.py` and reads from `./data` (mounted into the container as `/app/data`).

Open `http://localhost:8501`.

Notes:

- If feature Parquet doesn’t exist yet, it generates synthetic demo data so the UI still works.
- It expects chokepoint reference CSV at `data/chokepoints/chokepoints.csv` (generated by `scripts/generate_chokepoints.py`).

### 2) Plotly Dash “intelligence” dashboard (local app over Hive / cached tables)

The Dash app is `dashboard/app.py`. It can:

- Query HiveServer2 (default) using `.env` variables (`HIVE_HOST`, `HIVE_PORT`, etc.)
- Fall back to cached/parquet modes depending on `DASH_SQL_MODE` / `DASH_DATA_MODE`

Run locally:

```bash
python dashboard/app.py
```

Environment configuration lives in `.env.example` (copy to `.env`).

---

## Hive / analytics tables

Hive is optional but enables “SQL-backed” analytics and the Dash query panel.

- Hive setup helper: `scripts/hive_setup.py`
- Market price DDL: `scripts/hive_create_market_prices.sql`
- Regression v2 DDL: `scripts/hive_create_v2_tables.sql`
- Dashboard SQL entrypoints: `dashboard/queries.py`

Run the dashboard table DDL once after the stack is up:

```bash
docker exec -i hive-server /opt/hive/bin/beeline \
  -u 'jdbc:hive2://localhost:10000/default;auth=noSasl' \
  < scripts/hive_create_market_prices.sql

docker exec -i hive-server /opt/hive/bin/beeline \
  -u 'jdbc:hive2://localhost:10000/default;auth=noSasl' \
  < scripts/hive_create_v2_tables.sql
```

The demo dashboard is powered by these 8 Hive tables:

- `trigger_log` — daily chokepoint trigger signals
- `predictions` — legacy classification predictions
- `shap_attribution` — legacy aggregate SHAP attribution
- `event_study` — post-event lift / significance summaries
- `market_prices` — commodity price CSV external table over clean Hive staging path `/opt/data/commodities_hive/`
- `predictions_v2` — regression predictions from `/supply-chain/analytics/predictions_v2/`
- `shap_attribution_v2` — regression aggregate chokepoint SHAP from `/supply-chain/analytics/shap_attribution_v2/`
- `regression_metrics` — regression model metrics from `/supply-chain/analytics/regression_metrics/`

The dashboard map uses Plotly `Scattergeo` (no Mapbox token required) with the
dark land/ocean style defined in `dashboard/components.py`. The `market_prices`
table is CSV-backed rather than Parquet-backed; at roughly a few thousand rows,
the visible Hive query panel remains fast enough for the demo. Stage the CSV for
Hive before running the DDL:

```bash
docker exec namenode hdfs dfs -mkdir -p /opt/data/commodities_hive
docker exec namenode hdfs dfs -cp -f \
  /opt/data/commodities/commodity_prices.csv \
  /opt/data/commodities_hive/commodity_prices.csv
```

If you want remote access (e.g. Colab), use:

- **HiveServer2 tunnel**: `ngrok tcp 10000`
- **HttpFS tunnel** (WebHDFS single endpoint): `ngrok http 14000`

Then set the corresponding values in `.env`.

---

## EDA & notebooks

- EDA chart generator: `scripts/eda_charts.py`
- Notebooks live under `notebooks/` (Colab-friendly)

If you want to chart features locally, you can copy Parquet out of HDFS into `./data/parquet/` and run `scripts/eda_charts.py`.

---

## Model registry (versioned models on HDFS)

Models are versioned per-commodity with a text pointer file `CURRENT`.

See `docs/MODEL_REGISTRY.md`.

---

## Key files & entrypoints

| Path | What it does |
|------|-------------|
| `docker-compose.yml` | Local stack (HDFS, HttpFS, Spark, Kafka, Hive, Airflow, Streamlit demo) |
| `.env.example` | Ports/credentials; copy to `.env` |
| `scripts/download_gdelt.py` | Download daily GDELT exports for a date range |
| `scripts/run_tiered_ingest.py` | Spark-submit wrapper: ingest one day into hot/warm |
| `scripts/run_feature_refresh.py` | Spark-submit wrapper: build baseline features per commodity |
| `scripts/pipeline_ingest.py` | Tiered ingest implementation (`ingest_gdelt_tiered`) |
| `scripts/pipeline_features.py` | Cleaning/filtering + feature build |
| `scripts/schemas.py` | Schema contracts + validation for tiers/features |
| `dags/*.py` | Airflow orchestration DAGs |
| `dashboard/dashboard.py` | Streamlit demo dashboard (containerized) |
| `dashboard/app.py` | Plotly Dash intelligence dashboard (local) |

---

## Common issues & fixes

- **Containers exit with code 137 / Spark workers die**: almost always Docker OOM. Increase Docker RAM (13–16 GB recommended). See `docs/STACK_HEALTH.md`.
- **`airflow-init` exits (0)**: expected. It’s a one-time bootstrap container.
- **Spark driver UI (4040) is blank**: it only exists while a Spark application is running.
- **Hive connectivity from the Dash app fails**: check `.env` (`HIVE_HOST`, `HIVE_PORT`, `HIVE_AUTH=NOSASL`) and confirm the port is published in `docker-compose.yml`.

---

## Shutting down / reset

Stop services:

```bash
docker compose down
```

Full reset (deletes Docker volumes — you will need to re-ingest data):

```bash
docker compose down -v
```
