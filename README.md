# Supply Chain Disruption Intelligence

Predicting commodity price shocks from geopolitical events near global shipping chokepoints using GDELT, Spark, and HDFS.

**Course**: DATA 228 — Big Data Technologies  
**Status**: V1 pipeline runs end-to-end (mid-presentation ready)

---

## What this project does

We ingest 23M+ geopolitical events from GDELT, filter to supply-chain-relevant conflicts near 8 shipping chokepoints (Hormuz, Suez, Red Sea, etc.), engineer features from event intensity and frequency, join with crude oil prices and macroeconomic indicators, and train a Random Forest classifier to predict oil price shocks.

The full pipeline: **Raw CSVs → HDFS → Spark preprocessing → Feature engineering → ML training → EDA charts**

---

## Team roles

| Person | Role | Owns |
|--------|------|------|
| Person A | Introduction & narrative | Problem statement, dataset description, architecture diagram, presentation flow |
| Person B | Data acquisition & HDFS | `download_gdelt.py`, `download_markets.py`, `generate_chokepoints.py`, `docker-compose.yml`, HDFS ingestion |
| Person C | Spark preprocessing | `spark_pipeline.py` Steps 1-2 (ingest, dedup, CAMEO filter, geo-filter, chokepoint tagging) |
| Person D | EDA, ML & future direction | `spark_pipeline.py` Steps 3-4 (features, RF training), `eda_charts.py`, future scripts (Hive, Kafka, GraphFrames) |

---

## Prerequisites

- Docker Desktop installed (allocate at least 10GB RAM, 8 CPUs in Docker Desktop → Settings → Resources)
- Python 3.8+ on your Mac (for download scripts and EDA)
- ~15GB free disk space (for GDELT data + Docker volumes)

Install Python dependencies on your Mac:
```bash
pip install requests yfinance pandas matplotlib seaborn pyarrow
```

---

## Quick start (full setup from scratch)

### Step 1: Clone and navigate to project
```bash
git clone <your-repo-url>
cd Supply-chain-intel
```

### Step 2: Start the Docker cluster
```bash
docker compose up -d
```

Wait 30 seconds, then verify all 11 containers are running:
```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

You should see: namenode, datanode-1, datanode-2, spark-master, spark-worker-1, spark-worker-2, kafka, zookeeper, hive-server, hive-metastore-db, streamlit

### Step 3: Download GDELT data (6 months, ~9GB)
```bash
python3 ./scripts/download_gdelt.py --start 2024-01-01 --end 2024-06-30 --output ./data/gdelt
```
This takes 15-30 minutes depending on your connection. It downloads daily GDELT CSV files and auto-skips files already downloaded.

### Step 4: Download commodity prices and FRED macro data
```bash
python3 ./scripts/download_markets.py --start 2023-01-01 --end 2024-12-31 --output ./data
```

### Step 5: Generate chokepoint reference data
```bash
python3 ./scripts/generate_chokepoints.py
```

### Step 6: Push all data to HDFS
```bash
# Copy data into the namenode container
docker cp ./data/gdelt/. namenode:/tmp/gdelt/
docker cp ./data/commodities/ namenode:/tmp/commodities/
docker cp ./data/fred/ namenode:/tmp/fred/
docker cp ./data/chokepoints/ namenode:/tmp/chokepoints/

# Create HDFS directories and upload
docker exec -it namenode bash -c 'hdfs dfs -mkdir -p /opt/data/gdelt /opt/data/commodities /opt/data/fred /opt/data/chokepoints'
docker exec -it namenode bash -c 'hdfs dfs -put -f /tmp/gdelt/* /opt/data/gdelt/'
docker exec -it namenode bash -c 'hdfs dfs -put -f /tmp/commodities/* /opt/data/commodities/'
docker exec -it namenode bash -c 'hdfs dfs -put -f /tmp/fred/* /opt/data/fred/'
docker exec -it namenode bash -c 'hdfs dfs -put -f /tmp/chokepoints/* /opt/data/chokepoints/'
```

Verify data is on HDFS:
```bash
docker exec -it namenode bash -c 'hdfs dfs -ls /opt/data/gdelt/ | head'
docker exec -it namenode bash -c 'hdfs dfs -ls /opt/data/commodities/'
docker exec -it namenode bash -c 'hdfs dfs -ls /opt/data/fred/'
```

You can also check the Hadoop UI at http://localhost:9870 → Utilities → Browse the file system.

### Step 7: Run the Spark pipeline
```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --driver-memory 2g \
  --executor-memory 4g \
  /opt/scripts/spark_pipeline.py
```

This takes ~10 minutes. You'll see output for each step:
- **Step 1**: Ingests CSVs → Parquet (23.4M rows)
- **Step 2**: Dedup (33% removed), CAMEO filter (1.98M), geo-filter (20,872 near chokepoints)
- **Step 3**: Feature engineering (381 final rows after joining with oil prices + FRED)
- **Step 4**: Random Forest training with 3-fold cross-validation

Check Spark UI at http://localhost:8080 while it runs.

### Step 8: Generate EDA charts
```bash
# Pull features from HDFS to local
docker exec -it namenode bash -c 'hdfs dfs -get /supply-chain/features /tmp/features'
docker cp namenode:/tmp/features ./data/parquet/features
mkdir -p ./data/parquet

# Generate all 9 charts
python3 ./scripts/eda_charts.py --features ./data/parquet/features --output ./data/charts
```

Charts are saved to `./data/charts/`.

---

## What the pipeline does (detailed)

### Data sources
| Source | What | Size | Date range |
|--------|------|------|-----------|
| GDELT Events 2.0 | Global geopolitical events (protests, coercion, assault, violence) | 23.4M rows (~9GB) | Jan–Jun 2024 |
| Yahoo Finance | Crude oil futures daily close (CL=F) | ~500 rows | Jan 2023–Dec 2024 |
| FRED | 10-year Treasury yield, trade-weighted USD index | ~730 rows | Jan 2023–Dec 2024 |

### Preprocessing pipeline (Spark)
1. **Ingest**: Tab-delimited GDELT CSVs (58 columns) → Parquet partitioned by year/month
2. **Dedup**: Remove duplicate events (same URL + date + actor pair) — removes 33%
3. **CAMEO filter**: Keep only supply-chain-relevant event codes (14=protest, 17=coerce, 18=assault, 19=fight, 20=mass violence)
4. **Geo-filter**: Keep events within bounding boxes of 8 shipping chokepoints:
   - Hormuz, Suez, Red Sea, Black Sea, Malacca, Panama, Taiwan, Chile
5. **Chokepoint tagging**: Each event is tagged with which chokepoint it's near

### Feature engineering
- Daily aggregation per chokepoint: event count, avg Goldstein score, avg tone, total mentions, conflict ratio
- 7-day and 30-day rolling window averages for all metrics
- Oil price features: 5-day and 20-day lagged returns, 20-day rolling volatility
- FRED macro: Treasury yield and USD index (forward-filled to daily)
- Label: tercile-based classification of 5-day forward oil returns (negative shock / normal / positive shock)

### ML training
- Random Forest classifier (Spark MLlib), 100 trees, max depth 8
- 3-fold cross-validation with grid search over numTrees and maxDepth
- Time-based 60/20/20 split (train: Jan–mid Apr, val: mid Apr–late May, test: late May–Jun)
- Current results: preliminary (F1 ~0.22 on test) — expected with 381 samples, improving for final

---

## Key files

| File | What it does |
|------|-------------|
| `docker-compose.yml` | Defines all 11 containers (HDFS, Spark, Kafka, Hive, Streamlit) |
| `scripts/download_gdelt.py` | Downloads daily GDELT CSV files for a date range |
| `scripts/download_markets.py` | Downloads commodity prices (Yahoo Finance) and FRED macro data |
| `scripts/generate_chokepoints.py` | Generates chokepoint bounding boxes and commodity-region mapping |
| `scripts/spark_pipeline.py` | Main pipeline: ingest → clean → features → ML training |
| `scripts/eda_charts.py` | Generates 9 EDA and ML visualization charts |
| `scripts/graph_analysis.py` | GraphFrames analysis (future — for final presentation) |
| `scripts/hive_setup.py` | Hive external tables over HDFS Parquet (future) |
| `scripts/kafka_producer.py` | Kafka event streaming replay (future) |
| `scripts/spark_streaming.py` | Spark Structured Streaming consumer (future) |

---

## Web UIs

| Service | URL | What you see |
|---------|-----|-------------|
| Hadoop Namenode | http://localhost:9870 | HDFS file browser, cluster health, datanode status |
| Spark Master | http://localhost:8080 | Workers, running/completed applications |
| Spark Job UI | http://localhost:4040 | Stages, tasks, DAG visualization (only active during a job) |

---

## Common issues and fixes

**"No module named numpy"** when running spark-submit:
```bash
docker exec -it spark-master pip install numpy
docker exec -it spark-worker-1 pip install numpy
docker exec -it spark-worker-2 pip install numpy
```
This is already handled in docker-compose.yml but may be needed if containers were rebuilt.

**"Path does not exist" errors**: Data isn't on HDFS yet. Re-run Step 6.

**"Exit code 137" during pipeline**: A Spark executor got OOM-killed. The pipeline auto-recovers (tasks re-run on the surviving executor). If it keeps happening, increase Docker RAM allocation.

**Geo-filter returns 0 rows**: Schema misalignment. Make sure `spark_pipeline.py` has exactly 58 StructFields (no ADM2Code fields — those are GDELT v2 only, our data is v1 with 58 columns).

**Commodity CSV column error ("close" not found)**: Spark renames duplicate column headers. The pipeline uses `F.col("close1").cast("float")` — make sure you have the latest `spark_pipeline.py`.

---

## HDFS data layout

```
HDFS root
├── /opt/data/
│   ├── gdelt/                    ← Raw GDELT CSVs (landing zone)
│   │   ├── 20240101.export.CSV
│   │   ├── 20240102.export.CSV
│   │   └── ...
│   ├── commodities/
│   │   └── commodity_prices.csv
│   ├── fred/
│   │   └── fred_macro.csv
│   └── chokepoints/
│       ├── chokepoints.csv
│       └── commodity_region_map.csv
│
└── /supply-chain/                ← Pipeline outputs
    ├── gdelt_events/             ← Parquet (partitioned by year/month)
    │   ├── year=2024/month=1/
    │   ├── year=2024/month=2/
    │   └── ...
    ├── features/                 ← Feature table Parquet
    └── model_rf_v1/              ← Trained Random Forest model
```

---

## Shutting down

```bash
docker compose down
```

Data on HDFS persists across restarts (stored in Docker volumes). To fully reset:
```bash
docker compose down -v   # removes volumes — you'll need to re-upload data to HDFS
```

---

## For the mid-presentation

We present 4 charts:
1. **Event heatmap** (chart 01) — monthly event volume by chokepoint, shows Houthi crisis timeline
2. **Chokepoint bars** (chart 03) — all 8 chokepoints, oil-relevant ones highlighted
3. **Preprocessing funnel** (chart 06) — 23.4M raw → 381 features, shows Big Data justification
4. **Feature importance** (chart 07) — price/macro dominate, GDELT signal weak in V1, motivates final direction

Plus screenshots of: Hadoop UI, Spark UI, docker ps, GitHub commits.

---

## Future direction (for final presentation)

- Ingest 10 years of GDELT data (~30GB) for more training samples
- Use all 8 chokepoints (not just oil-relevant 3)
- Binary label (elevated volatility yes/no) instead of 3-class
- Weekly aggregation instead of daily
- GraphFrames: PageRank and centrality on region-commodity network, ablation study
- Kafka streaming: replay GDELT events, real-time inference
- Hive: SQL analytics over HDFS Parquet (additional Big Data tool not covered in class)
- Streamlit dashboard: live predictions, chokepoint map, alerts
- Data tiering: compress old GDELT data after training, keep feature summaries for long-term memory
