#!/usr/bin/env python3
"""
Kafka Producer — Streams GDELT events into Kafka topics.

Two modes:
  1. REPLAY: Reads historical GDELT CSVs and publishes them as if arriving in real-time
     (one day's file every N seconds — configurable speed)
  2. LIVE: Polls GDELT's 15-min update endpoint and publishes new files as they appear

For the demo, REPLAY mode is what you want — it makes the pipeline
look like a streaming system without depending on live internet.

Usage:
    # Replay mode (default): send 2024 data at 2-second intervals per day
    python kafka_producer.py --mode replay --data-dir ./data/gdelt --speed 2

    # Live mode: poll GDELT every 15 min for new files
    python kafka_producer.py --mode live
"""

import os
import sys
import time
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime

from kafka import KafkaProducer


def _kafka_bootstrap() -> str:
    """Prefer KAFKA_BROKER; else KAFKA_BROKER_HOST + KAFKA_HOST_PORT (host-side runs); else Docker DNS."""
    if os.environ.get("KAFKA_BROKER"):
        return os.environ["KAFKA_BROKER"]
    host, port = os.environ.get("KAFKA_BROKER_HOST"), os.environ.get("KAFKA_HOST_PORT")
    if host and port:
        return f"{host}:{port}"
    return "kafka:9092"


KAFKA_BROKER = _kafka_bootstrap()
TOPIC_RAW = "gdelt-events-raw"        # raw event rows
TOPIC_ALERTS = "gdelt-alerts"         # filtered high-severity events

# Same CAMEO roots as spark_pipeline.py
SC_CAMEO_ROOTS = {"14", "17", "18", "19", "20"}

# GDELT Events 2.0 column indices (tab-delimited, no header)
IDX_GLOBALEVENTID = 0
IDX_SQLDATE = 1
IDX_EVENT_ROOT_CODE = 28
IDX_GOLDSTEIN = 30
IDX_NUM_MENTIONS = 31
IDX_AVG_TONE = 34
IDX_ACTION_GEO_LAT = 56
IDX_ACTION_GEO_LONG = 57
IDX_SOURCEURL = 60


def create_producer():
    """Create Kafka producer with JSON serialization."""
    retries = 5
    for i in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                # Batching for throughput
                batch_size=16384,
                linger_ms=100,
            )
            print(f"Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except Exception as e:
            print(f"Kafka connection attempt {i+1}/{retries} failed: {e}")
            time.sleep(3)
    raise ConnectionError(f"Could not connect to Kafka at {KAFKA_BROKER}")


def parse_gdelt_line(line: str) -> dict | None:
    """Parse a single tab-delimited GDELT Events row into a dict."""
    fields = line.strip().split("\t")
    if len(fields) < 58:
        return None

    try:
        event = {
            "global_event_id": fields[IDX_GLOBALEVENTID],
            "sql_date": fields[IDX_SQLDATE],
            "event_root_code": fields[IDX_EVENT_ROOT_CODE],
            "goldstein_scale": float(fields[IDX_GOLDSTEIN]) if fields[IDX_GOLDSTEIN] else None,
            "num_mentions": int(fields[IDX_NUM_MENTIONS]) if fields[IDX_NUM_MENTIONS] else 0,
            "avg_tone": float(fields[IDX_AVG_TONE]) if fields[IDX_AVG_TONE] else None,
            "action_geo_lat": float(fields[IDX_ACTION_GEO_LAT]) if fields[IDX_ACTION_GEO_LAT] else None,
            "action_geo_long": float(fields[IDX_ACTION_GEO_LONG]) if fields[IDX_ACTION_GEO_LONG] else None,
            "source_url": fields[IDX_SOURCEURL] if len(fields) > IDX_SOURCEURL else "",
            "ingested_at": datetime.utcnow().isoformat(),
        }
        return event
    except (ValueError, IndexError):
        return None


def is_supply_chain_relevant(event: dict) -> bool:
    """Check if event matches supply-chain CAMEO codes."""
    return event.get("event_root_code") in SC_CAMEO_ROOTS


def is_high_severity(event: dict) -> bool:
    """High severity = conflict event with strong negative Goldstein + high mentions."""
    gs = event.get("goldstein_scale")
    mentions = event.get("num_mentions", 0)
    return (
        is_supply_chain_relevant(event)
        and gs is not None
        and gs < -5.0
        and mentions >= 10
    )


def replay_files(producer: KafkaProducer, data_dir: str, speed: float):
    """Replay historical GDELT CSVs into Kafka, one file at a time."""
    files = sorted(glob.glob(f"{data_dir}/*.export.CSV"))
    if not files:
        print(f"No GDELT CSV files found in {data_dir}")
        return

    print(f"Replaying {len(files)} files at {speed}s interval per file\n")

    total_sent = 0
    total_alerts = 0

    for filepath in files:
        filename = Path(filepath).name
        file_events = 0
        file_alerts = 0

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                event = parse_gdelt_line(line)
                if event is None:
                    continue

                # Send to raw topic (key = date for partition ordering)
                producer.send(
                    TOPIC_RAW,
                    key=event["sql_date"],
                    value=event,
                )
                file_events += 1

                # High-severity events also go to alerts topic
                if is_high_severity(event):
                    producer.send(
                        TOPIC_ALERTS,
                        key=event["sql_date"],
                        value=event,
                    )
                    file_alerts += 1

        producer.flush()
        total_sent += file_events
        total_alerts += file_alerts
        print(f"  {filename}: {file_events:,} events, {file_alerts} alerts")

        time.sleep(speed)  # pacing between files

    print(f"\nReplay complete: {total_sent:,} events, {total_alerts:,} alerts across {len(files)} files")


def live_poll(producer: KafkaProducer):
    """Poll GDELT for new 15-min update files (placeholder — needs internet)."""
    import requests

    GDELT_LASTUPDATE = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    seen_files = set()

    print("Live polling GDELT every 60 seconds...")
    while True:
        try:
            resp = requests.get(GDELT_LASTUPDATE, timeout=10)
            for line in resp.text.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 3 and "export" in parts[2].lower():
                    file_url = parts[2]
                    if file_url not in seen_files:
                        seen_files.add(file_url)
                        print(f"  New file: {file_url}")
                        # TODO: download, parse, and send to Kafka
                        # For now just log it
        except Exception as e:
            print(f"  Poll error: {e}")

        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="GDELT → Kafka producer")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--data-dir", default="./data/gdelt", help="Path to GDELT CSV files")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Seconds between files in replay mode")
    args = parser.parse_args()

    producer = create_producer()

    if args.mode == "replay":
        replay_files(producer, args.data_dir, args.speed)
    else:
        live_poll(producer)

    producer.close()


if __name__ == "__main__":
    main()
