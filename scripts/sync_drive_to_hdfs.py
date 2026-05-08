#!/usr/bin/env python3
"""Sync a Colab artifacts folder from local disk to HDFS.

DEPRECATED (tonight's flow): With HttpFS, the Colab notebook can write artifacts
directly to HDFS via WebHDFS, so this manual sync step is not required. Kept
for fallback/debugging.

Assumes you downloaded (or otherwise made available) the artifacts folder that
Colab wrote into Google Drive.

Usage:
    python scripts/sync_drive_to_hdfs.py /path/to/local/artifacts/<timestamp>/
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def _usage() -> None:
    print("Usage: sync_drive_to_hdfs.py <local_artifacts_dir>")


def hdfs_put_bytes(content: bytes, dst: str) -> None:
    subprocess.run(
        ["docker", "exec", "-i", "namenode", "hdfs", "dfs", "-put", "-f", "-", dst],
        input=content,
        check=True,
    )


def main() -> int:
    if len(sys.argv) != 2:
        _usage()
        return 2

    local_dir = Path(sys.argv[1]).expanduser().resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        print(f"Not a directory: {local_dir}")
        return 2

    timestamp = local_dir.name  # assumes dir name is the timestamp

    model_path = local_dir / "model.json"
    preds_path = local_dir / "predictions.parquet"
    metrics_path = local_dir / "metrics.json"

    for p in (model_path, preds_path, metrics_path):
        if not p.exists():
            print(f"Missing required file: {p}")
            return 2

    # Ensure parent dirs exist (hdfs dfs -put won't create intermediate paths reliably in all setups)
    subprocess.run(
        ["docker", "exec", "namenode", "hdfs", "dfs", "-mkdir", "-p", "/supply-chain/models/crude_oil"],
        check=False,
    )
    subprocess.run(
        ["docker", "exec", "namenode", "hdfs", "dfs", "-mkdir", "-p", "/supply-chain/predictions"],
        check=False,
    )
    subprocess.run(
        ["docker", "exec", "namenode", "hdfs", "dfs", "-mkdir", "-p", "/supply-chain/model_metrics"],
        check=False,
    )

    hdfs_put_bytes(model_path.read_bytes(), f"/supply-chain/models/crude_oil/{timestamp}/model.json")
    hdfs_put_bytes(preds_path.read_bytes(), f"/supply-chain/predictions/{timestamp}.parquet")
    hdfs_put_bytes(metrics_path.read_bytes(), f"/supply-chain/model_metrics/{timestamp}.json")

    # Update CURRENT pointer
    hdfs_put_bytes(timestamp.encode(), "/supply-chain/models/crude_oil/CURRENT")

    print(f"Synced artifacts for {timestamp} to HDFS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

