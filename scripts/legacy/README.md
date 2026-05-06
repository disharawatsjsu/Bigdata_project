## scripts/legacy

This directory contains **archived pipeline entrypoints** preserved for reference.

### Why this exists

During a refactor, multiple scripts accumulated overlapping end-to-end pipeline logic.
The current source of truth is `scripts/spark_pipeline.py`. The files in this folder
are kept to preserve prior iteration history and to aid comparison during reviews.

### What is here

- `pipeline_part1_ingest.py`: earlier “Part 2” (ingest/clean/features) entrypoint
- `pipeline_part3_full.py`: earlier “Part 3” full pipeline entrypoint

### How to treat these files

- Not used by the active pipeline.
- Not imported by the active pipeline.
- Kept intentionally; do not delete without an explicit decision.
