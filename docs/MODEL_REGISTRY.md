# Model Registry

## Layout

```
{HDFS_BASE}/models/
├── brent/
│   ├── CURRENT                    # text file: "2026-05-01T00:00:00"
│   ├── 2026-04-01T00:00:00/      # MLlib model dir
│   └── 2026-05-01T00:00:00/
├── nat_gas/
│   ├── CURRENT
│   └── ...
└── ... (one dir per commodity)
```

## Lifecycle

1. PR O training run produces a model artifact at
   `{registry}/{commodity}/{run_timestamp}/`.
2. After validation passes, atomically write the run_timestamp into
   `{registry}/{commodity}/CURRENT`.
3. Streaming inference (PR S) reads CURRENT to find the active model.
4. Old versions are kept indefinitely for rollback and comparison.

## Why a text file pointer instead of symlinks

HDFS doesn't handle symlinks consistently across clients. A one-line
text file is universally readable, atomic via overwrite, and trivially
versionable.

## Retention

Not enforced yet. If storage becomes a concern, a future DAG can prune
versions older than N months while preserving CURRENT and the most
recent K versions.
