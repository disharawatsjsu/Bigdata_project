#!/usr/bin/env python3
"""
Bulk parallel downloader for GDELT Events 2.0 daily exports.

Why this exists (vs downloading inside the ingest DAG):
- Downloads are network-bound; Spark ingest is CPU/memory-bound.
- Parallel download can saturate available bandwidth while Spark runs separately.
- This script is idempotent: it skips days whose CSV already exists and is non-empty.

Example:
  python scripts/bulk_download_gdelt.py \
      --start 2018-01-01 \
      --end 2020-12-31 \
      --output /opt/data/gdelt \
      --concurrency 12
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from download_gdelt import date_range, download_day


def _looks_like_retryable_http(err: BaseException) -> bool:
    msg = str(err)
    # download_day wraps requests.HTTPError into RuntimeError and includes the code in the message.
    return (" 429 " in msg) or (" 503 " in msg) or ("429 Client Error" in msg) or ("503 Server Error" in msg)


def _sleep_backoff(attempt: int) -> None:
    # Exponential backoff with jitter: ~1s, 2s, 4s, 8s, 16s (capped)
    base = min(16.0, 2 ** max(0, attempt - 1))
    time.sleep(base + random.random())


@dataclass(frozen=True)
class DayResult:
    day: date
    status: str  # ok | skipped | no_file_404 | failed
    bytes_downloaded: int = 0
    error: str | None = None


def bulk_download_range(
    start: date,
    end: date,
    output_dir: str | Path,
    *,
    concurrency: int = 12,
    max_attempts: int = 5,
) -> list[DayResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    days = list(date_range(start, end))
    total = len(days)
    if total == 0:
        return []

    concurrency = max(1, min(int(concurrency), 12))
    started = time.time()

    def _one(d: date) -> DayResult:
        ymd = d.strftime("%Y%m%d")
        csv_path = output_dir / f"{ymd}.export.CSV"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return DayResult(day=d, status="skipped", bytes_downloaded=0)

        for attempt in range(1, max_attempts + 1):
            try:
                out = download_day(d, output_dir)
                size = out.stat().st_size if out.exists() else 0
                return DayResult(day=d, status="ok", bytes_downloaded=int(size))
            except FileNotFoundError:
                return DayResult(day=d, status="no_file_404", bytes_downloaded=0)
            except RuntimeError as e:
                if _looks_like_retryable_http(e) and attempt < max_attempts:
                    _sleep_backoff(attempt)
                    continue
                return DayResult(day=d, status="failed", bytes_downloaded=0, error=str(e))

        return DayResult(day=d, status="failed", bytes_downloaded=0, error="exhausted retries")

    results: list[DayResult] = []
    done = 0
    bytes_total = 0
    ok = skipped = no404 = failed = 0

    print(
        f"[bulk_download] range={start.isoformat()}..{end.isoformat()} "
        f"days={total} concurrency={concurrency} output={output_dir}"
    )

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_one, d): d for d in days}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            bytes_total += r.bytes_downloaded

            if r.status == "ok":
                ok += 1
            elif r.status == "skipped":
                skipped += 1
            elif r.status == "no_file_404":
                no404 += 1
            else:
                failed += 1

            elapsed = max(0.001, time.time() - started)
            mb = bytes_total / 1e6
            mbps = mb / elapsed
            pct = (done / total) * 100.0

            print(
                f"[bulk_download] {done}/{total} ({pct:5.1f}%) "
                f"ok={ok} skipped={skipped} 404={no404} failed={failed} "
                f"— {mb:,.0f} MB downloaded — {mbps:,.1f} MB/s avg"
            )

            if r.status == "failed":
                print(f"[bulk_download] FAIL {r.day.isoformat()}: {r.error}")

    # stable sort for downstream consumers/log readability
    results.sort(key=lambda x: x.day)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk parallel downloader for GDELT Events 2.0 daily exports")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--output", required=True, help="Output directory (e.g. /opt/data/gdelt)")
    p.add_argument("--concurrency", type=int, default=12, help="Max parallel requests (cap=12)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out = Path(args.output)

    results = bulk_download_range(start, end, out, concurrency=args.concurrency)
    failed = [r for r in results if r.status == "failed"]

    if failed:
        print(f"[bulk_download] DONE with failures: {len(failed)} day(s) failed")
        return 2

    print("[bulk_download] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

