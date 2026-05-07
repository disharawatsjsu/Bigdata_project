#!/usr/bin/env python3
"""
Download GDELT Events 2.0 daily files for a date range.
Files are tab-delimited CSVs compressed as .zip.

Usage:
    python download_gdelt.py --start 2023-01-01 --end 2023-12-31 --output ./data/gdelt
    python download_gdelt.py --start 2024-01-01 --end 2024-06-30 --output ./data/gdelt

Resumes automatically — skips files already downloaded.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Iterator

import requests

# GDELT 2.0 Events daily export URL pattern
# Format: YYYYMMDD.export.CSV.zip
GDELT_URL_TEMPLATE = "http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"

# Column names for GDELT Events 2.0 (58 fields)
GDELT_EVENTS_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL"
]


def date_range(start: date, end: date) -> Iterator[date]:
    """Yield each date from start to end inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def download_day(target_date: date, output_dir: str | Path) -> Path:
    """Download GDELT export for one day. Returns the path to the extracted CSV.

    Idempotent: if the destination CSV already exists and is non-empty,
    returns immediately without re-downloading.

    Raises:
        FileNotFoundError: if GDELT returns 404 (file not yet published)
        RuntimeError: on other HTTP errors or malformed ZIP payload
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ymd = target_date.strftime("%Y%m%d")
    csv_path = output_dir / f"{ymd}.export.CSV"

    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path

    url = GDELT_URL_TEMPLATE.format(ymd=ymd)

    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"GDELT file not yet published for {target_date.isoformat()} "
                f"(URL returned 404: {url})"
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"HTTP error downloading {url}: {e}") from e

        try:
            with zipfile.ZipFile(BytesIO(resp.content), "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".CSV")]
                if not csv_names:
                    raise RuntimeError(
                        f"No .CSV file in GDELT zip for {target_date.isoformat()}"
                    )
                # GDELT zips contain exactly one .CSV file, but be defensive.
                zf.extract(csv_names[0], output_dir)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Bad ZIP payload from {url}: {e}") from e

        extracted = output_dir / csv_names[0]
        if extracted != csv_path:
            extracted.rename(csv_path)

        return csv_path

    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed downloading {target_date.isoformat()}: {e}") from e


def download_range(
    start: date,
    end: date,
    output_dir: str | Path,
    delay_seconds: float = 0.5,
) -> list[Path]:
    """Download GDELT exports for a date range. Polite delay between requests.

    Note: 404 (file not published) is treated as a warning and skipped.
    """
    paths: list[Path] = []
    for d in date_range(start, end):
        try:
            paths.append(download_day(d, output_dir))
        except FileNotFoundError as e:
            print(f"[download_range] WARN: {e}")
        time.sleep(delay_seconds)
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GDELT Events 2.0 daily files")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="./data/gdelt", help="Output directory")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between requests (politeness)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = Path(args.output)

    print(f"Downloading GDELT Events: {start.isoformat()} → {end.isoformat()}")
    print(f"Output: {output_dir.resolve()}\n")

    success, skip, warn = 0, 0, 0
    for d in date_range(start, end):
        csv_path = output_dir / f"{d.strftime('%Y%m%d')}.export.CSV"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            skip += 1
            continue
        try:
            out = download_day(d, output_dir)
            mb = out.stat().st_size / 1e6
            print(f"  {d.strftime('%Y%m%d')}: OK ({mb:.1f} MB)")
            success += 1
        except FileNotFoundError:
            print(f"  {d.strftime('%Y%m%d')}: no file (404)")
            warn += 1
        time.sleep(args.delay)  # be polite to GDELT servers

    print(f"\nDone: {success} downloaded, {skip} skipped, {warn} 404s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
