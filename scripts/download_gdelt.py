#!/usr/bin/env python3
"""
Download GDELT Events 2.0 daily files for a date range.
Files are tab-delimited CSVs compressed as .zip.

Usage:
    python download_gdelt.py --start 2023-01-01 --end 2023-12-31 --output ./data/gdelt
    python download_gdelt.py --start 2024-01-01 --end 2024-06-30 --output ./data/gdelt

Resumes automatically — skips files already downloaded.
"""

import os
import sys
import time
import argparse
import zipfile
import requests
from datetime import datetime, timedelta
from pathlib import Path

# GDELT 2.0 Events daily export URL pattern
# Format: YYYYMMDD.export.CSV.zip
BASE_URL = "http://data.gdeltproject.org/events"

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


def date_range(start: str, end: str):
    """Yield dates from start to end inclusive."""
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current <= end_dt:
        yield current
        current += timedelta(days=1)


def download_day(dt: datetime, output_dir: Path) -> str | None:
    """Download and extract one day's GDELT export. Returns CSV path or None."""
    date_str = dt.strftime("%Y%m%d")
    zip_name = f"{date_str}.export.CSV.zip"
    csv_name = f"{date_str}.export.CSV"
    csv_path = output_dir / csv_name

    # Skip if already extracted
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return str(csv_path)

    url = f"{BASE_URL}/{zip_name}"
    zip_path = output_dir / zip_name

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            # Some dates (weekends, holidays) may not have files
            print(f"  {date_str}: no file (404)")
            return None
        resp.raise_for_status()

        zip_path.write_bytes(resp.content)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)

        zip_path.unlink()  # clean up zip
        print(f"  {date_str}: OK ({csv_path.stat().st_size / 1e6:.1f} MB)")
        return str(csv_path)

    except Exception as e:
        print(f"  {date_str}: FAILED — {e}")
        if zip_path.exists():
            zip_path.unlink()
        return None


def main():
    parser = argparse.ArgumentParser(description="Download GDELT Events 2.0 daily files")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="./data/gdelt", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GDELT Events: {args.start} → {args.end}")
    print(f"Output: {output_dir.resolve()}\n")

    success, skip, fail = 0, 0, 0
    for dt in date_range(args.start, args.end):
        csv_path = output_dir / f"{dt.strftime('%Y%m%d')}.export.CSV"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            skip += 1
            continue
        result = download_day(dt, output_dir)
        if result:
            success += 1
        else:
            fail += 1
        time.sleep(0.5)  # be polite to GDELT servers

    print(f"\nDone: {success} downloaded, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
