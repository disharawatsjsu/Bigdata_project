#!/usr/bin/env python3
"""
Download commodity futures prices (Yahoo Finance) and FRED macro indicators.

Usage:
    python download_markets.py --start <START_DATE> --end <END_DATE> --output ./data

Outputs:
    ./data/commodities/commodity_prices.csv
    ./data/fred/fred_macro.csv
"""

import argparse
import os
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import yfinance as yf
import pandas as pd
import requests

# --- Commodity symbols ---
COMMODITIES = {
    "BZ=F": "brent",
    "CL=F": "wti",
    "HG=F": "copper",
    "GC=F": "gold",
    "ZW=F": "wheat",
    "ZS=F": "soybeans",
}

# --- FRED series (downloaded via public CSV endpoint, no API key needed) ---
FRED_SERIES = {
    "DGS10": "treasury_10y",          # 10-year Treasury yield
    "DTWEXBGS": "usd_index",          # Trade-weighted USD
    "VIXCLS": "vix",                  # CBOE volatility index
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def download_commodities(start: str, end: str, output_dir: Path):
    """Pull daily OHLCV for each commodity via yfinance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "commodity_prices.csv"
    yf_end = (date.fromisoformat(end) + timedelta(days=1)).isoformat()

    frames = []
    for symbol, name in COMMODITIES.items():
        print(f"  Fetching {name} ({symbol})...")
        df = yf.download(symbol, start=start, end=yf_end, progress=False)
        if df.empty:
            print(f"    WARNING: no data for {symbol}")
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        df = close.rename("close").to_frame()
        df["commodity"] = name
        df["symbol"] = symbol
        df.index.name = "date"
        frames.append(df.reset_index())

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"  → Saved {len(combined)} rows to {out_path}")
    return combined


def download_fred(start: str, end: str, output_dir: Path):
    """Pull FRED series via direct observation CSV download (no API key)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "fred_macro.csv"

    frames = []
    for series_id, col_name in FRED_SERIES.items():
        print(f"  Fetching FRED {series_id} ({col_name})...")
        # Direct CSV download URL — more reliable than the graph endpoint
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&cosd={start}&coed={end}"
        )
        try:
            # Read without assuming column names — detect them
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            # FRED CSVs have columns like DATE/date and SERIES_ID/series_id
            # Normalize: find the date column and the value column
            df.columns = [c.strip() for c in df.columns]
            date_col = next((c for c in df.columns if c.upper() == "DATE"), df.columns[0])
            val_col = next((c for c in df.columns if c != date_col), df.columns[1])
            df = df.rename(columns={date_col: "date", val_col: col_name})
            df["date"] = pd.to_datetime(df["date"])
            # FRED uses '.' for missing — coerce
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            frames.append(df[["date", col_name]])
            print(f"    {len(df)} rows")
        except Exception as e:
            print(f"    FAILED: {e}")
            print(f"    URL was: {url}")
            print(f"    Tip: try opening that URL in a browser to check the format")

    if frames:
        # Merge all FRED series on date (outer join — different frequencies)
        merged = frames[0]
        for f in frames[1:]:
            merged = pd.merge(merged, f, on="date", how="outer")
        merged = merged.sort_values("date").reset_index(drop=True)
        merged.to_csv(out_path, index=False)
        print(f"  → Saved {len(merged)} rows to {out_path}")
    return merged if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download commodity + FRED data")
    parser.add_argument("--start", default=os.environ.get("MARKET_START_DATE"))
    parser.add_argument("--end", default=os.environ.get("MARKET_END_DATE"))
    parser.add_argument("--output", default="./data")
    parser.add_argument(
        "--only",
        choices=["all", "commodities", "fred"],
        default="all",
        help="Limit download to commodity prices or FRED macro data.",
    )
    args = parser.parse_args()

    if not args.start or not args.end:
        parser.error("--start/--end are required unless MARKET_START_DATE/MARKET_END_DATE are set")

    base = Path(args.output)
    if args.only in ("all", "commodities"):
        print("=== Commodity Prices ===")
        download_commodities(args.start, args.end, base / "commodities")
    if args.only in ("all", "fred"):
        print("\n=== FRED Macro Indicators ===")
        download_fred(args.start, args.end, base / "fred")
    print("\nDone.")


if __name__ == "__main__":
    main()
