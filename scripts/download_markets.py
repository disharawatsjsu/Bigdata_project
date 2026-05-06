#!/usr/bin/env python3
"""
Download commodity futures prices (Yahoo Finance) and FRED macro indicators.

Usage:
    python download_markets.py --start 2015-01-01 --end 2024-12-31 --output ./data

Outputs:
    ./data/commodities/commodity_prices.csv
    ./data/fred/fred_macro.csv
"""

import argparse
from pathlib import Path

import yfinance as yf
import pandas as pd

# --- Commodity symbols ---
COMMODITIES = {
    "CL=F": "crude_oil",
    "ZW=F": "wheat",
    "HG=F": "copper",
    "NG=F": "natural_gas",
    "KC=F": "coffee",
    "GC=F": "gold",
}

# --- FRED series (downloaded via public CSV endpoint, no API key needed) ---
FRED_SERIES = {
    "DGS10": "treasury_10y",          # 10-year Treasury yield
    "DTWEXBGS": "usd_index",          # Trade-weighted USD
    "CPIAUCNS": "cpi",                # CPI (monthly)
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def download_commodities(start: str, end: str, output_dir: Path):
    """Pull daily OHLCV for each commodity via yfinance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "commodity_prices.csv"

    frames = []
    for symbol, name in COMMODITIES.items():
        print(f"  Fetching {name} ({symbol})...")
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            print(f"    WARNING: no data for {symbol}")
            continue
        df = df[["Close"]].rename(columns={"Close": "close"})
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
            df = pd.read_csv(url)
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
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", default="./data")
    args = parser.parse_args()

    base = Path(args.output)
    print("=== Commodity Prices ===")
    download_commodities(args.start, args.end, base / "commodities")
    print("\n=== FRED Macro Indicators ===")
    download_fred(args.start, args.end, base / "fred")
    print("\nDone.")


if __name__ == "__main__":
    main()
