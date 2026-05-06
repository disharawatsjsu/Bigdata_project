#!/usr/bin/env python3
"""
Generate the chokepoint reference data used for geo-filtering GDELT events
and for building the region-commodity graph.

Outputs:
    ./data/chokepoints/chokepoints.csv
    ./data/chokepoints/commodity_region_map.csv
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("./data/chokepoints")


def generate_chokepoints():
    """Key shipping chokepoints with bounding boxes for geo-filtering."""
    data = [
        # name, center_lat, center_lon, radius_deg (approx box half-width)
        ("Strait of Hormuz",    26.56,  56.25,  1.5),
        ("Suez Canal",          30.46,  32.34,  1.0),
        ("Panama Canal",         9.08, -79.68,  0.5),
        ("Strait of Malacca",    2.50, 101.50,  2.0),
        ("Bosphorus Strait",    41.12,  29.05,  0.5),
        ("Cape of Good Hope",  -34.35,  18.47,  2.0),
        ("Taiwan Strait",       24.50, 119.50,  1.5),
        ("Black Sea Ports",     44.60,  33.50,  3.0),  # Odessa/Sevastopol region
        ("Red Sea / Yemen",     14.00,  43.00,  3.0),  # Bab el-Mandeb + Houthi zone
        ("Chile Copper Belt",  -23.50, -69.50,  3.0),  # Atacama mining region
    ]

    df = pd.DataFrame(data, columns=["chokepoint", "lat", "lon", "radius_deg"])
    # Bounding box for quick filtering
    df["lat_min"] = df["lat"] - df["radius_deg"]
    df["lat_max"] = df["lat"] + df["radius_deg"]
    df["lon_min"] = df["lon"] - df["radius_deg"]
    df["lon_max"] = df["lon"] + df["radius_deg"]
    return df


def generate_commodity_region_map():
    """Which commodities are affected by disruptions at which chokepoints."""
    mappings = [
        ("crude_oil",    "Strait of Hormuz",   "primary"),
        ("crude_oil",    "Suez Canal",          "primary"),
        ("crude_oil",    "Red Sea / Yemen",     "primary"),
        ("crude_oil",    "Bosphorus Strait",    "secondary"),
        ("natural_gas",  "Strait of Hormuz",    "primary"),
        ("natural_gas",  "Red Sea / Yemen",     "secondary"),
        ("wheat",        "Black Sea Ports",     "primary"),
        ("wheat",        "Suez Canal",          "secondary"),
        ("copper",       "Chile Copper Belt",   "primary"),
        ("copper",       "Panama Canal",        "secondary"),
        ("coffee",       "Panama Canal",        "primary"),
        ("coffee",       "Suez Canal",          "secondary"),
        ("gold",         "Strait of Hormuz",    "secondary"),  # geopolitical hedge
        ("gold",         "Red Sea / Yemen",     "secondary"),
    ]
    return pd.DataFrame(mappings, columns=["commodity", "chokepoint", "link_type"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cp = generate_chokepoints()
    cp.to_csv(OUTPUT_DIR / "chokepoints.csv", index=False)
    print(f"Chokepoints: {len(cp)} entries → {OUTPUT_DIR / 'chokepoints.csv'}")

    crm = generate_commodity_region_map()
    crm.to_csv(OUTPUT_DIR / "commodity_region_map.csv", index=False)
    print(f"Commodity-region map: {len(crm)} entries → {OUTPUT_DIR / 'commodity_region_map.csv'}")


if __name__ == "__main__":
    main()
