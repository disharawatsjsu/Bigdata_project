#!/usr/bin/env python3
"""
EDA Visualizations for Supply Chain Disruption Intelligence.

Generates 9 charts for mid-presentation:
  1. Event volume heatmap by chokepoint × month
  2. Goldstein score distribution by chokepoint
  3. Chokepoint event volume bar chart
  4. Conflict ratio vs oil price overlay (FIXED)
  5. Label distribution
  6. Preprocessing funnel (raw → dedup → CAMEO → geo-filter)
  7. Feature importance bar chart
  8. Confusion matrix heatmaps (validation + test)
  9. Train/Val/Test split timeline

Run after spark_pipeline.py has produced the features parquet.

Usage:
    python eda_charts.py --features ./data/parquet/features --output ./data/charts
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (12, 6)

# --- Pipeline results from terminal output ---
PIPELINE_STATS = {
    "raw_rows": 23_395_578,
    "after_dedup": 15_687_177,
    "dedup_removed": 7_708_401,
    "supply_chain_cameo": 1_983_143,
    "after_geo_filter": 20_872,
    "daily_feature_rows": 1_383,
    "final_feature_rows": 381,
}

CHOKEPOINT_COUNTS = {
    "black_sea": 8200,
    "red_sea": 4764,
    "malacca": 2340,
    "hormuz": 2281,
    "taiwan": 1800,
    "panama": 1009,
    "suez": 459,
    "chile": 19,
}

FEATURE_IMPORTANCES = {
    "treasury_10y": 0.1940,
    "return_20d": 0.1418,
    "usd_index": 0.1327,
    "volatility_20d": 0.1209,
    "return_5d": 0.1208,
    "event_count_7d": 0.0366,
    "avg_goldstein_7d": 0.0359,
    "avg_tone_7d": 0.0352,
    "total_mentions_30d": 0.0303,
    "avg_goldstein_30d": 0.0280,
    "avg_tone_30d": 0.0268,
    "event_count_30d": 0.0262,
    "conflict_ratio_7d": 0.0259,
    "conflict_ratio_30d": 0.0248,
    "total_mentions_7d": 0.0202,
}

# Confusion matrices from terminal output
VAL_CM = {
    (0, 0): 16, (0, 1): 3,  (0, 2): 0,
    (1, 0): 45, (1, 1): 1,  (1, 2): 4,
    (2, 0): 3,  (2, 1): 0,  (2, 2): 0,
}

TEST_CM = {
    (0, 0): 9,  (0, 1): 2,  (0, 2): 1,
    (1, 0): 19, (1, 1): 0,  (1, 2): 3,
    (2, 0): 25, (2, 1): 1,  (2, 2): 8,
}


def load_data(features_path: str) -> pd.DataFrame:
    """Load feature parquet into pandas."""
    df = pd.read_parquet(features_path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


# =========================================================================
# CHART 1: Event heatmap — filter to 2024 only for clean presentation
# =========================================================================
def chart1_event_heatmap(df: pd.DataFrame, out: Path):
    df_2024 = df[df["event_date"] >= "2024-01-01"].copy()
    df_2024["ym"] = df_2024["event_date"].dt.to_period("M")
    pivot = df_2024.groupby(["chokepoint", "ym"]).agg({"event_count": "sum"}).reset_index()
    pivot["ym_str"] = pivot["ym"].astype(str)
    heat = pivot.pivot(index="chokepoint", columns="ym_str", values="event_count").fillna(0)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(heat, cmap="YlOrRd", ax=ax, annot=True, fmt=".0f",
                cbar_kws={"label": "Event Count"}, linewidths=0.5)
    ax.set_title("Monthly Supply-Chain Event Volume by Chokepoint (Jan–Jun 2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out / "01_event_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_event_heatmap.png")


# =========================================================================
# CHART 2: Goldstein distribution — KDE for smoother look
# =========================================================================
def chart2_goldstein_dist(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {"red_sea": "#e74c3c", "hormuz": "#e67e22", "suez": "#3498db"}
    for cp, color in colors.items():
        subset = df[df["chokepoint"] == cp]["avg_goldstein_7d"].dropna()
        if len(subset) > 5:
            subset.plot.kde(ax=ax, label=cp.replace("_", " ").title(), color=color, linewidth=2)
            ax.hist(subset, bins=30, alpha=0.15, color=color, density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.4, label="Neutral (0)")
    ax.set_xlabel("7-Day Avg Goldstein Score")
    ax.set_ylabel("Density")
    ax.set_title("Conflict Intensity Distribution by Oil Chokepoint", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "02_goldstein_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_goldstein_dist.png")


# =========================================================================
# CHART 3: All 8 chokepoints bar chart (from pipeline stats)
# =========================================================================
def chart3_chokepoint_bars(df: pd.DataFrame, out: Path):
    counts = pd.Series(CHOKEPOINT_COUNTS).sort_values(ascending=True)
    colors = ["#2ecc71" if k in ("hormuz", "red_sea", "suez") else "#95a5a6" for k in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate counts
    for bar, val in zip(bars, counts.values):
        ax.text(val + 100, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10)

    oil_patch = mpatches.Patch(color="#2ecc71", label="Oil-relevant (V1 scope)")
    other_patch = mpatches.Patch(color="#95a5a6", label="Other chokepoints")
    ax.legend(handles=[oil_patch, other_patch], loc="lower right")

    ax.set_xlabel("Total Supply-Chain Events (Jan–Jun 2024)")
    ax.set_title("Geo-Filtered Event Volume by Chokepoint Region", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(out / "03_chokepoint_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_chokepoint_bars.png")


# =========================================================================
# CHART 4: Conflict ratio vs oil price — FIXED
# =========================================================================
def chart4_conflict_vs_oil(df: pd.DataFrame, out: Path):
    # Use features data directly — conflict_ratio_7d from GDELT, return_5d as price proxy
    oil_regions = df[df["chokepoint"].isin(["hormuz", "red_sea", "suez"])]
    daily = (
        oil_regions.groupby("event_date")
        .agg({"conflict_ratio_7d": "mean", "event_count_7d": "mean"})
        .reset_index()
        .sort_values("event_date")
    )

    # Try reading actual oil prices from the commodity CSV
    oil = None
    try:
        prices = pd.read_csv("./data/commodities/commodity_prices.csv")
        # Handle the duplicate 'close' columns — first close column is crude oil
        prices.columns = ["date", "close_oil", "commodity", "symbol",
                          "close_wheat", "close_copper", "close_ng", "close_coffee", "close_gold"]
        prices = prices.dropna(subset=["commodity"])  # drop junk sub-header row
        prices["date"] = pd.to_datetime(prices["date"])
        prices["close_oil"] = pd.to_numeric(prices["close_oil"], errors="coerce")
        oil = prices[prices["commodity"] == "crude_oil"][["date", "close_oil"]].dropna().sort_values("date")
        # Filter to our date range
        oil = oil[(oil["date"] >= daily["event_date"].min()) & (oil["date"] <= daily["event_date"].max())]
    except Exception:
        oil = None

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Conflict ratio — filled area on left axis
    ax1.fill_between(daily["event_date"], daily["conflict_ratio_7d"],
                     alpha=0.35, color="#e74c3c", label="Conflict Ratio (7d avg)")
    ax1.plot(daily["event_date"], daily["conflict_ratio_7d"],
             color="#e74c3c", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("Conflict Ratio (7d avg)", color="#e74c3c", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_ylim(0, 1.05)

    # Oil price on right axis
    ax2 = ax1.twinx()
    if oil is not None and len(oil) > 0:
        ax2.plot(oil["date"], oil["close_oil"], color="#2980b9", linewidth=2, label="Crude Oil (USD)")
        ax2.set_ylabel("Crude Oil Price (USD)", color="#2980b9", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#2980b9")
    else:
        # Fallback: show event count on right axis
        ax2.bar(daily["event_date"], daily["event_count_7d"],
                alpha=0.3, color="#2980b9", width=1, label="Event Count (7d avg)")
        ax2.set_ylabel("Event Count (7d avg)", color="#2980b9", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#2980b9")

    ax1.set_title("Conflict Intensity near Oil Chokepoints vs Crude Oil Price",
                   fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out / "04_conflict_vs_oil.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_conflict_vs_oil.png")


# =========================================================================
# CHART 5: Label distribution — cleaner version
# =========================================================================
def chart5_label_distribution(df: pd.DataFrame, out: Path):
    label_map = {0: "Negative\nShock", 1: "Normal", 2: "Positive\nShock"}
    counts = df["label"].map(label_map).value_counts()
    order = ["Negative\nShock", "Normal", "Positive\nShock"]
    counts = counts.reindex(order)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    bars = ax.bar(order, counts.values, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 3,
                f"{val}\n({val/total:.1%})", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Count")
    ax.set_title("Price Shock Label Distribution (Crude Oil, Tercile-Based)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.25)
    plt.tight_layout()
    fig.savefig(out / "05_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_label_distribution.png")


# =========================================================================
# CHART 6: Preprocessing funnel — shows data reduction at each step
# =========================================================================
def chart6_preprocessing_funnel(out: Path):
    stages = [
        ("Raw GDELT\n(6 months)", PIPELINE_STATS["raw_rows"]),
        ("After Dedup\n(-33%)", PIPELINE_STATS["after_dedup"]),
        ("CAMEO Filter\n(14/17/18/19/20)", PIPELINE_STATS["supply_chain_cameo"]),
        ("Geo-Filter\n(8 chokepoints)", PIPELINE_STATS["after_geo_filter"]),
        ("Daily Features\n(date × region)", PIPELINE_STATS["daily_feature_rows"]),
        ("Final Features\n(oil join)", PIPELINE_STATS["final_feature_rows"]),
    ]

    labels, values = zip(*stages)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(values)))
    bars = ax.barh(range(len(values)), values, color=colors, edgecolor="white", height=0.6)

    for i, (bar, val) in enumerate(zip(bars, values)):
        # Label inside or outside depending on bar width
        if val > max(values) * 0.15:
            ax.text(val * 0.5, i, f"{val:,.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        else:
            ax.text(val + max(values) * 0.02, i, f"{val:,.0f}", ha="left", va="center",
                    fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Row Count")
    ax.set_title("Data Preprocessing Funnel: 23.4M Raw → 381 Training Features",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out / "06_preprocessing_funnel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 06_preprocessing_funnel.png")


# =========================================================================
# CHART 7: Feature importance — colored by feature category
# =========================================================================
def chart7_feature_importance(out: Path):
    feats = pd.Series(FEATURE_IMPORTANCES).sort_values(ascending=True)

    # Color by category
    def get_color(name):
        if name in ("treasury_10y", "usd_index"):
            return "#9b59b6"  # purple — macro
        elif name in ("return_5d", "return_20d", "volatility_20d"):
            return "#2980b9"  # blue — price
        else:
            return "#e74c3c"  # red — GDELT

    colors = [get_color(f) for f in feats.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feats.index, feats.values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, feats.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)

    # Legend
    macro_patch = mpatches.Patch(color="#9b59b6", label="Macro indicators")
    price_patch = mpatches.Patch(color="#2980b9", label="Price features")
    gdelt_patch = mpatches.Patch(color="#e74c3c", label="GDELT features")
    ax.legend(handles=[price_patch, macro_patch, gdelt_patch], loc="lower right", fontsize=10)

    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("Feature Importance — Price & Macro Dominate, GDELT Signal Is Weak (V1)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "07_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 07_feature_importance.png")


# =========================================================================
# CHART 8: Confusion matrices — validation and test side by side
# =========================================================================
def chart8_confusion_matrices(out: Path):
    label_names = ["Neg Shock", "Normal", "Pos Shock"]

    def cm_dict_to_array(cm_dict):
        arr = np.zeros((3, 3), dtype=int)
        for (true, pred), count in cm_dict.items():
            arr[true][pred] = count
        return arr

    val_arr = cm_dict_to_array(VAL_CM)
    test_arr = cm_dict_to_array(TEST_CM)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, arr, title, f1 in [
        (ax1, val_arr, "Validation", "F1: 0.128"),
        (ax2, test_arr, "Test", "F1: 0.223"),
    ]:
        sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_names, yticklabels=label_names,
                    linewidths=1, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{title} Set ({f1})", fontsize=13, fontweight="bold")

    fig.suptitle("Confusion Matrices — V1 Preliminary Results (381 samples, 208 train)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out / "08_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 08_confusion_matrices.png")


# =========================================================================
# CHART 9: Train/Val/Test timeline showing the time-based split
# =========================================================================
def chart9_train_val_test_split(df: pd.DataFrame, out: Path):
    df_2024 = df[df["event_date"] >= "2024-01-01"].copy()
    daily_counts = df_2024.groupby("event_date").agg({"event_count": "sum"}).reset_index()
    daily_counts = daily_counts.sort_values("event_date")

    # Split boundaries (from pipeline output: 178 days, 60/20/20)
    train_end = pd.Timestamp("2024-04-17")
    val_end = pd.Timestamp("2024-05-23")

    fig, ax = plt.subplots(figsize=(14, 4))

    for _, row in daily_counts.iterrows():
        d = row["event_date"]
        if d < train_end:
            color = "#2ecc71"
        elif d < val_end:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        ax.bar(d, row["event_count"], color=color, width=1, alpha=0.7)

    ax.axvline(train_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(val_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.text(train_end - pd.Timedelta(days=20), ax.get_ylim()[1] * 0.9,
            "Train (208)", fontsize=11, fontweight="bold", color="#2ecc71")
    ax.text(train_end + pd.Timedelta(days=5), ax.get_ylim()[1] * 0.9,
            "Val (72)", fontsize=11, fontweight="bold", color="#f39c12")
    ax.text(val_end + pd.Timedelta(days=5), ax.get_ylim()[1] * 0.9,
            "Test (68)", fontsize=11, fontweight="bold", color="#e74c3c")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Daily Event Count")
    ax.set_title("Time-Based Train/Val/Test Split (No Data Leakage)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "09_train_val_test_split.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 09_train_val_test_split.png")


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate EDA charts")
    parser.add_argument("--features", default="./data/parquet/features")
    parser.add_argument("--output", default="./data/charts")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = load_data(args.features)
    print(f"  {len(df):,} rows, date range: {df['event_date'].min()} → {df['event_date'].max()}")

    print("\nGenerating charts:")
    chart1_event_heatmap(df, out)
    chart2_goldstein_dist(df, out)
    chart3_chokepoint_bars(df, out)
    chart4_conflict_vs_oil(df, out)
    chart5_label_distribution(df, out)
    chart6_preprocessing_funnel(out)
    chart7_feature_importance(out)
    chart8_confusion_matrices(out)
    chart9_train_val_test_split(df, out)

    print(f"\nAll 9 charts saved to {out.resolve()}")


if __name__ == "__main__":
    main()