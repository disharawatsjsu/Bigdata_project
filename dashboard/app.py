"""Map-centered Dash dashboard for supply-chain chokepoint intelligence."""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots

from components import CATEGORIES, CHOKEPOINT_COORDS
from queries import load_all_tables, load_shap_local_v2, run_cached_sql, run_hive_sql
from theme import ACCENT, BACKGROUND, BORDER, COMMODITIES, FONT_MONO, MUTED, PANEL, RED, TEXT, YELLOW

CHOKEPOINT_ORDER = ["suez", "red_sea", "hormuz", "black_sea", "malacca", "taiwan", "panama", "chile"]
WINDOWS = {"1d": 1, "7d": 7, "30d": 30, "1y": 365, "3y": 1095}
QUERY_CACHE: dict[tuple[str, str], dict] = {}
DATA_CACHE: dict = {
    "tables": {},
    "shap_local_v2": {},
    "shap_errors": {},
    "timestamp": "not loaded",
    "error": None,
}


def _load_dotenv() -> None:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def refresh_cache() -> dict:
    tables, timestamp, error = load_all_tables()
    shap_frames, shap_errors = load_shap_local_v2()
    DATA_CACHE.update(
        {
            "tables": tables,
            "shap_local_v2": shap_frames,
            "shap_errors": shap_errors,
            "timestamp": timestamp,
            "error": error,
        }
    )
    QUERY_CACHE.clear()
    return DATA_CACHE


refresh_cache()

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Supply Chain Intelligence",
)
server = app.server


def get_table(name: str) -> pd.DataFrame:
    table = DATA_CACHE.get("tables", {}).get(name)
    return table if table is not None else pd.DataFrame()


def max_trigger_date() -> pd.Timestamp:
    trigger_log = get_table("trigger_log")
    if trigger_log.empty or "event_date" not in trigger_log:
        return pd.Timestamp.utcnow().normalize().tz_localize(None)
    return trigger_log["event_date"].dropna().max().normalize()


def empty_fig(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": MUTED})
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        height=height,
        margin={"l": 16, "r": 16, "t": 16, "b": 16},
        font={"family": FONT_MONO, "color": TEXT},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def window_bounds(window_key: str, custom_date: str | None) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    end_date = pd.to_datetime(custom_date).normalize() if custom_date else max_trigger_date()
    if window_key == "custom":
        days = 1
    else:
        days = WINDOWS.get(window_key or "30d", 30)
    return end_date - timedelta(days=days - 1), end_date, window_key or "30d"


def trigger_window(trigger_log: pd.DataFrame, cp: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trigger_log.empty:
        return pd.DataFrame()
    df = trigger_log.copy()
    return df[
        (df["chokepoint"].astype(str) == cp)
        & (df["event_date"].dt.normalize() >= start)
        & (df["event_date"].dt.normalize() <= end)
    ].copy()


def is_triggered(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    if "trigger_type" in frame.columns:
        return frame["trigger_type"].notna() & (frame["trigger_type"].astype(str) != "")
    return pd.to_numeric(frame.get("event_count_1d", 0), errors="coerce").fillna(0) > 0


def build_map_sparkline(window_key: str, custom_date: str | None, selected: list[str]) -> go.Figure:
    trigger_log = get_table("trigger_log")
    if trigger_log.empty:
        return empty_fig("Hive table trigger_log is empty.", 600)
    start, end, _ = window_bounds(window_key, custom_date)
    one_day_window = start.normalize() == end.normalize()
    timeline_center = start + (end - start) / 2

    rows = []
    for cp in CHOKEPOINT_ORDER:
        frame = trigger_window(trigger_log, cp, start, end)
        count = float(pd.to_numeric(frame.get("event_count_1d", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        gold = pd.to_numeric(frame.get("avg_goldstein_1d", pd.Series(dtype=float)), errors="coerce").mean()
        triggered = bool(is_triggered(frame).any())
        coord = CHOKEPOINT_COORDS[cp]
        rows.append(
            {
                "chokepoint": cp,
                "label": coord["label"],
                "lat": coord["lat"],
                "lon": coord["lon"],
                "event_count": count,
                "avg_goldstein": gold if pd.notna(gold) else 0,
                "triggered": triggered,
            }
        )
    map_df = pd.DataFrame(rows)

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "geo"}], [{"type": "xy"}]],
        row_heights=[0.70, 0.30],
        vertical_spacing=0.02,
    )
    if not map_df.empty:
        max_count = max(float(map_df["event_count"].max()), 1.0)
        map_df["size"] = 8 + (np.sqrt(map_df["event_count"].clip(lower=0)) / np.sqrt(max_count)) * 22
        map_df["size"] = map_df.apply(lambda row: row["size"] + 8 if row["chokepoint"] in selected else row["size"], axis=1)
        map_df["line_width"] = map_df["chokepoint"].map(lambda cp: 6 if cp in selected else 2)
        map_df["line_color"] = map_df["chokepoint"].map(lambda cp: YELLOW if cp in selected else ACCENT if bool(map_df.loc[map_df["chokepoint"] == cp, "triggered"].iloc[0]) else BORDER)
        map_df["label_text"] = map_df.apply(lambda row: f"SELECTED: {row['label']}" if row["chokepoint"] in selected else row["label"], axis=1)
        fig.add_trace(
            go.Scattergeo(
                lon=map_df["lon"],
                lat=map_df["lat"],
                mode="markers+text",
                text=map_df["label_text"],
                textposition="top center",
                marker={
                    "size": map_df["size"],
                    "color": map_df["avg_goldstein"].clip(-5, 0),
                    "colorscale": "Reds_r",
                    "cmin": -5,
                    "cmax": 0,
                    "line": {"color": map_df["line_color"], "width": map_df["line_width"]},
                    "colorbar": {"title": "Goldstein", "len": 0.58, "tickfont": {"family": FONT_MONO, "color": MUTED}},
                },
                customdata=np.column_stack([np.repeat("map", len(map_df)), map_df["chokepoint"], map_df["event_count"], map_df["avg_goldstein"], map_df["triggered"]]),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "events=%{customdata[2]:.0f}<br>"
                    "avg_goldstein=%{customdata[3]:.2f}<br>"
                    "triggered=%{customdata[4]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    for idx, cp in enumerate(CHOKEPOINT_ORDER):
        lane = len(CHOKEPOINT_ORDER) - idx
        cp_df = trigger_window(trigger_log, cp, start, end).sort_values("event_date").copy()
        fig.add_trace(
            go.Scatter(
                x=[timeline_center, timeline_center] if one_day_window else [start, end],
                y=[lane, lane],
                mode="lines",
                line={"color": "#1a2030", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        if cp_df.empty:
            continue
        events = pd.to_numeric(cp_df["event_count_1d"], errors="coerce").fillna(0)
        max_events = max(float(events.max()), 1.0)
        dot_color = YELLOW if cp in selected else RED
        fig.add_trace(
            go.Scatter(
                x=np.full(len(cp_df), timeline_center) if one_day_window else cp_df["event_date"],
                y=np.full(len(cp_df), lane),
                mode="markers",
                marker={
                    "size": 6 + (np.sqrt(events) / np.sqrt(max_events)) * (8 if cp in selected else 5),
                    "color": dot_color,
                    "opacity": 0.55 if cp not in selected else 0.95,
                    "line": {"color": BACKGROUND, "width": 0.5},
                },
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        if cp in selected and len(cp_df) >= 2:
            avg_y = lane + (events.rolling(7, min_periods=1).mean() / max_events) * 0.32
            fig.add_trace(
                go.Scatter(
                    x=cp_df["event_date"],
                    y=avg_y,
                    mode="lines",
                    line={"color": YELLOW, "width": 2.5},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#0a0e1a",
        showocean=True,
        oceancolor="#050810",
        showcoastlines=True,
        coastlinecolor="#1a2030",
        showcountries=True,
        countrycolor="#1a2030",
        showframe=False,
        bgcolor=BACKGROUND,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(1, len(CHOKEPOINT_ORDER) + 1)),
        ticktext=list(reversed(CHOKEPOINT_ORDER)),
        range=[0.5, len(CHOKEPOINT_ORDER) + 0.65],
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )
    if one_day_window:
        fig.update_xaxes(showgrid=False, showticklabels=False, ticks="", range=[timeline_center - timedelta(hours=12), timeline_center + timedelta(hours=12)], row=2, col=1)
    elif window_key == "7d":
        seven_days = pd.date_range(start=start, end=end, freq="D")
        fig.update_xaxes(showgrid=False, tickmode="array", tickvals=seven_days, ticktext=[d.strftime("%b %d") for d in seven_days], range=[start - timedelta(hours=12), end + timedelta(hours=12)], row=2, col=1)
    else:
        fig.update_xaxes(showgrid=False, tickformat="%b '%y", row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin={"l": 80, "r": 16, "t": 8, "b": 28},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font={"family": FONT_MONO, "color": TEXT},
        hoverlabel={"bgcolor": "#050810", "bordercolor": ACCENT, "font": {"family": FONT_MONO, "color": TEXT}},
        showlegend=False,
    )
    return fig


def metric_color(value: float | None) -> str:
    if value is None or pd.isna(value):
        return MUTED
    if value < -3:
        return RED
    if value <= 0:
        return YELLOW
    return ACCENT


def fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def shap_bars(chokepoint: str, target: str) -> dcc.Graph:
    shap = get_table("shap_attribution_v2")
    target_name = "abs_return_5d_fwd" if target == "5d" else "abs_return_20d_fwd"
    if shap.empty:
        rows = []
        for (commodity, local_target), frame in DATA_CACHE.get("shap_local_v2", {}).items():
            if local_target != target_name or frame.empty:
                continue
            subset = frame[frame["feature_name"].astype(str).str.startswith(f"{chokepoint}_")]
            if subset.empty:
                continue
            rows.append({"commodity": commodity, "mean_abs_shap": float(subset["shap_value"].abs().mean())})
        subset = pd.DataFrame(rows)
    else:
        subset = shap[(shap["chokepoint"] == chokepoint) & (shap["target_horizon"].astype(str).str.contains("5d" if target == "5d" else "20d"))].copy()
    if subset.empty:
        return dcc.Graph(figure=empty_fig(f"No {target} attribution", 180), config={"displayModeBar": False})
    subset = subset.sort_values("mean_abs_shap", ascending=True)
    fig = go.Figure(go.Bar(x=subset["mean_abs_shap"], y=subset["commodity"], orientation="h", marker={"color": ACCENT}))
    fig.update_layout(template="plotly_dark", height=180, margin={"l": 70, "r": 40, "t": 22, "b": 20}, title=f"{target} horizon", paper_bgcolor=PANEL, plot_bgcolor=PANEL, font={"family": FONT_MONO, "color": TEXT}, showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, className="mini-chart")


def top_features(chokepoint: str) -> html.Div:
    per_commodity = []
    for (commodity, target), frame in DATA_CACHE.get("shap_local_v2", {}).items():
        if target != "abs_return_5d_fwd" or frame.empty:
            continue
        subset = frame[frame["feature_name"].astype(str).str.startswith(f"{chokepoint}_")]
        if not subset.empty:
            feature_means = (
                subset.assign(abs_shap=subset["shap_value"].abs())
                .groupby("feature_name", as_index=False)["abs_shap"]
                .mean()
            )
            feature_means["commodity"] = commodity
            per_commodity.append(feature_means)
    if not per_commodity:
        return html.Div(
            [
                html.Div(f"Top 3 features for {chokepoint} (5d horizon):", className="block-title"),
                html.Div("No matching 5d SHAP features found.", className="muted-note"),
            ]
        )
    combined = pd.concat(per_commodity, ignore_index=True)
    top = combined.groupby("feature_name", as_index=False)["abs_shap"].mean().sort_values("abs_shap", ascending=False).head(3)
    return html.Div(
        [
            html.Div(f"Top 3 features for {chokepoint} (5d horizon):", className="block-title"),
            *[
                html.Div(
                    [
                        html.Span(row["feature_name"]),
                        html.Span(f"{row['abs_shap']:.4f}", style={"textAlign": "right"}),
                    ],
                    className="metric-line feature-line",
                )
                for _, row in top.iterrows()
            ],
        ]
    )


def local_shap_exposure(chokepoint: str, commodity: str, target: str = "abs_return_5d_fwd") -> float:
    frame = DATA_CACHE.get("shap_local_v2", {}).get((commodity, target))
    if frame is None or frame.empty:
        return 0.0
    subset = frame[frame["feature_name"].astype(str).str.startswith(f"{chokepoint}_")]
    if subset.empty:
        return 0.0
    return float(subset["shap_value"].abs().mean())


def detail_subpanel(chokepoint: str, window_key: str, custom_date: str | None) -> html.Div:
    trigger_log = get_table("trigger_log")
    start, end, _ = window_bounds(window_key, custom_date)
    frame = trigger_window(trigger_log, chokepoint, start, end)
    if frame.empty:
        metrics = {"events": 0, "trigger_days": 0, "gold": None, "tone": None, "extreme": "n/a"}
    else:
        events = pd.to_numeric(frame["event_count_1d"], errors="coerce").fillna(0)
        gold = pd.to_numeric(frame["avg_goldstein_1d"], errors="coerce")
        tone = pd.to_numeric(frame.get("avg_tone_1d", pd.Series(index=frame.index)), errors="coerce")
        extreme_row = frame.assign(_events=events, _gold=gold).sort_values(["_events", "_gold"], ascending=[False, True]).iloc[0]
        metrics = {
            "events": int(events.sum()),
            "trigger_days": int(is_triggered(frame).sum()),
            "gold": float(gold.mean()) if gold.notna().any() else None,
            "tone": float(tone.mean()) if tone.notna().any() else None,
            "extreme": f"{pd.to_datetime(extreme_row['event_date']).date()} ({int(extreme_row['_events'])} events, goldstein {fmt(extreme_row['_gold'])})",
        }
    label = CHOKEPOINT_COORDS.get(chokepoint, {}).get("label", chokepoint)
    return html.Div(
        [
            html.Div(f"{label.upper()} — window: {window_key}, ending {end.date()}", className="detail-title"),
            html.Div(
                [
                    html.Div([html.Span("Events in window"), html.Span(f"{metrics['events']:,}")], className="metric-line"),
                    html.Div([html.Span("Trigger days in window"), html.Span(f"{metrics['trigger_days']:,}")], className="metric-line"),
                    html.Div([html.Span("Avg goldstein"), html.Span(fmt(metrics["gold"]), style={"color": metric_color(metrics["gold"])})], className="metric-line"),
                    html.Div([html.Span("Avg tone"), html.Span(fmt(metrics["tone"]))], className="metric-line"),
                    html.Div([html.Span("Most extreme day"), html.Span(metrics["extreme"])], className="metric-line"),
                ],
                className="metrics-block",
            ),
            html.Div("Sector volatility outlook from this chokepoint", className="block-title"),
            html.Div([shap_bars(chokepoint, "5d"), shap_bars(chokepoint, "20d")], className="two-mini-charts"),
            top_features(chokepoint),
        ],
        className="compare-subpanel",
    )


def detail_panel(selected: list[str], window_key: str, custom_date: str | None) -> html.Div:
    if not selected:
        return html.Div("Click a chokepoint marker to inspect", className="placeholder")
    return detail_subpanel(selected[0], window_key, custom_date)


def market_sql(chokepoint: str, selected_date: str) -> str:
    start = pd.to_datetime(selected_date).date()
    end = (pd.to_datetime(selected_date) + timedelta(days=25)).date()
    return f"""SELECT commodity, price_date, close_price
FROM market_prices
WHERE price_date BETWEEN '{start}' AND '{end}'
ORDER BY commodity, price_date"""


def query_sql(chokepoint: str, action: str, selected_date: str | None = None) -> str:
    if action == "peak":
        return f"""SELECT event_date, event_count_1d, avg_goldstein_1d
FROM trigger_log
WHERE chokepoint = '{chokepoint}'
ORDER BY event_count_1d DESC
LIMIT 1"""
    if action == "last_sigma":
        return f"""WITH stats AS (
  SELECT AVG(event_count_1d) AS mu, STDDEV(event_count_1d) AS sigma
  FROM trigger_log WHERE chokepoint = '{chokepoint}'
)
SELECT event_date, event_count_1d, avg_goldstein_1d
FROM trigger_log, stats
WHERE chokepoint = '{chokepoint}' AND event_count_1d > stats.mu + stats.sigma
ORDER BY event_date DESC
LIMIT 1"""
    if action == "severity":
        return f"""SELECT event_date, event_count_1d, avg_goldstein_1d
FROM trigger_log
WHERE chokepoint = '{chokepoint}'
ORDER BY avg_goldstein_1d ASC
LIMIT 10"""
    date = pd.to_datetime(selected_date or max_trigger_date()).normalize()
    return f"""SELECT event_date, event_count_1d, avg_goldstein_1d
FROM trigger_log
WHERE chokepoint = '{chokepoint}'
  AND event_date BETWEEN '{(date - timedelta(days=7)).date()}' AND '{(date + timedelta(days=7)).date()}'
ORDER BY event_date"""


def run_dashboard_sql(sql: str, limit: int = 200) -> tuple[pd.DataFrame, str | None]:
    if os.environ.get("DASH_DATA_MODE", "auto").strip().lower() == "webhdfs":
        return run_cached_sql(DATA_CACHE.get("tables", {}), sql, limit=limit)
    frame, error = run_hive_sql(sql, limit=limit)
    if error and DATA_CACHE.get("tables"):
        cached_frame, cached_error = run_cached_sql(DATA_CACHE.get("tables", {}), sql, limit=limit)
        if cached_error is None:
            return cached_frame, None
    return frame, error


def run_visible_query(chokepoint: str, action: str, selected_date: str | None = None) -> dict:
    cache_key = (chokepoint, action)
    if action != "custom" and cache_key in QUERY_CACHE:
        cached = QUERY_CACHE[cache_key].copy()
        cached["cached"] = True
        return cached
    sql = query_sql(chokepoint, action, selected_date)
    start = time.perf_counter()
    frame, error = run_dashboard_sql(sql, limit=200)
    duration_ms = int((time.perf_counter() - start) * 1000)
    result = {"sql": sql, "rows": frame.to_dict("records"), "error": error, "duration_ms": duration_ms, "cached": False, "action": action}
    if action != "custom":
        QUERY_CACHE[cache_key] = result.copy()
    return result


def run_market_query(chokepoint: str, selected_date: str) -> dict:
    sql = market_sql(chokepoint, selected_date)
    start = time.perf_counter()
    frame, error = run_dashboard_sql(sql, limit=5000)
    duration_ms = int((time.perf_counter() - start) * 1000)
    return {"sql": sql, "rows": frame.to_dict("records"), "error": error, "duration_ms": duration_ms, "cached": False, "action": "market_prices"}


def commodity_panel(selected: list[str], selected_date: str | None, query_payload: dict | None) -> html.Div:
    if not selected or not selected_date:
        return html.Div("Select a chokepoint to view post-event sector movement.", className="placeholder")
    chokepoint = selected[0]
    if not query_payload or query_payload.get("action") != "market_prices":
        query_payload = run_market_query(chokepoint, selected_date)
    if query_payload.get("error"):
        return html.Div(f"market_prices query failed: {query_payload['error']}", className="error-banner", style={"display": "block"})
    prices = pd.DataFrame(query_payload.get("rows", []))
    if prices.empty:
        return html.Div("market_prices query returned no rows.", className="placeholder")
    prices["price_date"] = pd.to_datetime(prices["price_date"], errors="coerce")
    prices["close_price"] = pd.to_numeric(prices["close_price"], errors="coerce")
    event_date = pd.to_datetime(selected_date).normalize()
    rows = []
    attrs = get_table("shap_attribution_v2")
    for commodity in COMMODITIES:
        c = prices[prices["commodity"] == commodity].sort_values("price_date").copy()
        if c.empty:
            continue
        base = c[c["price_date"].dt.normalize() >= event_date].head(1)
        if base.empty:
            continue
        base_price = float(base.iloc[0]["close_price"])
        def future_return(days: int):
            target = c[c["price_date"].dt.normalize() >= event_date + timedelta(days=days)].head(1)
            if target.empty:
                return np.nan
            return float(target.iloc[0]["close_price"] / base_price - 1.0)
        r5 = future_return(5)
        r20 = future_return(20)
        hist = get_table("market_prices")
        hist_c = hist[hist["commodity"] == commodity].sort_values("price_date").copy()
        hist_c["ret20"] = hist_c["close_price"].pct_change(20)
        hist_c["mean30"] = hist_c["ret20"].rolling(30, min_periods=5).mean()
        hist_c["std30"] = hist_c["ret20"].rolling(30, min_periods=5).std()
        hist_row = hist_c[hist_c["price_date"].dt.normalize() <= event_date].tail(1)
        mean30 = float(hist_row["mean30"].iloc[0]) if not hist_row.empty and pd.notna(hist_row["mean30"].iloc[0]) else 0.0
        std30 = float(hist_row["std30"].iloc[0]) if not hist_row.empty and pd.notna(hist_row["std30"].iloc[0]) and hist_row["std30"].iloc[0] != 0 else np.nan
        sigma5 = (r5 - mean30) / std30 if pd.notna(std30) else np.nan
        sigma20 = (r20 - mean30) / std30 if pd.notna(std30) else np.nan
        attr_row = attrs[(attrs["chokepoint"] == chokepoint) & (attrs["commodity"] == commodity) & (attrs["target_horizon"].astype(str).str.contains("5d"))] if not attrs.empty else pd.DataFrame()
        attr = float(attr_row["mean_abs_shap"].mean()) if not attr_row.empty else local_shap_exposure(chokepoint, commodity)
        rows.append({"commodity": commodity, "price": base_price, "r5": r5, "s5": sigma5, "r20": r20, "s20": sigma20, "attr": attr})
    max_attr = max([r["attr"] for r in rows] or [1.0])
    rows = sorted(rows, key=lambda r: abs(r["s20"]) if pd.notna(r["s20"]) else -1, reverse=True)
    return html.Div(
        [
            html.Div("POST-EVENT SECTOR MOVEMENT", className="detail-title"),
            html.Div(f"Selected date: {event_date.date()}", className="detail-subtitle"),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th(c) for c in ["sector", "5d %", "5d σ", "20d %", "20d σ", "model attribution"]])),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(r["commodity"]),
                                    html.Td("" if pd.isna(r["r5"]) else f"{r['r5'] * 100:+.2f}%"),
                                    html.Td("" if pd.isna(r["s5"]) else f"{r['s5']:+.2f}", className="sigma-hot" if pd.notna(r["s5"]) and abs(r["s5"]) > 2 else "sigma-warm" if pd.notna(r["s5"]) and abs(r["s5"]) > 1 else ""),
                                    html.Td("" if pd.isna(r["r20"]) else f"{r['r20'] * 100:+.2f}%"),
                                    html.Td("" if pd.isna(r["s20"]) else f"{r['s20']:+.2f}", className="sigma-hot" if pd.notna(r["s20"]) and abs(r["s20"]) > 2 else "sigma-warm" if pd.notna(r["s20"]) and abs(r["s20"]) > 1 else ""),
                                    html.Td(html.Div(html.Div(style={"width": f"{(r['attr'] / max_attr * 100) if max_attr else 0:.1f}%"}), className="attr-bar")),
                                ]
                            )
                            for r in rows
                        ]
                    ),
                ],
                className="commodity-table",
            ),
            html.Details(
                [
                    html.Summary("Model performance note"),
                    performance_table(),
                    html.Div(
                        "Predictive performance was insufficient to integrate live forecasts. Model output is used here only for chokepoint-sector attribution patterns (SHAP), which capture historical exposure relationships.",
                        className="muted-note",
                    ),
                ],
                className="performance-note",
            ),
        ]
    )


def performance_table() -> dash_table.DataTable:
    metrics = get_table("regression_metrics")
    if metrics.empty:
        return dash_table.DataTable(data=[], columns=[])
    rows = []
    for commodity in COMMODITIES:
        item = {"sector": commodity}
        for horizon in ("5d", "20d"):
            subset = metrics[(metrics["commodity"] == commodity) & (metrics["target"].astype(str).str.contains(horizon))]
            if subset.empty:
                item[f"{horizon} R²"] = "n/a"
                item[f"{horizon} MAE vs naive"] = "n/a"
            else:
                row = subset.iloc[0]
                item[f"{horizon} R²"] = fmt(row.get("test_r2"), 3)
                item[f"{horizon} MAE vs naive"] = f"{fmt(row.get('test_mae'), 4)} / {fmt(row.get('naive_mae'), 4)}"
        rows.append(item)
    return dash_table.DataTable(data=rows, columns=[{"name": c, "id": c} for c in rows[0]], page_size=6, style_as_list_view=True, style_header=table_header_style(), style_cell=table_cell_style())


def table_header_style() -> dict:
    return {"backgroundColor": BACKGROUND, "color": ACCENT, "borderBottom": f"1px solid {BORDER}", "fontFamily": FONT_MONO}


def table_cell_style() -> dict:
    return {"backgroundColor": BACKGROUND, "color": TEXT, "border": f"1px solid {BORDER}", "fontFamily": FONT_MONO, "fontSize": "12px", "padding": "7px", "textAlign": "right"}


def query_panel(payload: dict | None) -> html.Div:
    if not payload:
        return html.Div("No query run yet — click a button or table row above", className="placeholder")
    rows = payload.get("rows", [])
    frame = pd.DataFrame(rows)
    title = f"HIVE QUERY — {payload.get('duration_ms', 0)}ms {'(cached)' if payload.get('cached') else '(fresh)'}"
    if payload.get("error"):
        body = html.Div(f"Query failed: {payload['error']}", className="error-banner", style={"display": "block"})
    elif frame.empty:
        body = html.Div("Query returned no rows.", className="placeholder")
    else:
        body = dash_table.DataTable(
            id="query-result-table",
            data=frame.to_dict("records"),
            columns=[{"name": c, "id": c} for c in frame.columns],
            page_size=10,
            row_selectable="single",
            style_as_list_view=True,
            style_header=table_header_style(),
            style_cell=table_cell_style(),
            style_data_conditional=[{"if": {"state": "selected"}, "backgroundColor": "#102437", "border": f"1px solid {ACCENT}"}],
        )
    return html.Div(
        [
            html.Div(title, className="panel-title"),
            html.Div([html.Pre(payload.get("sql", ""), className="sql-code"), html.Div(body, className="sql-result")], className="query-grid"),
        ]
    )


def build_layout() -> html.Div:
    error = DATA_CACHE.get("error")
    default_date = max_trigger_date().date().isoformat()
    return html.Div(
        [
            dcc.Store(id="refresh-token", data=0),
            dcc.Store(id="selected-chokepoints", data=[]),
            dcc.Store(id="selected-date", data=None),
            dcc.Store(id="selected-action", data="peak"),
            html.Div(
                [
                    html.Div("Supply Chain Intelligence", className="app-title"),
                    html.Div("", className="tagline"),
                    html.Div([html.Div(f"Last refresh: {DATA_CACHE['timestamp']}", id="last-refresh"), html.Button("Refresh", id="refresh-button", n_clicks=0, className="refresh-button")], className="refresh-strip"),
                ],
                className="header-strip",
            ),
            html.Div(error or "", id="error-banner", className="error-banner", style={"display": "block" if error else "none"}),
            dcc.Loading(html.Div(dcc.Graph(id="activity-map", config={"displayModeBar": False}), className="panel map-shell"), color=ACCENT, type="circle"),
            html.Div(
                [
                    html.Div([html.Span("WINDOW"), dcc.RadioItems(id="window-radio", value="30d", options=[{"label": k, "value": k} for k in ["1d", "7d", "30d", "1y", "3y"]], className="button-radio"), dcc.DatePickerSingle(id="custom-date", date=default_date, display_format="YYYY-MM-DD", className="date-picker")], className="control-group window-control-group"),
                    html.Div(
                        [
                            html.Span("HIVE QUERY"),
                            html.Button("All time", id="peak-button", n_clicks=0, className="ghost-button active-query"),
                            html.Button("Last trigger", id="sigma-button", n_clicks=0, className="ghost-button"),
                            html.Button("Top 10", id="severity-button", n_clicks=0, className="ghost-button"),
                            html.Button("Reset", id="reset-query-button", n_clicks=0, className="ghost-button"),
                        ],
                        className="control-group query-control-group",
                    ),
                ],
                className="control-row",
            ),
            html.Div(
                [
                    html.Div(id="left-detail-panel", className="panel row4-panel"),
                    html.Div(id="right-commodity-panel", className="panel row4-panel"),
                ],
                className="row4-grid",
            ),
            dcc.Loading(
                [
                    dcc.Store(id="last-query", data=None),
                    html.Div(id="hive-query-panel", className="panel query-panel"),
                ],
                color=ACCENT,
                type="cube",
                parent_className="hive-loading-shell",
            ),
        ],
        style={"backgroundColor": BACKGROUND, "minHeight": "100vh", "color": TEXT},
    )


app.layout = build_layout


@app.callback(
    Output("last-refresh", "children"),
    Output("error-banner", "children"),
    Output("error-banner", "style"),
    Output("refresh-token", "data"),
    Input("refresh-button", "n_clicks"),
    State("refresh-token", "data"),
    prevent_initial_call=True,
)
def refresh_data(_n_clicks: int, token: int):
    refresh_cache()
    error = DATA_CACHE.get("error")
    return f"Last refresh: {DATA_CACHE['timestamp']}", error or "", {"display": "block" if error else "none"}, (token or 0) + 1


@app.callback(
    Output("selected-chokepoints", "data"),
    Output("selected-date", "data"),
    Output("last-query", "data", allow_duplicate=True),
    Input("activity-map", "clickData"),
    State("selected-chokepoints", "data"),
    State("selected-action", "data"),
    prevent_initial_call=True,
)
def select_from_map(click_data, selected, selected_action):
    selected = selected or []
    if not click_data:
        return selected, no_update, no_update
    point = click_data["points"][0]
    custom = point.get("customdata") or []
    if custom and str(custom[0]) == "map" and str(custom[1]) in CHOKEPOINT_COORDS:
        cp = str(custom[1])
        query_payload = run_visible_query(cp, selected_action or "peak")
        rows = query_payload.get("rows", [])
        selected_date = pd.to_datetime(rows[0]["event_date"]).date().isoformat() if rows else no_update
        return [cp], selected_date, query_payload
    return selected, no_update, no_update


@app.callback(
    Output("activity-map", "figure"),
    Input("window-radio", "value"),
    Input("custom-date", "date"),
    Input("selected-chokepoints", "data"),
    Input("refresh-token", "data"),
)
def update_activity_map(window_key, custom_date, selected, _token):
    triggered = ctx.triggered_id
    use_custom = triggered == "custom-date"
    return build_map_sparkline(window_key, custom_date if use_custom else None, selected or [])


@app.callback(
    Output("left-detail-panel", "children"),
    Input("selected-chokepoints", "data"),
    Input("window-radio", "value"),
    Input("custom-date", "date"),
    Input("refresh-token", "data"),
)
def update_left_panel(selected, window_key, custom_date, _token):
    use_custom = ctx.triggered_id == "custom-date"
    return detail_panel(selected or [], "custom" if use_custom else window_key or "30d", custom_date if use_custom else None)


@app.callback(
    Output("selected-action", "data"),
    Output("peak-button", "className"),
    Output("sigma-button", "className"),
    Output("severity-button", "className"),
    Output("last-query", "data", allow_duplicate=True),
    Output("selected-date", "data", allow_duplicate=True),
    Input("peak-button", "n_clicks"),
    Input("sigma-button", "n_clicks"),
    Input("severity-button", "n_clicks"),
    State("selected-action", "data"),
    State("selected-chokepoints", "data"),
    prevent_initial_call=True,
)
def select_query_action(_peak, _sigma, _severity, selected_action, selected):
    action = {"peak-button": "peak", "sigma-button": "last_sigma", "severity-button": "severity"}.get(ctx.triggered_id, selected_action or "peak")
    classes = {
        "peak": "ghost-button active-query",
        "last_sigma": "ghost-button active-query",
        "severity": "ghost-button active-query",
    }
    query_payload = no_update
    selected_date = no_update
    if selected:
        query_payload = run_visible_query(selected[0], action)
        rows = query_payload.get("rows", [])
        selected_date = pd.to_datetime(rows[0]["event_date"]).date().isoformat() if rows and "event_date" in rows[0] else no_update
    return (
        action,
        classes["peak"] if action == "peak" else "ghost-button",
        classes["last_sigma"] if action == "last_sigma" else "ghost-button",
        classes["severity"] if action == "severity" else "ghost-button",
        query_payload,
        selected_date,
    )


@app.callback(
    Output("last-query", "data", allow_duplicate=True),
    Output("selected-date", "data", allow_duplicate=True),
    Output("selected-chokepoints", "data", allow_duplicate=True),
    Input("reset-query-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_query(_reset):
    return None, None, []


@app.callback(
    Output("selected-date", "data", allow_duplicate=True),
    Output("last-query", "data", allow_duplicate=True),
    Input("query-result-table", "selected_rows"),
    State("query-result-table", "data"),
    State("selected-chokepoints", "data"),
    prevent_initial_call=True,
)
def query_row_clicked(selected_rows, rows, selected):
    if not selected_rows or not rows or not selected:
        return no_update, no_update
    row = rows[selected_rows[0]]
    date_value = row.get("event_date") or row.get("price_date")
    if not date_value:
        return no_update, no_update
    selected_date = pd.to_datetime(date_value).date().isoformat()
    return selected_date, run_market_query(selected[0], selected_date)


@app.callback(
    Output("right-commodity-panel", "children"),
    Input("selected-chokepoints", "data"),
    Input("selected-date", "data"),
    Input("last-query", "data"),
    Input("refresh-token", "data"),
)
def update_right_panel(selected, selected_date, query_payload, _token):
    return commodity_panel(selected or [], selected_date, query_payload)


@app.callback(Output("hive-query-panel", "children"), Input("last-query", "data"))
def update_query_panel(payload):
    return query_panel(payload)


if __name__ == "__main__":
    debug = os.environ.get("DASH_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("DASH_PORT", "8050")))
