"""Reusable Dash/Plotly components for the supply-chain intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from plotly.subplots import make_subplots

from queries import commodity_metrics
from theme import (
    ACCENT,
    BACKGROUND,
    BORDER,
    CHOKEPOINT_ORDER,
    COMMODITIES,
    FONT_MONO,
    FONT_SANS,
    GRAY,
    GREEN,
    HOVER_LABEL,
    MUTED,
    PANEL_STYLE,
    RED,
    TEXT,
    YELLOW,
    apply_dark_layout,
    roc_color,
)

CHOKEPOINT_COORDS = {
    "suez": {"lat": 30.5, "lon": 32.3, "label": "Suez Canal"},
    "red_sea": {"lat": 12.6, "lon": 43.3, "label": "Red Sea / Bab-el-Mandeb"},
    "hormuz": {"lat": 26.5, "lon": 56.3, "label": "Strait of Hormuz"},
    "malacca": {"lat": 2.8, "lon": 101.5, "label": "Strait of Malacca"},
    "taiwan": {"lat": 24.0, "lon": 120.0, "label": "Taiwan Strait"},
    "panama": {"lat": 9.1, "lon": -79.7, "label": "Panama Canal"},
    "chile": {"lat": -33.0, "lon": -71.6, "label": "Chile Port"},
    "black_sea": {"lat": 41.0, "lon": 29.0, "label": "Black Sea / Bosphorus"},
}

CATEGORIES = {
    "Energy": ["brent", "wti"],
    "Metals": ["copper", "gold"],
    "Agriculture": ["wheat", "soybeans"],
}


def selected_date_from_ts(timestamp: int | float | None, trigger_log: pd.DataFrame) -> pd.Timestamp:
    if trigger_log is None or trigger_log.empty:
        return pd.Timestamp.utcnow().normalize().tz_localize(None)
    if timestamp is None:
        return trigger_log["event_date"].dropna().sort_values().iloc[len(trigger_log.dropna()) // 2].normalize()
    return pd.to_datetime(timestamp, unit="s").normalize()


def slider_bounds(trigger_log: pd.DataFrame) -> tuple[int, int, int, dict[int, str]]:
    dates = trigger_log["event_date"].dropna().sort_values()
    date_min = dates.min().normalize()
    date_max = dates.max().normalize()
    midpoint = dates.iloc[len(dates) // 2].normalize()
    marks: dict[int, str] = {}
    for mark in pd.date_range(date_min, date_max, freq="6MS"):
        marks[int(mark.timestamp())] = mark.strftime("%b '%y")
    marks[int(date_min.timestamp())] = date_min.strftime("%b '%y")
    marks[int(date_max.timestamp())] = date_max.strftime("%b '%y")
    return int(date_min.timestamp()), int(date_max.timestamp()), int(midpoint.timestamp()), marks


def chokepoint_map_figure(trigger_log: pd.DataFrame, selected_date: pd.Timestamp, selected_chokepoint: str | None) -> go.Figure:
    if trigger_log is None or trigger_log.empty:
        return empty_figure("Hive table trigger_log is empty.", height=500)

    day = trigger_log[trigger_log["event_date"].dt.normalize() == selected_date.normalize()].copy()
    latest_by_cp = day.groupby("chokepoint", as_index=False).agg(
        event_count_1d=("event_count_1d", "max"),
        avg_goldstein_1d=("avg_goldstein_1d", "mean"),
        triggered=("trigger_type", lambda values: values.notna().any()),
    )
    day_map = latest_by_cp.set_index("chokepoint").to_dict("index")

    rows = []
    for cp, coord in CHOKEPOINT_COORDS.items():
        values = day_map.get(cp, {})
        count = float(values.get("event_count_1d", 0) or 0)
        goldstein = values.get("avg_goldstein_1d")
        rows.append(
            {
                "chokepoint": cp,
                "label": coord["label"],
                "lat": coord["lat"],
                "lon": coord["lon"],
                "event_count_1d": count,
                "avg_goldstein_1d": goldstein if pd.notna(goldstein) else None,
                "triggered": bool(values.get("triggered", False)),
            }
        )
    df = pd.DataFrame(rows)
    max_count = max(float(df["event_count_1d"].max()), 1.0)
    df["size"] = 8 + (df["event_count_1d"].clip(lower=0).pow(0.5) / max_count**0.5) * 22
    df["color_value"] = df["avg_goldstein_1d"].fillna(0).clip(-5, 0)
    df["highlight"] = df["triggered"] | (df["chokepoint"] == selected_chokepoint)

    fig = go.Figure()
    for is_highlighted, line_color, line_width in ((False, BORDER, 1), (True, ACCENT, 4)):
        subset = df[df["highlight"] == is_highlighted]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=subset["lon"],
                lat=subset["lat"],
                mode="markers+text",
                text=subset["label"],
                textposition="top center",
                marker={
                    "size": subset["size"],
                    "color": subset["color_value"],
                    "colorscale": "Reds_r",
                    "cmin": -5,
                    "cmax": 0,
                    "line": {"color": line_color, "width": line_width},
                    "colorbar": {
                        "title": {"text": "Goldstein", "font": {"family": FONT_MONO, "color": MUTED}},
                        "tickfont": {"family": FONT_MONO, "color": MUTED},
                        "len": 0.72,
                    },
                    "showscale": not is_highlighted,
                },
                customdata=subset[["chokepoint", "event_count_1d", "avg_goldstein_1d", "triggered"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"date={selected_date:%Y-%m-%d}<br>"
                    "event_count_1d=%{customdata[1]:.0f}<br>"
                    "avg_goldstein_1d=%{customdata[2]:.2f}<br>"
                    "triggered=%{customdata[3]}<extra></extra>"
                ),
                textfont={"family": FONT_MONO, "size": 10, "color": MUTED},
                showlegend=False,
            )
        )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#0a0e1a",
        showocean=True,
        oceancolor="#050810",
        showcountries=True,
        countrycolor=BORDER,
        showcoastlines=True,
        coastlinecolor=BORDER,
        bgcolor=BACKGROUND,
        lataxis_showgrid=False,
        lonaxis_showgrid=False,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        height=500,
        font={"family": FONT_SANS, "color": TEXT},
        hoverlabel=HOVER_LABEL,
        showlegend=False,
    )
    return fig


def available_chokepoints(*frames: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    for frame in frames:
        if frame is not None and not frame.empty and "chokepoint" in frame.columns:
            seen.update(frame["chokepoint"].dropna().astype(str).unique())
    ordered = [cp for cp in CHOKEPOINT_ORDER if cp in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def format_name(value: str) -> str:
    return value.replace("_", " ").title()


def metric_text(value: float | None, *, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def empty_figure(message: str, *, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": MUTED, "size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_dark_layout(fig, show_grid=False, height=height)


def timeline_figure(trigger_log: pd.DataFrame) -> go.Figure:
    if trigger_log.empty:
        return empty_figure("Hive table trigger_log is empty.", height=250)

    df = trigger_log.dropna(subset=["event_date", "chokepoint"]).copy()
    order = available_chokepoints(df)
    df["chokepoint"] = pd.Categorical(df["chokepoint"], categories=order, ordered=True)
    counts = pd.to_numeric(df["event_count_1d"], errors="coerce").fillna(0)
    if counts.max() > counts.min():
        df["marker_size"] = 6 + (counts - counts.min()) / (counts.max() - counts.min()) * 12
    else:
        df["marker_size"] = 10

    fig = go.Figure()
    non_missing = df[df["avg_goldstein_1d"].notna()]
    missing = df[df["avg_goldstein_1d"].isna()]
    fig.add_trace(
        go.Scatter(
            x=non_missing["event_date"],
            y=non_missing["chokepoint"].astype(str),
            mode="markers",
            marker={
                "size": non_missing["marker_size"],
                "color": non_missing["avg_goldstein_1d"],
                "colorscale": "RdBu",
                "cmid": 0,
                "showscale": True,
                "colorbar": {
                    "title": {"text": "Goldstein", "font": {"family": FONT_MONO, "color": MUTED}},
                    "x": 0.995,
                    "len": 0.85,
                    "tickfont": {"family": FONT_MONO, "color": MUTED},
                },
                "line": {"color": BACKGROUND, "width": 0.5},
                "opacity": 0.88,
            },
            customdata=non_missing[["event_count_1d", "avg_goldstein_1d"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "date=%{x|%Y-%m-%d}<br>"
                "event_count_1d=%{customdata[0]:.0f}<br>"
                "avg_goldstein_1d=%{customdata[1]:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    if not missing.empty:
        fig.add_trace(
            go.Scatter(
                x=missing["event_date"],
                y=missing["chokepoint"].astype(str),
                mode="markers",
                marker={"size": missing["marker_size"], "color": GRAY, "opacity": 0.45},
                customdata=missing[["event_count_1d"]],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "date=%{x|%Y-%m-%d}<br>"
                    "event_count_1d=%{customdata[0]:.0f}<br>"
                    "avg_goldstein_1d=n/a<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)), automargin=True)
    fig.update_xaxes(dtick="M3", tickformat="%b %Y")
    fig.update_layout(hoverlabel=HOVER_LABEL)
    return apply_dark_layout(fig, show_grid=False, height=250)


def shap_heatmap_figure(shap: pd.DataFrame, selected: dict | None) -> go.Figure:
    if shap.empty:
        return empty_figure("Hive table shap_attribution is empty.", height=440)

    order = available_chokepoints(shap)
    pivot = (
        shap.pivot_table(
            index="chokepoint",
            columns="commodity",
            values="mean_abs_shap",
            aggfunc="mean",
        )
        .reindex(index=order, columns=COMMODITIES)
        .fillna(0)
    )
    z = pivot.values
    text = [[f"{value:.3f}" for value in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Viridis",
            text=text,
            texttemplate="%{text}",
            textfont={"family": FONT_MONO, "size": 11, "color": TEXT},
            hovertemplate=(
                "chokepoint=%{y}<br>"
                "commodity=%{x}<br>"
                "mean_abs_shap=%{z:.4f}<extra></extra>"
            ),
            colorbar={"title": "mean |SHAP|", "tickfont": {"family": FONT_MONO, "color": MUTED}},
        )
    )
    if selected:
        cp = selected.get("chokepoint")
        commodity = selected.get("commodity")
        if cp in pivot.index and commodity in pivot.columns:
            x_idx = list(pivot.columns).index(commodity)
            y_idx = list(pivot.index).index(cp)
            fig.add_shape(
                type="rect",
                x0=x_idx - 0.5,
                x1=x_idx + 0.5,
                y0=y_idx - 0.5,
                y1=y_idx + 0.5,
                xref="x",
                yref="y",
                line={"color": ACCENT, "width": 3},
            )
    return apply_dark_layout(fig, show_grid=False, height=440)


def default_selection(shap: pd.DataFrame) -> dict[str, str]:
    if shap.empty:
        return {"chokepoint": "suez", "commodity": "brent"}
    row = shap.sort_values("mean_abs_shap", ascending=False).iloc[0]
    return {"chokepoint": str(row["chokepoint"]), "commodity": str(row["commodity"])}


def performance_figure(predictions: pd.DataFrame, trigger_log: pd.DataFrame, commodity: str, chokepoint: str) -> go.Figure:
    if predictions.empty:
        return empty_figure("Hive table predictions is empty.", height=360)

    pred = predictions[predictions["commodity"] == commodity].sort_values("event_date").copy()
    if pred.empty:
        return empty_figure(f"No predictions for {commodity}.", height=360)

    triggers = trigger_log[trigger_log["chokepoint"] == chokepoint].sort_values("event_date").copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=pred["event_date"],
            y=pred["predicted_proba"],
            mode="lines",
            line={"color": ACCENT, "width": 2},
            name="predicted_proba",
            hovertemplate="date=%{x|%Y-%m-%d}<br>proba=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    label_colors = pred["actual_label"].map({1: GREEN, 0: GRAY}).fillna(GRAY)
    label_y = pred["actual_label"].map({1: 1.06, 0: -0.06}).fillna(-0.06)
    fig.add_trace(
        go.Scatter(
            x=pred["event_date"],
            y=label_y,
            mode="markers",
            marker={"size": 6, "color": label_colors},
            name="actual_label",
            hovertemplate="date=%{x|%Y-%m-%d}<br>actual_label=%{customdata}<extra></extra>",
            customdata=pred["actual_label"],
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color=MUTED, row=1, col=1)

    if not triggers.empty:
        fig.add_trace(
            go.Bar(
                x=triggers["event_date"],
                y=triggers["event_count_1d"],
                marker={"color": RED, "opacity": 0.75},
                name=f"{chokepoint} trigger day",
                hovertemplate=(
                    "trigger date=%{x|%Y-%m-%d}<br>"
                    "event_count_1d=%{y:.0f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title="proba", range=[-0.1, 1.1], row=1, col=1)
    fig.update_yaxes(title="triggers", row=2, col=1)
    fig.update_layout(showlegend=False)
    return apply_dark_layout(fig, show_grid=True, height=360)


def waterfall_figure(predictions: pd.DataFrame, shap_values: pd.DataFrame | None, commodity: str) -> go.Figure:
    if shap_values is None or shap_values.empty:
        return empty_figure("Local SHAP not yet computed. Run the notebook cell to populate SHAP parquet.", height=360)

    pred = predictions[predictions["commodity"] == commodity].sort_values("predicted_proba", ascending=False)
    if pred.empty:
        return empty_figure(f"No predictions for {commodity}.", height=360)

    target = pred.iloc[0]
    target_date = target["event_date"]
    day = shap_values[shap_values["event_date"] == target_date].copy()
    if day.empty:
        return empty_figure(f"No local SHAP values for {commodity} on {target_date:%Y-%m-%d}.", height=360)

    day = day.reindex(day["shap_value"].abs().sort_values(ascending=False).index).head(12)
    day = day.sort_values("shap_value", ascending=False)
    final_proba = float(target["predicted_proba"])
    base_value = final_proba - float(day["shap_value"].sum())
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(day),
            x=["base"] + day["feature_name"].astype(str).tolist(),
            y=[base_value] + day["shap_value"].astype(float).tolist(),
            connector={"line": {"color": MUTED}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": ACCENT}},
            hovertemplate="%{x}<br>contribution=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Feature contributions to {commodity} prediction on {target_date:%Y-%m-%d} (predicted: {final_proba:.2f})",
        xaxis_tickangle=-35,
    )
    return apply_dark_layout(fig, show_grid=True, height=360)


def trigger_table(trigger_log: pd.DataFrame, predictions: pd.DataFrame, commodity: str, chokepoint: str) -> dash_table.DataTable:
    triggers = trigger_log[trigger_log["chokepoint"] == chokepoint].copy()
    pred = predictions[predictions["commodity"] == commodity][["event_date", "actual_label", "predicted_proba"]].copy()
    if triggers.empty:
        rows: list[dict] = []
    else:
        merged = triggers.merge(pred, on="event_date", how="left")
        merged = merged.sort_values("event_count_1d", ascending=False).head(20)
        rows = [
            {
                "date": row["event_date"].strftime("%Y-%m-%d") if pd.notna(row["event_date"]) else "n/a",
                "event count": f"{row['event_count_1d']:.0f}" if pd.notna(row["event_count_1d"]) else "n/a",
                "goldstein": f"{row['avg_goldstein_1d']:.2f}" if pd.notna(row["avg_goldstein_1d"]) else "n/a",
                "next to": (
                    f"label={int(row['actual_label'])}, p={row['predicted_proba']:.3f}"
                    if pd.notna(row.get("actual_label")) and pd.notna(row.get("predicted_proba"))
                    else "no same-day prediction"
                ),
            }
            for _, row in merged.iterrows()
        ]

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": col, "id": col} for col in ["date", "event count", "goldstein", "next to"]],
        page_size=20,
        style_as_list_view=True,
        style_header={
            "backgroundColor": BACKGROUND,
            "color": ACCENT,
            "borderBottom": f"1px solid {BORDER}",
            "fontFamily": FONT_MONO,
        },
        style_cell={
            "backgroundColor": BACKGROUND,
            "color": TEXT,
            "border": f"1px solid {BORDER}",
            "fontFamily": FONT_MONO,
            "fontSize": "12px",
            "padding": "8px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "date"}, "textAlign": "left"},
            {"if": {"column_id": "next to"}, "textAlign": "left"},
            {"if": {"column_id": "event count"}, "textAlign": "right"},
            {"if": {"column_id": "goldstein"}, "textAlign": "right"},
        ],
    )


def metrics_row(predictions: pd.DataFrame, commodity: str) -> html.Div:
    metrics = commodity_metrics(predictions, commodity)
    return html.Div(
        [
            html.Span(f"Test ROC: {metric_text(metrics['roc'])}"),
            html.Span(f"Test PR: {metric_text(metrics['pr'])}"),
            html.Span(f"Naive baseline: {metric_text(metrics['naive_pr'])}"),
            html.Span(f"Lift: {metric_text(metrics['lift'], signed=True)}"),
        ],
        className="metrics-row",
    )


def commodity_cards(predictions: pd.DataFrame | None) -> list[html.Button]:
    cards: list[html.Button] = []
    predictions = predictions if predictions is not None else pd.DataFrame()
    for commodity in COMMODITIES:
        metrics = commodity_metrics(predictions, commodity) if not predictions.empty else {}
        subset = predictions[predictions["commodity"] == commodity].sort_values("event_date") if not predictions.empty else pd.DataFrame()
        sparkline = go.Figure()
        if not subset.empty:
            sparkline.add_trace(
                go.Scatter(
                    x=subset["event_date"],
                    y=subset["predicted_proba"],
                    mode="lines",
                    line={"color": ACCENT, "width": 1.5},
                    hoverinfo="skip",
                )
            )
        sparkline = apply_dark_layout(sparkline, show_grid=False, height=50)
        sparkline.update_layout(margin={"l": 0, "r": 0, "t": 2, "b": 0})
        sparkline.update_xaxes(visible=False)
        sparkline.update_yaxes(visible=False)

        cards.append(
            html.Button(
                [
                    html.Div(format_name(commodity), className="card-title"),
                    html.Div(
                        [
                            html.Div(["ROC", html.Span(metric_text(metrics.get("roc")))]),
                            html.Div(["PR", html.Span(metric_text(metrics.get("pr")))]),
                            html.Div(["Naive", html.Span(metric_text(metrics.get("naive_pr")))]),
                        ],
                        className="card-metrics",
                    ),
                    dcc.Graph(figure=sparkline, config={"displayModeBar": False}, className="sparkline"),
                ],
                id={"type": "commodity-card", "commodity": commodity},
                n_clicks=0,
                className="commodity-card",
                style={"borderColor": roc_color(metrics.get("roc"))},
            )
        )
    return cards


def event_study_summary(event_study: pd.DataFrame, commodity: str, chokepoint: str) -> html.Div:
    row = event_study[
        (event_study["commodity"] == commodity) & (event_study["chokepoint"] == chokepoint)
    ]
    if row.empty:
        return html.Div("No event study row for this selection.", className="muted-note")
    item = row.iloc[0]
    return html.Div(
        [
            html.Span(f"Count-trigger lift: {metric_text(item.get('lift_count_trigger'), signed=True)}"),
            html.Span(f"p={metric_text(item.get('pval_count_trigger'))}"),
            html.Span(f"n={int(item.get('n_count_trigger_days')) if pd.notna(item.get('n_count_trigger_days')) else 'n/a'}"),
            html.Span(f"Goldstein-trigger lift: {metric_text(item.get('lift_goldstein_trigger'), signed=True)}"),
            html.Span(f"p={metric_text(item.get('pval_goldstein_trigger'))}"),
        ],
        className="metrics-row",
    )
