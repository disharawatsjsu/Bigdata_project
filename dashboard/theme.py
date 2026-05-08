"""Shared visual constants for the Dash supply-chain intelligence dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

BACKGROUND = "#0a0e1a"
PANEL = "#0f1424"
BORDER = "#1a2030"
ACCENT = "#00d9ff"
TEXT = "#e0e6f0"
MUTED = "#7a8190"
GREEN = "#35d07f"
YELLOW = "#f7c948"
RED = "#ff5c5c"
GRAY = "#5d6474"

FONT_SANS = "'Inter', 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'Fira Code', 'JetBrains Mono', 'Courier New', monospace"

COMMODITIES = ["brent", "wti", "copper", "gold", "wheat", "soybeans"]
CHOKEPOINT_ORDER = [
    "suez",
    "red_sea",
    "hormuz",
    "bab_el_mandeb",
    "malacca",
    "taiwan",
    "panama",
    "chile",
    "black_sea",
]

PANEL_STYLE = {
    "backgroundColor": PANEL,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "padding": "14px",
    "transition": "border-color 200ms ease, background-color 200ms ease",
}

HOVER_LABEL = {
    "bgcolor": "#080b14",
    "bordercolor": ACCENT,
    "font": {"family": FONT_MONO, "color": TEXT, "size": 12},
}


def apply_dark_layout(fig: go.Figure, *, show_grid: bool = True, height: int | None = None) -> go.Figure:
    """Apply the dashboard's dark Plotly styling to a figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font={"family": FONT_SANS, "color": TEXT},
        margin={"l": 50, "r": 24, "t": 30, "b": 40},
        hoverlabel=HOVER_LABEL,
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(
        color=MUTED,
        gridcolor=BORDER if show_grid else BACKGROUND,
        zerolinecolor=BORDER,
        tickfont={"family": FONT_MONO, "color": MUTED},
    )
    fig.update_yaxes(
        color=MUTED,
        gridcolor=BORDER if show_grid else BACKGROUND,
        zerolinecolor=BORDER,
        tickfont={"family": FONT_MONO, "color": MUTED},
    )
    return fig


def roc_color(roc_auc: float | None) -> str:
    if roc_auc is None:
        return MUTED
    if roc_auc > 0.55:
        return GREEN
    if roc_auc >= 0.50:
        return YELLOW
    return RED
