from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import format_num, format_pct, slugify_symbol


LATEST_NAMES = {
    "cover_preview": "latest_cover_preview.png",
    "trend_momentum": "latest_trend_momentum.png",
    "volatility_regime": "latest_volatility_regime.png",
    "macro_correlation": "latest_macro_correlation.png",
    "backtest_equity": "latest_backtest_equity.png",
    "monthly_box": "latest_monthly_returns_box.png",
    "regime_box": "latest_regime_returns_box.png",
    "signal_box": "latest_signal_returns_box.png",
}


def _export_figure(fig: go.Figure, output_path: Path, latest_name: str | None = None) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), width=1400, height=800, scale=2)
    if latest_name:
        shutil.copyfile(output_path, output_path.parent / latest_name)
    return str(output_path)


def _apply_layout(fig: go.Figure, settings: dict, title: str) -> go.Figure:
    fig.update_layout(
        template=settings["charts"]["theme"],
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=30, t=70, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def build_cover_figure(settings: dict, dataset: dict) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=settings["charts"]["theme"],
        width=900,
        height=1280,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="#f7f7f2",
        plot_bgcolor="#f7f7f2",
    )
    fig.add_annotation(text=settings["report"]["title"], x=0.5, y=0.83, showarrow=False, font=dict(size=28, family="Georgia"))
    fig.add_annotation(text=settings["report"]["subtitle"], x=0.5, y=0.77, showarrow=False, font=dict(size=16))
    fig.add_annotation(text=settings["project"]["research_question"], x=0.5, y=0.67, showarrow=False, font=dict(size=14), xref="paper", yref="paper")
    fig.add_annotation(text=f"Primary Asset: {dataset['summary']['selected_symbol']}", x=0.5, y=0.55, showarrow=False, font=dict(size=18))
    fig.add_annotation(text=f"Coverage: {dataset['summary']['start']} to {dataset['summary']['end']}", x=0.5, y=0.49, showarrow=False, font=dict(size=15))
    fig.add_annotation(text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", x=0.5, y=0.43, showarrow=False, font=dict(size=15))
    fig.add_annotation(text=f"Author: {settings['report']['author']}", x=0.5, y=0.33, showarrow=False, font=dict(size=15))
    return fig


def build_trend_figure(settings: dict, dataset: dict, trend: dict) -> go.Figure:
    frame = trend["data"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["asset_close"], mode="lines", name=dataset["summary"]["selected_symbol"], line=dict(width=2.2, color="#1f4e79")))
    for window in settings["windows"]["sma"]:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[f"sma_{window}"], mode="lines", name=f"SMA {window}", line=dict(width=1.2)))
    for window in settings["windows"]["ema"]:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[f"ema_{window}"], mode="lines", name=f"EMA {window}", line=dict(width=1, dash="dot")))

    if "close_GC_F" in frame.columns and dataset["summary"]["selected_symbol"] != "GC=F":
        fig.add_trace(go.Scatter(x=frame.index, y=frame["close_GC_F"], mode="lines", name="GC=F comparison", line=dict(width=1.1, color="#8c6d31"), opacity=0.65, yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Comparison"))

    _apply_layout(fig, settings, "Price, Trend Filters, and Comparison Context")
    fig.update_yaxes(title="Primary Asset Price")
    return fig


def build_volatility_figure(settings: dict, volatility: dict) -> go.Figure:
    frame = volatility["data"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["realized_vol_21d"], mode="lines", name="21D realized vol", line=dict(color="#6a3d9a", width=2)))
    colors = {"low": "rgba(182, 215, 168, 0.28)", "medium": "rgba(255, 229, 153, 0.28)", "high": "rgba(244, 204, 204, 0.32)"}
    regime = frame["vol_regime"].ffill()
    start = None
    current = None
    for idx, label in regime.items():
        if label != current:
            if current is not None and start is not None:
                fig.add_vrect(x0=start, x1=idx, fillcolor=colors.get(current, "rgba(220,220,220,0.2)"), opacity=0.25, line_width=0)
            start = idx
            current = label
    if current is not None and start is not None:
        fig.add_vrect(x0=start, x1=frame.index.max(), fillcolor=colors.get(current, "rgba(220,220,220,0.2)"), opacity=0.25, line_width=0)
    _apply_layout(fig, settings, "Volatility Regime Detection")
    fig.update_yaxes(title="Annualized Volatility")
    return fig


def build_macro_figure(settings: dict, macro: dict) -> go.Figure:
    frame = macro["data"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Gold vs DXY", "Gold vs Real Yield"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["corr_dxy_pearson"], mode="lines", name="DXY Pearson", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["corr_dxy_spearman"], mode="lines", name="DXY Spearman", line=dict(color="#17becf", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["corr_real_yield_pearson"], mode="lines", name="Real yield Pearson", line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["corr_real_yield_spearman"], mode="lines", name="Real yield Spearman", line=dict(color="#ff9896", dash="dot")), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#444", row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#444", row=2, col=1)
    _apply_layout(fig, settings, "Rolling Factor Exposure")
    return fig


def build_backtest_figure(settings: dict, backtest: dict) -> go.Figure:
    frame = backtest["data"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["strategy_equity"], mode="lines", name="Strategy (net)", line=dict(color="#0b6e4f", width=2.2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["strategy_equity_gross"], mode="lines", name="Strategy (gross)", line=dict(color="#74c69d", dash="dot")))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["asset_equity"], mode="lines", name="Buy & Hold", line=dict(color="#7f7f7f", width=1.6)))
    _apply_layout(fig, settings, "Walk-Forward Signal Review")
    fig.update_yaxes(title="Equity Curve")
    return fig


def build_box_figures(settings: dict, backtest: dict) -> dict[str, go.Figure]:
    frame = backtest["data"].copy()
    monthly = frame["strategy_simple_return"].resample("ME").apply(lambda values: (1 + values).prod() - 1).dropna()
    monthly_frame = monthly.to_frame(name="monthly_return").reset_index()
    monthly_frame["series"] = "Strategy Monthly Returns"
    monthly_fig = px.box(monthly_frame, x="series", y="monthly_return", points="all", template=settings["charts"]["theme"], title="Monthly Return Distribution")

    regime_source = frame[["asset_simple_return", "vol_regime"]].dropna()
    if regime_source.empty:
        regime_source = pd.DataFrame({"vol_regime": ["unavailable"], "asset_simple_return": [0.0]})
    regime_fig = px.box(regime_source, x="vol_regime", y="asset_simple_return", color="vol_regime", template=settings["charts"]["theme"], title="Asset Returns by Volatility Regime")

    signal_source = frame[["asset_simple_return", "walk_forward_signal"]].dropna()
    if signal_source.empty:
        signal_source = pd.DataFrame({"walk_forward_signal": [0.0], "asset_simple_return": [0.0]})
    signal_source["signal_state"] = signal_source["walk_forward_signal"].map({0.0: "Defensive", 1.0: "Risk-on"}).fillna("Unknown")
    signal_fig = px.box(signal_source, x="signal_state", y="asset_simple_return", color="signal_state", template=settings["charts"]["theme"], title="Asset Returns by Signal State")

    for fig in [monthly_fig, regime_fig, signal_fig]:
        fig.update_layout(margin=dict(l=40, r=30, t=70, b=40))
        fig.update_yaxes(tickformat=".1%")
    return {
        "monthly_box": monthly_fig,
        "regime_box": regime_fig,
        "signal_box": signal_fig,
    }


def build_all_figures(settings: dict, run_id: str, dataset: dict, trend: dict, volatility: dict, macro: dict, backtest: dict) -> dict:
    charts_dir = Path(settings["output"]["charts_dir"])
    figures = {
        "cover_preview": build_cover_figure(settings, dataset),
        "trend_momentum": build_trend_figure(settings, dataset, trend),
        "volatility_regime": build_volatility_figure(settings, volatility),
        "macro_correlation": build_macro_figure(settings, macro),
        "backtest_equity": build_backtest_figure(settings, backtest),
    }
    figures.update(build_box_figures(settings, backtest))

    artifacts = {}
    for key, fig in figures.items():
        path = charts_dir / f"{key}_{run_id}.png"
        artifacts[key] = _export_figure(fig, path, LATEST_NAMES.get(key))

    return {"plotly": figures, "artifacts": artifacts}

