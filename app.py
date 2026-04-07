from __future__ import annotations

from dash import Dash, Input, Output, State, dash_table, dcc, html

from src.pipeline import prepare_settings, run_pipeline
from src.utils import format_num, format_pct


settings = prepare_settings()
app = Dash(__name__)
app.title = settings["gui"]["title"]


def metrics_table(rows: list[dict[str, str]]):
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": "Metric", "id": "metric"}, {"name": "Value", "id": "value"}, {"name": "Comment", "id": "comment"}],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1d3557", "color": "white", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "8px", "fontFamily": "Arial", "fontSize": 13},
    )


def regression_table(rows: list[dict]):
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": "Term", "id": "term"}, {"name": "Coefficient", "id": "coefficient"}, {"name": "t-stat", "id": "t_stat"}, {"name": "p-value", "id": "p_value"}],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1d3557", "color": "white", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "8px", "fontFamily": "Arial", "fontSize": 13},
    )


def build_tabs(result: dict) -> html.Div:
    modules = result["modules"]
    figures = result["figures"]["plotly"]
    dataset = modules["dataset"]
    backtest = modules["backtest"]
    macro = modules["macro"]
    volatility = modules["volatility"]

    overview_rows = [
        {"metric": "Primary Asset", "value": dataset["summary"]["selected_symbol"], "comment": "Current GUI selection"},
        {"metric": "Coverage", "value": f"{dataset['summary']['start']} to {dataset['summary']['end']}", "comment": "Business-day aligned panel"},
        {"metric": "Latest Vol Regime", "value": str(volatility["summary"]["latest_vol_regime"]), "comment": "Percentile-based regime label"},
        {"metric": "Net Sharpe", "value": format_num(backtest["summary"]["sharpe_ratio"]), "comment": "After transaction costs"},
    ]
    signal_rows = [
        {"metric": "Latest Price", "value": format_num(modules["trend"]["summary"]["latest_price"], 2), "comment": dataset["summary"]["selected_symbol"]},
        {"metric": "Dual MA Signal", "value": format_num(modules["trend"]["summary"]["latest_dual_ma_signal"], 0), "comment": modules["trend"]["summary"]["signal_interpretation"]},
        {"metric": "21D Momentum", "value": format_pct(modules["trend"]["summary"]["latest_momentum_21"]), "comment": "1-month lookback"},
        {"metric": "63D Momentum", "value": format_pct(modules["trend"]["summary"]["latest_momentum_63"]), "comment": "1-quarter lookback"},
    ]
    backtest_rows = [
        {"metric": "Annualized Return (Net)", "value": format_pct(backtest["summary"]["annualized_return"]), "comment": "After transaction costs"},
        {"metric": "Annualized Return (Gross)", "value": format_pct(backtest["summary"]["annualized_return_gross"]), "comment": "Before transaction costs"},
        {"metric": "Maximum Drawdown", "value": format_pct(backtest["summary"]["max_drawdown"]), "comment": "Net equity curve"},
        {"metric": "Total Cost", "value": format_pct(backtest["summary"]["total_transaction_cost"]), "comment": f"{backtest['summary']['transaction_cost_bps']:.1f} bps model"},
    ]

    regression_rows = macro["summary"]["regression_table_preview"] or [{"term": "Unavailable", "coefficient": "N/A", "t_stat": "N/A", "p_value": "N/A"}]

    return html.Div(
        [
            dcc.Tabs(
                value="Overview",
                children=[
                    dcc.Tab(
                        label="Overview",
                        value="Overview",
                        children=[
                            dcc.Graph(figure=figures["trend_momentum"]),
                            metrics_table(overview_rows),
                        ],
                    ),
                    dcc.Tab(
                        label="Macro",
                        value="Macro",
                        children=[
                            dcc.Graph(figure=figures["macro_correlation"]),
                            regression_table(regression_rows),
                        ],
                    ),
                    dcc.Tab(
                        label="Signals",
                        value="Signals",
                        children=[
                            dcc.Graph(figure=figures["volatility_regime"]),
                            metrics_table(signal_rows),
                        ],
                    ),
                    dcc.Tab(
                        label="Backtest",
                        value="Backtest",
                        children=[
                            dcc.Graph(figure=figures["backtest_equity"]),
                            metrics_table(backtest_rows),
                        ],
                    ),
                    dcc.Tab(
                        label="Distribution / Robustness",
                        value="Distribution / Robustness",
                        children=[
                            dcc.Graph(figure=figures["monthly_box"]),
                            dcc.Graph(figure=figures["regime_box"]),
                            dcc.Graph(figure=figures["signal_box"]),
                        ],
                    ),
                    dcc.Tab(
                        label="Methodology",
                        value="Methodology",
                        children=[
                            html.Div(
                                [
                                    html.H4("Research Objective"),
                                    html.P(result["metadata"]["research_question"]),
                                    html.H4("As-of Macro Handling"),
                                    html.Ul([html.Li(text) for text in dataset["metadata"]["macro_assumptions"].values()]),
                                    html.H4("Narrative Layer"),
                                    html.P("The LLM layer is descriptive only and falls back to deterministic templates when no Anthropic API key is available."),
                                ],
                                style={"padding": "16px", "lineHeight": "1.7"},
                            )
                        ],
                    ),
                ],
            )
        ]
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H2(settings["gui"]["title"]),
                html.P("Interactive research terminal for factor exposure, regime behavior, and signal review."),
                html.Label("Primary Asset"),
                dcc.Dropdown(
                    id="asset-symbol",
                    options=[{"label": symbol, "value": symbol} for symbol in settings["asset"]["supported_symbols"]],
                    value=settings["asset"]["default_symbol"],
                    clearable=False,
                ),
                html.Label("Start Date", style={"marginTop": "14px"}),
                dcc.DatePickerSingle(id="start-date", date=settings["date_range"]["start"]),
                html.Label("End Date", style={"marginTop": "14px"}),
                dcc.DatePickerSingle(id="end-date", date=settings["date_range"]["end"]),
                html.Label("Transaction Cost (bps)", style={"marginTop": "14px"}),
                dcc.Input(id="transaction-cost", type="number", value=settings["backtest"]["transaction_cost_bps"], min=0, step=0.5),
                html.Label("Enable HMM", style={"marginTop": "14px"}),
                dcc.Checklist(id="enable-hmm", options=[{"label": "Use 2-state HMM", "value": "hmm"}], value=["hmm"] if settings["optional_features"]["hmm"] else []),
                html.Label("Generate PDF", style={"marginTop": "14px"}),
                dcc.Checklist(id="generate-pdf", options=[{"label": "Export PDF on run", "value": "pdf"}], value=[]),
                html.Button("Run Analysis", id="run-analysis", n_clicks=0, style={"marginTop": "18px", "width": "100%", "padding": "10px"}),
                html.Div(id="run-status", style={"marginTop": "14px", "fontSize": 13, "color": "#1d3557"}),
            ],
            style={"width": "22%", "padding": "20px", "backgroundColor": "#f5f7fa", "minHeight": "100vh", "boxSizing": "border-box"},
        ),
        html.Div(id="analysis-content", style={"width": "78%", "padding": "20px", "boxSizing": "border-box"}),
    ],
    style={"display": "flex", "fontFamily": "Arial, sans-serif"},
)


@app.callback(
    Output("analysis-content", "children"),
    Output("run-status", "children"),
    Input("run-analysis", "n_clicks"),
    State("asset-symbol", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("transaction-cost", "value"),
    State("enable-hmm", "value"),
    State("generate-pdf", "value"),
)
def run_dashboard_analysis(n_clicks, asset_symbol, start_date, end_date, transaction_cost, enable_hmm, generate_pdf):
    if not n_clicks:
        return html.Div("Select parameters and click Run Analysis to generate the research terminal."), "Awaiting first run."

    overrides = {
        "asset": {"default_symbol": asset_symbol},
        "date_range": {"start": start_date, "end": end_date},
        "backtest": {"transaction_cost_bps": float(transaction_cost or 0.0)},
        "optional_features": {"hmm": "hmm" in (enable_hmm or [])},
    }
    result = run_pipeline(overrides=overrides, generate_report=("pdf" in (generate_pdf or [])))
    report_path = result["artifacts"]["report_path"]
    status = f"Run complete for {asset_symbol}. PDF: {report_path}" if report_path else f"Run complete for {asset_symbol}. PDF export skipped."
    return build_tabs(result), status


if __name__ == "__main__":
    app.run(host=settings["gui"]["host"], port=settings["gui"]["port"], debug=False)

