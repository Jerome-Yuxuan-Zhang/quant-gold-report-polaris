from __future__ import annotations

from pathlib import Path
from typing import Any

from src.backtest import run as run_backtest
from src.charting import build_all_figures
from src.data_layer import run as run_data_layer
from src.llm_narrative import run as run_llm
from src.macro_factor import run as run_macro
from src.report_assembly import build as build_report
from src.trend_momentum import run as run_trend
from src.utils import deep_merge, ensure_directories, load_settings, timestamp_slug
from src.volatility_regime import run as run_volatility


def prepare_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = load_settings("config/settings.yaml")
    if overrides:
        settings = deep_merge(settings, overrides)
    ensure_directories(settings["output"].values())
    return settings


def run_pipeline(
    *,
    settings: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
    generate_report: bool = True,
    dataset_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_settings = settings or prepare_settings(overrides)
    if settings and overrides:
        runtime_settings = deep_merge(settings, overrides)
    run_id = run_id or timestamp_slug()

    dataset = dataset_override or run_data_layer(settings=runtime_settings, run_id=run_id)
    trend = run_trend(settings=runtime_settings, dataset=dataset, run_id=run_id)
    volatility = run_volatility(settings=runtime_settings, dataset=dataset, run_id=run_id)
    macro = run_macro(settings=runtime_settings, dataset=dataset, run_id=run_id)
    backtest = run_backtest(settings=runtime_settings, dataset=dataset, trend=trend, volatility=volatility, run_id=run_id)
    figures = build_all_figures(
        settings=runtime_settings,
        run_id=run_id,
        dataset=dataset,
        trend=trend,
        volatility=volatility,
        macro=macro,
        backtest=backtest,
    )
    narrative = run_llm(
        settings=runtime_settings,
        dataset=dataset,
        trend=trend,
        volatility=volatility,
        macro=macro,
        backtest=backtest,
    )

    report_path: Path | None = None
    if generate_report:
        report_path = build_report(
            settings=runtime_settings,
            run_id=run_id,
            dataset=dataset,
            trend=trend,
            volatility=volatility,
            macro=macro,
            backtest=backtest,
            figures=figures,
            narrative=narrative,
        )

    return {
        "settings": runtime_settings,
        "run_id": run_id,
        "modules": {
            "dataset": dataset,
            "trend": trend,
            "volatility": volatility,
            "macro": macro,
            "backtest": backtest,
            "narrative": narrative,
        },
        "figures": figures,
        "artifacts": {"report_path": str(report_path) if report_path else None},
        "metadata": {
            "default_symbol": runtime_settings["asset"]["default_symbol"],
            "comparison_symbols": runtime_settings["asset"]["comparison_symbols"],
            "research_question": runtime_settings["project"]["research_question"],
        },
    }

