from __future__ import annotations

from pathlib import Path

from src.backtest import run as run_backtest
from src.data_layer import run as run_data_layer
from src.llm_narrative import run as run_llm
from src.macro_factor import run as run_macro
from src.report_assembly import build as build_report
from src.trend_momentum import run as run_trend
from src.utils import ensure_directories, load_settings, timestamp_slug
from src.volatility_regime import run as run_volatility


def main() -> Path:
    settings = load_settings("config/settings.yaml")
    ensure_directories(settings["output"].values())

    run_id = timestamp_slug()
    dataset = run_data_layer(settings=settings, run_id=run_id)
    trend = run_trend(settings=settings, dataset=dataset, run_id=run_id)
    backtest = run_backtest(settings=settings, dataset=dataset, trend=trend, run_id=run_id)
    volatility = run_volatility(settings=settings, dataset=dataset, run_id=run_id)
    macro = run_macro(settings=settings, dataset=dataset, run_id=run_id)
    narrative = run_llm(
        settings=settings,
        dataset=dataset,
        trend=trend,
        volatility=volatility,
        macro=macro,
        backtest=backtest,
    )
    report_path = build_report(
        settings=settings,
        run_id=run_id,
        dataset=dataset,
        trend=trend,
        volatility=volatility,
        macro=macro,
        backtest=backtest,
        narrative=narrative,
    )
    print(f"Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    main()

