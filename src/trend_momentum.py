from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import safe_last


def _plot_trend(price_frame: pd.DataFrame, chart_path: str | Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_frame.index, price_frame["asset_close"], label="Asset Close", linewidth=1.4)
    for column in [col for col in price_frame.columns if col.startswith("sma_")]:
        ax.plot(price_frame.index, price_frame[column], label=column.upper(), linewidth=1.0, alpha=0.9)
    ax.set_title("Price With Trend Filters")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def run(settings: dict, dataset: dict, run_id: str) -> dict:
    frame = dataset["data"].copy()
    windows = settings["windows"]

    for window in windows["sma"]:
        frame[f"sma_{window}"] = frame["asset_close"].rolling(window).mean()

    for window in windows["ema"]:
        frame[f"ema_{window}"] = frame["asset_close"].ewm(span=window, adjust=False).mean()

    for lookback in windows["momentum"]:
        frame[f"momentum_{lookback}"] = frame["asset_close"] / frame["asset_close"].shift(lookback) - 1

    fast_window = windows["sma"][0]
    slow_window = windows["sma"][-1]
    frame["dual_ma_signal"] = (frame[f"sma_{fast_window}"] > frame[f"sma_{slow_window}"]).astype(float)

    chart_path = Path(settings["output"]["charts_dir"]) / f"trend_momentum_{run_id}.png"
    _plot_trend(frame, chart_path)
    shutil.copyfile(chart_path, Path(settings["output"]["charts_dir"]) / "latest_trend_momentum.png")

    return {
        "data": frame,
        "artifacts": {"trend_chart": str(chart_path)},
        "summary": {
            "latest_price": safe_last(frame["asset_close"]),
            "latest_dual_ma_signal": safe_last(frame["dual_ma_signal"]),
            "latest_momentum_21": safe_last(frame.get("momentum_21")),
            "latest_momentum_63": safe_last(frame.get("momentum_63")),
            "latest_momentum_126": safe_last(frame.get("momentum_126")),
            "signal_interpretation": "Risk-on trend regime" if safe_last(frame["dual_ma_signal"]) == 1.0 else "Trend filter is defensive",
        },
    }
