from __future__ import annotations

import pandas as pd

from src.utils import safe_last


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

    latest_signal = safe_last(frame["dual_ma_signal"])
    return {
        "data": frame,
        "summary": {
            "latest_price": safe_last(frame["asset_close"]),
            "latest_dual_ma_signal": latest_signal,
            "latest_momentum_21": safe_last(frame.get("momentum_21")),
            "latest_momentum_63": safe_last(frame.get("momentum_63")),
            "latest_momentum_126": safe_last(frame.get("momentum_126")),
            "signal_interpretation": "Risk-on trend regime" if latest_signal == 1.0 else "Trend filter is defensive",
        },
        "artifacts": {},
        "figures": {},
        "metadata": {"fast_window": fast_window, "slow_window": slow_window},
    }

