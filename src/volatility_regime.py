from __future__ import annotations

import contextlib
import io
from pathlib import Path
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import safe_last


def percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    def _rank(window: np.ndarray) -> float:
        return float(pd.Series(window).rank(pct=True).iloc[-1])

    return series.rolling(lookback).apply(_rank, raw=True)


def label_regime(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    if value < 0.33:
        return "low"
    if value < 0.67:
        return "medium"
    return "high"


def fit_hmm_states(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()
    if len(clean) < 100:
        return pd.Series(index=returns.index, dtype="float64")

    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return pd.Series(index=returns.index, dtype="float64")

    try:
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=42)
        values = clean.to_numpy().reshape(-1, 1)
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            model.fit(values)
            predicted = model.predict(values)
        states = pd.Series(predicted, index=clean.index)
        state_vol = {state: clean.loc[states[states == state].index].std() for state in states.unique()}
        ordered = sorted(state_vol, key=state_vol.get)
        mapping = {ordered[0]: 0, ordered[-1]: 1}
        return states.map(mapping).reindex(returns.index)
    except Exception:
        return pd.Series(index=returns.index, dtype="float64")


def plot_volatility(frame: pd.DataFrame, chart_path: str | Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frame.index, frame["realized_vol_21d"], label="21D Realized Vol", linewidth=1.4)

    colors = {"low": "#b8e0d2", "medium": "#f7d794", "high": "#f5b7b1"}
    regime = frame["vol_regime"].ffill()
    start = None
    current = None
    for idx, label in regime.items():
        if label != current:
            if current is not None and start is not None:
                ax.axvspan(start, idx, color=colors.get(current, "#eeeeee"), alpha=0.25)
            start = idx
            current = label
    if current is not None and start is not None:
        ax.axvspan(start, frame.index.max(), color=colors.get(current, "#eeeeee"), alpha=0.25)

    ax.set_title("Volatility Regime Detection")
    ax.set_ylabel("Annualized Volatility")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def run(settings: dict, dataset: dict, run_id: str) -> dict:
    frame = dataset["data"].copy()
    vol_window = int(settings["windows"]["vol"])
    rank_window = int(settings["windows"]["vol_rank"])

    frame["realized_vol_21d"] = frame["asset_log_return"].rolling(vol_window).std() * np.sqrt(252)
    frame["vol_percentile"] = percentile_rank(frame["realized_vol_21d"], rank_window)
    frame["vol_regime"] = frame["vol_percentile"].apply(label_regime)
    frame["hmm_state"] = fit_hmm_states(frame["asset_log_return"]) if settings["optional_features"].get("hmm", False) else np.nan

    chart_path = Path(settings["output"]["charts_dir"]) / f"volatility_regime_{run_id}.png"
    plot_volatility(frame, chart_path)
    shutil.copyfile(chart_path, Path(settings["output"]["charts_dir"]) / "latest_volatility_regime.png")

    latest_regime = None
    non_na_regime = frame["vol_regime"].dropna()
    if not non_na_regime.empty:
        latest_regime = non_na_regime.iloc[-1]

    return {
        "data": frame,
        "artifacts": {"vol_chart": str(chart_path)},
        "summary": {
            "latest_realized_vol": safe_last(frame["realized_vol_21d"]),
            "latest_vol_regime": latest_regime,
            "latest_hmm_state": safe_last(frame["hmm_state"]),
        },
    }
