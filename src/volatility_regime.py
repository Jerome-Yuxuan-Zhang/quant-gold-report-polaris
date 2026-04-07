from __future__ import annotations

import contextlib
import io
import warnings

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


def run(settings: dict, dataset: dict, run_id: str) -> dict:
    frame = dataset["data"].copy()
    vol_window = int(settings["windows"]["vol"])
    rank_window = int(settings["windows"]["vol_rank"])

    frame["realized_vol_21d"] = frame["asset_log_return"].rolling(vol_window).std() * np.sqrt(252)
    frame["vol_percentile"] = percentile_rank(frame["realized_vol_21d"], rank_window)
    frame["vol_regime"] = frame["vol_percentile"].apply(label_regime)
    frame["hmm_state"] = fit_hmm_states(frame["asset_log_return"]) if settings["optional_features"].get("hmm", False) else np.nan

    non_na_regime = frame["vol_regime"].dropna()
    latest_regime = non_na_regime.iloc[-1] if not non_na_regime.empty else None
    return {
        "data": frame,
        "summary": {
            "latest_realized_vol": safe_last(frame["realized_vol_21d"]),
            "latest_vol_regime": latest_regime,
            "latest_hmm_state": safe_last(frame["hmm_state"]),
        },
        "artifacts": {},
        "figures": {},
        "metadata": {"vol_window": vol_window, "rank_window": rank_window},
    }

