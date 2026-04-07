from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils import safe_last


def rolling_spearman(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    ranked_a = a.rank()
    ranked_b = b.rank()
    return ranked_a.rolling(window).corr(ranked_b)


def run_regression(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    regressors = ["dxy_log_return", "real_yield_change"]
    subset = frame[["asset_log_return"] + regressors].dropna()
    if len(subset) < 50:
        return pd.DataFrame(), {"status": "insufficient_data"}

    X = sm.add_constant(subset[regressors])
    y = subset["asset_log_return"]
    model = sm.OLS(y, X).fit()
    table = pd.DataFrame({"coefficient": model.params, "t_stat": model.tvalues, "p_value": model.pvalues})
    summary = {
        "status": "ok",
        "r_squared": float(model.rsquared),
        "observations": int(model.nobs),
        "coefficients": {key: float(value) for key, value in model.params.to_dict().items()},
    }
    return table, summary


def shanghai_summary(frame: pd.DataFrame) -> dict:
    if "shanghai_gold_relative_spread" not in frame.columns:
        return {"status": "unavailable"}
    spread = frame["shanghai_gold_relative_spread"].dropna()
    if spread.empty:
        return {"status": "unavailable"}
    return {
        "status": "ok",
        "latest_spread": float(spread.iloc[-1]),
        "average_spread": float(spread.mean()),
        "spread_std": float(spread.std()),
    }


def run(settings: dict, dataset: dict, run_id: str) -> dict:
    frame = dataset["data"].copy()
    window = int(settings["windows"]["corr"])
    for column in ["dxy_log_return", "real_yield_change"]:
        if column not in frame.columns:
            frame[column] = np.nan

    frame["corr_dxy_pearson"] = frame["asset_log_return"].rolling(window).corr(frame["dxy_log_return"])
    frame["corr_real_yield_pearson"] = frame["asset_log_return"].rolling(window).corr(frame["real_yield_change"])
    frame["corr_dxy_spearman"] = rolling_spearman(frame["asset_log_return"], frame["dxy_log_return"], window)
    frame["corr_real_yield_spearman"] = rolling_spearman(frame["asset_log_return"], frame["real_yield_change"], window)

    regression_table, regression_summary = run_regression(frame)
    return {
        "data": frame,
        "summary": {
            "latest_corr_dxy": safe_last(frame["corr_dxy_pearson"]),
            "latest_corr_real_yield": safe_last(frame["corr_real_yield_pearson"]),
            "regression_r_squared": regression_summary.get("r_squared"),
            "regression_status": regression_summary.get("status"),
            "dxy_beta": regression_summary.get("coefficients", {}).get("dxy_log_return"),
            "real_yield_beta": regression_summary.get("coefficients", {}).get("real_yield_change"),
            "shanghai_analysis": shanghai_summary(frame),
            "regression_table_preview": regression_table.reset_index().rename(columns={"index": "term"}).to_dict(orient="records"),
        },
        "artifacts": {"regression_table": regression_table.reset_index().rename(columns={"index": "term"})},
        "figures": {},
        "metadata": {"corr_window": window},
    }

