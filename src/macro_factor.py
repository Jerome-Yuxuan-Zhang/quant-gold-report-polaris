from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
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
    table = pd.DataFrame(
        {
            "coefficient": model.params,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
        }
    )
    summary = {
        "status": "ok",
        "r_squared": float(model.rsquared),
        "observations": int(model.nobs),
        "coefficients": {key: float(value) for key, value in model.params.to_dict().items()},
    }
    return table, summary


def plot_correlations(frame: pd.DataFrame, chart_path: str | Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(frame.index, frame["corr_dxy_pearson"], label="Pearson", linewidth=1.2)
    axes[0].plot(frame.index, frame["corr_dxy_spearman"], label="Spearman", linewidth=1.0, alpha=0.85)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Gold vs DXY Rolling Correlation")
    axes[0].legend(loc="upper left")

    axes[1].plot(frame.index, frame["corr_real_yield_pearson"], label="Pearson", linewidth=1.2)
    axes[1].plot(frame.index, frame["corr_real_yield_spearman"], label="Spearman", linewidth=1.0, alpha=0.85)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Gold vs Real Yield Rolling Correlation")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


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
    chart_path = Path(settings["output"]["charts_dir"]) / f"macro_correlation_{run_id}.png"
    plot_correlations(frame, chart_path)
    shutil.copyfile(chart_path, Path(settings["output"]["charts_dir"]) / "latest_macro_correlation.png")

    return {
        "data": frame,
        "artifacts": {
            "macro_chart": str(chart_path),
            "regression_table": regression_table.reset_index().rename(columns={"index": "term"}),
        },
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
    }
