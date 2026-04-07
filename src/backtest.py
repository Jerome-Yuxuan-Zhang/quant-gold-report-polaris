from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import safe_last


def generate_dual_ma_signal(price: pd.Series, fast: int, slow: int) -> pd.Series:
    return (price.rolling(fast).mean() > price.rolling(slow).mean()).astype(float)


def annualized_return(returns: pd.Series) -> float | None:
    clean = returns.dropna()
    if clean.empty:
        return None
    return (1 + clean).prod() ** (252 / len(clean)) - 1


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float | None:
    clean = returns.dropna()
    if clean.empty or clean.std() == 0:
        return None
    excess = clean - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / clean.std()


def max_drawdown(equity_curve: pd.Series) -> float | None:
    clean = equity_curve.dropna()
    if clean.empty:
        return None
    running_max = clean.cummax()
    return float((clean / running_max - 1).min())


def choose_best_pair(train_frame: pd.DataFrame, candidates: list[list[int]], risk_free_rate: float) -> tuple[int, int]:
    best_pair = tuple(candidates[0])
    best_score = -np.inf
    for fast, slow in candidates:
        if slow <= fast:
            continue
        signal = generate_dual_ma_signal(train_frame["asset_close"], fast, slow).shift(1).fillna(0)
        strategy_returns = signal * train_frame["asset_simple_return"].fillna(0)
        score = sharpe_ratio(strategy_returns, risk_free_rate) or -np.inf
        if score > best_score:
            best_score = score
            best_pair = (fast, slow)
    return best_pair


def plot_equity_curve(frame: pd.DataFrame, chart_path: str | Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frame.index, frame["strategy_equity"], label="Strategy", linewidth=1.5)
    ax.plot(frame.index, frame["asset_equity"], label="Buy & Hold", linewidth=1.2, alpha=0.75)
    ax.set_title("Walk-Forward Dual-MA Backtest")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def run(settings: dict, dataset: dict, trend: dict, run_id: str) -> dict:
    frame = trend["data"].copy()
    frame["selected_fast"] = np.nan
    frame["selected_slow"] = np.nan
    frame["walk_forward_signal"] = 0.0

    candidates = settings["windows"]["walk_forward_candidates"]
    refit_every = int(settings["windows"]["walk_forward_refit"])
    risk_free_rate = float(settings["report"]["risk_free_rate"])
    max_slow = max(slow for _, slow in candidates)
    start_idx = max(refit_every, max_slow)
    selection_log: list[dict] = []

    if len(frame) > start_idx:
        for start_pos in range(start_idx, len(frame), refit_every):
            train_frame = frame.iloc[:start_pos].copy()
            chosen_fast, chosen_slow = choose_best_pair(train_frame, candidates, risk_free_rate)
            segment_end = min(start_pos + refit_every, len(frame))
            signal = generate_dual_ma_signal(frame["asset_close"], chosen_fast, chosen_slow).iloc[start_pos:segment_end]
            frame.iloc[start_pos:segment_end, frame.columns.get_loc("walk_forward_signal")] = signal.values
            frame.iloc[start_pos:segment_end, frame.columns.get_loc("selected_fast")] = chosen_fast
            frame.iloc[start_pos:segment_end, frame.columns.get_loc("selected_slow")] = chosen_slow
            selection_log.append(
                {
                    "start": frame.index[start_pos].date().isoformat(),
                    "end": frame.index[segment_end - 1].date().isoformat(),
                    "fast": chosen_fast,
                    "slow": chosen_slow,
                }
            )

    frame["strategy_simple_return"] = frame["walk_forward_signal"].shift(1).fillna(0) * frame["asset_simple_return"].fillna(0)
    frame["strategy_equity"] = (1 + frame["strategy_simple_return"]).cumprod()
    frame["asset_equity"] = (1 + frame["asset_simple_return"].fillna(0)).cumprod()

    monthly = frame["strategy_simple_return"].resample("ME").apply(lambda values: (1 + values).prod() - 1)
    hit_rate = float((monthly > 0).mean()) if not monthly.empty else None

    summary = {
        "annualized_return": annualized_return(frame["strategy_simple_return"]),
        "sharpe_ratio": sharpe_ratio(frame["strategy_simple_return"], risk_free_rate),
        "max_drawdown": max_drawdown(frame["strategy_equity"]),
        "hit_rate": hit_rate,
        "latest_equity": safe_last(frame["strategy_equity"]),
    }

    chart_path = Path(settings["output"]["charts_dir"]) / f"backtest_equity_{run_id}.png"
    plot_equity_curve(frame, chart_path)
    shutil.copyfile(chart_path, Path(settings["output"]["charts_dir"]) / "latest_backtest_equity.png")

    return {
        "data": frame,
        "artifacts": {"equity_curve_chart": str(chart_path), "parameter_log": selection_log},
        "summary": summary,
    }
