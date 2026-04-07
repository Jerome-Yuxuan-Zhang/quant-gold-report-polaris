from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import safe_last


def generate_dual_ma_signal(price: pd.Series, fast: int, slow: int) -> pd.Series:
    return (price.rolling(fast).mean() > price.rolling(slow).mean()).astype(float)


def compute_turnover(signal: pd.Series) -> pd.Series:
    return signal.fillna(0).diff().abs().fillna(signal.fillna(0).abs())


def apply_strategy_returns(signal: pd.Series, returns: pd.Series, transaction_cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series]:
    lagged_signal = signal.shift(1).fillna(0)
    turnover = compute_turnover(signal)
    transaction_cost = turnover * (transaction_cost_bps / 10000.0)
    strategy_returns = lagged_signal * returns.fillna(0) - transaction_cost
    return strategy_returns, transaction_cost


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


def choose_best_pair(train_frame: pd.DataFrame, candidates: list[list[int]], risk_free_rate: float, transaction_cost_bps: float) -> tuple[int, int]:
    best_pair = tuple(candidates[0])
    best_score = -np.inf
    for fast, slow in candidates:
        if slow <= fast:
            continue
        signal = generate_dual_ma_signal(train_frame["asset_close"], fast, slow)
        strategy_returns, _ = apply_strategy_returns(signal, train_frame["asset_simple_return"], transaction_cost_bps)
        score = sharpe_ratio(strategy_returns, risk_free_rate) or -np.inf
        if score > best_score:
            best_score = score
            best_pair = (fast, slow)
    return best_pair


def run(settings: dict, dataset: dict, trend: dict, volatility: dict, run_id: str) -> dict:
    frame = trend["data"].copy()
    frame["selected_fast"] = np.nan
    frame["selected_slow"] = np.nan
    frame["walk_forward_signal"] = 0.0

    candidates = settings["windows"]["walk_forward_candidates"]
    refit_every = int(settings["windows"]["walk_forward_refit"])
    risk_free_rate = float(settings["backtest"]["risk_free_rate"])
    transaction_cost_bps = float(settings["backtest"]["transaction_cost_bps"])
    max_slow = max(slow for _, slow in candidates)
    start_idx = max(refit_every, max_slow)
    selection_log: list[dict] = []

    if len(frame) > start_idx:
        for start_pos in range(start_idx, len(frame), refit_every):
            train_frame = frame.iloc[:start_pos].copy()
            chosen_fast, chosen_slow = choose_best_pair(train_frame, candidates, risk_free_rate, transaction_cost_bps)
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

    gross_strategy_returns, transaction_cost = apply_strategy_returns(
        frame["walk_forward_signal"],
        frame["asset_simple_return"],
        transaction_cost_bps=0.0,
    )
    net_strategy_returns, transaction_cost = apply_strategy_returns(
        frame["walk_forward_signal"],
        frame["asset_simple_return"],
        transaction_cost_bps=transaction_cost_bps,
    )

    frame["turnover"] = compute_turnover(frame["walk_forward_signal"])
    frame["transaction_cost"] = transaction_cost
    frame["strategy_simple_return_gross"] = gross_strategy_returns
    frame["strategy_simple_return"] = net_strategy_returns
    frame["strategy_equity"] = (1 + frame["strategy_simple_return"]).cumprod()
    frame["strategy_equity_gross"] = (1 + frame["strategy_simple_return_gross"]).cumprod()
    frame["asset_equity"] = (1 + frame["asset_simple_return"].fillna(0)).cumprod()

    monthly = frame["strategy_simple_return"].resample("ME").apply(lambda values: (1 + values).prod() - 1)
    hit_rate = float((monthly > 0).mean()) if not monthly.empty else None

    if "vol_regime" in volatility["data"].columns:
        frame["vol_regime"] = volatility["data"]["vol_regime"]

    return {
        "data": frame,
        "summary": {
            "annualized_return": annualized_return(frame["strategy_simple_return"]),
            "annualized_return_gross": annualized_return(frame["strategy_simple_return_gross"]),
            "sharpe_ratio": sharpe_ratio(frame["strategy_simple_return"], risk_free_rate),
            "sharpe_ratio_gross": sharpe_ratio(frame["strategy_simple_return_gross"], risk_free_rate),
            "max_drawdown": max_drawdown(frame["strategy_equity"]),
            "max_drawdown_gross": max_drawdown(frame["strategy_equity_gross"]),
            "hit_rate": hit_rate,
            "latest_equity": safe_last(frame["strategy_equity"]),
            "transaction_cost_bps": transaction_cost_bps,
            "total_turnover": float(frame["turnover"].sum()),
            "total_transaction_cost": float(frame["transaction_cost"].sum()),
        },
        "artifacts": {"parameter_log": selection_log},
        "figures": {},
        "metadata": {"selection_log": selection_log},
    }

