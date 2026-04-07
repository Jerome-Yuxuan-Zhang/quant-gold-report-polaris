from __future__ import annotations

import pandas as pd

from src.backtest import apply_strategy_returns, max_drawdown, sharpe_ratio


def test_strategy_uses_lagged_signal():
    signal = pd.Series([0, 1, 1], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    returns = pd.Series([0.0, 0.10, -0.05], index=signal.index)
    strategy_returns, _ = apply_strategy_returns(signal, returns, transaction_cost_bps=0.0)
    assert strategy_returns.iloc[1] == 0.0
    assert round(strategy_returns.iloc[2], 6) == -0.05


def test_transaction_cost_is_deducted_on_turnover():
    signal = pd.Series([0, 1, 0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    returns = pd.Series([0.0, 0.02, 0.03], index=signal.index)
    strategy_returns, transaction_cost = apply_strategy_returns(signal, returns, transaction_cost_bps=10.0)
    assert round(transaction_cost.iloc[1], 6) == 0.001
    assert round(transaction_cost.iloc[2], 6) == 0.001
    assert round(strategy_returns.iloc[1], 6) == -0.001


def test_sharpe_ratio_matches_simple_case():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert sharpe_ratio(returns) is None


def test_max_drawdown_matches_expected_value():
    equity_curve = pd.Series([1.0, 1.1, 0.9, 1.05])
    assert round(max_drawdown(equity_curve), 6) == -0.181818

