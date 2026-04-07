from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_dataset():
    dates = pd.bdate_range("2022-01-03", periods=420)
    base = np.linspace(100, 160, len(dates))
    cycle = 4 * np.sin(np.linspace(0, 12 * np.pi, len(dates)))
    asset_close = base + cycle
    dxy_close = 100 + 2 * np.cos(np.linspace(0, 8 * np.pi, len(dates)))
    nominal_yield = 0.02 + 0.003 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
    real_yield = 0.01 + 0.002 * np.cos(np.linspace(0, 6 * np.pi, len(dates)))
    cny_close = 6.6 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
    shanghai_gold = asset_close * cny_close * 1.02

    frame = pd.DataFrame(
        {
            "asset_close": asset_close,
            "tradable_benchmark_close": asset_close * 0.995,
            "dxy_close": dxy_close,
            "dxy_log_return": np.log(dxy_close / np.roll(dxy_close, 1)),
            "nominal_yield": nominal_yield,
            "real_yield": real_yield,
            "real_yield_change": pd.Series(real_yield, index=dates).diff().values,
            "cpi_yoy": 0.03 + 0.001 * np.sin(np.linspace(0, 5 * np.pi, len(dates))),
            "cny_close": cny_close,
            "shanghai_gold_price": shanghai_gold,
            "asset_simple_return": pd.Series(asset_close, index=dates).pct_change().values,
            "asset_log_return": np.log(asset_close / np.roll(asset_close, 1)),
        },
        index=dates,
    )
    frame.index.name = "date"
    frame.iloc[0, frame.columns.get_loc("asset_simple_return")] = 0.0
    frame.iloc[0, frame.columns.get_loc("asset_log_return")] = 0.0
    frame.iloc[0, frame.columns.get_loc("dxy_log_return")] = 0.0
    frame["shanghai_gold_relative_spread"] = frame["shanghai_gold_price"] / (frame["asset_close"] * frame["cny_close"]) - 1

    return {
        "data": frame,
        "summary": {
            "start": dates.min().date().isoformat(),
            "end": dates.max().date().isoformat(),
            "observations": len(dates),
            "selected_symbol": "GLD",
            "asof_mode": "release_lag",
        },
        "artifacts": {},
        "figures": {},
        "metadata": {
            "selected_symbol": "GLD",
            "comparison_symbols": ["GC=F", "Au99.99"],
            "roles": {"GLD": "Tradable ETF benchmark", "GC=F": "COMEX futures comparison", "Au99.99": "Shanghai gold comparison"},
            "macro_asof_mode": "release_lag",
            "macro_assumptions": {
                "dgs10": "Daily FRED yield assumed available on same business day close.",
                "dfii10": "Daily FRED real yield assumed available on same business day close.",
                "cpi_index": "Monthly CPI mapped into daily panel using release-lag availability, not naive forward-fill from observation month.",
            },
        },
    }
