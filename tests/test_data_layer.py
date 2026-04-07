from __future__ import annotations

import pandas as pd

from src.data_layer import _available_date


def test_cpi_available_date_uses_release_lag():
    frame = pd.DataFrame({"cpi_index": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    available = _available_date(frame, "cpi_index", "release_lag")
    assert available[0] > pd.Timestamp("2024-01-31")
    assert available[1] > pd.Timestamp("2024-02-29")


def test_daily_series_available_same_day():
    frame = pd.DataFrame({"dgs10": [4.0]}, index=pd.to_datetime(["2024-01-10"]))
    available = _available_date(frame, "dgs10", "release_lag")
    assert available[0] == pd.Timestamp("2024-01-10")

