from __future__ import annotations

import copy
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def load_settings(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_directories(paths: Iterable[str | Path]) -> None:
    for raw_path in paths:
        Path(raw_path).mkdir(parents=True, exist_ok=True)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def format_num(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def latest_matching_file(directory: str | Path, pattern: str) -> Path | None:
    candidates = sorted(Path(directory).glob(pattern))
    return candidates[-1] if candidates else None


def safe_last(series) -> float | None:
    if series is None:
        return None
    non_na = series.dropna()
    if non_na.empty:
        return None
    return float(non_na.iloc[-1])


def to_iso_date(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def slugify_symbol(symbol: str) -> str:
    return symbol.replace("=", "_").replace(".", "_").replace("-", "_")


def metric_dict(label: str, value: str, comment: str = "") -> dict[str, str]:
    return {"metric": label, "value": value, "comment": comment}

