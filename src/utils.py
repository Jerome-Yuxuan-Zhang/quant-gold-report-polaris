from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_settings(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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

