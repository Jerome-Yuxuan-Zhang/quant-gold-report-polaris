from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import date
import io
from io import StringIO
from pathlib import Path
from typing import Callable

import akshare as ak
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.utils import latest_matching_file, slugify_symbol


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass
class SeriesResult:
    name: str
    frame: pd.DataFrame
    source: str
    status: str
    note: str = ""


def _normalize_history(history: pd.DataFrame, column_name: str) -> pd.DataFrame:
    history = history.copy()
    if history.empty:
        raise ValueError("Empty history")
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history.index = pd.to_datetime(history.index).tz_localize(None)
    if "Adj Close" in history.columns:
        series = history["Adj Close"]
    elif "Close" in history.columns:
        series = history["Close"]
    else:
        series = history.iloc[:, 0]
    cleaned = series.rename(column_name).to_frame()
    cleaned.index.name = "date"
    return cleaned


def _find_column(frame: pd.DataFrame, candidates: list[str], fallback_index: int | None = None) -> str:
    normalized = {str(column).strip(): column for column in frame.columns}
    for candidate in candidates:
        for text, original in normalized.items():
            if candidate.lower() == text.lower() or candidate.lower() in text.lower():
                return original
    if fallback_index is not None:
        return frame.columns[fallback_index]
    raise KeyError(f"Missing expected columns: {candidates}")


def download_yfinance_series(symbol: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        history = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    return _normalize_history(history, column_name)


def download_fred_series(series_id: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    response = requests.get(FRED_URL, params={"id": series_id, "cosd": start, "coed": end}, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    date_col = "DATE" if "DATE" in frame.columns else "observation_date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.rename(columns={date_col: "date", series_id: column_name})
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.set_index("date")
    return frame[[column_name]]


def download_akshare_foreign_futures(symbol: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = ak.futures_foreign_hist(symbol=symbol.replace("=F", ""))
    date_col = _find_column(frame, ["date", "日期"], fallback_index=0)
    close_col = _find_column(frame, ["close", "收盘"], fallback_index=4)
    frame = frame.rename(columns={date_col: "date", close_col: column_name})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
    frame = frame.set_index("date")[[column_name]]
    frame.index.name = "date"
    return frame


def download_akshare_us_stock(symbol: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    code = f"107.{symbol}"
    frame = ak.stock_us_hist(
        symbol=code,
        period="daily",
        start_date=pd.Timestamp(start).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end).strftime("%Y%m%d"),
        adjust="",
    )
    date_col = _find_column(frame, ["date", "日期"], fallback_index=0)
    close_col = _find_column(frame, ["close", "收盘"], fallback_index=2)
    frame = frame.rename(columns={date_col: "date", close_col: column_name})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.set_index("date")[[column_name]]
    frame.index.name = "date"
    return frame


def download_akshare_dxy(start: str, end: str, column_name: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = ak.index_global_hist_em(symbol="美元指数")
    date_col = _find_column(frame, ["date", "日期"], fallback_index=0)
    close_col = _find_column(frame, ["latest", "最新"], fallback_index=4)
    frame = frame.rename(columns={date_col: "date", close_col: column_name})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
    frame = frame.set_index("date")[[column_name]]
    frame.index.name = "date"
    return frame


def download_shanghai_gold_series(symbol: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    candidates: list[Callable[[], pd.DataFrame]] = [
        lambda: ak.spot_hist_sge(symbol=symbol),
        lambda: ak.spot_sge_hist(symbol=symbol),
        lambda: ak.spot_golden_benchmark_sge(),
    ]
    last_error: Exception | None = None

    for loader in candidates:
        try:
            frame = loader()
            break
        except Exception as exc:  # pragma: no cover
            last_error = exc
    else:
        raise RuntimeError(f"Could not download Shanghai gold series for {symbol}") from last_error

    date_col = _find_column(frame, ["date", "日期"], fallback_index=0)
    close_col = _find_column(frame, ["close", "收盘", "价格"], fallback_index=min(1, len(frame.columns) - 1))
    frame = frame.rename(columns={date_col: "date", close_col: column_name})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
    frame = frame.set_index("date")[[column_name]]
    frame.index.name = "date"
    return frame


def download_market_series(symbol: str, start: str, end: str, column_name: str, settings: dict) -> pd.DataFrame:
    strategies: list[Callable[[], pd.DataFrame]] = []

    if symbol == settings["asset"]["shanghai_symbol"]:
        strategies.append(lambda: download_shanghai_gold_series(symbol, start, end, column_name))
    elif symbol.endswith("=F"):
        strategies.append(lambda: download_akshare_foreign_futures(symbol, start, end, column_name))
    elif symbol.isupper() and "=" not in symbol and "." not in symbol:
        strategies.append(lambda: download_akshare_us_stock(symbol, start, end, column_name))
    elif symbol == settings["macro_symbols"]["dxy"]:
        strategies.append(lambda: download_akshare_dxy(start, end, column_name))
    elif symbol == settings["asset"]["optional_fx_symbol"]:
        strategies.append(lambda: download_fred_series(settings["macro_symbols"]["cny"], start, end, column_name))

    if symbol != settings["asset"]["shanghai_symbol"]:
        strategies.append(lambda: download_yfinance_series(symbol, start, end, column_name))

    last_error: Exception | None = None
    for strategy in strategies:
        try:
            return strategy()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download market series for {symbol}") from last_error


def persist_raw_snapshot(frame: pd.DataFrame, directory: str | Path, dataset_name: str, run_id: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{dataset_name}_{run_id}.parquet"
    frame.to_parquet(output_path)
    return output_path


def load_latest_snapshot(directory: str | Path, dataset_names: list[str]) -> pd.DataFrame | None:
    candidates = []
    for dataset_name in dataset_names:
        latest = latest_matching_file(directory, f"{dataset_name}_*.parquet")
        if latest is not None:
            candidates.append(latest)
    if not candidates:
        return None
    latest_path = sorted(candidates)[-1]
    return pd.read_parquet(latest_path)


def fetch_with_cache(
    *,
    dataset_name: str,
    downloader: Callable[[], pd.DataFrame],
    raw_dir: str | Path,
    run_id: str,
    cache_aliases: list[str] | None = None,
) -> SeriesResult:
    try:
        frame = downloader()
        persist_raw_snapshot(frame, raw_dir, dataset_name, run_id)
        return SeriesResult(name=dataset_name, frame=frame, source="remote", status="ok")
    except Exception as exc:
        cached = load_latest_snapshot(raw_dir, [dataset_name, *(cache_aliases or [])])
        if cached is not None:
            cached.index = pd.to_datetime(cached.index)
            return SeriesResult(dataset_name, cached, "cache", "fallback", str(exc))
        return SeriesResult(dataset_name, pd.DataFrame(), "remote", "missing", str(exc))


def _available_date(frame: pd.DataFrame, series_name: str, asof_mode: str) -> pd.Series:
    base_index = pd.to_datetime(frame.index)
    if asof_mode != "release_lag":
        return base_index
    if series_name == "cpi_index":
        return base_index + pd.offsets.MonthEnd(1) + pd.offsets.BDay(10)
    return base_index


def _merge_macro_series(master: pd.DataFrame, frame: pd.DataFrame, value_columns: list[str], series_name: str, asof_mode: str) -> pd.DataFrame:
    if frame.empty:
        return master
    macro = frame.copy().sort_index()
    macro["available_date"] = _available_date(macro, series_name, asof_mode)
    left = master.reset_index().rename(columns={"index": "date"})
    right = macro.reset_index().rename(columns={macro.index.name or "index": "observation_date"})
    merge_columns = ["available_date"] + value_columns
    merged = pd.merge_asof(
        left.sort_values("date"),
        right[merge_columns].sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    merged = merged.drop(columns=["available_date"])
    return merged.set_index("date")


def _prepare_cpi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    monthly = frame.copy().sort_index()
    monthly["cpi_yoy"] = monthly["cpi_index"].pct_change(12, fill_method=None)
    return monthly


def build_master_dataframe(results: dict[str, SeriesResult], settings: dict) -> tuple[pd.DataFrame, dict]:
    start = settings["date_range"]["start"]
    end = settings["date_range"]["end"] or date.today().isoformat()
    master = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="B"))
    master.index.name = "date"

    asof_mode = settings["macro_symbols"].get("asof_mode", "release_lag")
    asset_cfg = settings["asset"]
    selected_symbol = asset_cfg["default_symbol"]

    # Primary and comparison assets
    ordered_symbols = list(dict.fromkeys([selected_symbol, *asset_cfg["comparison_symbols"], asset_cfg["tradable_benchmark_symbol"]]))
    for symbol in ordered_symbols:
        key = f"symbol_{slugify_symbol(symbol)}"
        result = results.get(key)
        if result is None or result.frame.empty:
            continue
        col_name = f"close_{slugify_symbol(symbol)}"
        joined = result.frame.rename(columns={result.frame.columns[0]: col_name})
        master = master.join(joined, how="left")
        master[col_name] = master[col_name].ffill()

    primary_col = f"close_{slugify_symbol(selected_symbol)}"
    benchmark_col = f"close_{slugify_symbol(asset_cfg['tradable_benchmark_symbol'])}"
    if primary_col in master.columns:
        master["asset_close"] = master[primary_col]
    if benchmark_col in master.columns:
        master["tradable_benchmark_close"] = master[benchmark_col]

    # Daily market series
    for name, target in [("dxy", "dxy_close"), ("cny", "cny_close")]:
        result = results.get(name)
        if result and not result.frame.empty:
            master = master.join(result.frame.rename(columns={result.frame.columns[0]: target}), how="left")
            master[target] = master[target].ffill()

    # Macro series with as-of handling
    macro_map = {
        "dgs10": ["dgs10"],
        "dfii10": ["dfii10"],
        "cpi": ["cpi_index", "cpi_yoy"],
    }
    if results.get("dgs10") and not results["dgs10"].frame.empty:
        master = _merge_macro_series(master, results["dgs10"].frame, macro_map["dgs10"], "dgs10", asof_mode)
    if results.get("dfii10") and not results["dfii10"].frame.empty:
        master = _merge_macro_series(master, results["dfii10"].frame, macro_map["dfii10"], "dfii10", asof_mode)
    if results.get("cpi") and not results["cpi"].frame.empty:
        master = _merge_macro_series(master, _prepare_cpi_frame(results["cpi"].frame), macro_map["cpi"], "cpi_index", asof_mode)

    if results.get("symbol_Au99_99") and not results["symbol_Au99_99"].frame.empty:
        frame = results["symbol_Au99_99"].frame.rename(columns={results["symbol_Au99_99"].frame.columns[0]: "shanghai_gold_price"})
        master = master.join(frame, how="left")
        master["shanghai_gold_price"] = master["shanghai_gold_price"].ffill()

    if "asset_close" in master.columns:
        master["asset_log_return"] = np.log(master["asset_close"] / master["asset_close"].shift(1))
        master["asset_simple_return"] = master["asset_close"].pct_change()

    if "tradable_benchmark_close" in master.columns:
        master["benchmark_simple_return"] = master["tradable_benchmark_close"].pct_change()

    if "dxy_close" in master.columns:
        master["dxy_log_return"] = np.log(master["dxy_close"] / master["dxy_close"].shift(1))

    if "dgs10" in master.columns:
        master["nominal_yield"] = master["dgs10"] / 100.0
    if "dfii10" in master.columns:
        master["real_yield"] = master["dfii10"] / 100.0
    elif {"nominal_yield", "cpi_yoy"}.issubset(master.columns):
        master["real_yield"] = master["nominal_yield"] - master["cpi_yoy"]
    if "real_yield" in master.columns:
        master["real_yield_change"] = master["real_yield"].diff()
    if {"nominal_yield", "real_yield"}.issubset(master.columns):
        master["breakeven_inflation"] = master["nominal_yield"] - master["real_yield"]

    if {"asset_close", "shanghai_gold_price", "cny_close"}.issubset(master.columns):
        usd_gold_cny = master["asset_close"] * master["cny_close"]
        master["shanghai_gold_relative_spread"] = master["shanghai_gold_price"] / usd_gold_cny - 1

    metadata = {
        "selected_symbol": selected_symbol,
        "comparison_symbols": asset_cfg["comparison_symbols"],
        "roles": asset_cfg["roles"],
        "macro_asof_mode": asof_mode,
        "macro_assumptions": {
            "dgs10": "Daily FRED yield assumed available on same business day close.",
            "dfii10": "Daily FRED real yield assumed available on same business day close.",
            "cpi_index": "Monthly CPI mapped into daily panel using release-lag availability, not naive forward-fill from observation month.",
        },
    }
    return master, metadata


def summarize_dataset(master: pd.DataFrame, results: dict[str, SeriesResult], metadata: dict) -> dict:
    coverage = {
        key: {
            "status": result.status,
            "source": result.source,
            "rows": int(len(result.frame)),
            "note": result.note,
        }
        for key, result in results.items()
    }
    return {
        "start": master.index.min().date().isoformat() if not master.empty else None,
        "end": master.index.max().date().isoformat() if not master.empty else None,
        "observations": int(master["asset_close"].dropna().shape[0]) if "asset_close" in master.columns else 0,
        "coverage": coverage,
        "available_columns": list(master.columns),
        "selected_symbol": metadata["selected_symbol"],
        "asof_mode": metadata["macro_asof_mode"],
    }


def run(settings: dict, run_id: str) -> dict:
    raw_dir = settings["output"]["raw_dir"]
    processed_dir = Path(settings["output"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    start = settings["date_range"]["start"]
    end = settings["date_range"]["end"] or date.today().isoformat()
    asset_cfg = settings["asset"]
    macro_cfg = settings["macro_symbols"]

    symbols_to_fetch = {asset_cfg["default_symbol"], asset_cfg["tradable_benchmark_symbol"], *asset_cfg["comparison_symbols"]}
    results: dict[str, SeriesResult] = {}
    symbol_cache_aliases = {
        "GLD": ["flow_proxy", "asset"],
        "GC=F": ["asset"],
        asset_cfg["shanghai_symbol"]: ["shanghai_gold"],
    }

    for symbol in symbols_to_fetch:
        dataset_name = f"symbol_{slugify_symbol(symbol)}"
        column_name = f"{dataset_name}_close"
        results[dataset_name] = fetch_with_cache(
            dataset_name=dataset_name,
            downloader=lambda symbol=symbol, column_name=column_name: download_market_series(symbol, start, end, column_name, settings),
            raw_dir=raw_dir,
            run_id=run_id,
            cache_aliases=symbol_cache_aliases.get(symbol, []),
        )

    results["dxy"] = fetch_with_cache(
        dataset_name="dxy",
        downloader=lambda: download_market_series(macro_cfg["dxy"], start, end, "dxy_close", settings),
        raw_dir=raw_dir,
        run_id=run_id,
    )
    results["dgs10"] = fetch_with_cache(
        dataset_name="dgs10",
        downloader=lambda: download_fred_series(macro_cfg["dgs10"], start, end, "dgs10"),
        raw_dir=raw_dir,
        run_id=run_id,
    )
    results["cpi"] = fetch_with_cache(
        dataset_name="cpi",
        downloader=lambda: download_fred_series(macro_cfg["cpi"], start, end, "cpi_index"),
        raw_dir=raw_dir,
        run_id=run_id,
    )
    results["dfii10"] = fetch_with_cache(
        dataset_name="dfii10",
        downloader=lambda: download_fred_series(macro_cfg["dfii10"], start, end, "dfii10"),
        raw_dir=raw_dir,
        run_id=run_id,
    )
    results["cny"] = fetch_with_cache(
        dataset_name="cny",
        downloader=lambda: download_market_series(asset_cfg["optional_fx_symbol"], start, end, "cny_close", settings),
        raw_dir=raw_dir,
        run_id=run_id,
    )

    master, metadata = build_master_dataframe(results, settings)
    if "asset_close" not in master.columns or master["asset_close"].dropna().empty:
        raise RuntimeError("Primary asset price history is unavailable; cannot build report.")

    processed_path = processed_dir / f"master_dataset_{run_id}.parquet"
    master.to_parquet(processed_path)

    return {
        "data": master,
        "summary": summarize_dataset(master, results, metadata),
        "artifacts": {"processed_dataset_path": str(processed_path)},
        "figures": {},
        "metadata": metadata,
    }
