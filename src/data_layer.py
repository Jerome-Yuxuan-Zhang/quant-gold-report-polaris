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

from src.utils import latest_matching_file


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


def download_yfinance_series(symbol: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        history = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    return _normalize_history(history, column_name)


def download_fred_series(series_id: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    response = requests.get(
        FRED_URL,
        params={"id": series_id, "cosd": start, "coed": end},
        timeout=30,
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    date_col = "DATE" if "DATE" in frame.columns else "observation_date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.rename(columns={date_col: "date", series_id: column_name})
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.set_index("date")
    return frame[[column_name]]


def download_akshare_foreign_futures(symbol: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    mapped_symbol = symbol.replace("=F", "")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = ak.futures_foreign_hist(symbol=mapped_symbol)
    frame = frame.rename(columns={"date": "date", "close": column_name})
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
    frame = frame.rename(columns={"日期": "date", "收盘": column_name})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.set_index("date")[[column_name]]
    frame.index.name = "date"
    return frame


def download_akshare_dxy(start: str, end: str, column_name: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = ak.index_global_hist_em(symbol="\u7f8e\u5143\u6307\u6570")
    frame = frame.rename(columns={"日期": "date", "最新价": column_name})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
    frame = frame.set_index("date")[[column_name]]
    frame.index.name = "date"
    return frame


def download_market_series(symbol: str, start: str, end: str, column_name: str) -> pd.DataFrame:
    strategies: list[Callable[[], pd.DataFrame]] = []

    if symbol.endswith("=F"):
        strategies.append(lambda: download_akshare_foreign_futures(symbol, start, end, column_name))
    if symbol.isupper() and "=" not in symbol and "." not in symbol:
        strategies.append(lambda: download_akshare_us_stock(symbol, start, end, column_name))
    if symbol == "DX-Y.NYB":
        strategies.append(lambda: download_akshare_dxy(start, end, column_name))
    if symbol == "CNY=X":
        strategies.append(lambda: download_fred_series("DEXCHUS", start, end, column_name))

    strategies.append(lambda: download_yfinance_series(symbol, start, end, column_name))

    last_error: Exception | None = None
    for strategy in strategies:
        try:
            return strategy()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download market series for {symbol}") from last_error


def download_shanghai_gold_series(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    candidates: list[Callable[[], pd.DataFrame]] = [
        lambda: ak.spot_hist_sge(symbol=symbol),
        lambda: ak.spot_sge_hist(symbol=symbol),
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

    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    date_column = next((col for col in frame.columns if "日期" in col or col.lower() == "date"), None)
    price_column = next((col for col in frame.columns if "收盘" in col or "close" in col.lower() or "价格" in col), None)
    if date_column is None or price_column is None:
        raise RuntimeError("Unsupported Shanghai gold dataframe format")

    frame[date_column] = pd.to_datetime(frame[date_column])
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    frame = frame.loc[(frame[date_column] >= start_ts) & (frame[date_column] <= end_ts)]
    frame = frame.set_index(date_column)[[price_column]].rename(columns={price_column: "shanghai_gold_price"})
    frame.index.name = "date"
    return frame


def persist_raw_snapshot(frame: pd.DataFrame, directory: str | Path, dataset_name: str, run_id: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{dataset_name}_{run_id}.parquet"
    frame.to_parquet(output_path)
    return output_path


def load_latest_snapshot(directory: str | Path, dataset_name: str) -> pd.DataFrame | None:
    latest = latest_matching_file(directory, f"{dataset_name}_*.parquet")
    if latest is None:
        return None
    return pd.read_parquet(latest)


def fetch_with_cache(
    *,
    dataset_name: str,
    downloader: Callable[[], pd.DataFrame],
    raw_dir: str | Path,
    run_id: str,
) -> SeriesResult:
    try:
        frame = downloader()
        persist_raw_snapshot(frame, raw_dir, dataset_name, run_id)
        return SeriesResult(name=dataset_name, frame=frame, source="remote", status="ok")
    except Exception as exc:
        cached = load_latest_snapshot(raw_dir, dataset_name)
        if cached is not None:
            cached.index = pd.to_datetime(cached.index)
            return SeriesResult(dataset_name, cached, "cache", "fallback", str(exc))
        return SeriesResult(dataset_name, pd.DataFrame(), "remote", "missing", str(exc))


def build_master_dataframe(results: dict[str, SeriesResult], settings: dict) -> pd.DataFrame:
    start = settings["date_range"]["start"]
    end = settings["date_range"]["end"] or date.today().isoformat()
    index = pd.date_range(start=start, end=end, freq="B")
    master = pd.DataFrame(index=index)
    master.index.name = "date"

    for result in results.values():
        if result.frame.empty:
            continue
        frame = result.frame.copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        master = master.join(frame, how="left")

    macro_cols = [col for col in ["dgs10", "cpi_index", "dfii10"] if col in master.columns]
    if macro_cols:
        master[macro_cols] = master[macro_cols].ffill()

    price_cols = [col for col in ["asset_close", "flow_proxy_close", "dxy_close", "cny_close", "shanghai_gold_price"] if col in master.columns]
    if price_cols:
        master[price_cols] = master[price_cols].ffill()

    if "asset_close" in master.columns:
        master["asset_log_return"] = np.log(master["asset_close"] / master["asset_close"].shift(1))
        master["asset_simple_return"] = master["asset_close"].pct_change()

    if "flow_proxy_close" in master.columns:
        master["flow_proxy_log_return"] = np.log(master["flow_proxy_close"] / master["flow_proxy_close"].shift(1))

    if "dxy_close" in master.columns:
        master["dxy_log_return"] = np.log(master["dxy_close"] / master["dxy_close"].shift(1))

    if "cpi_index" in master.columns:
        master["cpi_yoy"] = master["cpi_index"].pct_change(252)

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

    return master


def summarize_dataset(master: pd.DataFrame, results: dict[str, SeriesResult]) -> dict:
    coverage = {}
    for key, result in results.items():
        coverage[key] = {
            "status": result.status,
            "source": result.source,
            "rows": int(len(result.frame)),
            "note": result.note,
        }

    return {
        "start": master.index.min().date().isoformat() if not master.empty else None,
        "end": master.index.max().date().isoformat() if not master.empty else None,
        "observations": int(master["asset_close"].dropna().shape[0]) if "asset_close" in master.columns else 0,
        "coverage": coverage,
        "available_columns": list(master.columns),
    }


def run(settings: dict, run_id: str) -> dict:
    raw_dir = settings["output"]["raw_dir"]
    processed_dir = Path(settings["output"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    start = settings["date_range"]["start"]
    end = settings["date_range"]["end"] or date.today().isoformat()
    asset_cfg = settings["asset"]
    macro_cfg = settings["macro_symbols"]
    optional_cfg = settings["optional_features"]

    results = {
        "asset": fetch_with_cache(
            dataset_name="asset",
            downloader=lambda: download_market_series(asset_cfg["primary_symbol"], start, end, "asset_close"),
            raw_dir=raw_dir,
            run_id=run_id,
        ),
        "flow_proxy": fetch_with_cache(
            dataset_name="flow_proxy",
            downloader=lambda: download_market_series(asset_cfg["flow_proxy_symbol"], start, end, "flow_proxy_close"),
            raw_dir=raw_dir,
            run_id=run_id,
        ),
        "dxy": fetch_with_cache(
            dataset_name="dxy",
            downloader=lambda: download_market_series(macro_cfg["dxy"], start, end, "dxy_close"),
            raw_dir=raw_dir,
            run_id=run_id,
        ),
        "dgs10": fetch_with_cache(
            dataset_name="dgs10",
            downloader=lambda: download_fred_series(macro_cfg["dgs10"], start, end, "dgs10"),
            raw_dir=raw_dir,
            run_id=run_id,
        ),
        "cpi": fetch_with_cache(
            dataset_name="cpi",
            downloader=lambda: download_fred_series(macro_cfg["cpi"], start, end, "cpi_index"),
            raw_dir=raw_dir,
            run_id=run_id,
        ),
        "dfii10": fetch_with_cache(
            dataset_name="dfii10",
            downloader=lambda: download_fred_series(macro_cfg["dfii10"], start, end, "dfii10"),
            raw_dir=raw_dir,
            run_id=run_id,
        ),
        "cny": fetch_with_cache(
            dataset_name="cny",
            downloader=lambda: download_market_series(asset_cfg["optional_fx_symbol"], start, end, "cny_close"),
            raw_dir=raw_dir,
            run_id=run_id,
        ),
    }

    if optional_cfg.get("shanghai_premium", False):
        results["shanghai_gold"] = fetch_with_cache(
            dataset_name="shanghai_gold",
            downloader=lambda: download_shanghai_gold_series(asset_cfg["shanghai_symbol"], start, end),
            raw_dir=raw_dir,
            run_id=run_id,
        )

    master = build_master_dataframe(results, settings)
    if "asset_close" not in master.columns or master["asset_close"].dropna().empty:
        raise RuntimeError("Primary asset price history is unavailable; cannot build report.")
    processed_path = processed_dir / f"master_dataset_{run_id}.parquet"
    master.to_parquet(processed_path)

    return {
        "data": master,
        "artifacts": {"processed_dataset_path": str(processed_path)},
        "summary": summarize_dataset(master, results),
    }
