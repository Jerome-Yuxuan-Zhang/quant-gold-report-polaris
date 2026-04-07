from __future__ import annotations

import json
import os
from typing import Any

from src.utils import format_num, format_pct


def build_payload(dataset: dict, trend: dict, volatility: dict, macro: dict, backtest: dict) -> dict[str, Any]:
    return {
        "dataset": dataset["summary"],
        "dataset_metadata": dataset.get("metadata", {}),
        "trend": trend["summary"],
        "volatility": volatility["summary"],
        "macro": macro["summary"],
        "backtest": backtest["summary"],
    }


def build_prompt(payload: dict[str, Any]) -> str:
    instructions = """
You are writing a gold research report using only the supplied statistics.
Do not invent facts, forecasts, or causal claims that are not directly supported by the data.
Return JSON with keys: executive_summary, macro_analysis, signal_review, limitations_note.
Each value should be one short paragraph.
"""
    return f"{instructions}\nDATA:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"


def fallback_sections(payload: dict[str, Any]) -> dict[str, str]:
    trend = payload["trend"]
    vol = payload["volatility"]
    macro = payload["macro"]
    backtest = payload["backtest"]
    dataset = payload["dataset"]
    asset_label = dataset.get("selected_symbol", "the selected asset")

    return {
        "executive_summary": (
            f"{asset_label} closed near {format_num(trend.get('latest_price'), 2)} with the dual moving-average filter "
            f"currently reading {format_num(trend.get('latest_dual_ma_signal'), 0)}. "
            f"The 21-day realized volatility is {format_pct(vol.get('latest_realized_vol'))}, placing the asset in a "
            f"{vol.get('latest_vol_regime', 'unclassified')} volatility regime."
        ),
        "macro_analysis": (
            f"The latest 63-day Pearson correlation between {asset_label} returns and DXY is {format_num(macro.get('latest_corr_dxy'))}, "
            f"while the correlation with real-yield changes is {format_num(macro.get('latest_corr_real_yield'))}. "
            f"The regression beta on DXY is {format_num(macro.get('dxy_beta'))} and the beta on real yields is "
            f"{format_num(macro.get('real_yield_beta'))}, with R-squared {format_num(macro.get('regression_r_squared'))}."
        ),
        "signal_review": (
            f"The walk-forward trend strategy delivered an annualized return of {format_pct(backtest.get('annualized_return'))}, "
            f"a Sharpe ratio of {format_num(backtest.get('sharpe_ratio'))}, and a maximum drawdown of "
            f"{format_pct(backtest.get('max_drawdown'))}. Monthly hit rate was {format_pct(backtest.get('hit_rate'))}."
        ),
        "limitations_note": (
            "This narrative is template-generated because no Anthropic API key was supplied. "
            "It only summarizes observed statistics and should not be treated as a forecast."
        ),
    }


def try_anthropic(settings: dict, prompt: str) -> dict[str, str] | None:
    api_key = os.getenv(settings["llm"]["api_key_env"])
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
    except Exception:
        return None

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=settings["llm"]["model"],
            max_tokens=settings["llm"]["max_tokens"],
            temperature=settings["llm"]["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        parsed = json.loads(text)
        required = {"executive_summary", "macro_analysis", "signal_review", "limitations_note"}
        return parsed if required.issubset(parsed) else None
    except Exception:
        return None


def run(settings: dict, dataset: dict, trend: dict, volatility: dict, macro: dict, backtest: dict) -> dict:
    payload = build_payload(dataset, trend, volatility, macro, backtest)
    prompt = build_prompt(payload)
    anthropic_sections = try_anthropic(settings, prompt)
    sections = anthropic_sections or fallback_sections(payload)
    mode = "anthropic" if anthropic_sections else "fallback"
    return {
        "data": payload,
        "artifacts": {"prompt": prompt},
        "summary": {"mode": mode, **sections},
    }
