% 00_project-requirements.md

# Gold Research Report Automation — Project Requirements

## Project Identity
- **Name**: quant-gold-report
- **Tagline**: An end-to-end automated gold research report pipeline combining quantitative analysis, macro factor modeling, and LLM-generated narrative — built as a portfolio project for MFE graduate applications.
- **Author**: Jerome Yuxuan Zhang
- **Stack**: Python 3.11+, free data sources only, AI-assisted workflow

---

## Purpose
This project has three simultaneous goals:
1. Produce a genuinely useful, demo-able automated research report on gold
2. Demonstrate the intersection of financial domain knowledge + quantitative engineering + AI workflow
3. Serve as a teachable codebase where every mathematical choice is documented and explainable

Every module must satisfy two tests:
- **It runs**: produces a real output from real data
- **It can be explained**: the math and economic logic behind it is documented in plain language

---

## Repo Structure
```
quant-gold-report/
├── config/
│   └── settings.yaml          # asset, date range, window params — swap asset here
├── data/
│   ├── raw/                   # never modified after download
│   └── processed/             # cleaned, log-return transformed
├── src/
│   ├── data_layer.py          # data pulling and cleaning
│   ├── trend_momentum.py      # MA, momentum signals
│   ├── volatility_regime.py   # rolling vol, HMM regime detection
│   ├── macro_factor.py        # correlation, OLS regression
│   ├── backtest.py            # walk-forward, Sharpe, drawdown
│   ├── llm_narrative.py       # prompt construction, API call, hallucination guard
│   └── report_assembly.py     # chart + table + narrative → PDF
├── notebooks/
│   └── exploration.ipynb      # scratchpad, not production
├── output/
│   └── reports/               # generated PDFs land here
├── docs/
│   └── math_notes.md          # mathematical derivations for every module
├── main.py                    # single entry point: python main.py → PDF
├── requirements.txt
└── README.md
```

---

## Module 1: Data Layer

### What to build
- Pull the following series automatically using free sources:
  - Gold spot price (yfinance: `GC=F` or `GLD`)
  - DXY Dollar Index (`DX-Y.NYB`)
  - US 10Y Treasury yield (FRED: `DGS10`)
  - US CPI YoY (FRED: `CPIAUCSL`)
  - US 10Y Breakeven Inflation / Real Yield (FRED: `DFII10`)
  - Gold ETF: GLD (as flow proxy)
  - Optional: Shanghai gold price via akshare for CNY premium analysis
- Store raw data as parquet, never overwrite
- Produce a cleaned dataframe aligned on business days, forward-filled for macro series

### What to document (in docs/math_notes.md)
- Why these variables? Economic mechanism for each:
  - Real yield channel (Fisher equation: r = i - π)
  - Dollar denomination effect
  - Inflation hedge hypothesis
- What is stationarity and why raw prices are non-stationary
- Formula and intuition for log returns: r_t = ln(P_t / P_{t-1})
- Why log returns rather than simple returns (variance stabilization, time additivity)

---

## Module 2: Trend & Momentum

### What to build
- Simple Moving Average (SMA): 20, 50, 200 day
- Exponential Moving Average (EMA): 20, 50 day
- Price momentum: r_{t,k} = P_t / P_{t-k} - 1 for k = 21, 63, 126 days
- A simple dual-MA crossover signal (20/200) as the baseline trading signal
- Output: dataframe of signals + matplotlib chart

### What to document
- SMA formula: SMA_t = (1/n) Σ P_{t-i}
- Why longer window = more smoothing (low-pass filter intuition)
- EMA formula and the decay factor λ
- Momentum definition, behavioral explanation (underreaction), risk explanation (compensation for crash risk)
- Difference between price momentum and risk-adjusted momentum

---

## Module 3: Volatility Regime Detection

### What to build
- Rolling 21-day realized volatility: σ_t = std(r_{t-20}...r_t) × √252
- Volatility percentile rank over trailing 252 days (high/medium/low regime label)
- Optional stretch goal: 2-state Gaussian HMM on log returns using hmmlearn
- Output: regime label series + vol chart with regime shading

### What to document
- Rolling vol formula and its limitations (backward-looking, no forward info)
- HMM intuition: hidden states (regimes), transition matrix, emission probabilities
- Why HMM is appropriate for financial regime detection
- Difference between realized vol and implied vol (VIX), why it matters
- Which parts of this module are genuinely sophisticated vs standard

---

## Module 4: Macro Factor Analysis

### What to build
- Rolling 63-day Pearson correlation: gold returns vs real yield changes, DXY changes
- Rolling 63-day Spearman correlation as robustness check
- OLS regression: gold_return = α + β₁×ΔDXY + β₂×Δreal_yield + ε
- Output: rolling correlation chart, regression summary table

### What to document
- Pearson correlation formula and assumptions (linearity, normality)
- Why Spearman rank correlation is more robust for financial data
- OLS formula derivation: β = (XᵀX)⁻¹Xᵀy
- What β means economically (e.g. β on DXY = sensitivity of gold to dollar moves)
- Multicollinearity: why DXY and real yields are correlated, what this does to β estimates
- Rolling correlation: why more informative than static (regime-dependence of relationships)

---

## Module 5: Walk-Forward Backtest

### What to build
- Apply dual-MA crossover signal from Module 2
- Walk-forward validation: expanding window, refit signal parameters every 252 days
- Compute for each out-of-sample period:
  - Annualized return
  - Sharpe Ratio: SR = (R_p - R_f) / σ_p × √252
  - Maximum Drawdown: MDD = max(1 - V_t / max_{s≤t} V_s)
  - Hit rate (% of months strategy is positive)
- Output: equity curve chart, performance summary table

### What to document
- In-sample vs out-of-sample: why in-sample always looks better
- Walk-forward definition: expanding vs rolling window, which is more conservative
- Sharpe Ratio formula, assumptions, limitations (assumes normality, penalizes upside vol)
- Maximum Drawdown formula and why it matters beyond Sharpe
- Overfitting: degrees of freedom argument, multiple testing problem
- Data snooping bias: must disclose in report limitations section

---

## Module 6: LLM Narrative Layer

### What to build
- A prompt constructor that:
  - Takes actual numbers from modules 2–5 as inputs
  - Structures them into a data-grounded prompt
  - Requests specific report sections (Executive Summary, Macro Analysis, Signal Review, Limitations)
- A hallucination guard: LLM is instructed to only describe what numbers show, never extrapolate
- API call to Claude claude-sonnet-4-20250514 via Anthropic API
- Output: structured text blocks per report section

### Prompt design principles to document
- Why injecting actual numbers reduces hallucination (token prediction vs factual retrieval)
- What prompt engineering is doing: constraining the conditional distribution P(output|input)
- Why the limitations section must be written by the human, not the LLM

---

## Module 7: Report Assembly

### What to build
- Combine: charts (PNG) + tables (formatted) + LLM narrative (text) into a single PDF
- Recommended library: WeasyPrint (HTML→PDF) or ReportLab
- Report sections in order:
  1. Cover Page (asset, date range, generated timestamp)
  2. Executive Summary (LLM, 1 paragraph, data-grounded)
  3. Market Overview (price chart, vol chart)
  4. Macro Factor Analysis (correlation charts, regression table)
  5. Quantitative Signal Review (equity curve, performance table)
  6. Risk & Limitations (human-written, honest)
  7. Appendix (data sources, methodology notes)
- Single entry point: `python main.py` → produces dated PDF in output/reports/

---

## Demo Strategy

### 3-minute live demo structure
1. Run `python main.py` live (or show pre-run output if time-sensitive)
2. Show the PDF opening: cover page → executive summary
3. Point to one chart and explain the math behind it (rolling correlation or equity curve)
4. Show the limitations section: explain why it exists and what data snooping bias means
5. Show the config file: explain how swapping the asset ticker generates a new report

### For a non-technical finance professor
- Lead with the PDF output and the narrative quality
- Emphasize the macro factor logic (real yield channel, dollar effect)
- Show the limitations section as evidence of research integrity

### For a quant-oriented professor
- Lead with the walk-forward backtest methodology
- Explain the difference between expanding and rolling window
- Discuss the HMM regime detection if implemented
- Be honest about what a dual-MA crossover signal actually is (simple, not alpha)

---

## Honest Limitations to Include in Report

- This is a single-asset, single-strategy backtest — no diversification
- Dual-MA crossover is a well-known, widely-arbitraged signal — alpha is unlikely to persist
- Data snooping: signal parameters were chosen with knowledge of historical data
- No transaction costs, no slippage modeled
- LLM narrative is generated from structured inputs — it describes, it does not predict
- Free data sources may have gaps or errors not fully corrected

---

## README Requirements

The README must contain:
1. One-paragraph plain-language description of what this project does and why
2. Architecture diagram (simple ASCII or mermaid)
3. Quickstart: `pip install -r requirements.txt` → `python main.py`
4. Section: "What I learned" — written in first person, explains the math behind each module in 2–3 sentences each
5. Sample output: embed a screenshot of the generated PDF cover and one chart
6. Honest limitations section mirroring the report

---

## Success Criteria

This project is complete when:
- [ ] `python main.py` runs end-to-end without errors and produces a PDF
- [ ] Every module has corresponding math documentation in docs/math_notes.md
- [ ] The README "What I learned" section can be read aloud in a 5-minute interview explanation
- [ ] The limitations section is honest enough that a professor cannot accuse you of overclaiming
- [ ] The config file works: swapping GLD for an oil ETF produces a different valid report
