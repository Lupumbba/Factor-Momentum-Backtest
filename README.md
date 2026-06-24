# Factor Momentum Backtest (12–1 Momentum, Long-Only)

## Source and Acknowledgement

This project is based on and extended from:
[Xuanyu6Chen/Factor-Momentum-Backtest](https://github.com/Xuanyu6Chen/Factor-Momentum-Backtest).

The original repository provides the 12–1 momentum backtest foundation. This version adds:
- automatic latest-stock selection,
- conservative entry filters,
- transaction-cost comparison output,
- multi-universe comparison,
- ETF dual momentum rotation,
- low-volatility quality momentum defensive selection,
- and a local beginner-friendly website for reading the results.

## 🎯 Overview
This project builds a small, reproducible “quant research” pipeline that tests a classic trading signal — **momentum** — on historical US stock data. Each month-end, it ranks stocks by 12–1 momentum (past 12 months return excluding the most recent month), forms a long-only portfolio, and holds it until the next rebalance. The backtest includes turnover-based transaction costs and outputs daily returns and an equity curve.

The goal is to learn (and demonstrate) how to:
- turn an idea into a precise trading rule,
- simulate it fairly on historical data (a backtest),
- and evaluate results using standard return + risk metrics.

The current strategy library follows the practical priority suggested in the research conversation:

| Role | Strategy | Current status |
| --- | --- | --- |
| Stable version | ETF dual momentum | Implemented as a separate ETF rotation dashboard |
| Aggressive version | US stock momentum | Implemented through 12–1 stock momentum and multi-universe comparison |
| Defensive version | Low-volatility / quality momentum | Implemented as a defensive stock dashboard with a current fundamental-quality overlay |

**Signal (12–1 momentum) meaning:** 
- Look back **12 months**
- Skip the most recent **1 month**
- Use the return from **t−12 months to t−1 month** as the momentum score

For each stock $i$ at rebalance date $t$,

$$
\text{Momentum}_{i,t} = \frac{P_{i,t-1}}{P_{i,t-12}} - 1
$$

where $P_{i,t-1}$ is the price one month before $t$ and $P_{i,t-12}$ is the price twelve months before $t$.

## 💡 Strategy Definition
### ETF dual momentum strategy
This strategy is the stable-version dashboard.

Asset pool:
- `SPY`
- `QQQ`
- `IWM`
- `TLT`
- `GLD`
- `SHY`

Monthly rule:
1. Compare the past 6-month return of `SPY`, `QQQ`, `IWM`, `TLT`, and `GLD`.
2. Select the strongest asset.
3. If the strongest asset has positive 6-month return, hold it at 100%.
4. If the strongest asset has non-positive 6-month return, switch to `SHY`.
5. Rebalance monthly and apply the same transaction-cost stress test grid.

The generated dashboard is:
```bash
Results/site/etf_dual_momentum/index.html
```

### Low-volatility quality momentum strategy
This strategy is the defensive-version dashboard.

Universe:
- `sector_balanced_100`

Monthly price rule:
1. Compute 12–1 momentum.
2. Keep stocks with positive momentum.
3. Keep stocks trading above their 200-day moving average.
4. Keep stocks whose recent 126-trading-day volatility is not in the high-volatility group.
5. Rank the remaining stocks by a blend of momentum strength and low volatility.
6. Select up to 8 stocks.
7. If fewer than 5 stocks pass the rule, keep the unused allocation in `CASH`.

Current quality overlay:
1. Pull a latest fundamental snapshot from `yfinance`.
2. Score quality using profitability, cash-flow yield, and debt-pressure fields.
3. Keep the top half of quality scores as quality-pass stocks.
4. The latest defensive candidate list must pass both the price rule and the quality rule.

Important limitation:
- The historical backtest does **not** use today's fundamental snapshot for past dates.
- The backtest validates the price-based defensive rule.
- The latest dashboard then applies current fundamental quality to the current month selection.

The generated dashboard is:
```bash
Results/site/low_vol_quality_momentum/index.html
```

### Stock momentum strategy
### Universe
- The default universe is the original **50 large-cap US tickers** from the source project.
- The pipeline can also compare larger or more sector-balanced universes.
- Each universe is kept fixed during the backtest to make the experiment reproducible.

### Built-in universes

| Universe | Meaning | Purpose |
| --- | --- | --- |
| `author_50` | Original 50-stock universe | Baseline comparison |
| `large_cap_100` | Expanded 100-stock large-cap universe | Tests whether a wider opportunity set improves the strategy |
| `sector_balanced_100` | 100-stock universe spread across major sectors | Tests whether reducing sector concentration improves robustness |

### Rebalancing schedule
- **Monthly** rebalancing using month-end dates.

### Signal
- For each ticker at each month-end: compute the **12–1 momentum** score.

### Portfolio construction (long-only)
At each rebalance:
1. Rank the 50 tickers by their momentum score (highest to lowest).
2. Select the **top 10** tickers (top 20%).
3. Allocate weights **equally** across the selected tickers.

So if 10 tickers are selected, each gets weight:
- 10% per ticker
- weights sum to 1.0 (100% invested)

### Holding period
- Hold these weights until the next month-end rebalance.

## 📂 Project Pipeline
**Produce the core market tables (Prices and Returns):**

1) Download prices for tickers — `Src/factor_momentum/data_fetch.py`  
2) Clean prices into a daily panel — `Src/factor_momentum/data_clean.py`  
3) Compute daily returns — `Src/factor_momentum/returns.py`  

**Compute the signal from prices (using only past data):**

4) Compute 12–1 momentum scores at each rebalance date — `Src/factor_momentum/signals/momentum_12_1.py`  

**Convert signal into a tradeable portfolio:**

5) Convert momentum scores to long-only, normalized portfolio weights (rebalance-date weights) — `Src/factor_momentum/portfolio.py`  

**Run the backtest (no look-ahead):**

6) Align rebalance-date weights to the daily trading calendar and apply starting the next trading day — `Src/factor_momentum/backtest.py`  
7) Compute daily **gross** portfolio returns from daily weights and asset returns — `Src/factor_momentum/backtest.py`  

**Apply trading frictions and report results:**

8) Compute turnover and subtract transaction costs to obtain **net** returns — `Src/factor_momentum/costs.py` + `Src/factor_momentum/backtest.py`  
9) Evaluate performance (equity curve, metrics) and save outputs — `Src/factor_momentum/evaluate.py`
10) Generate latest stock-selection output — `Src/factor_momentum/selection.py`
11) Compare all configured universes — `Src/factor_momentum/pipeline.py`
12) Generate the local website — `Src/factor_momentum/web_report.py`
13) Run ETF dual momentum and generate its dashboard — `Src/factor_momentum/strategies/etf_dual_momentum.py` + `Src/factor_momentum/etf_dual_momentum_report.py`
14) Run low-volatility quality momentum and generate its dashboard — `Src/factor_momentum/strategies/low_vol_quality_momentum.py` + `Src/factor_momentum/low_vol_quality_report.py`

## 🚀 Repo Structure
- `Src/factor_momentum/` — core pipeline and strategy code
- `Data/Raw/` — raw pulled price data (ignored; regenerated by running the pipeline)
- `Data/Processed/` — cleaned returns/signal/weights (ignored; regenerated by running the pipeline)
- `Data/Universes/` — per-universe data for non-default universes (ignored; regenerated)
- `Results/` — backtest outputs (ignored by default)
- `Results/universe_comparison.csv` — comparison table across configured universes
- `Results/universes/` — per-universe backtest outputs for non-default universes
- `Results/strategies/etf_dual_momentum/` — ETF dual momentum summary CSV outputs
- `Results/strategies/low_vol_quality_momentum/` — defensive low-volatility quality momentum summary CSV outputs
- `Results/assets/` — lightweight, committed sample artifacts (plots / summary tables)
- `Results/site/index.html` — generated local website with the latest readable stock-selection dashboard
- `Results/site/<universe>/index.html` — generated local website for each non-default universe
- `Results/site/etf_dual_momentum/index.html` — generated ETF dual momentum dashboard
- `Results/site/low_vol_quality_momentum/index.html` — generated low-volatility quality momentum dashboard

## 📊 Results & Comparison

![Equity Curve (Cost Grid)](Results/assets/equity_curve_cost_grid.png)

![Drawdown (Cost Grid)](Results/assets/drawdown_cost_grid.png)

## 💻 How to Run

### 1) Clone the repo and open the project
```bash
git clone <REPO_URL>
cd Factor-Momentum-Backtest
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Configure the experiment
Edit:
```bash
Src/factor_momentum/config.py
```
Common things to change:
- UNIVERSES / built-in ticker lists (stock universes)
- COST_BPS_GRID (e.g., [0, 10, 50])
- RESULTS_DIR (where results are saved)

### 5) Run the full pipeline (end-to-end)
```bash
PYTHONPATH=Src python -m factor_momentum.pipeline
```

After the run finishes, open:
```bash
Results/site/index.html
```

This local page shows the latest momentum ranking, conservative entry candidates,
risk-filtered names, equity curve, drawdown, transaction-cost sensitivity, and
the multi-universe comparison table.

The full pipeline now runs:
- `author_50`
- `large_cap_100`
- `sector_balanced_100`
- `etf_dual_momentum`
- `low_vol_quality_momentum`

Open the default page:
```bash
Results/site/index.html
```

Open the expanded-universe pages:
```bash
Results/site/large_cap_100/index.html
Results/site/sector_balanced_100/index.html
```

Open the ETF dual momentum page:
```bash
Results/site/etf_dual_momentum/index.html
```

Open the low-volatility quality momentum page:
```bash
Results/site/low_vol_quality_momentum/index.html
```

Read the direct comparison table:
```bash
Results/universe_comparison.csv
```

## Beginner Usage Guide

### What to open first

Open the generated website:
```bash
Results/site/index.html
```

The page is designed as the first place to read the result. You do not need to open
Parquet files or Python tables manually.

### How to read the website

- **优先研究**: stocks that passed the conservative filters. These are not automatic buy orders; they are the first names to research further.
- **动量强但风险高**: stocks with strong momentum but high recent volatility or other risk flags. Avoid chasing them blindly.
- **完整股票排名**: the full ranked universe, including tickers that are only worth observing or temporarily avoiding.
- **策略净值曲线**: historical portfolio value of the original top-10 momentum strategy.
- **历史回撤**: how much the strategy fell from previous highs during bad periods.
- **交易成本压力测试**: checks whether the strategy still looks acceptable after adding trading costs.
- **股票池对比**: compares the original 50-stock universe with the expanded and sector-balanced universes.
- **ETF 双动量看板**: shows the current ETF target, recent monthly rotations, equity curve, drawdown, and transaction-cost sensitivity.
- **低波动质量动量看板**: shows the current defensive stock candidates, quality coverage, cash reserve, equity curve, drawdown, and transaction-cost sensitivity.

### Automatic stock-selection rules

The latest-selection report uses these rules:

1. Rank stocks by 12–1 momentum.
2. Keep the top 10 stocks with positive momentum as the original strategy signal.
3. Mark a stock as a conservative entry candidate only when it also:
   - trades above its 200-day moving average,
   - is not in the highest-volatility group based on recent 63 trading days.
4. Split the conservative candidates into equal weights for reference.

### Defensive stock-selection rules

The low-volatility quality momentum dashboard uses these rules:

1. Start from the sector-balanced 100-stock universe.
2. Keep stocks with positive 12–1 momentum.
3. Keep stocks above the 200-day moving average.
4. Keep stocks outside the high-volatility group based on recent 126 trading days.
5. Rank by momentum plus low volatility.
6. Overlay the latest fundamental quality score.
7. Select up to 8 defensive candidates and assign equal reference weights.
8. If too few stocks pass, reserve the remaining weight as cash.

### Important warning

This project is a research and learning tool. The website helps narrow down a watchlist,
but it is not financial advice and does not guarantee future returns. Before real trading,
check company fundamentals, market conditions, position size, stop-loss rules, and your own risk tolerance.
