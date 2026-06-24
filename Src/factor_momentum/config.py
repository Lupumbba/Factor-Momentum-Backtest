from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict


class UniverseSpec(TypedDict):
    name: str
    tickers: tuple[str, ...]


def _validate_universe(slug: str, tickers: tuple[str, ...], expected_count: int) -> tuple[str, ...]:
    unique_tickers: set[str] = set(tickers)
    if len(tickers) != expected_count:
        raise ValueError(
            f"Universe '{slug}' expected {expected_count} tickers but received {len(tickers)}."
        )
    if len(unique_tickers) != len(tickers):
        duplicate_count: int = len(tickers) - len(unique_tickers)
        raise ValueError(f"Universe '{slug}' contains {duplicate_count} duplicate tickers.")

    return tickers


# ========= Project root =========
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ========= Folders =========
DATA_DIR: Final[Path] = PROJECT_ROOT / "Data"
DATA_DIR_RAW: Final[Path] = DATA_DIR / "Raw"
DATA_DIR_PROCESSED: Final[Path] = DATA_DIR / "Processed"
DATA_DIR_UNIVERSES: Final[Path] = DATA_DIR / "Universes"
DATA_DIR_STRATEGIES: Final[Path] = DATA_DIR / "Strategies"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "Results"
UNIVERSE_RESULTS_DIR: Final[Path] = RESULTS_DIR / "universes"
STRATEGY_RESULTS_DIR: Final[Path] = RESULTS_DIR / "strategies"

# ========= Universe definitions =========
AUTHOR_50_TICKERS: Final[tuple[str, ...]] = _validate_universe(
    "author_50",
    (
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "BRK-B",
        "WMT", "LLY", "JPM", "V", "XOM", "JNJ", "ORCL", "MA", "MU", "COST",
        "AMD", "ABBV", "PLTR", "HD", "BAC", "NFLX", "PG", "CVX", "KO", "GE",
        "CSCO", "LRCX", "CAT", "MS", "GS", "PM", "IBM", "WFC", "RTX", "MRK",
        "AMAT", "UNH", "AXP", "TMO", "MCD", "INTC", "CRM", "KLAC", "LIN", "TMUS",
    ),
    50,
)

LARGE_CAP_100_TICKERS: Final[tuple[str, ...]] = _validate_universe(
    "large_cap_100",
    tuple(ticker for ticker in AUTHOR_50_TICKERS if ticker != "PLTR")
    + (
        "ADBE", "ACN", "QCOM", "TXN", "NOW", "INTU", "AMGN", "PEP", "DIS", "NKE",
        "LOW", "SBUX", "BKNG", "BA", "HON", "UPS", "DE", "SPGI", "BLK", "C",
        "SCHW", "CB", "ADP", "GILD", "ABT", "ISRG", "DHR", "SYK", "MDT",
        "BSX", "PFE", "BMY", "LMT", "NOC", "ELV", "CI", "CVS", "SO", "DUK",
        "NEE", "AEP", "SLB", "EOG", "COP", "OXY", "MO", "CL", "MDLZ", "EL",
        "ADI", "ICE",
    ),
    100,
)

SECTOR_BALANCED_100_TICKERS: Final[tuple[str, ...]] = _validate_universe(
    "sector_balanced_100",
    (
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "IBM", "CRM", "ADBE", "ACN",
        "GOOGL", "GOOG", "META", "NFLX", "DIS", "TMUS", "VZ", "T", "CMCSA", "EA",
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "ORLY",
        "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL", "MDLZ", "EL",
        "LLY", "JNJ", "ABBV", "UNH", "MRK", "TMO", "ABT", "AMGN", "PFE", "ISRG",
        "JPM", "V", "MA", "BAC", "MS", "GS", "WFC", "AXP", "BLK", "SCHW",
        "GE", "CAT", "RTX", "HON", "UPS", "DE", "LMT", "NOC", "BA", "ADP",
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO", "KMI",
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DD", "PPG", "MLM", "VMC",
        "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "D", "PLD", "AMT", "EQIX",
    ),
    100,
)

DEFAULT_UNIVERSE_SLUG: Final[str] = "author_50"
UNIVERSE_ORDER: Final[tuple[str, ...]] = (
    "author_50",
    "large_cap_100",
    "sector_balanced_100",
)
UNIVERSES: Final[dict[str, UniverseSpec]] = {
    "author_50": {
        "name": "作者原始 50 只",
        "tickers": AUTHOR_50_TICKERS,
    },
    "large_cap_100": {
        "name": "扩展大盘 100 只",
        "tickers": LARGE_CAP_100_TICKERS,
    },
    "sector_balanced_100": {
        "name": "行业均衡 100 只",
        "tickers": SECTOR_BALANCED_100_TICKERS,
    },
}

# Kept for compatibility with modules or notebooks that import TICKERS directly.
TICKERS: Final[tuple[str, ...]] = UNIVERSES[DEFAULT_UNIVERSE_SLUG]["tickers"]

# ========= Date range =========
START_DATE: Final[str] = "2016-01-01"
END_DATE: Final[str | None] = None
YFINANCE_RETRY_COUNT: Final[int] = 3
YFINANCE_RETRY_SLEEP_SECONDS: Final[float] = 2.0

# ========= Common file paths =========
WEIGHTS_PATH: Final[Path] = DATA_DIR_PROCESSED / "weights_top10_eq.parquet"
RETURNS_PATH: Final[Path] = DATA_DIR_PROCESSED / "returns.parquet"
PRICES_PATH: Final[Path] = DATA_DIR_PROCESSED / "prices.parquet"
MOMENTUM_SCORES_PATH: Final[Path] = DATA_DIR_PROCESSED / "mom12_1_scores.parquet"

# ========= Robustness grid =========
COST_BPS_GRID: Final[list[int]] = [0, 10, 50]

# ========= Output filenames =========
RAW_PRICES_FILE: Final[str] = "prices_raw.parquet"
PRICES_FILE: Final[str] = "prices.parquet"
RETURNS_FILE: Final[str] = "returns.parquet"
MONTH_END_PRICES_FILE: Final[str] = "prices_month_end.parquet"
MOMENTUM_SCORES_FILE: Final[str] = "mom12_1_scores.parquet"
WEIGHTS_FILE: Final[str] = "weights_top10_eq.parquet"
STRATEGY_RETURNS_FILE: Final[str] = "strategy_returns.parquet"
EQUITY_CURVE_FILE: Final[str] = "equity_curve.parquet"
META_FILE: Final[str] = "run_meta.txt"
COST_GRID_METRICS_FILE: Final[str] = "cost_grid_metrics.csv"
UNIVERSE_COMPARISON_FILE: Final[str] = "universe_comparison.csv"

# ========= ETF dual momentum strategy =========
ETF_DUAL_MOMENTUM_SLUG: Final[str] = "etf_dual_momentum"
ETF_DUAL_MOMENTUM_NAME: Final[str] = "ETF 双动量轮动"
ETF_DUAL_MOMENTUM_TICKERS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
    "SHY",
)
ETF_DUAL_MOMENTUM_RISK_TICKERS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
)
ETF_DUAL_MOMENTUM_DEFENSIVE_TICKER: Final[str] = "SHY"
ETF_DUAL_MOMENTUM_LOOKBACK_MONTHS: Final[int] = 6
ETF_DUAL_MOMENTUM_SIGNALS_FILE: Final[str] = "dual_momentum_signals.csv"
ETF_DUAL_MOMENTUM_RANKING_FILE: Final[str] = "dual_momentum_latest_ranking.csv"
ETF_DUAL_MOMENTUM_WEIGHTS_FILE: Final[str] = "dual_momentum_weights.parquet"
ETF_DUAL_MOMENTUM_DATA_DIR: Final[Path] = DATA_DIR_STRATEGIES / ETF_DUAL_MOMENTUM_SLUG
ETF_DUAL_MOMENTUM_RESULTS_DIR: Final[Path] = STRATEGY_RESULTS_DIR / ETF_DUAL_MOMENTUM_SLUG
ETF_DUAL_MOMENTUM_SITE_DIR: Final[Path] = RESULTS_DIR / "site" / ETF_DUAL_MOMENTUM_SLUG

# ========= Low-volatility quality momentum strategy =========
LOW_VOL_QUALITY_MOMENTUM_SLUG: Final[str] = "low_vol_quality_momentum"
LOW_VOL_QUALITY_MOMENTUM_NAME: Final[str] = "低波动质量动量防守版"
LOW_VOL_QUALITY_UNIVERSE_SLUG: Final[str] = "sector_balanced_100"
LOW_VOL_QUALITY_TICKERS: Final[tuple[str, ...]] = SECTOR_BALANCED_100_TICKERS
LOW_VOL_QUALITY_TOP_N: Final[int] = 8
LOW_VOL_QUALITY_MIN_FULL_POSITIONS: Final[int] = 5
LOW_VOL_QUALITY_CASH_TICKER: Final[str] = "CASH"
LOW_VOL_QUALITY_MIN_MOMENTUM: Final[float] = 0.0
LOW_VOL_QUALITY_SMA_WINDOW: Final[int] = 200
LOW_VOL_QUALITY_VOL_WINDOW: Final[int] = 126
LOW_VOL_QUALITY_MAX_VOL_QUANTILE: Final[float] = 0.60
LOW_VOL_QUALITY_MIN_FACTOR_COUNT: Final[int] = 4
LOW_VOL_QUALITY_PASS_QUANTILE: Final[float] = 0.50
LOW_VOL_QUALITY_MIN_COVERAGE: Final[float] = 0.65
LOW_VOL_QUALITY_PRICE_WEIGHT: Final[float] = 0.60
LOW_VOL_QUALITY_QUALITY_WEIGHT: Final[float] = 0.40
LOW_VOL_QUALITY_WEIGHTS_FILE: Final[str] = "low_vol_quality_weights.parquet"
LOW_VOL_QUALITY_RETURNS_FILE: Final[str] = "returns_with_cash.parquet"
LOW_VOL_QUALITY_PRICE_SCORES_FILE: Final[str] = "price_defensive_scores.parquet"
LOW_VOL_QUALITY_SELECTION_FILE: Final[str] = "low_vol_quality_latest_selection.csv"
LOW_VOL_QUALITY_SNAPSHOT_FILE: Final[str] = "quality_snapshot.csv"
LOW_VOL_QUALITY_DATA_DIR: Final[Path] = DATA_DIR_STRATEGIES / LOW_VOL_QUALITY_MOMENTUM_SLUG
LOW_VOL_QUALITY_RESULTS_DIR: Final[Path] = STRATEGY_RESULTS_DIR / LOW_VOL_QUALITY_MOMENTUM_SLUG
LOW_VOL_QUALITY_SITE_DIR: Final[Path] = RESULTS_DIR / "site" / LOW_VOL_QUALITY_MOMENTUM_SLUG

# ========= Latest selection report =========
SELECTION_TOP_N: Final[int] = 10
SELECTION_MIN_MOMENTUM: Final[float] = 0.0
SELECTION_SMA_WINDOW: Final[int] = 200
SELECTION_VOL_WINDOW: Final[int] = 63
SELECTION_HIGH_VOL_QUANTILE: Final[float] = 0.75
SELECTION_FILE: Final[str] = "latest_selection.csv"

# ========= Local website =========
SITE_DIR: Final[Path] = RESULTS_DIR / "site"
SITE_INDEX_FILE: Final[str] = "index.html"
SITE_BASE_COST_BPS: Final[int] = 10
SITE_CHART_POINTS: Final[int] = 220
