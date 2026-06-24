from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd

from .backtest import run_backtest_from_paths
from .config import (
    COST_BPS_GRID,
    COST_GRID_METRICS_FILE,
    DATA_DIR_PROCESSED,
    DATA_DIR_RAW,
    DATA_DIR_UNIVERSES,
    DEFAULT_UNIVERSE_SLUG,
    END_DATE,
    EQUITY_CURVE_FILE,
    ETF_DUAL_MOMENTUM_LOOKBACK_MONTHS,
    ETF_DUAL_MOMENTUM_NAME,
    ETF_DUAL_MOMENTUM_RANKING_FILE,
    ETF_DUAL_MOMENTUM_RESULTS_DIR,
    ETF_DUAL_MOMENTUM_SIGNALS_FILE,
    ETF_DUAL_MOMENTUM_SITE_DIR,
    MOMENTUM_SCORES_FILE,
    MONTH_END_PRICES_FILE,
    PRICES_FILE,
    RAW_PRICES_FILE,
    RESULTS_DIR,
    RETURNS_FILE,
    SELECTION_FILE,
    SELECTION_HIGH_VOL_QUANTILE,
    SELECTION_MIN_MOMENTUM,
    SELECTION_SMA_WINDOW,
    SELECTION_TOP_N,
    SELECTION_VOL_WINDOW,
    SITE_BASE_COST_BPS,
    SITE_CHART_POINTS,
    SITE_DIR,
    SITE_INDEX_FILE,
    START_DATE,
    STRATEGY_RETURNS_FILE,
    UNIVERSE_COMPARISON_FILE,
    UNIVERSE_ORDER,
    UNIVERSE_RESULTS_DIR,
    UNIVERSES,
    UniverseSpec,
    WEIGHTS_FILE,
    YFINANCE_RETRY_COUNT,
    YFINANCE_RETRY_SLEEP_SECONDS,
)
from .data_clean import clean_prices
from .data_fetch import fetch_prices
from .etf_dual_momentum_report import build_etf_dual_momentum_site
from .evaluate import export_cost_grid_report
from .portfolio import build_and_save_weights
from .returns import compute_and_save_returns
from .selection import build_and_save_latest_selection
from .signals.momentum_12_1 import build_and_save_momentum_scores
from .strategies.etf_dual_momentum import run_strategy as run_etf_dual_momentum_strategy
from .web_report import build_site


class UniversePaths(TypedDict):
    raw_dir: Path
    processed_dir: Path
    results_dir: Path
    site_dir: Path


class UniverseSummary(TypedDict):
    universe_slug: str
    universe_name: str
    requested_tickers: int
    used_tickers: int
    dropped_tickers: int
    signal_date: str
    conservative_candidates: int
    strategy_candidates: int
    cost_bps: int
    total_return: float
    CAGR: float
    annual_vol: float
    Sharpe: float
    max_drawdown: float


def _universe_paths(slug: str) -> UniversePaths:
    if slug == DEFAULT_UNIVERSE_SLUG:
        return {
            "raw_dir": DATA_DIR_RAW,
            "processed_dir": DATA_DIR_PROCESSED,
            "results_dir": RESULTS_DIR,
            "site_dir": SITE_DIR,
        }

    return {
        "raw_dir": DATA_DIR_UNIVERSES / slug / "Raw",
        "processed_dir": DATA_DIR_UNIVERSES / slug / "Processed",
        "results_dir": UNIVERSE_RESULTS_DIR / slug,
        "site_dir": SITE_DIR / slug,
    }


def _metric_row(metrics: pd.DataFrame, cost_bps: int, universe_slug: str) -> pd.Series:
    matched: pd.DataFrame = metrics[metrics["cost_bps"] == cost_bps]
    if len(matched) != 1:
        raise ValueError(
            f"Expected one metrics row for universe_slug={universe_slug}, "
            f"cost_bps={cost_bps}; found {len(matched)}."
        )

    return matched.iloc[0]


def _signal_date(selection: pd.DataFrame, universe_slug: str) -> str:
    signal_dates: list[str] = sorted(str(value) for value in selection["signal_date"].unique())
    if len(signal_dates) != 1:
        raise ValueError(f"Expected one signal date for universe_slug={universe_slug}; found {signal_dates}.")

    return signal_dates[0]


def _build_summary(
    slug: str,
    name: str,
    requested_tickers: int,
    prices: pd.DataFrame,
    selection: pd.DataFrame,
    metrics: pd.DataFrame,
    cost_bps: int,
) -> UniverseSummary:
    base_metric: pd.Series = _metric_row(metrics, cost_bps, slug)
    used_tickers: int = len(prices.columns)
    return {
        "universe_slug": slug,
        "universe_name": name,
        "requested_tickers": requested_tickers,
        "used_tickers": used_tickers,
        "dropped_tickers": requested_tickers - used_tickers,
        "signal_date": _signal_date(selection, slug),
        "conservative_candidates": int(selection["conservative_entry_signal"].sum()),
        "strategy_candidates": int(selection["strategy_signal"].sum()),
        "cost_bps": int(cost_bps),
        "total_return": float(base_metric["total_return"]),
        "CAGR": float(base_metric["CAGR"]),
        "annual_vol": float(base_metric["annual_vol"]),
        "Sharpe": float(base_metric["Sharpe"]),
        "max_drawdown": float(base_metric["max_drawdown"]),
    }


def _run_universe(slug: str, spec: UniverseSpec, paths: UniversePaths) -> UniverseSummary:
    raw_prices_path: Path = paths["raw_dir"] / RAW_PRICES_FILE
    prices_path: Path = paths["processed_dir"] / PRICES_FILE
    returns_path: Path = paths["processed_dir"] / RETURNS_FILE
    momentum_scores_path: Path = paths["processed_dir"] / MOMENTUM_SCORES_FILE
    month_end_prices_path: Path = paths["processed_dir"] / MONTH_END_PRICES_FILE
    weights_path: Path = paths["processed_dir"] / WEIGHTS_FILE
    selection_path: Path = paths["results_dir"] / SELECTION_FILE

    tickers: tuple[str, ...] = spec["tickers"]
    print(f"\n=== Running universe: {spec['name']} ({slug}) ===")
    fetch_prices(
        tickers,
        START_DATE,
        END_DATE,
        raw_prices_path,
        YFINANCE_RETRY_COUNT,
        YFINANCE_RETRY_SLEEP_SECONDS,
    )
    prices: pd.DataFrame = clean_prices(raw_prices_path, prices_path, 0.95)
    compute_and_save_returns(prices_path, returns_path)
    build_and_save_momentum_scores(prices_path, momentum_scores_path, month_end_prices_path)
    build_and_save_weights(momentum_scores_path, returns_path, weights_path, SELECTION_TOP_N)

    for bps in COST_BPS_GRID:
        out_dir: Path = paths["results_dir"] / f"cost_{bps}bps"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_backtest_from_paths(weights_path, returns_path, bps, out_dir)

    metrics: pd.DataFrame = export_cost_grid_report(paths["results_dir"], COST_BPS_GRID, 0.0)
    selection: pd.DataFrame = build_and_save_latest_selection(
        momentum_scores_path,
        prices_path,
        returns_path,
        selection_path,
        SELECTION_TOP_N,
        SELECTION_MIN_MOMENTUM,
        SELECTION_SMA_WINDOW,
        SELECTION_VOL_WINDOW,
        SELECTION_HIGH_VOL_QUANTILE,
    )

    return _build_summary(
        slug,
        spec["name"],
        len(tickers),
        prices,
        selection,
        metrics,
        SITE_BASE_COST_BPS,
    )


def _save_comparison(summaries: list[UniverseSummary], out_path: Path) -> pd.DataFrame:
    if not summaries:
        raise ValueError("Cannot save universe comparison with no summaries.")

    comparison: pd.DataFrame = pd.DataFrame(summaries)
    comparison = comparison.sort_values(["Sharpe", "CAGR"], ascending=[False, False]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=False)
    print(f"Saved universe comparison: {out_path}")
    print(comparison.to_string(index=False))
    return comparison


def _build_all_sites(comparison_path: Path) -> None:
    for slug in UNIVERSE_ORDER:
        spec: UniverseSpec = UNIVERSES[slug]
        paths: UniversePaths = _universe_paths(slug)
        out_path: Path = build_site(
            paths["results_dir"],
            paths["site_dir"],
            SELECTION_FILE,
            COST_GRID_METRICS_FILE,
            EQUITY_CURVE_FILE,
            STRATEGY_RETURNS_FILE,
            SITE_INDEX_FILE,
            SITE_BASE_COST_BPS,
            SITE_CHART_POINTS,
            SELECTION_SMA_WINDOW,
            SELECTION_VOL_WINDOW,
            comparison_path,
            slug,
            spec["name"],
        )
        print(f"Saved local website: {out_path}")


def _build_etf_dual_momentum_site() -> None:
    out_path: Path = build_etf_dual_momentum_site(
        ETF_DUAL_MOMENTUM_RESULTS_DIR,
        ETF_DUAL_MOMENTUM_SITE_DIR,
        ETF_DUAL_MOMENTUM_SIGNALS_FILE,
        ETF_DUAL_MOMENTUM_RANKING_FILE,
        COST_GRID_METRICS_FILE,
        EQUITY_CURVE_FILE,
        STRATEGY_RETURNS_FILE,
        SITE_INDEX_FILE,
        SITE_BASE_COST_BPS,
        SITE_CHART_POINTS,
        ETF_DUAL_MOMENTUM_LOOKBACK_MONTHS,
        ETF_DUAL_MOMENTUM_NAME,
    )
    print(f"Saved ETF dual momentum website: {out_path}")


def main() -> None:
    summaries: list[UniverseSummary] = []
    for slug in UNIVERSE_ORDER:
        spec: UniverseSpec = UNIVERSES[slug]
        paths: UniversePaths = _universe_paths(slug)
        summaries.append(_run_universe(slug, spec, paths))

    comparison_path: Path = RESULTS_DIR / UNIVERSE_COMPARISON_FILE
    _save_comparison(summaries, comparison_path)
    _build_all_sites(comparison_path)
    run_etf_dual_momentum_strategy()
    _build_etf_dual_momentum_site()


if __name__ == "__main__":
    main()
