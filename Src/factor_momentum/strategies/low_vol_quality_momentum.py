from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from ..backtest import run_backtest_from_paths
from ..config import (
    COST_BPS_GRID,
    END_DATE,
    LOW_VOL_QUALITY_CASH_TICKER,
    LOW_VOL_QUALITY_DATA_DIR,
    LOW_VOL_QUALITY_MAX_VOL_QUANTILE,
    LOW_VOL_QUALITY_MIN_COVERAGE,
    LOW_VOL_QUALITY_MIN_FACTOR_COUNT,
    LOW_VOL_QUALITY_MIN_FULL_POSITIONS,
    LOW_VOL_QUALITY_MIN_MOMENTUM,
    LOW_VOL_QUALITY_PASS_QUANTILE,
    LOW_VOL_QUALITY_PRICE_SCORES_FILE,
    LOW_VOL_QUALITY_PRICE_WEIGHT,
    LOW_VOL_QUALITY_QUALITY_WEIGHT,
    LOW_VOL_QUALITY_RESULTS_DIR,
    LOW_VOL_QUALITY_RETURNS_FILE,
    LOW_VOL_QUALITY_SELECTION_FILE,
    LOW_VOL_QUALITY_SMA_WINDOW,
    LOW_VOL_QUALITY_SNAPSHOT_FILE,
    LOW_VOL_QUALITY_TICKERS,
    LOW_VOL_QUALITY_TOP_N,
    LOW_VOL_QUALITY_VOL_WINDOW,
    LOW_VOL_QUALITY_WEIGHTS_FILE,
    MOMENTUM_SCORES_FILE,
    MONTH_END_PRICES_FILE,
    PRICES_FILE,
    RAW_PRICES_FILE,
    RETURNS_FILE,
    START_DATE,
    YFINANCE_RETRY_COUNT,
    YFINANCE_RETRY_SLEEP_SECONDS,
)
from ..data_clean import clean_prices
from ..data_fetch import fetch_prices
from ..evaluate import export_cost_grid_report
from ..fundamentals import fetch_and_save_quality_snapshot
from ..returns import compute_and_save_returns
from ..signals.momentum_12_1 import build_and_save_momentum_scores


class LowVolQualityPaths(TypedDict):
    raw_prices_path: Path
    prices_path: Path
    stock_returns_path: Path
    strategy_returns_path: Path
    momentum_scores_path: Path
    month_end_prices_path: Path
    price_scores_path: Path
    weights_path: Path
    quality_snapshot_path: Path
    selection_path: Path
    results_dir: Path


class WeightAllocation(TypedDict):
    stock_weight: float
    cash_weight: float


def strategy_paths(data_dir: Path, results_dir: Path) -> LowVolQualityPaths:
    processed_dir: Path = data_dir / "Processed"
    raw_dir: Path = data_dir / "Raw"
    return {
        "raw_prices_path": raw_dir / RAW_PRICES_FILE,
        "prices_path": processed_dir / PRICES_FILE,
        "stock_returns_path": processed_dir / RETURNS_FILE,
        "strategy_returns_path": processed_dir / LOW_VOL_QUALITY_RETURNS_FILE,
        "momentum_scores_path": processed_dir / MOMENTUM_SCORES_FILE,
        "month_end_prices_path": processed_dir / MONTH_END_PRICES_FILE,
        "price_scores_path": processed_dir / LOW_VOL_QUALITY_PRICE_SCORES_FILE,
        "weights_path": processed_dir / LOW_VOL_QUALITY_WEIGHTS_FILE,
        "quality_snapshot_path": results_dir / LOW_VOL_QUALITY_SNAPSHOT_FILE,
        "selection_path": results_dir / LOW_VOL_QUALITY_SELECTION_FILE,
        "results_dir": results_dir,
    }


def _validate_columns(frame: pd.DataFrame, tickers: tuple[str, ...], label: str) -> None:
    missing: list[str] = sorted(set(tickers).difference(set(frame.columns)))
    if missing:
        raise ValueError(
            f"{label} is missing required tickers; missing={missing}; "
            f"available_columns={list(frame.columns)}."
        )


def _prepare_indexed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared: pd.DataFrame = frame.copy()
    prepared.index = pd.to_datetime(prepared.index)
    return prepared.sort_index()


def _rank_pct(values: pd.Series, ascending: bool) -> pd.Series:
    numeric_values: pd.Series = pd.to_numeric(values, errors="coerce")
    return numeric_values.rank(pct=True, ascending=ascending)


def _score_records_for_date(
    signal_date: pd.Timestamp,
    tickers: tuple[str, ...],
    prices: pd.DataFrame,
    moving_average: pd.DataFrame,
    annualized_volatility: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    min_momentum: float,
    max_vol_quantile: float,
    price_weight: float,
    low_vol_weight: float,
) -> list[dict[str, object]]:
    momentum: pd.Series = pd.to_numeric(momentum_scores.loc[signal_date, list(tickers)], errors="coerce")
    latest_prices: pd.Series = pd.to_numeric(prices.loc[signal_date, list(tickers)], errors="coerce")
    sma: pd.Series = pd.to_numeric(moving_average.loc[signal_date, list(tickers)], errors="coerce")
    vol: pd.Series = pd.to_numeric(annualized_volatility.loc[signal_date, list(tickers)], errors="coerce")
    valid_vol: pd.Series = vol.dropna()
    vol_threshold: float = float(valid_vol.quantile(max_vol_quantile)) if not valid_vol.empty else float("nan")

    momentum_rank: pd.Series = _rank_pct(momentum, True)
    low_vol_rank: pd.Series = _rank_pct(vol, False)
    composite_score: pd.Series = price_weight * momentum_rank + low_vol_weight * low_vol_rank
    price_rank: pd.Series = composite_score.rank(ascending=False, method="first")

    records: list[dict[str, object]] = []
    for ticker in tickers:
        ticker_momentum: float = float(momentum.loc[ticker])
        ticker_price: float = float(latest_prices.loc[ticker])
        ticker_sma: float = float(sma.loc[ticker])
        ticker_vol: float = float(vol.loc[ticker])
        above_sma: bool = bool(np.isfinite(ticker_price) and np.isfinite(ticker_sma) and ticker_price > ticker_sma)
        volatility_ok: bool = bool(
            np.isfinite(ticker_vol)
            and np.isfinite(vol_threshold)
            and ticker_vol <= vol_threshold
        )
        positive_momentum: bool = bool(np.isfinite(ticker_momentum) and ticker_momentum > min_momentum)
        price_eligible: bool = positive_momentum and above_sma and volatility_ok
        records.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "ticker": ticker,
                "price_rank": float(price_rank.loc[ticker]),
                "momentum_12_1": ticker_momentum,
                "momentum_rank_pct": float(momentum_rank.loc[ticker]),
                "latest_price": ticker_price,
                "sma": ticker_sma,
                "annualized_volatility": ticker_vol,
                "volatility_threshold": vol_threshold,
                "low_vol_rank_pct": float(low_vol_rank.loc[ticker]),
                "price_composite_score": float(composite_score.loc[ticker]),
                "positive_momentum": positive_momentum,
                "above_sma": above_sma,
                "volatility_ok": volatility_ok,
                "price_eligible": price_eligible,
            }
        )
    return records


def build_price_defensive_scores(
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    tickers: tuple[str, ...],
    sma_window: int,
    vol_window: int,
    min_momentum: float,
    max_vol_quantile: float,
    price_weight: float,
    low_vol_weight: float,
) -> pd.DataFrame:
    if sma_window <= 0:
        raise ValueError(f"sma_window must be positive. Received: {sma_window}.")
    if vol_window <= 0:
        raise ValueError(f"vol_window must be positive. Received: {vol_window}.")
    if max_vol_quantile <= 0.0 or max_vol_quantile >= 1.0:
        raise ValueError(f"max_vol_quantile must be between 0 and 1. Received: {max_vol_quantile}.")
    if not np.isclose(price_weight + low_vol_weight, 1.0):
        raise ValueError(
            "price_weight and low_vol_weight must sum to 1.0; "
            f"price_weight={price_weight}; low_vol_weight={low_vol_weight}."
        )

    prepared_prices: pd.DataFrame = _prepare_indexed_frame(prices)
    prepared_returns: pd.DataFrame = _prepare_indexed_frame(daily_returns)
    prepared_momentum: pd.DataFrame = _prepare_indexed_frame(momentum_scores)
    _validate_columns(prepared_prices, tickers, "prices")
    _validate_columns(prepared_returns, tickers, "daily returns")
    _validate_columns(prepared_momentum, tickers, "momentum scores")

    if len(prepared_prices) < sma_window:
        raise ValueError(f"Not enough price rows for sma_window={sma_window}; rows={len(prepared_prices)}.")
    if len(prepared_returns) < vol_window:
        raise ValueError(f"Not enough return rows for vol_window={vol_window}; rows={len(prepared_returns)}.")

    moving_average: pd.DataFrame = prepared_prices.loc[:, list(tickers)].rolling(sma_window).mean()
    annualized_volatility: pd.DataFrame = (
        prepared_returns.loc[:, list(tickers)].rolling(vol_window).std(ddof=1) * np.sqrt(252.0)
    )
    signal_dates: pd.DatetimeIndex = pd.DatetimeIndex(
        prepared_momentum.dropna(how="all").index.intersection(prepared_prices.index)
    )
    if signal_dates.empty:
        raise ValueError("No overlapping signal dates between momentum scores and prices.")

    all_records: list[dict[str, object]] = []
    for signal_date in signal_dates:
        all_records.extend(
            _score_records_for_date(
                pd.Timestamp(signal_date),
                tickers,
                prepared_prices,
                moving_average,
                annualized_volatility,
                prepared_momentum,
                min_momentum,
                max_vol_quantile,
                price_weight,
                low_vol_weight,
            )
        )

    price_scores: pd.DataFrame = pd.DataFrame(all_records)
    return price_scores.sort_values(["signal_date", "price_rank"]).reset_index(drop=True)


def _allocation_for_selected_count(
    selected_count: int,
    min_full_positions: int,
) -> WeightAllocation:
    if selected_count < 0:
        raise ValueError(f"selected_count cannot be negative. Received: {selected_count}.")
    if min_full_positions <= 0:
        raise ValueError(f"min_full_positions must be positive. Received: {min_full_positions}.")
    if selected_count == 0:
        return {"stock_weight": 0.0, "cash_weight": 1.0}
    if selected_count < min_full_positions:
        stock_weight: float = 1.0 / min_full_positions
        return {"stock_weight": stock_weight, "cash_weight": 1.0 - selected_count * stock_weight}

    return {"stock_weight": 1.0 / selected_count, "cash_weight": 0.0}


def add_cash_returns(
    daily_returns: pd.DataFrame,
    cash_ticker: str,
) -> pd.DataFrame:
    if cash_ticker in set(daily_returns.columns):
        raise ValueError(f"cash_ticker already exists in daily returns: {cash_ticker}.")

    strategy_returns: pd.DataFrame = daily_returns.copy()
    strategy_returns[cash_ticker] = 0.0
    return strategy_returns


def build_defensive_weights(
    price_scores: pd.DataFrame,
    daily_returns: pd.DataFrame,
    top_n: int,
    min_full_positions: int,
    cash_ticker: str,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError(f"top_n must be positive. Received: {top_n}.")
    if cash_ticker not in set(daily_returns.columns):
        raise ValueError(f"cash_ticker={cash_ticker} is missing from daily returns columns.")

    tickers: tuple[str, ...] = tuple(
        str(value)
        for value in price_scores["ticker"].drop_duplicates().tolist()
        if str(value) != cash_ticker
    )
    weight_columns: list[str] = list(tickers) + [cash_ticker]
    signal_dates: list[str] = sorted(str(value) for value in price_scores["signal_date"].unique())
    records: list[dict[str, float]] = []
    index_values: list[pd.Timestamp] = []

    for signal_date in signal_dates:
        signal_scores: pd.DataFrame = price_scores[price_scores["signal_date"] == signal_date]
        eligible: pd.DataFrame = signal_scores[
            signal_scores["price_eligible"].astype(bool)
            & signal_scores["price_composite_score"].notna()
        ].sort_values("price_composite_score", ascending=False)
        selected: pd.DataFrame = eligible.head(top_n)
        allocation: WeightAllocation = _allocation_for_selected_count(len(selected), min_full_positions)
        row: dict[str, float] = {column: 0.0 for column in weight_columns}
        for ticker in selected["ticker"]:
            row[str(ticker)] = allocation["stock_weight"]
        row[cash_ticker] = allocation["cash_weight"]
        records.append(row)
        index_values.append(pd.Timestamp(signal_date))

    weights: pd.DataFrame = pd.DataFrame(records, index=pd.DatetimeIndex(index_values), columns=weight_columns)
    return weights.sort_index()


def _reason_for_latest_row(row: pd.Series) -> str:
    if bool(row["defensive_entry_signal"]):
        return "defensive_entry_signal"

    reasons: list[str] = []
    if not bool(row["quality_data_available"]):
        reasons.append("quality_data_missing")
    elif not bool(row["quality_pass"]):
        reasons.append("quality_score_below_threshold")
    if not bool(row["positive_momentum"]):
        reasons.append("nonpositive_momentum")
    if not bool(row["above_sma"]):
        reasons.append("below_200d_sma")
    if not bool(row["volatility_ok"]):
        reasons.append("above_defensive_volatility_limit")
    if not reasons:
        reasons.append("outside_top_candidates")
    return ";".join(reasons)


def _status_for_latest_row(row: pd.Series) -> str:
    if bool(row["defensive_entry_signal"]):
        return "优先研究"
    if bool(row["price_eligible"]) and not bool(row["quality_pass"]):
        return "价格通过但质量不足"
    if bool(row["quality_pass"]) and not bool(row["price_eligible"]):
        return "质量可以但价格不稳"
    return "暂不考虑"


def build_latest_low_vol_quality_selection(
    price_scores: pd.DataFrame,
    quality_snapshot: pd.DataFrame,
    top_n: int,
    min_full_positions: int,
    price_weight: float,
    quality_weight: float,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError(f"top_n must be positive. Received: {top_n}.")
    if not np.isclose(price_weight + quality_weight, 1.0):
        raise ValueError(
            "price_weight and quality_weight must sum to 1.0; "
            f"price_weight={price_weight}; quality_weight={quality_weight}."
        )

    latest_signal_date: str = str(pd.to_datetime(price_scores["signal_date"]).max().date().isoformat())
    latest_scores: pd.DataFrame = price_scores[price_scores["signal_date"] == latest_signal_date].copy()
    required_quality_columns: set[str] = {
        "ticker",
        "sector",
        "quality_score",
        "quality_pass",
        "quality_data_available",
        "quality_factor_count",
        "quality_threshold",
        "return_on_equity",
        "profit_margin",
        "free_cash_flow_yield",
        "debt_to_market_cap",
        "missing_quality_fields",
    }
    missing_quality_columns: list[str] = sorted(required_quality_columns.difference(set(quality_snapshot.columns)))
    if missing_quality_columns:
        raise ValueError(f"Quality snapshot is missing columns: {missing_quality_columns}.")

    merged: pd.DataFrame = latest_scores.merge(quality_snapshot, on="ticker", how="left", validate="one_to_one")
    missing_quality_rows: list[str] = sorted(
        str(ticker)
        for ticker in merged.loc[merged["quality_score"].isna(), "ticker"].tolist()
    )
    if missing_quality_rows:
        raise ValueError(f"Latest selection is missing quality rows for tickers: {missing_quality_rows}.")

    merged["final_score"] = (
        price_weight * pd.to_numeric(merged["price_composite_score"], errors="coerce")
        + quality_weight * pd.to_numeric(merged["quality_score"], errors="coerce")
    )
    merged["quality_entry_eligible"] = merged["price_eligible"].astype(bool) & merged["quality_pass"].astype(bool)

    eligible: pd.DataFrame = merged[
        merged["quality_entry_eligible"]
        & merged["final_score"].notna()
    ].sort_values("final_score", ascending=False)
    selected_tickers: set[str] = set(str(ticker) for ticker in eligible.head(top_n)["ticker"].tolist())
    merged["defensive_entry_signal"] = [str(ticker) in selected_tickers for ticker in merged["ticker"]]

    allocation: WeightAllocation = _allocation_for_selected_count(len(selected_tickers), min_full_positions)
    merged["target_weight"] = np.where(
        merged["defensive_entry_signal"],
        allocation["stock_weight"],
        0.0,
    )
    merged["cash_weight"] = allocation["cash_weight"]
    merged["final_rank"] = merged["final_score"].rank(ascending=False, method="first")
    merged["status"] = [_status_for_latest_row(row) for _, row in merged.iterrows()]
    merged["reason"] = [_reason_for_latest_row(row) for _, row in merged.iterrows()]

    output_columns: list[str] = [
        "signal_date",
        "ticker",
        "sector",
        "final_rank",
        "price_rank",
        "status",
        "defensive_entry_signal",
        "target_weight",
        "cash_weight",
        "final_score",
        "price_composite_score",
        "quality_score",
        "quality_threshold",
        "quality_pass",
        "quality_data_available",
        "quality_factor_count",
        "momentum_12_1",
        "annualized_volatility",
        "volatility_threshold",
        "latest_price",
        "sma",
        "positive_momentum",
        "above_sma",
        "volatility_ok",
        "price_eligible",
        "return_on_equity",
        "profit_margin",
        "free_cash_flow_yield",
        "debt_to_market_cap",
        "missing_quality_fields",
        "reason",
    ]
    return merged.loc[:, output_columns].sort_values(
        ["defensive_entry_signal", "final_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def save_strategy_outputs(
    strategy_returns: pd.DataFrame,
    price_scores: pd.DataFrame,
    weights: pd.DataFrame,
    selection: pd.DataFrame,
    strategy_returns_path: Path,
    price_scores_path: Path,
    weights_path: Path,
    selection_path: Path,
) -> None:
    strategy_returns_path.parent.mkdir(parents=True, exist_ok=True)
    price_scores_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_returns.to_parquet(strategy_returns_path)
    price_scores.to_parquet(price_scores_path)
    weights.to_parquet(weights_path)
    selection.to_csv(selection_path, index=False)


def run_strategy() -> LowVolQualityPaths:
    paths: LowVolQualityPaths = strategy_paths(
        LOW_VOL_QUALITY_DATA_DIR,
        LOW_VOL_QUALITY_RESULTS_DIR,
    )
    fetch_prices(
        LOW_VOL_QUALITY_TICKERS,
        START_DATE,
        END_DATE,
        paths["raw_prices_path"],
        YFINANCE_RETRY_COUNT,
        YFINANCE_RETRY_SLEEP_SECONDS,
    )
    prices: pd.DataFrame = clean_prices(paths["raw_prices_path"], paths["prices_path"], 0.95)
    stock_returns: pd.DataFrame = compute_and_save_returns(
        paths["prices_path"],
        paths["stock_returns_path"],
    )
    momentum_scores: pd.DataFrame = build_and_save_momentum_scores(
        paths["prices_path"],
        paths["momentum_scores_path"],
        paths["month_end_prices_path"],
    )
    price_scores: pd.DataFrame = build_price_defensive_scores(
        prices,
        stock_returns,
        momentum_scores,
        tuple(str(column) for column in prices.columns),
        LOW_VOL_QUALITY_SMA_WINDOW,
        LOW_VOL_QUALITY_VOL_WINDOW,
        LOW_VOL_QUALITY_MIN_MOMENTUM,
        LOW_VOL_QUALITY_MAX_VOL_QUANTILE,
        LOW_VOL_QUALITY_PRICE_WEIGHT,
        1.0 - LOW_VOL_QUALITY_PRICE_WEIGHT,
    )
    strategy_returns: pd.DataFrame = add_cash_returns(stock_returns, LOW_VOL_QUALITY_CASH_TICKER)
    weights: pd.DataFrame = build_defensive_weights(
        price_scores,
        strategy_returns,
        LOW_VOL_QUALITY_TOP_N,
        LOW_VOL_QUALITY_MIN_FULL_POSITIONS,
        LOW_VOL_QUALITY_CASH_TICKER,
    )
    quality_snapshot: pd.DataFrame = fetch_and_save_quality_snapshot(
        tuple(str(column) for column in prices.columns),
        paths["quality_snapshot_path"],
        YFINANCE_RETRY_COUNT,
        YFINANCE_RETRY_SLEEP_SECONDS,
        LOW_VOL_QUALITY_MIN_FACTOR_COUNT,
        LOW_VOL_QUALITY_PASS_QUANTILE,
        LOW_VOL_QUALITY_MIN_COVERAGE,
    )
    selection: pd.DataFrame = build_latest_low_vol_quality_selection(
        price_scores,
        quality_snapshot,
        LOW_VOL_QUALITY_TOP_N,
        LOW_VOL_QUALITY_MIN_FULL_POSITIONS,
        LOW_VOL_QUALITY_PRICE_WEIGHT,
        LOW_VOL_QUALITY_QUALITY_WEIGHT,
    )
    save_strategy_outputs(
        strategy_returns,
        price_scores,
        weights,
        selection,
        paths["strategy_returns_path"],
        paths["price_scores_path"],
        paths["weights_path"],
        paths["selection_path"],
    )

    for cost_bps in COST_BPS_GRID:
        out_dir: Path = paths["results_dir"] / f"cost_{cost_bps}bps"
        run_backtest_from_paths(
            paths["weights_path"],
            paths["strategy_returns_path"],
            cost_bps,
            out_dir,
        )

    export_cost_grid_report(paths["results_dir"], COST_BPS_GRID, 0.0)
    print(f"Saved low-vol quality selection: {paths['selection_path']}")
    print("Low-vol quality defensive candidates:")
    print(
        selection.loc[
            selection["defensive_entry_signal"],
            ["ticker", "sector", "final_rank", "target_weight", "quality_score", "reason"],
        ].to_string(index=False)
    )
    return paths


def main() -> None:
    run_strategy()


if __name__ == "__main__":
    main()
