from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd

from ..backtest import run_backtest_from_paths
from ..config import (
    COST_BPS_GRID,
    ETF_DUAL_MOMENTUM_DATA_DIR,
    ETF_DUAL_MOMENTUM_DEFENSIVE_TICKER,
    ETF_DUAL_MOMENTUM_LOOKBACK_MONTHS,
    ETF_DUAL_MOMENTUM_RANKING_FILE,
    ETF_DUAL_MOMENTUM_RESULTS_DIR,
    ETF_DUAL_MOMENTUM_RISK_TICKERS,
    ETF_DUAL_MOMENTUM_SIGNALS_FILE,
    ETF_DUAL_MOMENTUM_TICKERS,
    ETF_DUAL_MOMENTUM_WEIGHTS_FILE,
    PRICES_FILE,
    RAW_PRICES_FILE,
    RETURNS_FILE,
    START_DATE,
    END_DATE,
    YFINANCE_RETRY_COUNT,
    YFINANCE_RETRY_SLEEP_SECONDS,
)
from ..data_clean import clean_prices
from ..data_fetch import fetch_prices
from ..evaluate import export_cost_grid_report
from ..returns import compute_and_save_returns
from ..signals.momentum_12_1 import make_month_end_prices


class EtfDualMomentumPaths(TypedDict):
    raw_prices_path: Path
    prices_path: Path
    returns_path: Path
    weights_path: Path
    signals_path: Path
    ranking_path: Path
    results_dir: Path


def strategy_paths(data_dir: Path, results_dir: Path) -> EtfDualMomentumPaths:
    processed_dir: Path = data_dir / "Processed"
    raw_dir: Path = data_dir / "Raw"
    return {
        "raw_prices_path": raw_dir / RAW_PRICES_FILE,
        "prices_path": processed_dir / PRICES_FILE,
        "returns_path": processed_dir / RETURNS_FILE,
        "weights_path": processed_dir / ETF_DUAL_MOMENTUM_WEIGHTS_FILE,
        "signals_path": results_dir / ETF_DUAL_MOMENTUM_SIGNALS_FILE,
        "ranking_path": results_dir / ETF_DUAL_MOMENTUM_RANKING_FILE,
        "results_dir": results_dir,
    }


def momentum_score(month_end_prices: pd.DataFrame, lookback_months: int) -> pd.DataFrame:
    if lookback_months <= 0:
        raise ValueError(f"lookback_months must be positive. Received: {lookback_months}.")

    scores: pd.DataFrame = month_end_prices / month_end_prices.shift(lookback_months) - 1.0
    return scores


def _validate_asset_columns(
    prices: pd.DataFrame,
    tickers: tuple[str, ...],
    risk_tickers: tuple[str, ...],
    defensive_ticker: str,
) -> None:
    missing_tickers: list[str] = sorted(set(tickers).difference(set(prices.columns)))
    missing_risk_tickers: list[str] = sorted(set(risk_tickers).difference(set(prices.columns)))
    missing_defensive: bool = defensive_ticker not in set(prices.columns)
    if missing_tickers or missing_risk_tickers or missing_defensive:
        raise ValueError(
            "ETF dual momentum prices are missing required assets; "
            f"missing_tickers={missing_tickers}; "
            f"missing_risk_tickers={missing_risk_tickers}; "
            f"defensive_ticker={defensive_ticker}; "
            f"missing_defensive={missing_defensive}; "
            f"available_columns={list(prices.columns)}."
        )


def _select_asset(
    score_row: pd.Series,
    risk_tickers: tuple[str, ...],
    defensive_ticker: str,
) -> tuple[str, float, bool]:
    risk_scores: pd.Series = score_row.loc[list(risk_tickers)].dropna().astype(float)
    if risk_scores.empty:
        raise ValueError(f"Risk score row has no valid values. date={score_row.name}.")

    ranked_scores: pd.Series = risk_scores.sort_values(ascending=False)
    top_ticker: str = str(ranked_scores.index[0])
    top_score: float = float(ranked_scores.iloc[0])
    if top_score > 0.0:
        return top_ticker, top_score, False

    return defensive_ticker, top_score, True


def build_dual_momentum_outputs(
    prices: pd.DataFrame,
    tickers: tuple[str, ...],
    risk_tickers: tuple[str, ...],
    defensive_ticker: str,
    lookback_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_asset_columns(prices, tickers, risk_tickers, defensive_ticker)

    month_end_prices: pd.DataFrame = make_month_end_prices(prices.loc[:, list(tickers)])
    scores: pd.DataFrame = momentum_score(month_end_prices, lookback_months)
    valid_scores: pd.DataFrame = scores.dropna(subset=list(risk_tickers), how="any")
    if valid_scores.empty:
        raise ValueError(
            "No valid ETF dual momentum signal rows after lookback calculation; "
            f"lookback_months={lookback_months}; risk_tickers={risk_tickers}."
        )

    signal_records: list[dict[str, object]] = []
    weight_records: list[dict[str, float]] = []

    for signal_date, score_row in valid_scores.iterrows():
        selected_ticker, selected_score, defensive_signal = _select_asset(
            score_row,
            risk_tickers,
            defensive_ticker,
        )
        risk_scores: pd.Series = score_row.loc[list(risk_tickers)].dropna().astype(float)
        best_risk_ticker: str = str(risk_scores.sort_values(ascending=False).index[0])
        weight_record: dict[str, float] = {ticker: 0.0 for ticker in tickers}
        weight_record[selected_ticker] = 1.0

        signal_records.append(
            {
                "signal_date": pd.Timestamp(signal_date).date().isoformat(),
                "selected_ticker": selected_ticker,
                "selected_score": selected_score,
                "best_risk_ticker": best_risk_ticker,
                "best_risk_score": selected_score,
                "defensive_signal": defensive_signal,
            }
        )
        weight_records.append(weight_record)

    signals: pd.DataFrame = pd.DataFrame(signal_records)
    weights: pd.DataFrame = pd.DataFrame(weight_records, index=valid_scores.index)
    latest_signal_date: pd.Timestamp = pd.Timestamp(valid_scores.index.max())
    latest_risk_scores: pd.Series = valid_scores.loc[latest_signal_date, list(risk_tickers)].astype(float)
    latest_selected: str = str(signals.iloc[-1]["selected_ticker"])
    ranking: pd.DataFrame = pd.DataFrame(
        {
            "signal_date": latest_signal_date.date().isoformat(),
            "ticker": latest_risk_scores.sort_values(ascending=False).index,
            "rank": range(1, len(latest_risk_scores) + 1),
            f"momentum_{lookback_months}m": latest_risk_scores.sort_values(ascending=False).values,
        }
    )
    ranking["selected"] = ranking["ticker"] == latest_selected
    ranking["target_weight"] = ranking["selected"].astype(float)
    ranking["role"] = "risk_asset"
    if latest_selected == defensive_ticker:
        defensive_row: pd.DataFrame = pd.DataFrame(
            {
                "signal_date": [latest_signal_date.date().isoformat()],
                "ticker": [defensive_ticker],
                "rank": [len(ranking) + 1],
                f"momentum_{lookback_months}m": [0.0],
                "selected": [True],
                "target_weight": [1.0],
                "role": ["defensive_asset"],
            }
        )
        ranking = pd.concat([ranking, defensive_row], ignore_index=True)

    return signals, weights, ranking


def save_dual_momentum_outputs(
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    ranking: pd.DataFrame,
    signals_path: Path,
    weights_path: Path,
    ranking_path: Path,
) -> None:
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(signals_path, index=False)
    weights.to_parquet(weights_path)
    ranking.to_csv(ranking_path, index=False)


def run_strategy() -> EtfDualMomentumPaths:
    paths: EtfDualMomentumPaths = strategy_paths(
        ETF_DUAL_MOMENTUM_DATA_DIR,
        ETF_DUAL_MOMENTUM_RESULTS_DIR,
    )
    fetch_prices(
        ETF_DUAL_MOMENTUM_TICKERS,
        START_DATE,
        END_DATE,
        paths["raw_prices_path"],
        YFINANCE_RETRY_COUNT,
        YFINANCE_RETRY_SLEEP_SECONDS,
    )
    prices: pd.DataFrame = clean_prices(paths["raw_prices_path"], paths["prices_path"], 0.95)
    compute_and_save_returns(paths["prices_path"], paths["returns_path"])
    signals, weights, ranking = build_dual_momentum_outputs(
        prices,
        ETF_DUAL_MOMENTUM_TICKERS,
        ETF_DUAL_MOMENTUM_RISK_TICKERS,
        ETF_DUAL_MOMENTUM_DEFENSIVE_TICKER,
        ETF_DUAL_MOMENTUM_LOOKBACK_MONTHS,
    )
    save_dual_momentum_outputs(
        signals,
        weights,
        ranking,
        paths["signals_path"],
        paths["weights_path"],
        paths["ranking_path"],
    )

    for cost_bps in COST_BPS_GRID:
        out_dir: Path = paths["results_dir"] / f"cost_{cost_bps}bps"
        run_backtest_from_paths(
            paths["weights_path"],
            paths["returns_path"],
            cost_bps,
            out_dir,
        )

    export_cost_grid_report(paths["results_dir"], COST_BPS_GRID, 0.0)
    print(f"Saved ETF dual momentum signals: {paths['signals_path']}")
    print(f"Saved ETF dual momentum ranking: {paths['ranking_path']}")
    return paths


def main() -> None:
    run_strategy()


if __name__ == "__main__":
    main()
