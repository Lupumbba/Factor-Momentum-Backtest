from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MOMENTUM_SCORES_PATH,
    PRICES_PATH,
    RETURNS_PATH,
    RESULTS_DIR,
    SELECTION_FILE,
    SELECTION_HIGH_VOL_QUANTILE,
    SELECTION_MIN_MOMENTUM,
    SELECTION_SMA_WINDOW,
    SELECTION_TOP_N,
    SELECTION_VOL_WINDOW,
)


def _load_frame(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    frame: pd.DataFrame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)

    return frame.sort_index()


def _latest_signal_date(scores: pd.DataFrame) -> pd.Timestamp:
    valid_scores: pd.DataFrame = scores.dropna(how="all")
    if valid_scores.empty:
        raise ValueError("Momentum score table has no valid rows.")

    return pd.Timestamp(valid_scores.index.max())


def _validate_history(history: pd.DataFrame, window: int, label: str) -> None:
    if len(history) < window:
        raise ValueError(
            f"Not enough {label} history for window={window}. "
            f"Available rows={len(history)}."
        )


def _selection_reason(
    strategy_signal: bool,
    above_sma: bool,
    high_volatility: bool,
    top_n: int,
) -> str:
    if not strategy_signal:
        return f"outside_top_{top_n}"
    if not above_sma:
        return "strategy_signal_but_below_sma"
    if high_volatility:
        return "strategy_signal_but_high_volatility"
    return "conservative_entry_signal"


def build_latest_selection(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    top_n: int,
    min_momentum: float,
    sma_window: int,
    vol_window: int,
    high_vol_quantile: float,
) -> pd.DataFrame:
    signal_date: pd.Timestamp = _latest_signal_date(scores)
    ranked_scores: pd.Series = scores.loc[signal_date].dropna().sort_values(ascending=False)
    common_tickers: pd.Index = ranked_scores.index.intersection(prices.columns).intersection(daily_returns.columns)
    if common_tickers.empty:
        raise ValueError("No overlapping tickers between scores, prices, and returns.")

    ranked_scores = ranked_scores.loc[common_tickers].sort_values(ascending=False)
    price_history: pd.DataFrame = prices.loc[:signal_date, ranked_scores.index]
    return_history: pd.DataFrame = daily_returns.loc[:signal_date, ranked_scores.index]

    _validate_history(price_history, sma_window, "price")
    _validate_history(return_history, vol_window, "return")

    latest_prices: pd.Series = price_history.iloc[-1].astype(float)
    moving_average: pd.Series = price_history.rolling(sma_window).mean().iloc[-1].astype(float)
    annualized_volatility: pd.Series = (
        return_history.tail(vol_window).std(ddof=1).astype(float) * np.sqrt(252.0)
    )
    volatility_threshold: float = float(annualized_volatility.quantile(high_vol_quantile))

    selection: pd.DataFrame = pd.DataFrame(
        {
            "signal_date": signal_date.date().isoformat(),
            "ticker": ranked_scores.index,
            "rank": range(1, len(ranked_scores) + 1),
            "mom12_1": ranked_scores.astype(float).values,
            "latest_price": latest_prices.loc[ranked_scores.index].values,
            f"sma_{sma_window}": moving_average.loc[ranked_scores.index].values,
            f"annualized_vol_{vol_window}d": annualized_volatility.loc[ranked_scores.index].values,
            "high_vol_threshold": volatility_threshold,
        }
    )

    selection["above_sma"] = selection["latest_price"] > selection[f"sma_{sma_window}"]
    selection["high_volatility"] = selection[f"annualized_vol_{vol_window}d"] > volatility_threshold
    selection["strategy_signal"] = (selection["rank"] <= top_n) & (selection["mom12_1"] > min_momentum)
    selection["conservative_entry_signal"] = (
        selection["strategy_signal"] & selection["above_sma"] & ~selection["high_volatility"]
    )

    strategy_count: int = int(selection["strategy_signal"].sum())
    conservative_count: int = int(selection["conservative_entry_signal"].sum())
    strategy_weight: float = 1.0 / strategy_count if strategy_count > 0 else 0.0
    conservative_weight: float = 1.0 / conservative_count if conservative_count > 0 else 0.0

    selection["strategy_equal_weight"] = np.where(selection["strategy_signal"], strategy_weight, 0.0)
    selection["conservative_equal_weight"] = np.where(
        selection["conservative_entry_signal"], conservative_weight, 0.0
    )
    selection["reason"] = [
        _selection_reason(
            bool(row["strategy_signal"]),
            bool(row["above_sma"]),
            bool(row["high_volatility"]),
            top_n,
        )
        for _, row in selection.iterrows()
    ]

    return selection


def save_selection(selection: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out_path, index=False)


def main() -> None:
    scores: pd.DataFrame = _load_frame(MOMENTUM_SCORES_PATH, "momentum scores")
    prices: pd.DataFrame = _load_frame(PRICES_PATH, "processed prices")
    daily_returns: pd.DataFrame = _load_frame(RETURNS_PATH, "daily returns")

    selection: pd.DataFrame = build_latest_selection(
        scores,
        prices,
        daily_returns,
        SELECTION_TOP_N,
        SELECTION_MIN_MOMENTUM,
        SELECTION_SMA_WINDOW,
        SELECTION_VOL_WINDOW,
        SELECTION_HIGH_VOL_QUANTILE,
    )
    out_path: Path = Path(RESULTS_DIR) / SELECTION_FILE
    save_selection(selection, out_path)

    candidates: pd.DataFrame = selection[selection["conservative_entry_signal"]]
    print(f"Saved latest selection: {out_path}")
    print("Conservative entry candidates:")
    print(
        candidates[
            [
                "ticker",
                "rank",
                "mom12_1",
                "conservative_equal_weight",
                "reason",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
