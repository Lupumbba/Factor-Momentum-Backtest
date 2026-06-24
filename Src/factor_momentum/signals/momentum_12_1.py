from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import (
    DATA_DIR_PROCESSED,
    MOMENTUM_SCORES_FILE,
    MONTH_END_PRICES_FILE,
    PRICES_FILE,
)


def load_prices_wide(path: Path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df[~df.index.duplicated(keep="last")]
    return df


def make_month_end_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Month-end prices derived from daily prices, indexed by the last trading day each month.
    """
    sorted_prices: pd.DataFrame = daily_prices.sort_index()
    month_periods: pd.PeriodIndex = sorted_prices.index.to_period("M")
    month_end_prices: pd.DataFrame = sorted_prices.groupby(month_periods).tail(1)
    return month_end_prices


def momentum_12_1(month_end_prices: pd.DataFrame) -> pd.DataFrame:
    """
    12–1 momentum at month-end t:
      mom_t = P_{t-1} / P_{t-12} - 1
    Implemented as shift(1)/shift(12) - 1 using month-end prices.
    """
    p_t_1 = month_end_prices.shift(1)
    p_t_12 = month_end_prices.shift(12)
    mom = (p_t_1 / p_t_12) - 1.0
    return mom


def coverage_report(scores: pd.DataFrame) -> pd.DataFrame:
    """
    For each month-end, how many tickers have a valid score and what % coverage that is.
    """
    n_tickers = scores.shape[1]
    valid = scores.notna().sum(axis=1)
    pct = valid / n_tickers
    rep = pd.DataFrame({"valid_tickers": valid, "pct_coverage": pct})
    return rep


def build_and_save_momentum_scores(
    cleaned_prices_path: Path,
    out_scores_path: Path,
    out_monthly_prices_path: Path,
) -> pd.DataFrame:
    daily: pd.DataFrame = load_prices_wide(cleaned_prices_path)
    month_end: pd.DataFrame = make_month_end_prices(daily)
    scores: pd.DataFrame = momentum_12_1(month_end)

    out_monthly_prices_path.parent.mkdir(parents=True, exist_ok=True)
    month_end.to_parquet(out_monthly_prices_path)
    scores.to_parquet(out_scores_path)

    rep: pd.DataFrame = coverage_report(scores)

    print("Saved month-end prices:", out_monthly_prices_path)
    print("Saved 12–1 momentum scores:", out_scores_path)
    print("\nCoverage (last 12 months):")
    print(rep.tail(12).to_string())

    print("\nFirst 15 rows non-null counts:")
    print(scores.notna().sum(axis=1).head(15).to_string())
    return scores


def main() -> None:
    build_and_save_momentum_scores(
        DATA_DIR_PROCESSED / PRICES_FILE,
        DATA_DIR_PROCESSED / MOMENTUM_SCORES_FILE,
        DATA_DIR_PROCESSED / MONTH_END_PRICES_FILE,
    )


if __name__ == "__main__":
    main()
