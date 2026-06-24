from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import (
    DATA_DIR_RAW,
    END_DATE,
    RAW_PRICES_FILE,
    START_DATE,
    TICKERS,
    YFINANCE_RETRY_COUNT,
    YFINANCE_RETRY_SLEEP_SECONDS,
)


def _download_validation_error(
    frame: pd.DataFrame | None,
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str | None,
) -> RuntimeError | None:
    if frame is None or frame.empty:
        return RuntimeError(
            f"yfinance returned empty data for tickers={list(tickers)}, "
            f"start={start_date}, end={end_date}."
        )
    if not isinstance(frame.columns, pd.MultiIndex):
        return RuntimeError(f"Expected MultiIndex columns but got: {type(frame.columns)}")

    level0: pd.Index = frame.columns.get_level_values(0)
    if "Adj Close" not in set(level0):
        return RuntimeError("Adj Close not found. See printed 'Fields returned' above.")

    adjusted_close: pd.DataFrame = frame["Adj Close"]
    returned_tickers: set[str] = {str(ticker) for ticker in adjusted_close.columns}
    requested_tickers: set[str] = set(tickers)
    missing_tickers: list[str] = sorted(requested_tickers.difference(returned_tickers))
    empty_tickers: list[str] = sorted(
        ticker
        for ticker in tickers
        if ticker in adjusted_close.columns and adjusted_close[ticker].dropna().empty
    )
    if missing_tickers or empty_tickers:
        return RuntimeError(
            "yfinance returned a partial download; "
            f"missing_tickers={missing_tickers}; empty_tickers={empty_tickers}; "
            f"tickers={list(tickers)}; start={start_date}; end={end_date}."
        )

    return None


def _download_with_retries(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str | None,
    retry_count: int,
    retry_sleep_seconds: float,
) -> pd.DataFrame:
    if retry_count <= 0:
        raise ValueError(f"retry_count must be positive. Received: {retry_count}.")

    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            df: pd.DataFrame = yf.download(
                list(tickers),
                start=start_date,
                end=end_date,
                progress=False,
                group_by="column",
                auto_adjust=False,
                actions=False,
            )
            validation_error: RuntimeError | None = _download_validation_error(
                df,
                tickers,
                start_date,
                end_date,
            )
            if validation_error is None:
                return df

            last_error = validation_error
            warnings.warn(
                "yfinance download validation failed; "
                f"attempt={attempt}; retry_count={retry_count}; "
                f"error={validation_error}",
                RuntimeWarning,
                stacklevel=2,
            )
            if attempt < retry_count:
                time.sleep(retry_sleep_seconds)
        except Exception as exc:
            last_error = exc
            warnings.warn(
                "yfinance download failed; "
                f"attempt={attempt}; retry_count={retry_count}; "
                f"tickers={list(tickers)}; start={start_date}; end={end_date}; "
                f"error_type={type(exc).__name__}; error={exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            if attempt < retry_count:
                time.sleep(retry_sleep_seconds)

    if last_error is None:
        raise RuntimeError(
            f"yfinance download did not run for tickers={list(tickers)}, "
            f"start={start_date}, end={end_date}."
        )
    raise last_error


def fetch_prices(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str | None,
    out_path: Path,
    retry_count: int,
    retry_sleep_seconds: float,
) -> pd.DataFrame:
    df: pd.DataFrame = _download_with_retries(
        tickers,
        start_date,
        end_date,
        retry_count,
        retry_sleep_seconds,
    )
    level0: pd.Index = df.columns.get_level_values(0)
    print("Fields returned:", sorted(set(level0)))

    prices: pd.DataFrame = df["Adj Close"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path)
    print(f"Saved: {out_path}")
    print(prices.head())
    return prices


def main() -> None:
    out_path: Path = DATA_DIR_RAW / RAW_PRICES_FILE
    fetch_prices(
        TICKERS,
        START_DATE,
        END_DATE,
        out_path,
        YFINANCE_RETRY_COUNT,
        YFINANCE_RETRY_SLEEP_SECONDS,
    )


if __name__ == "__main__":
    main()
