from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Literal, Mapping, TypedDict

import numpy as np
import pandas as pd
import yfinance as yf


QualityDirection = Literal["higher", "lower"]


class FundamentalRecord(TypedDict):
    ticker: str
    retrieved_at: str
    quote_type: str
    sector: str
    industry: str
    market_cap: float
    return_on_equity: float
    return_on_assets: float
    profit_margin: float
    operating_margin: float
    free_cash_flow: float
    operating_cash_flow: float
    total_debt: float
    total_cash: float
    debt_to_equity: float
    free_cash_flow_yield: float
    debt_to_market_cap: float
    net_debt_to_market_cap: float


QUALITY_FACTOR_DIRECTIONS: Mapping[str, QualityDirection] = {
    "return_on_equity": "higher",
    "return_on_assets": "higher",
    "profit_margin": "higher",
    "operating_margin": "higher",
    "free_cash_flow_yield": "higher",
    "debt_to_market_cap": "lower",
    "net_debt_to_market_cap": "lower",
    "debt_to_equity": "lower",
}


def _numeric_or_nan(info: Mapping[str, object], key: str) -> float:
    value: object | None = info.get(key)
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        stripped: str = value.strip()
        if not stripped:
            return float("nan")
        try:
            return float(stripped)
        except ValueError:
            return float("nan")

    return float("nan")


def _text_or_empty(info: Mapping[str, object], key: str) -> str:
    value: object | None = info.get(key)
    if value is None:
        return ""
    return str(value)


def _ratio_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _ticker_info_error(
    info: object,
    ticker: str,
) -> RuntimeError | None:
    if not isinstance(info, dict):
        return RuntimeError(f"yfinance info for ticker={ticker} is not a dict: {type(info)}.")
    if not info:
        return RuntimeError(f"yfinance info for ticker={ticker} is empty.")
    return None


def fetch_ticker_info_with_retries(
    ticker: str,
    retry_count: int,
    retry_sleep_seconds: float,
) -> Mapping[str, object]:
    if retry_count <= 0:
        raise ValueError(f"retry_count must be positive. Received: {retry_count}.")

    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            info: object = yf.Ticker(ticker).get_info()
            validation_error: RuntimeError | None = _ticker_info_error(info, ticker)
            if validation_error is None:
                return info

            last_error = validation_error
            warnings.warn(
                "yfinance fundamentals validation failed; "
                f"ticker={ticker}; attempt={attempt}; retry_count={retry_count}; "
                f"error={validation_error}",
                RuntimeWarning,
                stacklevel=2,
            )
            if attempt < retry_count:
                time.sleep(retry_sleep_seconds)
        except Exception as exc:
            last_error = exc
            warnings.warn(
                "yfinance fundamentals fetch failed; "
                f"ticker={ticker}; attempt={attempt}; retry_count={retry_count}; "
                f"error_type={type(exc).__name__}; error={exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            if attempt < retry_count:
                time.sleep(retry_sleep_seconds)

    if last_error is None:
        raise RuntimeError(f"yfinance fundamentals fetch did not run for ticker={ticker}.")
    raise last_error


def _fundamental_record(
    ticker: str,
    info: Mapping[str, object],
    retrieved_at: str,
) -> FundamentalRecord:
    market_cap: float = _numeric_or_nan(info, "marketCap")
    free_cash_flow: float = _numeric_or_nan(info, "freeCashflow")
    total_debt: float = _numeric_or_nan(info, "totalDebt")
    total_cash: float = _numeric_or_nan(info, "totalCash")
    net_debt: float = total_debt - total_cash

    return {
        "ticker": ticker,
        "retrieved_at": retrieved_at,
        "quote_type": _text_or_empty(info, "quoteType"),
        "sector": _text_or_empty(info, "sector"),
        "industry": _text_or_empty(info, "industry"),
        "market_cap": market_cap,
        "return_on_equity": _numeric_or_nan(info, "returnOnEquity"),
        "return_on_assets": _numeric_or_nan(info, "returnOnAssets"),
        "profit_margin": _numeric_or_nan(info, "profitMargins"),
        "operating_margin": _numeric_or_nan(info, "operatingMargins"),
        "free_cash_flow": free_cash_flow,
        "operating_cash_flow": _numeric_or_nan(info, "operatingCashflow"),
        "total_debt": total_debt,
        "total_cash": total_cash,
        "debt_to_equity": _numeric_or_nan(info, "debtToEquity"),
        "free_cash_flow_yield": _ratio_or_nan(free_cash_flow, market_cap),
        "debt_to_market_cap": _ratio_or_nan(total_debt, market_cap),
        "net_debt_to_market_cap": _ratio_or_nan(net_debt, market_cap),
    }


def fetch_fundamental_snapshot(
    tickers: tuple[str, ...],
    retry_count: int,
    retry_sleep_seconds: float,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("Cannot fetch fundamentals for an empty ticker list.")

    retrieved_at: str = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    records: list[FundamentalRecord] = []
    for ticker in tickers:
        info: Mapping[str, object] = fetch_ticker_info_with_retries(
            ticker,
            retry_count,
            retry_sleep_seconds,
        )
        records.append(_fundamental_record(ticker, info, retrieved_at))

    return pd.DataFrame(records)


def _factor_rank(
    frame: pd.DataFrame,
    column: str,
    direction: QualityDirection,
) -> pd.Series:
    values: pd.Series = pd.to_numeric(frame[column], errors="coerce")
    if direction == "higher":
        return values.rank(pct=True)
    if direction == "lower":
        return (-values).rank(pct=True)

    raise ValueError(f"Unsupported quality direction: {direction}.")


def _missing_fields(row: pd.Series, factor_columns: tuple[str, ...]) -> str:
    missing: list[str] = [
        column
        for column in factor_columns
        if not np.isfinite(float(row[column]))
    ]
    return ",".join(missing)


def add_quality_scores(
    fundamentals: pd.DataFrame,
    min_factor_count: int,
    pass_quantile: float,
) -> pd.DataFrame:
    if fundamentals.empty:
        raise ValueError("Cannot score an empty fundamentals table.")
    if min_factor_count <= 0:
        raise ValueError(f"min_factor_count must be positive. Received: {min_factor_count}.")
    if pass_quantile <= 0.0 or pass_quantile >= 1.0:
        raise ValueError(f"pass_quantile must be between 0 and 1. Received: {pass_quantile}.")

    scored: pd.DataFrame = fundamentals.copy()
    factor_columns: tuple[str, ...] = tuple(QUALITY_FACTOR_DIRECTIONS.keys())
    missing_columns: list[str] = sorted(set(factor_columns).difference(set(scored.columns)))
    if missing_columns:
        raise ValueError(f"Fundamentals table is missing quality factor columns: {missing_columns}.")

    rank_columns: list[str] = []
    for column, direction in QUALITY_FACTOR_DIRECTIONS.items():
        rank_column: str = f"{column}_quality_rank"
        scored[rank_column] = _factor_rank(scored, column, direction)
        rank_columns.append(rank_column)

    scored["quality_factor_count"] = scored.loc[:, list(factor_columns)].notna().sum(axis=1)
    scored["quality_score"] = scored.loc[:, rank_columns].mean(axis=1, skipna=True)
    scored["quality_data_available"] = scored["quality_factor_count"] >= min_factor_count

    available_scores: pd.Series = scored.loc[scored["quality_data_available"], "quality_score"].dropna()
    if available_scores.empty:
        raise ValueError(
            "No tickers have enough quality factors; "
            f"min_factor_count={min_factor_count}; tickers={list(scored['ticker'])}."
        )

    threshold: float = float(available_scores.quantile(pass_quantile))
    scored["quality_threshold"] = threshold
    scored["quality_pass"] = scored["quality_data_available"] & (scored["quality_score"] >= threshold)
    scored["missing_quality_fields"] = [
        _missing_fields(row, factor_columns)
        for _, row in scored.iterrows()
    ]
    return scored


def validate_quality_coverage(
    scored: pd.DataFrame,
    min_coverage: float,
) -> None:
    if min_coverage <= 0.0 or min_coverage > 1.0:
        raise ValueError(f"min_coverage must be in (0, 1]. Received: {min_coverage}.")

    available_count: int = int(scored["quality_data_available"].sum())
    coverage: float = available_count / len(scored)
    if coverage < min_coverage:
        missing: pd.DataFrame = scored.loc[
            ~scored["quality_data_available"],
            ["ticker", "quality_factor_count", "missing_quality_fields"],
        ]
        raise ValueError(
            "Fundamental quality coverage is too low; "
            f"coverage={coverage:.3f}; min_coverage={min_coverage:.3f}; "
            f"available_count={available_count}; total={len(scored)}; "
            f"missing_rows={missing.to_dict(orient='records')}."
        )


def save_quality_snapshot(
    scored: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)


def fetch_and_save_quality_snapshot(
    tickers: tuple[str, ...],
    out_path: Path,
    retry_count: int,
    retry_sleep_seconds: float,
    min_factor_count: int,
    pass_quantile: float,
    min_coverage: float,
) -> pd.DataFrame:
    fundamentals: pd.DataFrame = fetch_fundamental_snapshot(
        tickers,
        retry_count,
        retry_sleep_seconds,
    )
    scored: pd.DataFrame = add_quality_scores(
        fundamentals,
        min_factor_count,
        pass_quantile,
    )
    validate_quality_coverage(scored, min_coverage)
    save_quality_snapshot(scored, out_path)
    print(f"Saved quality snapshot: {out_path}")
    print(
        scored[
            [
                "ticker",
                "sector",
                "quality_score",
                "quality_pass",
                "quality_factor_count",
            ]
        ].sort_values("quality_score", ascending=False).head(15).to_string(index=False)
    )
    return scored
