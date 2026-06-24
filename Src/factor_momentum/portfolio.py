from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR_PROCESSED,
    MOMENTUM_SCORES_FILE,
    RETURNS_FILE,
    SELECTION_TOP_N,
    WEIGHTS_FILE,
)


def _ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If columns are MultiIndex, keep the last level (often the ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    return df


def build_weights_topk_equal(
    mom_scores: pd.DataFrame,
    daily_returns: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Build a long-only Top-K equal-weight portfolio from momentum scores.

    Concept:
      - At each rebalance date (the dates in mom_scores), rank tickers by momentum.
      - Select the Top-K tickers (highest scores).
      - Assign equal weights to the selected tickers (1/K each; 0 for others).
      - Hold these weights every day until the next rebalance date (forward-fill).
      - Return a DAILY weights DataFrame aligned to daily_returns.index.

    Inputs:
      mom_scores:
        DataFrame indexed by rebalance dates (typically month-end trading days),
        columns = tickers, values = momentum scores (higher = better).
      daily_returns:
        DataFrame indexed by daily trading days,
        columns = tickers, values = daily returns.
      top_k:
        Number of tickers to hold at each rebalance.

    Output:
      weights:
        DataFrame indexed by daily trading days (same index as daily_returns),
        columns = tickers, values = portfolio weights (long-only, sums to ~1 when invested).
    """
    mom_scores = _ensure_flat_columns(mom_scores).sort_index()
    daily_returns = _ensure_flat_columns(daily_returns).sort_index()

    # Keeps only tickers that exist in both tables.
    tickers: pd.Index = mom_scores.columns.intersection(daily_returns.columns)
    if len(tickers) == 0:
        raise ValueError("No overlapping tickers between momentum scores and returns.")

    mom_scores = mom_scores[tickers]
    daily_returns = daily_returns[tickers]

    # Check whether rebalance dates align with trading dates.
    rebalance_dates: pd.Index = mom_scores.index.intersection(daily_returns.index)
    if len(rebalance_dates) == 0:
        raise ValueError("No momentum-score dates match daily returns dates. Check alignment.")

    w_rebal: pd.DataFrame = pd.DataFrame(0.0, index=rebalance_dates, columns=tickers)

    # For each rebalance date: select Top-K winners and assign equal weights.
    for d in rebalance_dates:
        scores: pd.Series = mom_scores.loc[d].dropna()
        if scores.empty:
            continue
        winners: pd.Index = scores.nlargest(min(top_k, len(scores))).index
        w_rebal.loc[d, winners] = 1.0 / len(winners)

    weights: pd.DataFrame = w_rebal.reindex(daily_returns.index).ffill().fillna(0.0)

    # Normalize to sum to 1 when invested
    s: pd.Series = weights.sum(axis=1)
    invested: pd.Series = s > 0
    weights.loc[invested] = weights.loc[invested].div(s[invested], axis=0)

    return weights


def build_and_save_weights(
    momentum_scores_path: Path,
    returns_path: Path,
    out_path: Path,
    top_k: int,
) -> pd.DataFrame:
    mom: pd.DataFrame = pd.read_parquet(momentum_scores_path)
    rets: pd.DataFrame = pd.read_parquet(returns_path)
    weights: pd.DataFrame = build_weights_topk_equal(mom, rets, top_k=top_k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(out_path)

    print(f"Saved: {out_path}")
    print("Last 5 days weight sums:")
    print(weights.sum(axis=1).tail())
    print("Last 5 days #positions:")
    print((weights > 0).sum(axis=1).tail())
    return weights


def main() -> None:
    build_and_save_weights(
        DATA_DIR_PROCESSED / MOMENTUM_SCORES_FILE,
        DATA_DIR_PROCESSED / RETURNS_FILE,
        DATA_DIR_PROCESSED / WEIGHTS_FILE,
        SELECTION_TOP_N,
    )
