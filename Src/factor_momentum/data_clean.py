from pathlib import Path

import pandas as pd

from .config import DATA_DIR_RAW, DATA_DIR_PROCESSED, PRICES_FILE, RAW_PRICES_FILE


def clean_prices(raw_prices_path: Path, out_path: Path, min_coverage: float) -> pd.DataFrame:
    prices: pd.DataFrame = pd.read_parquet(raw_prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    prices = prices[~prices.index.duplicated(keep="first")]

    coverage: pd.Series = prices.notna().mean(axis=0)
    keep: pd.Index = coverage[coverage >= min_coverage].index
    if len(keep) == 0:
        raise ValueError(
            f"No tickers passed min_coverage={min_coverage} for raw_prices_path={raw_prices_path}."
        )

    prices = prices[keep]
    prices = prices.ffill()
    prices = prices.dropna(how="all")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path)
    print(f"Saved processed prices: {out_path}")
    print("Final shape:", prices.shape)
    print(prices.head())
    print(prices.tail())
    return prices


def main() -> None:
    clean_prices(DATA_DIR_RAW / RAW_PRICES_FILE, DATA_DIR_PROCESSED / PRICES_FILE, 0.95)

if __name__ == "__main__":
    main()
