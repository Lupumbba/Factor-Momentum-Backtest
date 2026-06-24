from pathlib import Path

import pandas as pd

from .config import DATA_DIR_PROCESSED, PRICES_FILE, RETURNS_FILE


def compute_and_save_returns(prices_path: Path, out_path: Path) -> pd.DataFrame:
    prices: pd.DataFrame = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    returns: pd.DataFrame = prices.pct_change()
    returns = returns.dropna(how="all")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(out_path)
    print(f"Saved returns: {out_path}")
    print("Returns shape:", returns.shape)
    print(returns.head())
    print(returns.tail())
    return returns


def main() -> None:
    compute_and_save_returns(DATA_DIR_PROCESSED / PRICES_FILE, DATA_DIR_PROCESSED / RETURNS_FILE)

if __name__ == "__main__":
    main()
