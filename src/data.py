"""Step 1 — Data: download adjusted prices and compute daily log returns."""
import numpy as np
import pandas as pd

from src.config import DATA_DIR

TICKERS = ["SPY", "TLT", "GLD"]  # US stocks, long-term Treasuries, gold
START = "2005-01-01"
END = "2026-09-01"  # fixed end date so every rerun uses identical data

RAW_PATH = DATA_DIR / "raw" / "prices.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "returns.csv"


def download_prices(tickers=TICKERS, start=START, end=END, force=False):
    """Return adjusted close prices. Loads the saved CSV if it exists, else downloads."""
    if RAW_PATH.exists() and not force:
        return pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)

    import yfinance as yf  # imported here so the rest of the module works offline

    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = df["Close"][tickers].dropna()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(RAW_PATH)
    return prices


def compute_returns(prices, save=True):
    """Daily log returns: r_t = ln(P_t / P_{t-1})."""
    returns = np.log(prices / prices.shift(1)).dropna()

    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        returns.to_csv(PROCESSED_PATH)
    return returns


def summary_stats(returns, periods_per_year=252):
    """Annualized return, volatility and Sharpe (risk-free rate = 0) per asset."""
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    return pd.DataFrame({
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol,
    })
