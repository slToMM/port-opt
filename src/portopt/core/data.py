from __future__ import annotations
import pandas as pd
import yfinance as yf
import numpy as np
from typing import List, Union
from datetime import date

def fetch_prices(
    tickers: List[str],
    start: Union[str, date],
    end: Union[str, date],
    interval: str = "Daily",
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance via yfinance.
    Uses a per-ticker fallback if the multi-ticker call returns empty.
    """
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    yf_interval = interval_map.get(interval, "1d")

    # First try a multi-ticker download
    df = yf.download(
        tickers=tickers,
        start=str(start),
        end=str(end),
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out = None
    if isinstance(df.columns, pd.MultiIndex) and len(tickers) > 1 and len(df) > 0:
        # Build from multi-index result
        closes = {}
        for t in tickers:
            try:
                cols = df[t].columns
                col_name = "Close" if "Close" in cols else ("Adj Close" if "Adj Close" in cols else None)
                if col_name is None:
                    continue
                closes[t] = df[t][col_name]
            except Exception:
                continue
        if closes:
            out = pd.DataFrame(closes)

    elif len(tickers) == 1 and len(df) > 0:
        # Single ticker shape
        series = df["Close"] if "Close" in df.columns else df.get("Adj Close", None)
        if series is not None:
            out = pd.DataFrame(series).rename(columns={series.name: tickers[0]})

    # Fallback: loop per ticker
    if out is None or out.empty:
        frames = []
        for t in tickers:
            try:
                one = yf.download(
                    t, start=str(start), end=str(end),
                    interval=yf_interval, auto_adjust=True, progress=False
                )
                if len(one) == 0:
                    continue
                col_name = "Close" if "Close" in one.columns else ("Adj Close" if "Adj Close" in one.columns else None)
                if col_name is None:
                    continue
                s = one[col_name].rename(t)
                frames.append(s)
            except Exception:
                continue
        if frames:
            out = pd.concat(frames, axis=1)

    if out is None or out.empty:
        # Nothing succeeded
        return pd.DataFrame()

    out.index.name = "Date"
    return out.sort_index()


def align_and_clean(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    # drop tickers that are fully missing
    df = df.dropna(axis=1, how="all")
    # forward fill small gaps
    df = df.ffill(limit=3)
    # keep rows with enough data relative to number of assets
    if df.shape[1] >= 2:
        thresh = max(2, int(np.ceil(0.5 * df.shape[1])))  # 50% coverage threshold
        df = df.dropna(thresh=thresh)
    # drop any remaining rows that are fully NaN
    df = df.dropna(how="all")
    return df